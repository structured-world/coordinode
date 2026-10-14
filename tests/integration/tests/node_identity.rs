//! Node identity as an operator meets it on real processes: a configuration
//! the server cannot honour refuses the start, renewed credentials take
//! effect without a restart while a broken set is refused, a member whose
//! certificate lapses is cut off, and a port that requires client
//! certificates serves only the clients that present one.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant, SystemTime};

use coordinode_integration::harness::{
    CoordinodeProcess, free_port, member_addr, serve_args_expecting_refusal,
};
use coordinode_integration::proto::admin::{
    GetClusterStatusRequest, JoinNodeRequest, NodeRole,
    cluster_service_client::ClusterServiceClient,
};
use coordinode_integration::proto::query::ExecuteCypherRequest;
use coordinode_integration::proto::query::cypher_service_client::CypherServiceClient;
use coordinode_integration::proto::replication::raft_service_client::RaftServiceClient;
use coordinode_integration::proto::replication::{RaftPayload, ReadConcern};
use coordinode_test_fixtures::{TestPki, tls_client_channel};
use tonic::transport::Channel;

/// `path` as a quoted YAML scalar.
fn quoted(path: &std::path::Path) -> String {
    serde_json::to_string(&path.display().to_string()).expect("a string")
}

/// Write `yaml` as the config file `name` in `dir`; returns its path as an
/// argument.
fn config_file(dir: &std::path::Path, name: &str, yaml: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, yaml).expect("write the config");
    path.display().to_string()
}

/// Each configuration the server cannot honour refuses the start with a
/// message naming what to fix, instead of serving without the protection
/// the operator asked for.
#[tokio::test]
async fn a_node_identity_the_server_cannot_honour_refuses_the_start() {
    let pki = TestPki::new();
    let dir = tempfile::tempdir().expect("dir");
    let (cert, key) = pki.write_credentials(dir.path(), "node", "spiffe://cluster.test/n1");
    let bundle = pki.write_bundle(dir.path());
    let data = |name: &str| dir.path().join(name).display().to_string();
    let peer = member_addr(free_port());

    // Peers without an identity: there is no plain-text peer channel.
    let (status, printed) =
        serve_args_expecting_refusal(&["--node-id", "1", "--peers", &peer, "--data", &data("a")])
            .await;
    assert!(!status.success(), "{printed}");
    assert!(printed.contains("needs node_identity"), "{printed}");

    // A second server certificate beside the node identity.
    let identity = format!(
        "node_identity:\n  cert: {}\n  key: {}\n  trust_bundles:\n    cluster.test: {}\n",
        quoted(&cert),
        quoted(&key),
        quoted(&bundle)
    );
    let both = config_file(
        dir.path(),
        "both.conf",
        &format!(
            "{identity}tls_cert: {}\ntls_key: {}\n",
            quoted(&cert),
            quoted(&key)
        ),
    );
    let (status, printed) = serve_args_expecting_refusal(&[
        "--config",
        &both,
        "--node-id",
        "1",
        "--peers",
        &peer,
        "--data",
        &data("b-data"),
    ])
    .await;
    assert!(!status.success(), "{printed}");
    assert!(printed.contains("tls_cert / tls_key"), "{printed}");

    // Client certificates required with nothing to verify them against.
    let mtls = config_file(dir.path(), "mtls.conf", "tls_require_client_auth: true\n");
    let (status, printed) =
        serve_args_expecting_refusal(&["--config", &mtls, "--data", &data("c-data")]).await;
    assert!(!status.success(), "{printed}");
    assert!(printed.contains("requires tls_ca"), "{printed}");

    // A certificate the configured bundle does not trust: the same trust
    // domain, another CA.
    let impostor = TestPki::new();
    let elsewhere = dir.path().join("impostor");
    std::fs::create_dir_all(&elsewhere).expect("a directory");
    let foreign = impostor.write_bundle(&elsewhere);
    let untrusted = config_file(
        dir.path(),
        "untrusted.conf",
        &format!(
            "node_identity:\n  cert: {}\n  key: {}\n  trust_bundles:\n    cluster.test: {}\n",
            quoted(&cert),
            quoted(&key),
            quoted(&foreign)
        ),
    );
    let (status, printed) = serve_args_expecting_refusal(&[
        "--config",
        &untrusted,
        "--node-id",
        "1",
        "--peers",
        &peer,
        "--data",
        &data("d-data"),
    ])
    .await;
    assert!(!status.success(), "{printed}");
    assert!(printed.contains("node_identity"), "{printed}");
}

/// On a running cluster: a reload that finds credentials of another
/// identity is refused and the member goes on under its last valid set; a
/// member whose certificate lapses is cut off from its group; one whose
/// certificate was renewed in place and reloaded before it lapsed (by
/// `SIGHUP` where there are signals, by the ops endpoint elsewhere) stays in,
/// with no restart.
#[tokio::test(flavor = "multi_thread")]
async fn renewal_takes_effect_live_and_a_lapsed_member_is_cut_off() {
    let (p1, p2, p3) = (free_port(), free_port(), free_port());
    // Long enough to form the group on a slow machine; the members' first
    // connections are made under these certificates.
    let lapse = SystemTime::now() + Duration::from_secs(30);
    let n1 = CoordinodeProcess::start_cluster_member(1, p1, &[p2, p3]).await;
    let n2 = CoordinodeProcess::start_cluster_member_until(2, p2, &[p1, p3], lapse).await;
    let n3 = CoordinodeProcess::start_cluster_member_until(3, p3, &[p1, p2], lapse).await;
    let mut leader = n1.cluster_client().await;
    for (id, member) in [(2, &n2), (3, &n3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .expect("JoinNode must be accepted");
        wait_for_voters(&mut leader, usize::try_from(id).expect("small")).await;
    }

    // Credentials proving another member's identity are refused; the
    // leader keeps proving its own and the group keeps replicating.
    n1.install_identity_of(p2);
    assert_eq!(n1.reload(), 500, "another identity is not a renewal");
    write_on(&n1, "CREATE (:Lapse {v: 1})").await;
    for member in [&n2, &n3] {
        wait_for_rows(member, 1).await;
    }
    n1.renew_identity(None);
    assert_eq!(n1.reload(), 200, "its own identity renewed");

    // Node 2 renews in place; node 3 does not.
    n2.renew_identity(None);
    #[cfg(unix)]
    n2.send_sighup();
    #[cfg(not(unix))]
    assert_eq!(n2.reload(), 200);
    let left = lapse
        .duration_since(SystemTime::now())
        .expect("the group formed before the certificates lapsed");
    assert!(
        left > Duration::from_secs(3),
        "too little time left to tell renewal from lapse: {left:?}"
    );

    // Past the old certificates' end, a write commits with nodes 1 and 2.
    tokio::time::sleep(left + Duration::from_secs(3)).await;
    write_on(&n1, "CREATE (:Lapse {v: 2})").await;
    wait_for_rows(&n2, 2).await;
    // Node 3 proves an expired identity: no member takes its connections or
    // keeps one to it, so nothing reaches it.
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(rows_on(&n3).await, Some(1), "the lapsed member was cut off");
}

/// A member's port that requires client certificates serves a client that
/// presents one from its client CA, refuses one in plain text, one without a
/// certificate and one from another CA, and never treats a client
/// certificate as a node.
#[tokio::test(flavor = "multi_thread")]
async fn a_port_that_requires_client_certificates_serves_only_those_clients() {
    let clients = TestPki::with_trust_domain("clients.test");
    let dir = tempfile::tempdir().expect("dir");
    let bundle = clients.write_bundle(dir.path());
    let port = free_port();
    let member = CoordinodeProcess::start_cluster_member_with_settings(
        1,
        port,
        &[free_port()],
        &format!(
            "tls_ca: {}\ntls_require_client_auth: true\n",
            quoted(&bundle)
        ),
    )
    .await;
    let addr = member_addr(port);
    let app = || {
        tls_client_channel(
            &addr,
            Some(clients.credentials("spiffe://clients.test/app", None)),
        )
    };

    // Served once the member leads.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let answer = CypherServiceClient::new(app())
            .execute_cypher(ExecuteCypherRequest {
                query: "RETURN 1".to_string(),
                ..Default::default()
            })
            .await;
        if answer.is_ok() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "a client with a certificate was never served: {answer:?}"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Its certificate makes it a client, not a node.
    let vote = RaftServiceClient::new(app())
        .vote(RaftPayload::default())
        .await
        .expect_err("a client certificate is not a node");
    assert_eq!(vote.code(), tonic::Code::Unauthenticated, "{vote:?}");

    let refused = |channel: Channel, who: &'static str| async move {
        let answer = CypherServiceClient::new(channel)
            .execute_cypher(ExecuteCypherRequest {
                query: "RETURN 1".to_string(),
                ..Default::default()
            })
            .await;
        assert!(answer.is_err(), "{who} was served");
    };
    refused(
        Channel::from_shared(member.endpoint())
            .expect("endpoint")
            .connect_lazy(),
        "a plain-text client",
    )
    .await;
    refused(
        tls_client_channel(&addr, None),
        "a client without a certificate",
    )
    .await;
    let stranger = TestPki::with_trust_domain("clients.test");
    refused(
        tls_client_channel(
            &addr,
            Some(stranger.credentials("spiffe://clients.test/app", None)),
        ),
        "a client of another CA",
    )
    .await;
}

/// Write `query` through `node`, retrying while the group settles.
async fn write_on(node: &CoordinodeProcess, query: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let answer = node
            .cypher_client()
            .await
            .execute_cypher(ExecuteCypherRequest {
                query: query.to_string(),
                ..Default::default()
            })
            .await;
        if answer.is_ok() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{query} never committed: {answer:?}"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// The `:Lapse` nodes `node` holds itself; `None` when it does not answer.
async fn rows_on(node: &CoordinodeProcess) -> Option<usize> {
    node.cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:Lapse) RETURN n.v".to_string(),
            read_preference: 5, // NEAREST
            read_concern: Some(ReadConcern {
                level: 1, // LOCAL
                after_index: 0,
                at_timestamp: 0,
            }),
            ..Default::default()
        })
        .await
        .ok()
        .map(|r| r.into_inner().rows.len())
}

/// Wait until `node` holds `expected` `:Lapse` nodes.
async fn wait_for_rows(node: &CoordinodeProcess, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let rows = rows_on(node).await;
        if rows == Some(expected) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "member on {} never held {expected}: {rows:?}",
            node.port
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Poll the leader until it reports `expected` members, all voters.
async fn wait_for_voters(client: &mut ClusterServiceClient<Channel>, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        if let Ok(resp) = client.get_cluster_status(GetClusterStatusRequest {}).await {
            let s = resp.into_inner();
            let voters = s
                .nodes
                .iter()
                .filter(|n| n.role != NodeRole::Learner as i32)
                .count();
            if s.nodes.len() == expected && voters == expected {
                return;
            }
        }
        assert!(
            Instant::now() < deadline,
            "the group did not reach {expected} voters"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}
