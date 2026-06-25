#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end inter-node mutual-TLS integration test.
//!
//! Brings up a 2-node embedded Raft cluster where every node serves the
//! `RaftService` over mutual TLS (server cert + required client cert, both
//! chaining to one in-memory CA) and every outbound peer dial presents the
//! node's certificate via the process-wide client TLS config. A proposal on the
//! leader must replicate to the follower through the encrypted, mutually
//! authenticated channel — proving the full TLS + compression-codec wire path,
//! not just an isolated rustls handshake (that is covered in `coordinode-wire`).
//!
//! Single process, real localhost ports (bind-to-zero), no Docker. nextest runs
//! each test in its own process, so the process-global client TLS set here does
//! not leak into other tests.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::{RaftGrpcHandler, RaftNode};
use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use rcgen::{BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair};
use tonic::transport::{Certificate, Identity, Server, ServerTlsConfig};

const TEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Allocate a free localhost port (bind-to-zero, then release).
fn alloc_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

/// Generate an in-memory CA and one node certificate (PEM) signed by it. The
/// node cert carries the `127.0.0.1` IP SAN and both ServerAuth + ClientAuth
/// usages, so a single cert serves the listener and authenticates outbound
/// dials. All nodes in this single-process test share it (each is verified
/// against the same CA). Returns `(ca_pem, node_cert_pem, node_key_pem)`.
fn gen_node_pki() -> (String, String, String) {
    let ca_key = KeyPair::generate().expect("ca key");
    let mut ca_params = CertificateParams::new(Vec::new()).expect("ca params");
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let ca_cert = ca_params.self_signed(&ca_key).expect("ca self-sign");
    let ca_pem = ca_cert.pem();
    // rcgen 0.14: leaf certs are signed by an `Issuer` bundling the CA params + key.
    let ca_issuer = rcgen::Issuer::new(ca_params, ca_key);

    let node_key = KeyPair::generate().expect("node key");
    let mut node_params =
        CertificateParams::new(vec!["127.0.0.1".to_string()]).expect("node params");
    node_params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ServerAuth,
        ExtendedKeyUsagePurpose::ClientAuth,
    ];
    let node_cert = node_params
        .signed_by(&node_key, &ca_issuer)
        .expect("node sign");

    (ca_pem, node_cert.pem(), node_key.serialize_pem())
}

/// Open a fresh single-endpoint storage engine in a temp dir.
fn open_engine() -> (tempfile::TempDir, Arc<StorageEngine>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open engine"));
    (dir, engine)
}

/// Mount the Raft handler on a tonic server that requires mutual TLS (server
/// identity + client cert verified against the CA) and serve it on `addr`.
/// Returns the shutdown trigger.
fn spawn_mtls_server(
    handler: RaftGrpcHandler,
    addr: SocketAddr,
    node_cert: &str,
    node_key: &str,
    ca: &str,
) -> tokio::sync::oneshot::Sender<()> {
    let tls = ServerTlsConfig::new()
        .identity(Identity::from_pem(node_cert, node_key))
        .client_ca_root(Certificate::from_pem(ca));
    let server = Server::builder()
        .tls_config(tls)
        .expect("server tls config")
        .add_service(RaftServiceServer::new(handler));
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let graceful = server.serve_with_shutdown(addr, async {
            let _ = rx.await;
        });
        if let Err(e) = graceful.await {
            tracing::error!(%e, "mTLS raft server failed");
        }
    });
    tx
}

/// A leader and a follower interconnect over mutual TLS; a write proposed on the
/// leader replicates to the follower through the encrypted channel.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_mtls_bootstrap_and_replicate() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    // Install the pure-Rust crypto provider as the process default, exactly as
    // the server binary does at startup; tonic's TLS builders need it.
    coordinode_wire::tls::install_ce_crypto_provider();

    let (ca, node_cert, node_key) = gen_node_pki();
    // Outbound dials present this node's cert and verify peers against the CA.
    coordinode_wire::set_wire_client_tls(coordinode_wire::build_client_tls(
        ca.as_bytes(),
        Some((
            node_cert.clone().into_bytes(),
            node_key.clone().into_bytes(),
        )),
    ));

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let addr1: SocketAddr = format!("127.0.0.1:{p1}").parse().expect("addr1");
        let addr2: SocketAddr = format!("127.0.0.1:{p2}").parse().expect("addr2");

        // Leader (initializes) + follower (waits to be added), each embedded
        // behind its own mTLS tonic server, exactly as the server binary mounts
        // them on :7080.
        let (dir1, engine1) = open_engine();
        let (n1, h1) = RaftNode::open_cluster_embedded(
            1,
            Arc::clone(&engine1),
            format!("https://127.0.0.1:{p1}"),
        )
        .await
        .expect("leader");
        let s1 = spawn_mtls_server(h1, addr1, &node_cert, &node_key, &ca);

        let (dir2, engine2) = open_engine();
        let (n2, h2) = RaftNode::open_joining_embedded(2, Arc::clone(&engine2))
            .await
            .expect("follower");
        let s2 = spawn_mtls_server(h2, addr2, &node_cert, &node_key, &ca);

        tokio::time::sleep(Duration::from_millis(800)).await;
        assert!(n1.is_leader().await, "node 1 should be leader");

        // Adding the learner + promoting to voter both require the leader to
        // reach the follower — over mTLS. A failed handshake hangs to timeout.
        n1.add_node(2, format!("https://127.0.0.1:{p2}"))
            .await
            .expect("add node 2 over mTLS");
        n1.change_membership(vec![1, 2])
            .await
            .expect("change membership over mTLS");

        tokio::time::sleep(Duration::from_millis(800)).await;

        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:mtls-test".to_vec(),
                value: b"encrypted!".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline
            .propose_and_wait(&proposal)
            .expect("propose on leader");

        tokio::time::sleep(Duration::from_millis(800)).await;

        let replicated = engine2
            .get(Partition::Node, b"node:1:mtls-test")
            .expect("read follower");
        assert_eq!(
            replicated.as_deref(),
            Some(b"encrypted!".as_slice()),
            "write must replicate to the follower over the mutual-TLS channel"
        );

        n1.shutdown().await.expect("shutdown leader");
        n2.shutdown().await.expect("shutdown follower");
        let _ = s1.send(());
        let _ = s2.send(());
        drop(dir1);
        drop(dir2);
    })
    .await;

    assert!(
        result.is_ok(),
        "mTLS cluster test TIMED OUT after {TEST_TIMEOUT:?} — TLS handshake or \
         replication over the encrypted channel is not working"
    );
}
