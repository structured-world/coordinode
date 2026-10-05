//! A group moves to a new engine format by majority, on real server
//! processes: members are updated one at a time under a continuous read and
//! write workload, no acknowledged write is lost, reads are served
//! throughout, and the group pauses its writes exactly once.
//!
//! The server binary must be built with its `test-format-bump` feature (a
//! workspace build with `--all-features`): `COORDINODE_TEST_ENGINE_FORMAT_BUMP`
//! then raises the engine format version a process runs.
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server --features test-format-bump
//! cargo nextest run -P cluster -p coordinode-integration --features test-format-bump --test version_moves
//! ```

#![cfg(feature = "test-format-bump")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use coordinode_integration::harness::{
    CoordinodeProcess, free_port, start_cluster_member_expecting_refusal_with_env,
};
use coordinode_integration::proto::admin::{
    GetClusterStatusRequest, JoinNodeRequest, NodeRole,
    cluster_service_client::ClusterServiceClient,
};
use coordinode_integration::proto::common::{PropertyValue, property_value::Value as Pv};
use coordinode_integration::proto::query::{
    ExecuteCypherRequest, cypher_service_client::CypherServiceClient,
};
use coordinode_integration::proto::replication::{
    Journal, ReadConcern, WriteConcern, WriteConcernMode, write_concern::W,
};

const BUMP: &str = "COORDINODE_TEST_ENGINE_FORMAT_BUMP";

fn at(bump: &str) -> [(&str, &str); 1] {
    [(BUMP, bump)]
}

/// The engine format a member's report says it runs.
fn engine_of(member: &CoordinodeProcess) -> u64 {
    member.version_report()["pair"]["engine"]
        .as_u64()
        .expect("the report names the engine format")
}

async fn cypher_client(port: u16) -> Option<CypherServiceClient<tonic::transport::Channel>> {
    let channel = tonic::transport::Endpoint::from_shared(format!("http://[::1]:{port}"))
        .expect("endpoint")
        .connect_timeout(Duration::from_millis(500))
        .timeout(Duration::from_secs(5))
        .connect()
        .await
        .ok()?;
    Some(CypherServiceClient::new(channel))
}

/// Create node `seq` of `:Move` on the member at `port`, acknowledged at
/// majority with the journal.
async fn write_seq(port: u16, seq: i64) -> bool {
    let Some(mut client) = cypher_client(port).await else {
        return false;
    };
    let mut parameters = HashMap::new();
    parameters.insert(
        "seq".to_string(),
        PropertyValue {
            value: Some(Pv::IntValue(seq)),
        },
    );
    client
        .execute_cypher(ExecuteCypherRequest {
            query: "CREATE (:Move {seq: $seq})".to_string(),
            parameters,
            read_preference: 0,
            read_concern: None,
            write_concern: Some(WriteConcern {
                w: Some(W::Mode(WriteConcernMode::Majority as i32)),
                journal: Journal::Journal as i32,
                timeout_ms: 0,
            }),
            transaction_id: 0,
            ..Default::default()
        })
        .await
        .is_ok()
}

/// The `:Move` sequence numbers the member at `port` holds, read locally
/// from itself; `None` when it does not answer.
async fn local_seqs(port: u16) -> Option<BTreeSet<i64>> {
    let mut client = cypher_client(port).await?;
    let rows = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:Move) RETURN n.seq AS seq".to_string(),
            parameters: HashMap::new(),
            read_preference: 5, // NEAREST: answered where the request lands
            read_concern: Some(ReadConcern {
                level: 1, // LOCAL
                after_index: 0,
                at_timestamp: 0,
            }),
            write_concern: None,
            transaction_id: 0,
            ..Default::default()
        })
        .await
        .ok()?
        .into_inner()
        .rows;
    Some(
        rows.iter()
            .map(|row| match row.values[0].value {
                Some(Pv::IntValue(seq)) => seq,
                ref other => panic!("unexpected seq {other:?}"),
            })
            .collect(),
    )
}

async fn wait_for_voters(client: &mut ClusterServiceClient<tonic::transport::Channel>, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        if let Ok(resp) = client.get_cluster_status(GetClusterStatusRequest {}).await {
            let s = resp.into_inner();
            let voters = s
                .nodes
                .iter()
                .filter(|m| m.role != NodeRole::Learner as i32)
                .count();
            if s.nodes.len() == n && voters == n {
                return;
            }
        }
        assert!(Instant::now() < deadline, "no group of {n} voters");
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Three members at format bump 0, member 1 leading.
async fn group_of_three() -> ([CoordinodeProcess; 3], [u16; 3]) {
    let ports = [free_port(), free_port(), free_port()];
    let [p1, p2, p3] = ports;
    let m1 = CoordinodeProcess::start_cluster_member_with_env(1, p1, &[p2, p3], &at("0")).await;
    let m2 = CoordinodeProcess::start_cluster_member_with_env(2, p2, &[p1, p3], &at("0")).await;
    let m3 = CoordinodeProcess::start_cluster_member_with_env(3, p3, &[p1, p2], &at("0")).await;
    let mut leader = m1.cluster_client().await;
    for (id, member) in [(2u64, &m2), (3u64, &m3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
            })
            .await
            .unwrap_or_else(|e| panic!("join {id}: {e}"));
        wait_for_voters(&mut leader, usize::try_from(id).expect("small")).await;
    }
    ([m1, m2, m3], ports)
}

fn peers_of(ports: [u16; 3], i: usize) -> Vec<u16> {
    ports
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, p)| *p)
        .collect()
}

/// What the workload saw: every acknowledged write with when it was
/// acknowledged, and every local read a member refused with when.
#[derive(Default)]
struct Observed {
    acked: Vec<(i64, Instant)>,
    refused_reads: Vec<(u16, Instant)>,
}

/// Move a group of three from format bump 0 to 1, member 3 first, member 2
/// completing the majority (killed when `kill_completing`), member 1 last,
/// under a continuous workload; check what the move promises.
async fn move_under_workload(kill_completing: bool) {
    let ([m1, m2, m3], ports) = group_of_three().await;
    let base = engine_of(&m1);

    let stop = Arc::new(AtomicBool::new(false));
    let observed = Arc::new(Mutex::new(Observed::default()));
    let writer = {
        let (stop, observed) = (Arc::clone(&stop), Arc::clone(&observed));
        tokio::spawn(async move {
            let mut seq = 0i64;
            while !stop.load(Ordering::Acquire) {
                for port in ports {
                    seq += 1;
                    if write_seq(port, seq).await {
                        observed.lock().unwrap().acked.push((seq, Instant::now()));
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
    };
    let reader = {
        let (stop, observed) = (Arc::clone(&stop), Arc::clone(&observed));
        tokio::spawn(async move {
            while !stop.load(Ordering::Acquire) {
                for port in ports {
                    if local_seqs(port).await.is_none() {
                        observed
                            .lock()
                            .unwrap()
                            .refused_reads
                            .push((port, Instant::now()));
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
    };
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Each member is down from the start of its update until it serves
    // again; a read it refuses is allowed only then.
    let mut down: Vec<(u16, Instant, Instant)> = Vec::new();

    let started = Instant::now();
    let m3 = m3
        .restart_member(3, &peers_of(ports, 2), &at("1"), false)
        .await;
    down.push((ports[2], started, Instant::now()));
    assert_eq!(
        engine_of(&m3),
        base + 1,
        "the server honours the format bump"
    );
    tokio::time::sleep(Duration::from_secs(2)).await;

    let completing = Instant::now();
    let m2 = m2
        .restart_member(2, &peers_of(ports, 1), &at("1"), kill_completing)
        .await;
    let update_time = completing.elapsed();
    down.push((ports[1], completing, Instant::now()));
    tokio::time::sleep(Duration::from_secs(5)).await;

    let last = Instant::now();
    let m1 = m1
        .restart_member(1, &peers_of(ports, 0), &at("1"), false)
        .await;
    down.push((ports[0], last, Instant::now()));
    tokio::time::sleep(Duration::from_secs(3)).await;

    stop.store(true, Ordering::Release);
    writer.await.expect("writer");
    reader.await.expect("reader");
    let observed = std::mem::take(&mut *observed.lock().unwrap());

    // No acknowledged write is lost, and the member updated last catches up.
    let acked: BTreeSet<i64> = observed.acked.iter().map(|(seq, _)| *seq).collect();
    assert!(
        acked.len() > 50,
        "the workload wrote ({} acks)",
        acked.len()
    );
    let deadline = Instant::now() + Duration::from_secs(40);
    for (member, port) in [(&m1, ports[0]), (&m2, ports[1]), (&m3, ports[2])] {
        loop {
            let held = local_seqs(port).await.unwrap_or_default();
            let lost: Vec<&i64> = acked.difference(&held).collect();
            if lost.is_empty() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "member on {port} lacks {} acknowledged writes: {:?}",
                lost.len(),
                &lost[..lost.len().min(10)]
            );
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        let report = member.version_report();
        assert_eq!(engine_of(member), base + 1);
        assert!(report["read_only"].is_null(), "{report}");
    }

    // Reads are served throughout: a member refuses one only while down.
    let slack = Duration::from_secs(3);
    for (port, when) in &observed.refused_reads {
        assert!(
            down.iter()
                .any(|(p, from, to)| p == port && *when >= *from && *when <= *to + slack),
            "member on {port} refused a local read while it was up"
        );
    }

    // Exactly one write pause, about as long as the completing update.
    let pauses: Vec<Duration> = observed
        .acked
        .windows(2)
        .map(|w| w[1].1 - w[0].1)
        .filter(|gap| *gap > Duration::from_millis(1500))
        .collect();
    assert_eq!(pauses.len(), 1, "one write pause, got {pauses:?}");
    assert!(
        pauses[0] <= update_time + Duration::from_secs(20),
        "pause {:?} against an update of {update_time:?}",
        pauses[0]
    );
    eprintln!(
        "write pause {:?}, completing member's update {update_time:?}",
        pauses[0]
    );
}

/// Three members updated one at a time under a continuous workload.
#[tokio::test(flavor = "multi_thread")]
async fn an_engine_format_move_loses_no_acknowledged_write() {
    move_under_workload(false).await;
}

/// The same with the completing member killed, so its directory carries an
/// unapplied and partly uncommitted tail when it opens at the new format.
#[tokio::test(flavor = "multi_thread")]
async fn a_move_with_the_completing_member_killed_loses_no_acknowledged_write() {
    move_under_workload(true).await;
}

/// Take the reachable members 1 and 2 of a group from format bump `from` to
/// `to`, member 2 first.
async fn move_two(
    m1: CoordinodeProcess,
    m2: CoordinodeProcess,
    ports: [u16; 3],
    to: &str,
) -> (CoordinodeProcess, CoordinodeProcess) {
    let m2 = m2
        .restart_member(2, &peers_of(ports, 1), &at(to), false)
        .await;
    let m1 = m1
        .restart_member(1, &peers_of(ports, 0), &at(to), false)
        .await;
    (m1, m2)
}

/// Write `seq` on whichever of `ports` leads, within 40 s.
async fn write_on_leader(ports: &[u16], seq: i64) {
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        for port in ports {
            if write_seq(*port, seq).await {
                return;
            }
        }
        assert!(Instant::now() < deadline, "no member took write {seq}");
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

async fn wait_holds(port: u16, seqs: &BTreeSet<i64>) {
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        if local_seqs(port)
            .await
            .is_some_and(|held| seqs.is_subset(&held))
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "member on {port} never caught up"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// A member away through two moves of its group comes back two formats
/// behind: refused by name, it rejoins through the intermediate format
/// (read-only and behind there, matched and caught up at the group's).
#[tokio::test(flavor = "multi_thread")]
async fn a_member_two_formats_behind_rejoins_through_the_intermediate_format() {
    let ([m1, m2, m3], ports) = group_of_three().await;
    let base = engine_of(&m1);
    write_on_leader(&ports[..2], 1).await;
    let away = m3.stop_keeping_data().await;

    let (m1, m2) = move_two(m1, m2, ports, "1").await;
    write_on_leader(&ports[..2], 2).await;
    let (m1, m2) = move_two(m1, m2, ports, "2").await;
    write_on_leader(&ports[..2], 3).await;

    let (status, printed) = start_cluster_member_expecting_refusal_with_env(
        3,
        ports[2],
        &peers_of(ports, 2),
        away.path(),
        &at("2"),
    )
    .await;
    assert!(!status.success(), "two formats behind refuses: {printed}");
    assert!(
        printed.contains(&format!("engine format {base}")),
        "the refusal names the directory's format: {printed}"
    );

    let m3 = CoordinodeProcess::start_cluster_member_over(
        3,
        ports[2],
        &peers_of(ports, 2),
        away,
        &at("1"),
    )
    .await;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let report = m3.version_report();
        if report["read_only"]["behind"].as_bool() == Some(true) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "never read-only behind: {report}"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    let m3 = m3
        .restart_member(3, &peers_of(ports, 2), &at("2"), false)
        .await;
    wait_holds(ports[2], &BTreeSet::from([1, 2, 3])).await;
    let report = m3.version_report();
    assert_eq!(engine_of(&m3), base + 2);
    assert!(report["read_only"].is_null(), "{report}");
    drop((m1, m2));
}

/// The other way back for a member two formats behind: removed while away,
/// the removal committed first, its directory discarded after, and added
/// again empty at the group's format.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_two_formats_behind_is_removed_and_re_added_empty() {
    use coordinode_integration::proto::admin::DecommissionNodeRequest;

    let ([m1, m2, m3], ports) = group_of_three().await;
    write_on_leader(&ports[..2], 1).await;
    let away = m3.stop_keeping_data().await;
    let (m1, m2) = move_two(m1, m2, ports, "1").await;
    let (m1, m2) = move_two(m1, m2, ports, "2").await;
    write_on_leader(&ports[..2], 2).await;

    // Removed through whichever member leads, then the directory goes.
    let deadline = Instant::now() + Duration::from_secs(40);
    let leader = loop {
        let mut removed = None;
        for member in [&m1, &m2] {
            let decommission = member
                .cluster_client()
                .await
                .decommission_node(DecommissionNodeRequest {
                    node_id: 3,
                    pruning: false,
                    force: false,
                    skip_confirmation: false,
                })
                .await;
            if decommission.is_ok() {
                removed = Some(member);
                break;
            }
        }
        if let Some(member) = removed {
            break member;
        }
        assert!(Instant::now() < deadline, "the removal never committed");
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    let mut admin = leader.cluster_client().await;
    wait_for_voters(&mut admin, 2).await;
    drop(away);

    let m3 = CoordinodeProcess::start_cluster_member_with_env(
        3,
        ports[2],
        &peers_of(ports, 2),
        &at("2"),
    )
    .await;
    admin
        .join_node(JoinNodeRequest {
            node_id: 3,
            address: m3.member_addr(),
            pre_seeded: false,
        })
        .await
        .expect("join 3 again");
    wait_for_voters(&mut admin, 3).await;
    wait_holds(ports[2], &BTreeSet::from([1, 2])).await;
    assert!(m3.version_report()["read_only"].is_null());
    drop((m1, m2));
}

/// A directory two engine formats behind is refused by name and left as it
/// was; taken through the intermediate format it opens, holding its data.
/// A single-member group is the case where that is the only way.
#[tokio::test(flavor = "multi_thread")]
async fn a_directory_two_formats_behind_goes_through_the_intermediate_one() {
    let (p1, p2) = (free_port(), free_port());
    let m1 = CoordinodeProcess::start_cluster_member_with_env(1, p1, &[p2], &at("0")).await;
    m1.wait_for_leader(Duration::from_secs(20)).await;
    let base = engine_of(&m1);
    assert!(write_seq(p1, 1).await, "the single member writes");
    let dir = m1.stop_keeping_data().await;

    let (status, printed) =
        start_cluster_member_expecting_refusal_with_env(1, p1, &[p2], dir.path(), &at("2")).await;
    assert!(!status.success(), "two formats ahead refuses: {printed}");
    assert!(
        printed.contains(&format!("engine format {base}"))
            && printed.contains(&format!("runs format {}", base + 2)),
        "the refusal names both formats: {printed}"
    );

    let m1 = CoordinodeProcess::start_cluster_member_over(1, p1, &[p2], dir, &at("1")).await;
    m1.wait_for_leader(Duration::from_secs(20)).await;
    assert_eq!(engine_of(&m1), base + 1);
    let dir = m1.stop_keeping_data().await;
    let m1 = CoordinodeProcess::start_cluster_member_over(1, p1, &[p2], dir, &at("2")).await;
    m1.wait_for_leader(Duration::from_secs(20)).await;
    assert_eq!(engine_of(&m1), base + 2);
    assert_eq!(
        local_seqs(p1).await.expect("it answers"),
        BTreeSet::from([1]),
        "the data came through both migrations"
    );
    assert!(
        write_seq(p1, 2).await,
        "and the member writes at the new format"
    );
}
