//! One member's slow disk, on real server processes: the member's own Raft
//! log syncs are held while the other members' disks answer.
//!
//! A follower's slow disk must not hold the group back: the leader and the
//! other follower form a majority, so writes keep committing at their usual
//! pace and the slow member catches up once its disk answers. A leader's
//! slow disk may hold writes back or move the leadership; what must hold is
//! that no acknowledged write is lost, writes resume once the disk answers,
//! and every member ends holding them under one leader. A disk shared by the
//! whole group is a different case, with its own requirements (the format
//! move tests stall that one).
//!
//! The server binary must be built with its `test-faults` feature (a
//! workspace build with `--all-features`).

#![cfg(all(unix, feature = "test-faults"))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use coordinode_integration::harness::{CoordinodeProcess, free_port};
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

/// Each sync of the stalled member's log waits this long, past any election
/// timeout, so a group that waited on that member would commit at most one
/// write per such wait.
const OWN_STALL: [(&str, &str); 1] = [("COORDINODE_TEST_DISK_STALL_OWN_MS", "2000")];
/// How long the member's disk stays slow.
const STALL: Duration = Duration::from_secs(6);

async fn channel(port: u16, timeout: Duration) -> Option<tonic::transport::Channel> {
    tonic::transport::Endpoint::from_shared(format!("http://[::1]:{port}"))
        .expect("endpoint")
        .connect_timeout(Duration::from_millis(500))
        .timeout(timeout)
        .connect()
        .await
        .ok()
}

/// The leader and term the member at `port` reports, if it answers.
async fn leadership_of(port: u16) -> Option<(String, u64)> {
    let mut client = ClusterServiceClient::new(channel(port, Duration::from_millis(500)).await?);
    let status = client
        .get_cluster_status(GetClusterStatusRequest {})
        .await
        .ok()?
        .into_inner();
    Some((status.leader_id, status.raft_term))
}

/// Create `:Stalled {seq}` through the member at `port`, acknowledged at
/// majority with the journal; why not when it is refused.
async fn write_seq(port: u16, seq: i64) -> Result<(), String> {
    let mut client = CypherServiceClient::new(
        channel(port, Duration::from_secs(10))
            .await
            .ok_or_else(|| "no connection".to_string())?,
    );
    let mut parameters = HashMap::new();
    parameters.insert(
        "seq".to_string(),
        PropertyValue {
            value: Some(Pv::IntValue(seq)),
        },
    );
    client
        .execute_cypher(ExecuteCypherRequest {
            query: "CREATE (:Stalled {seq: $seq})".to_string(),
            parameters,
            write_concern: Some(WriteConcern {
                w: Some(W::Mode(WriteConcernMode::Majority as i32)),
                journal: Journal::Journal as i32,
                timeout_ms: 0,
            }),
            ..Default::default()
        })
        .await
        .map(drop)
        .map_err(|status| format!("{:?}: {}", status.code(), status.message()))
}

/// The `:Stalled` numbers the member at `port` holds, read from itself.
async fn held(port: u16) -> Result<BTreeSet<i64>, String> {
    let mut client = CypherServiceClient::new(
        channel(port, Duration::from_secs(5))
            .await
            .ok_or_else(|| "no connection".to_string())?,
    );
    let rows = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:Stalled) RETURN n.seq AS seq".to_string(),
            read_preference: 5, // NEAREST: answered where the request lands
            read_concern: Some(ReadConcern {
                level: 1, // LOCAL
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .map_err(|status| format!("{:?}: {}", status.code(), status.message()))?
        .into_inner()
        .rows;
    Ok(rows
        .iter()
        .map(|row| match &row.values[0].value {
            Some(Pv::IntValue(seq)) => *seq,
            other => panic!("unexpected seq {other:?}"),
        })
        .collect())
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

/// Three members that can each be given a slow disk of their own, member 1
/// leading.
async fn group_of_three() -> ([CoordinodeProcess; 3], [u16; 3]) {
    let ports = [free_port(), free_port(), free_port()];
    let [p1, p2, p3] = ports;
    let m1 = CoordinodeProcess::start_cluster_member_with_env(1, p1, &[p2, p3], &OWN_STALL).await;
    let m2 = CoordinodeProcess::start_cluster_member_with_env(2, p2, &[p1, p3], &OWN_STALL).await;
    let m3 = CoordinodeProcess::start_cluster_member_with_env(3, p3, &[p1, p2], &OWN_STALL).await;
    let mut leader = m1.cluster_client().await;
    for (id, member) in [(2u64, &m2), (3u64, &m3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .unwrap_or_else(|e| panic!("join {id}: {e}"));
        wait_for_voters(&mut leader, usize::try_from(id).expect("small")).await;
    }
    ([m1, m2, m3], ports)
}

/// Write numbers one after another for `run`, each through `first` and,
/// when that member refuses it, through the others: every acknowledged
/// number with when it was acknowledged.
async fn write_for(ports: [u16; 3], first: u16, run: Duration) -> Vec<(i64, Instant)> {
    let until = Instant::now() + run;
    let order: Vec<u16> = std::iter::once(first)
        .chain(ports.into_iter().filter(|p| *p != first))
        .collect();
    let mut acked = Vec::new();
    let mut seq = 0i64;
    while Instant::now() < until {
        seq += 1;
        for port in &order {
            if write_seq(*port, seq).await.is_ok() {
                acked.push((seq, Instant::now()));
                break;
            }
        }
    }
    acked
}

/// Every member holds every acknowledged number and reports the same leader,
/// within the time a member needs to catch up.
async fn every_member_holds(ports: [u16; 3], acked: &BTreeSet<i64>) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let mut missing = Vec::new();
        for port in ports {
            match held(port).await {
                Ok(seqs) => {
                    let lacks: Vec<i64> = acked.difference(&seqs).copied().collect();
                    if !lacks.is_empty() {
                        missing.push(format!("{port} lacks {} of them", lacks.len()));
                    }
                }
                Err(why) => missing.push(format!("{port}: {why}")),
            }
        }
        let mut leaders = BTreeSet::new();
        for port in ports {
            leaders.insert(leadership_of(port).await);
        }
        let one_leader = leaders.len() == 1
            && leaders
                .iter()
                .all(|l| l.as_ref().is_some_and(|(id, _)| !id.is_empty()));
        if missing.is_empty() && one_leader {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "members did not converge on {} acknowledged writes: {missing:?}; leaders {leaders:?}",
            acked.len()
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// A follower whose disk stalls does not hold the group's writes back: the
/// others keep committing at their pace while its syncs wait, the leader
/// keeps its term, and the follower catches up once its disk answers.
#[tokio::test(flavor = "multi_thread")]
async fn a_followers_slow_disk_does_not_hold_the_group_back() {
    let (members, ports) = group_of_three().await;
    let leadership = leadership_of(ports[0]).await.expect("the leader answers");
    assert_eq!(leadership.0, "1", "member 1 leads");

    let writer = tokio::spawn(write_for(ports, ports[0], STALL + Duration::from_secs(4)));
    tokio::time::sleep(Duration::from_secs(2)).await;
    let stalled_from = Instant::now();
    members[1].stall_disk_for(STALL).await;
    let stalled_to = Instant::now();
    let acked = writer.await.expect("writer");

    // Waiting on the stalled follower would commit at most one write per
    // held sync, three in the stall; the majority without it commits on.
    let during = acked
        .iter()
        .filter(|(_, at)| *at > stalled_from && *at < stalled_to)
        .count();
    assert!(
        during >= 10,
        "{during} writes committed in the {STALL:?} the follower's disk stalled"
    );
    assert_eq!(
        leadership_of(ports[0]).await,
        Some(leadership),
        "the leader keeps leading in its term"
    );
    let acked: BTreeSet<i64> = acked.into_iter().map(|(seq, _)| seq).collect();
    every_member_holds(ports, &acked).await;
}

/// A leader whose disk stalls may hold writes back or give up its
/// leadership, but loses nothing acknowledged: once its disk answers, writes
/// commit again, and every member holds them under one leader.
#[tokio::test(flavor = "multi_thread")]
async fn a_leaders_slow_disk_loses_no_acknowledged_write() {
    let (members, ports) = group_of_three().await;
    assert_eq!(
        leadership_of(ports[0]).await.map(|(id, _)| id),
        Some("1".to_string()),
        "member 1 leads"
    );

    let writer = tokio::spawn(write_for(ports, ports[0], STALL + Duration::from_secs(8)));
    tokio::time::sleep(Duration::from_secs(2)).await;
    members[0].stall_disk_for(STALL).await;
    let released = Instant::now();
    let acked = writer.await.expect("writer");

    let after = acked
        .iter()
        .filter(|(_, at)| *at > released + Duration::from_secs(2))
        .count();
    assert!(
        after > 0,
        "no write committed in the last {:?} after the leader's disk answered",
        Duration::from_secs(6)
    );
    let acked: BTreeSet<i64> = acked.into_iter().map(|(seq, _)| seq).collect();
    every_member_holds(ports, &acked).await;
}
