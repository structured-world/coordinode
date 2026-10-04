#![allow(clippy::unwrap_used, clippy::expect_used)]
//! DERIVED index maintenance across a replicated group: the log carries the
//! sealed work and its inputs, and every member derives the same entries,
//! whether it applies the entry from the log, after a leader change, or on
//! top of a snapshot.

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::graph::node::NodeRecord;
use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::{IndexInterpretation, KEY_CODEC, PropertyRef, entry};
use coordinode_core::index::encoding::encode_tuple;
use coordinode_core::txn::proposal::{
    DerivedIndexWork, DerivedSource, IndexBinding, Mutation, PartitionId, ProposalError,
    ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::alloc_port;

struct Member {
    node: RaftNode,
    engine: Arc<StorageEngine>,
    _dir: tempfile::TempDir,
}

fn open_engine(dir: &tempfile::TempDir) -> Arc<StorageEngine> {
    Arc::new(
        StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]))
        .expect("open"),
    )
}

async fn await_leadership(node: &RaftNode) {
    for _ in 0..150 {
        if node.is_leader().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(node.is_leader().await, "node {} never led", node.node_id());
}

async fn bootstrap_3() -> (Member, Member, Member) {
    let ports = [alloc_port(), alloc_port(), alloc_port()];
    let dir1 = tempfile::tempdir().expect("d1");
    let e1 = open_engine(&dir1);
    let n1 = RaftNode::open_cluster(
        1,
        Arc::clone(&e1),
        format!("127.0.0.1:{}", ports[0]).parse().expect("addr"),
        format!("http://127.0.0.1:{}", ports[0]),
    )
    .await
    .expect("leader");
    let mut joined = Vec::new();
    for (id, port) in [(2u64, ports[1]), (3, ports[2])] {
        let dir = tempfile::tempdir().expect("dir");
        let engine = open_engine(&dir);
        let node = RaftNode::open_joining(
            id,
            Arc::clone(&engine),
            format!("127.0.0.1:{port}").parse().expect("addr"),
        )
        .await
        .expect("joining");
        joined.push(Member {
            node,
            engine,
            _dir: dir,
        });
    }
    await_leadership(&n1).await;
    for (id, port) in [(2u64, ports[1]), (3, ports[2])] {
        n1.add_node(id, format!("http://127.0.0.1:{port}"))
            .await
            .expect("add");
    }
    n1.change_membership(vec![1, 2, 3])
        .await
        .expect("membership");
    let n3 = joined.pop().expect("n3");
    let n2 = joined.pop().expect("n2");
    (
        Member {
            node: n1,
            engine: e1,
            _dir: dir1,
        },
        n2,
        n3,
    )
}

fn unique_email() -> IndexBinding {
    IndexBinding {
        epoch: 1,
        interpretation: IndexInterpretation {
            codec: KEY_CODEC,
            name: "u_email".into(),
            unique: true,
            sparse: false,
            properties: vec![PropertyRef {
                field: Some(1),
                name: "email".into(),
            }],
            filter: None,
        },
    }
}

/// The entry key and value of `node_id` holding `email`.
fn claim(node_id: u64, email: &str) -> (Vec<u8>, Vec<u8>) {
    let tuple = encode_tuple(&[Value::String(email.into())]).expect("tuple");
    entry(
        "u_email",
        true,
        &tuple,
        coordinode_core::index::derive::EntryOwner::node(node_id),
    )
}

/// A unit that writes `node_id`'s record with `email` and derives its entry
/// from that record, moving it from `old` when the node held one.
fn write_email(
    id_gen: &ProposalIdGenerator,
    node_id: u64,
    email: &str,
    old: Option<&str>,
    ts: u64,
) -> RaftProposal {
    let mut record = NodeRecord::new("U");
    record.set(1, Value::String(email.into()));
    RaftProposal {
        id: id_gen.next(),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:0:{node_id}").into_bytes(),
                value: record.to_msgpack().expect("record"),
            },
            Mutation::Derive(DerivedIndexWork {
                binding: unique_email(),
                node_id,
                valid_from: None,
                old: old.map(|o| vec![Value::String(o.into())]),
                new: DerivedSource::UnitRecord(0),
            }),
        ],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts - 1),
        bypass_rate_limiter: false,
    }
}

/// Every entry of the index a member holds.
fn entries(engine: &StorageEngine) -> Vec<(Vec<u8>, Vec<u8>)> {
    use coordinode_storage::Guard as _;
    let prefix = coordinode_core::index::encoding::unique_index_prefix("u_email");
    engine
        .prefix_scan(Partition::Idx, &prefix)
        .expect("scan")
        .map(|item| {
            let (k, v) = item.into_inner().expect("kv");
            (k.to_vec(), v.to_vec())
        })
        .collect()
}

/// Block until `engine` holds exactly `expected`.
async fn await_entries(engine: &StorageEngine, expected: &[(Vec<u8>, Vec<u8>)], what: &str) {
    for _ in 0..200 {
        if entries(engine) == expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(entries(engine), expected, "{what}");
}

fn propose(node: &RaftNode, proposal: &RaftProposal) {
    let pipeline = node.pipeline();
    tokio::task::block_in_place(|| pipeline.propose_and_wait(proposal)).expect("commit");
}

/// Every member derives the same entries from the logged work, and a new
/// leader continues from them: the unique value a node moves off is free,
/// the one it moves to is held.
#[tokio::test(flavor = "multi_thread")]
async fn members_derive_the_same_entries_across_a_leader_change() {
    let result = tokio::time::timeout(Duration::from_secs(90), async {
        let (n1, n2, n3) = bootstrap_3().await;
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        propose(&n1.node, &write_email(&id_gen, 1, "a@x", None, 100));
        propose(&n1.node, &write_email(&id_gen, 2, "b@x", None, 110));

        let mut expected = vec![claim(1, "a@x"), claim(2, "b@x")];
        expected.sort();
        for (member, who) in [(&n1, "n1"), (&n2, "n2"), (&n3, "n3")] {
            await_entries(&member.engine, &expected, who).await;
        }

        n1.node.shutdown().await.expect("shutdown leader");
        drop(n1);
        let mut leader = None;
        for _ in 0..150 {
            if n2.node.is_leader().await {
                leader = Some((&n2, &n3));
                break;
            }
            if n3.node.is_leader().await {
                leader = Some((&n3, &n2));
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let (leader, follower) = leader.expect("a survivor leads");
        let id_gen = ProposalIdGenerator::with_base(2u64 << 48);
        propose(
            &leader.node,
            &write_email(&id_gen, 1, "c@x", Some("a@x"), 200),
        );

        let mut expected = vec![claim(1, "c@x"), claim(2, "b@x")];
        expected.sort();
        await_entries(&leader.engine, &expected, "the new leader").await;
        await_entries(&follower.engine, &expected, "the follower").await;

        n2.node.shutdown().await.expect("s2");
        n3.node.shutdown().await.expect("s3");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: members_derive_the_same_entries_across_a_leader_change"
    );
}

/// Work no member could derive is refused before it reaches the log, and
/// the group goes on committing: a unit naming no record as its source
/// would otherwise stop every member at the same entry.
#[tokio::test(flavor = "multi_thread")]
async fn underivable_work_is_refused_before_the_log() {
    let result = tokio::time::timeout(Duration::from_secs(90), async {
        let (n1, n2, n3) = bootstrap_3().await;
        let id_gen = ProposalIdGenerator::with_base(3u64 << 48);
        let mut bad = write_email(&id_gen, 1, "a@x", None, 100);
        bad.mutations.swap(0, 1);
        let pipeline = n1.node.pipeline();
        let refused = tokio::task::block_in_place(|| pipeline.propose_and_wait(&bad));
        assert!(
            matches!(refused, Err(ProposalError::Unencodable(_))),
            "the source names no earlier record: {refused:?}"
        );

        propose(&n1.node, &write_email(&id_gen, 2, "b@x", None, 110));
        let expected = vec![claim(2, "b@x")];
        for (member, who) in [(&n1, "n1"), (&n2, "n2"), (&n3, "n3")] {
            await_entries(&member.engine, &expected, who).await;
        }

        n1.node.shutdown().await.expect("s1");
        n2.node.shutdown().await.expect("s2");
        n3.node.shutdown().await.expect("s3");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: underivable_work_is_refused_before_the_log"
    );
}

/// A member that joins after the log was purged takes the entries derived
/// before the snapshot from it and derives the rest from the log.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_joining_from_a_snapshot_derives_the_log_tail() {
    let result = tokio::time::timeout(Duration::from_secs(90), async {
        let (p1, p2) = (alloc_port(), alloc_port());
        let dir1 = tempfile::tempdir().expect("d1");
        let e1 = open_engine(&dir1);
        let n1 = RaftNode::open_cluster_with_options(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("addr"),
            format!("http://127.0.0.1:{p1}"),
            coordinode_raft::cluster::NodeOptions {
                snapshots: coordinode_raft::cluster::SnapshotTriggerConfig {
                    min_interval: Duration::from_secs(3600),
                    log_bytes: u64::MAX,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .expect("leader");
        await_leadership(&n1).await;
        let id_gen = ProposalIdGenerator::with_base(4u64 << 48);

        // In the snapshot only.
        propose(&n1, &write_email(&id_gen, 1, "a@x", None, 100));
        n1.raft().trigger().snapshot().await.expect("snapshot");
        tokio::time::sleep(Duration::from_secs(1)).await;
        let applied = n1.applied_index();
        n1.raft().trigger().purge_log(applied).await.expect("purge");
        tokio::time::sleep(Duration::from_millis(500)).await;
        // In the log only: one new claim and one move.
        propose(&n1, &write_email(&id_gen, 2, "b@x", None, 200));
        propose(&n1, &write_email(&id_gen, 1, "c@x", Some("a@x"), 210));

        let dir2 = tempfile::tempdir().expect("d2");
        let e2 = open_engine(&dir2);
        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("addr"),
        )
        .await
        .expect("joining");
        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add n2");

        let mut expected = vec![claim(1, "c@x"), claim(2, "b@x")];
        expected.sort();
        await_entries(&e1, &expected, "the leader").await;
        await_entries(&e2, &expected, "the joined member").await;

        n2.shutdown().await.expect("s2");
        n1.shutdown().await.expect("s1");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: a_member_joining_from_a_snapshot_derives_the_log_tail"
    );
}
