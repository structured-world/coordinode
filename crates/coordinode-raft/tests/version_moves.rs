#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! A group moves to a new version by majority: members are updated one at a
//! time, a member that does not run the version its group runs takes no
//! writes and names why, the side holding the majority writes, and no write
//! the group acknowledged is lost on the way.
//!
//! The members here differ in their host format epoch, half of the version
//! pair members are matched on; the engine format version is the same in one
//! process, and a move of it goes through the same handshake, record and
//! refusal.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalError, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_core::version::VersionPair;
use coordinode_raft::cluster::version::MemberState;
use coordinode_raft::cluster::{NodeOptions, RaftNode};
use coordinode_raft::read_fence::{ReadConcern, ReadFenceError, ReadPreference};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::alloc_port;

const TEST_TIMEOUT: Duration = Duration::from_secs(120);

fn open_engine(path: &Path) -> Arc<StorageEngine> {
    Arc::new(
        StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]))
        .expect("open engine"),
    )
}

fn options(host_epoch: u64) -> NodeOptions {
    NodeOptions {
        host_epoch,
        ..Default::default()
    }
}

fn pair(host_epoch: u64) -> VersionPair {
    VersionPair::current(host_epoch)
}

/// Every write of a test draws its id from one generator: the state machine
/// drops a repeated id as a replay.
fn write(
    node: &RaftNode,
    ids: &ProposalIdGenerator,
    key: &str,
    commit_ts: u64,
) -> Result<(), ProposalError> {
    let proposal = RaftProposal {
        id: ids.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.as_bytes().to_vec(),
            value: b"v".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    node.pipeline().propose_and_wait(&proposal).map(|_| ())
}

fn holds(engine: &StorageEngine, key: &str) -> bool {
    engine
        .get(Partition::Node, key.as_bytes())
        .expect("read")
        .is_some()
}

async fn eventually(what: &str, mut check: impl FnMut() -> bool) {
    for _ in 0..300 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("never: {what}");
}

/// Open member `id` on `port` over the directory it already holds.
async fn reopen(id: u64, engine: &Arc<StorageEngine>, port: u16, host_epoch: u64) -> RaftNode {
    let mut last = None;
    for _ in 0..20 {
        match RaftNode::open_cluster_with_options(
            id,
            Arc::clone(engine),
            format!("127.0.0.1:{port}").parse().expect("addr"),
            format!("http://127.0.0.1:{port}"),
            options(host_epoch),
        )
        .await
        {
            Ok(node) => return node,
            Err(e) => {
                last = Some(e);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    panic!("reopen node {id} on {port}: {last:?}");
}

struct Member {
    id: u64,
    port: u16,
    engine: Arc<StorageEngine>,
    node: Option<RaftNode>,
    _dir: tempfile::TempDir,
}

impl Member {
    fn node(&self) -> &RaftNode {
        self.node.as_ref().expect("member is running")
    }

    /// Stop the member and open its node again at `host_epoch` over the
    /// directory it holds. The host epoch is no format of the engine's, so
    /// the engine itself stays open.
    async fn update(&mut self, host_epoch: u64) {
        if let Some(node) = self.node.take() {
            node.shutdown().await.expect("shutdown");
        }
        self.node = Some(reopen(self.id, &self.engine, self.port, host_epoch).await);
    }
}

/// A group of three at `host_epoch`, member 1 leading, its pair recorded.
async fn group_of_three(host_epoch: u64) -> [Member; 3] {
    let dirs = [
        tempfile::tempdir().expect("d1"),
        tempfile::tempdir().expect("d2"),
        tempfile::tempdir().expect("d3"),
    ];
    let ports = [alloc_port(), alloc_port(), alloc_port()];
    let [d1, d2, d3] = dirs;
    let e1 = open_engine(d1.path());
    let e2 = open_engine(d2.path());
    let e3 = open_engine(d3.path());
    let n1 = RaftNode::open_cluster_with_options(
        1,
        Arc::clone(&e1),
        format!("127.0.0.1:{}", ports[0]).parse().expect("addr"),
        format!("http://127.0.0.1:{}", ports[0]),
        options(host_epoch),
    )
    .await
    .expect("open 1");
    let n2 = RaftNode::open_joining_with_options(
        2,
        Arc::clone(&e2),
        format!("127.0.0.1:{}", ports[1]).parse().expect("addr"),
        options(host_epoch),
    )
    .await
    .expect("open 2");
    let n3 = RaftNode::open_joining_with_options(
        3,
        Arc::clone(&e3),
        format!("127.0.0.1:{}", ports[2]).parse().expect("addr"),
        options(host_epoch),
    )
    .await
    .expect("open 3");
    eventually("member 1 records its pair", || {
        n1.version().group_pair().map(|r| r.pair) == Some(pair(host_epoch))
    })
    .await;
    n1.add_node(2, format!("http://127.0.0.1:{}", ports[1]))
        .await
        .expect("add 2");
    n1.add_node(3, format!("http://127.0.0.1:{}", ports[2]))
        .await
        .expect("add 3");
    n1.change_membership(vec![1, 2, 3])
        .await
        .expect("three voters");
    [
        Member {
            id: 1,
            port: ports[0],
            _dir: d1,
            engine: e1,
            node: Some(n1),
        },
        Member {
            id: 2,
            port: ports[1],
            _dir: d2,
            engine: e2,
            node: Some(n2),
        },
        Member {
            id: 3,
            port: ports[2],
            _dir: d3,
            engine: e3,
            node: Some(n3),
        },
    ]
}

fn mismatch(result: Result<(), ProposalError>) -> coordinode_core::version::Mismatch {
    match result {
        Err(ProposalError::Mismatched(m)) => m,
        other => panic!("expected a read-only refusal, got {other:?}"),
    }
}

/// Three members move from epoch 0 to epoch 1 one at a time. The first one
/// updated is read-only and frozen while the old side writes; the second
/// completes the new majority, which records its pair; the old leader is then
/// read-only, behind, and names the new leader; updated last, it catches up.
/// Every write the group acknowledged is held by the new side.
#[tokio::test(flavor = "multi_thread")]
async fn a_group_moves_by_majority_one_member_at_a_time() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let ids = ProposalIdGenerator::with_base(1u64 << 48);
        let [mut m1, mut m2, mut m3] = group_of_three(0).await;
        write(m1.node(), &ids, "node:before", 100).expect("write before the move");

        // Member 3 first: ahead of its group, read-only, frozen.
        m3.update(1).await;
        let ahead = mismatch(write(m3.node(), &ids, "node:on-3", 110));
        assert!(!ahead.behind);
        assert_eq!((ahead.own, ahead.group), (pair(1), pair(0)));
        // It serves what it holds, labelled as of its last applied commit,
        // and refuses the reads that would have to be current.
        let mut fence = m3.node().read_fence();
        fence
            .apply_default(ReadPreference::Nearest, ReadConcern::Local)
            .await
            .expect("a local read is served");
        assert_eq!(fence.as_of(), Some(ahead.as_of));
        for (preference, concern) in [
            (ReadPreference::Nearest, ReadConcern::Majority),
            (ReadPreference::Nearest, ReadConcern::Linearizable),
            (ReadPreference::Primary, ReadConcern::Local),
        ] {
            let refused = m3
                .node()
                .read_fence()
                .apply_default(preference, concern)
                .await;
            assert!(
                matches!(refused, Err(ReadFenceError::ReadOnly(_))),
                "{preference:?}/{concern:?}: {refused:?}"
            );
        }
        assert!(
            matches!(
                m3.node()
                    .read_fence()
                    .wait_for_index(u64::MAX, Duration::from_secs(30))
                    .await,
                Err(ReadFenceError::ReadOnly(_))
            ),
            "a causal wait on a position it will never reach is refused at once"
        );
        write(m1.node(), &ids, "node:during", 120).expect("the old side still holds a majority");
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert!(
            !holds(&m3.engine, "node:during"),
            "a member of another version receives nothing"
        );

        // Member 2 completes the new majority: one of the two leads and
        // records the new pair.
        m2.update(1).await;
        eventually("the new side records its pair", || {
            [&m2, &m3]
                .iter()
                .any(|m| m.node().version().group_pair().map(|r| r.pair) == Some(pair(1)))
        })
        .await;
        let new_leader = if m2.node().version().state() == MemberState::Matched
            && write(m2.node(), &ids, "node:after", 130).is_ok()
        {
            &m2
        } else {
            eventually("member 3 matches", || {
                m3.node().version().state() == MemberState::Matched
            })
            .await;
            write(m3.node(), &ids, "node:after", 130).expect("the new side writes");
            &m3
        };
        for key in ["node:before", "node:during", "node:after"] {
            assert!(holds(&new_leader.engine, key), "the new side lost {key}");
        }

        // The old leader learns the group moved past it.
        let behind = loop {
            match write(m1.node(), &ids, "node:on-1", 140) {
                Err(ProposalError::Mismatched(m)) => break m,
                _ => tokio::time::sleep(Duration::from_millis(200)).await,
            }
        };
        assert!(behind.behind);
        assert_eq!((behind.own, behind.group), (pair(0), pair(1)));
        assert_eq!(behind.leader.map(|(id, _)| id), Some(new_leader.id));

        // Its report says the same, and the new leader's shows the group
        // writing at the new pair.
        let old = m1.node().version_report();
        assert_eq!(old.pair, pair(0));
        assert!(old.read_only.as_ref().is_some_and(|r| r.behind));
        let led = new_leader.node().version_report();
        assert_eq!(led.group_pair.map(|r| r.pair), Some(pair(1)));
        assert_eq!(led.majority_pair, Some(pair(1)));
        assert_eq!(led.pause_ms, None);
        assert!(led.read_only.is_none());

        // Updated last, it matches and catches up.
        m1.update(1).await;
        eventually("member 1 matches", || {
            m1.node().version().state() == MemberState::Matched
        })
        .await;
        eventually("member 1 catches up", || holds(&m1.engine, "node:after")).await;
        assert!(
            !holds(&m1.engine, "node:on-1"),
            "a refused write never lands"
        );

        for m in [&mut m1, &mut m2, &mut m3] {
            if let Some(node) = m.node.take() {
                node.shutdown().await.expect("shutdown");
            }
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// Write `key` on `node` off the runtime, giving up after `wait`: a write
/// that cannot reach a majority neither commits nor fails at once.
async fn try_write(
    node: &RaftNode,
    ids: &ProposalIdGenerator,
    key: &str,
    commit_ts: u64,
    wait: Duration,
) -> Option<Result<(), ProposalError>> {
    let pipeline = node.pipeline();
    let proposal = RaftProposal {
        id: ids.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.as_bytes().to_vec(),
            value: b"v".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    let task = tokio::task::spawn_blocking(move || pipeline.propose_and_wait(&proposal).map(drop));
    tokio::time::timeout(wait, task)
        .await
        .ok()
        .map(|joined| joined.expect("write task"))
}

/// With one member unreachable throughout, the group cannot write from the
/// moment the first reachable member is updated until the second one is: the
/// pause is reported while it lasts and ends once both reachable members run
/// the new pair. The unreachable member, updated and back later, catches up.
#[tokio::test(flavor = "multi_thread")]
async fn a_move_with_one_member_unreachable_pauses_until_both_others_move() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let ids = ProposalIdGenerator::with_base(2u64 << 48);
        let [mut m1, mut m2, mut m3] = group_of_three(0).await;
        write(m1.node(), &ids, "node:before", 100).expect("write before the move");

        // Member 3 is unreachable from here on.
        m3.node
            .take()
            .expect("running")
            .shutdown()
            .await
            .expect("shutdown 3");

        m2.update(1).await;
        let paused = try_write(m1.node(), &ids, "node:paused", 110, Duration::from_secs(3)).await;
        assert!(
            !matches!(paused, Some(Ok(()))),
            "no majority runs one pair: {paused:?}"
        );
        eventually("the pause is reported", || {
            m1.node().version_report().pause_ms.is_some()
        })
        .await;

        let updated_at = std::time::Instant::now();
        m1.update(1).await;
        // Whichever of the two leads takes the write; the other refuses it.
        let leader = loop {
            if let Some(m) = [&m1, &m2].into_iter().find(|m| {
                m.node().version().state() == MemberState::Matched
                    && m.node().version().group_pair().map(|r| r.pair) == Some(pair(1))
                    && write(m.node(), &ids, "node:after", 130).is_ok()
            }) {
                break m;
            }
            assert!(
                updated_at.elapsed() < Duration::from_secs(60),
                "the two reachable members never wrote at the new pair"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        };
        assert!(holds(&leader.engine, "node:before"));
        let resumed = updated_at.elapsed();
        assert_eq!(leader.node().version_report().pause_ms, None, "it writes");
        assert!(
            resumed < Duration::from_secs(30),
            "writes resumed {resumed:?} after the second member was updated"
        );

        // The unreachable member comes back updated and catches up.
        m3.node = Some(reopen(3, &m3.engine, m3.port, 1).await);
        eventually("member 3 matches", || {
            m3.node().version().state() == MemberState::Matched
        })
        .await;
        eventually("member 3 catches up", || holds(&m3.engine, "node:after")).await;

        for m in [&mut m1, &mut m2, &mut m3] {
            if let Some(node) = m.node.take() {
                node.shutdown().await.expect("shutdown");
            }
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}
