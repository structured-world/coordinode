//! Integration test: a Raft log write that fails, as on a full disk, stops
//! consensus instead of being written past.
//!
//! The node then commits nothing more, and its consensus reports the fatal
//! error a server watches for to stop reporting ready and exit; reopened on
//! the same store, it starts, with every write acknowledged before the
//! failure kept and nothing acknowledged past it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{Mutation, PartitionId, ProposalId, RaftProposal};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::storage::Request;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn open_engine(dir: &std::path::Path) -> Arc<StorageEngine> {
    Arc::new(
        StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]))
        .expect("open engine"),
    )
}

/// A write of `key` through consensus.
fn write(key: &str, id: u64) -> Request {
    // Wall-clock microseconds, as the timestamp domain counts them.
    let ts = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("after the epoch")
            .as_micros(),
    )
    .expect("fits");
    Request::single(RaftProposal {
        id: ProposalId::from_raw(id),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.as_bytes().to_vec(),
            value: b"v".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts - 1),
        bypass_rate_limiter: false,
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_log_write_stops_consensus_and_the_node_restarts_whole() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let engine = open_engine(dir.path());
        let node = RaftNode::open(1, Arc::clone(&engine))
            .await
            .expect("open node");
        node.raft()
            .client_write(write("node:before", 1))
            .await
            .expect("a write before the failure commits");

        node.fail_log_writes();
        let refused = tokio::time::timeout(
            Duration::from_secs(10),
            node.raft().client_write(write("node:after", 2)),
        )
        .await;
        assert!(
            !matches!(refused, Ok(Ok(_))),
            "a write the log could not hold is not acknowledged: {refused:?}"
        );
        let fatal = tokio::time::timeout(Duration::from_secs(10), node.consensus_stopped())
            .await
            .expect("consensus reports its stop");
        assert!(
            fatal.is_some(),
            "a failed log write is a fatal stop, which takes the node out of service"
        );
        drop(node.shutdown().await);
    }
    let engine = open_engine(dir.path());
    let node = RaftNode::open(1, Arc::clone(&engine))
        .await
        .expect("the node starts again on the same store");
    assert!(
        engine
            .get(Partition::Node, b"node:before")
            .expect("read")
            .is_some(),
        "the write acknowledged before the failure is kept"
    );
    node.raft()
        .client_write(write("node:again", 3))
        .await
        .expect("the restarted node commits");
    node.shutdown().await.expect("shutdown");
}
