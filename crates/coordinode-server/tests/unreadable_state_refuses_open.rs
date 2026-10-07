//! A replicated member opened the way the server opens it (a Raft node over
//! the engine, the database over its proposal pipeline) refuses stored state
//! it cannot read instead of serving without it: an index definition (the
//! index and the uniqueness it enforces would be gone) or the applied
//! membership (the member would forget its group).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::ProposalPipeline;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_embed::db::DatabaseError;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_storage::Guard as _;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::StorageError;
use coordinode_test_fixtures::alloc_port;

fn open_engine(dir: &std::path::Path, oracle: &Arc<TimestampOracle>) -> Arc<StorageEngine> {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    Arc::new(StorageEngine::open_with_oracle(&config, Arc::clone(oracle)).unwrap())
}

/// A single-voter member over `engine`, once it leads.
async fn open_member(engine: &Arc<StorageEngine>, port: u16) -> Result<RaftNode, String> {
    let (raft, _handler) =
        RaftNode::open_cluster_embedded(1, Arc::clone(engine), format!("http://127.0.0.1:{port}"))
            .await
            .map_err(|e| e.to_string())?;
    for _ in 0..150 {
        if raft.is_leader().await {
            return Ok(raft);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("the single voter never led");
}

fn pipeline(raft: &RaftNode) -> Arc<dyn ProposalPipeline> {
    Arc::new(RaftProposalPipeline::new(Arc::clone(raft.raft())))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_member_refuses_an_unreadable_index_definition() {
    let port = alloc_port();
    let dir = tempfile::tempdir().unwrap();
    let planted = {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = open_engine(dir.path(), &oracle);
        let raft = open_member(&engine, port).await.expect("open member");
        let mut db =
            Database::from_engine(dir.path(), Arc::clone(&engine), oracle, pipeline(&raft))
                .expect("open database");
        db.execute_cypher("CREATE CONSTRAINT user_key FOR (u:User) REQUIRE u.email IS UNIQUE")
            .expect("unique");
        // The definition in the shape the earlier catalog stored it: a
        // MessagePack array opening with the index name,
        // ["user_key", "User", ["email"], true].
        let old_shape: &[u8] = b"\x94\xa8user_key\xa4User\x91\xa5email\xc3";
        let key = engine
            .prefix_scan(Partition::Schema, b"schema:idx:")
            .expect("scan")
            .next()
            .expect("the constraint's index definition")
            .into_inner()
            .expect("read")
            .0
            .to_vec();
        engine
            .put(Partition::Schema, &key, old_shape)
            .expect("plant");
        drop(db);
        raft.shutdown().await.expect("shutdown");
        key
    };

    let oracle = Arc::new(TimestampOracle::new());
    let engine = open_engine(dir.path(), &oracle);
    let raft = open_member(&engine, port).await.expect("reopen member");
    match Database::from_engine(dir.path(), Arc::clone(&engine), oracle, pipeline(&raft)) {
        Err(DatabaseError::Storage(StorageError::UnreadableCatalog { key, .. })) => {
            assert_eq!(key, coordinode_storage::error::printable_key(&planted));
        }
        Err(other) => panic!("refused for another reason: {other}"),
        Ok(_) => panic!("served without the unique index it cannot read"),
    }
    raft.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_member_refuses_an_unreadable_applied_membership() {
    let port = alloc_port();
    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let engine = open_engine(dir.path(), &oracle);
    engine
        .put(
            Partition::Schema,
            b"raft:sm:membership",
            b"not-msgpack-bytes",
        )
        .expect("plant");

    match open_member(&engine, port).await {
        Err(e) => assert!(e.contains("raft:sm:membership"), "got: {e}"),
        Ok(_) => panic!("a member opened with an empty group in place of its membership"),
    }
}
