use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_replicate::{ConsumerKind, SeqnoConsumerRegistry};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use tonic::Request;

use super::ChangeEventServiceImpl;
use crate::proto::replication::cdc::change_stream_service_server::ChangeStreamService;
use crate::proto::replication::cdc::SubscribeRequest;
use crate::registry::{build_consumer_registry, RegistryTuning};

/// Open a fresh single-endpoint engine in a temp directory, mirroring the
/// registry-construction tests so the CDC service exercises the same wiring
/// production runs.
fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle).expect("open engine"));
    (engine, dir)
}

/// A CDC subscription registers an `oplog_events` consumer for its shard so the
/// oplog retention floor is held at the reader's position (ADR-028), and the
/// registration is released when the stream is dropped (client disconnect).
/// This is the per-consumer retention contract R137d wires; without it the
/// registry would only enforce the static time-travel window.
#[tokio::test]
async fn subscribe_registers_then_unregisters_cdc_consumer() {
    let (engine, _engine_dir) = open_engine();
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));
    // Hold `_bg` for the test's lifetime: dropping it stops the background
    // service that flushes heartbeats and drives eviction.
    let (registry, _bg) = build_consumer_registry(engine, pipeline, 1, RegistryTuning::default());

    let data_dir = tempfile::tempdir().expect("data dir");
    let service = ChangeEventServiceImpl::new(data_dir.path().to_path_buf(), registry.clone());

    // No oplog dir exists yet, so the stream is empty (caught up immediately)
    // — registration happens synchronously inside `subscribe` regardless.
    let response = service
        .subscribe(Request::new(SubscribeRequest {
            resume_token: None,
            filters: None,
        }))
        .await
        .expect("subscribe");

    let consumers = registry.list_consumers();
    let cdc: Vec<_> = consumers
        .iter()
        .filter(|c| c.kind == ConsumerKind::OplogEvents && c.consumer_id.starts_with("cdc-"))
        .collect();
    assert_eq!(
        cdc.len(),
        1,
        "subscribe must register exactly one cdc oplog_events consumer, got {consumers:?}"
    );

    // Dropping the response drops the receiver stream; the tailing task notices
    // the closed channel on its next poll and unregisters.
    drop(response);

    // Poll interval is 100ms; give the task several cycles to observe the
    // disconnect and release the registration.
    let mut released = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let still_present = registry
            .list_consumers()
            .iter()
            .any(|c| c.kind == ConsumerKind::OplogEvents && c.consumer_id.starts_with("cdc-"));
        if !still_present {
            released = true;
            break;
        }
    }
    assert!(
        released,
        "cdc consumer must be unregistered after the stream is dropped"
    );
}
