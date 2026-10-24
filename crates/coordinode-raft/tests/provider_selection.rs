#![allow(clippy::expect_used)]
//! Local consensus does not choose a process-wide TLS backend for its host.

use std::sync::Arc;

use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_wire::tls::CryptoProvider;

// Deployment identity allocation must work without occupying the host's TLS slot.
#[tokio::test(flavor = "multi_thread")]
async fn local_consensus_does_not_select_a_tls_provider() {
    assert!(CryptoProvider::get_default().is_none());
    let directory = tempfile::tempdir().expect("storage directory");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        directory.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("storage engine"));
    let node = RaftNode::single_node(engine).await.expect("local consensus");
    let deployment = node.deployment();
    node.shutdown().await.expect("shutdown");
    assert!(deployment.is_some(), "the deployment identity was recorded");
    assert!(CryptoProvider::get_default().is_none());
}
