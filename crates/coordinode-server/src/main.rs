//! CoordiNode server binary.
//!
//! Usage:
//!   coordinode serve [--addr ADDR] [--data DIR]
//!   coordinode version
//!   coordinode verify [--data DIR] [--deep]
//!
//! # Cluster-ready notes
//! - gRPC server is stateless — all state in CoordiNode storage.
//! - In CE 3-node HA: each node runs identical gRPC server.
//! - Inter-node communication uses the same :7080 port (distributed mode).

use std::net::SocketAddr;
use std::sync::Arc;

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, Tier};
use coordinode_storage::Guard;
use tonic::transport::Server;
use tracing::info;

pub mod proto {
    pub mod common {
        tonic::include_proto!("coordinode.v1.common");
    }
    pub mod graph {
        tonic::include_proto!("coordinode.v1.graph");
    }
    pub mod query {
        tonic::include_proto!("coordinode.v1.query");
    }
    pub mod health {
        tonic::include_proto!("coordinode.v1.health");
    }
    pub mod replication {
        pub mod cdc {
            tonic::include_proto!("coordinode.v1.replication");
        }
        // Re-export replication types at this level so generated code for other
        // proto packages that import coordinode.v1.replication can resolve them
        // via `super::replication::TypeName`.
        pub use cdc::ReadConcern;
        pub use cdc::ReadConcernLevel;
        pub use cdc::ReadPreference;
        pub use cdc::WriteConcern;
        pub use cdc::WriteConcernLevel;
    }
    pub mod admin {
        pub mod cluster {
            tonic::include_proto!("coordinode.v1.admin");
        }
    }
}

mod cli;
mod config;
mod grpc;
mod logging;
mod metrics_catalog;
mod ops;
mod services;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command = cli::parse_args();

    match command {
        cli::Command::Version => {
            println!("coordinode v{}", env!("CARGO_PKG_VERSION"));
        }

        cli::Command::Verify { data_dir, deep } => {
            logging::init_logging();
            info!(data_dir = %data_dir, deep = deep, "verifying storage integrity");

            let config = coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
                EndpointConfig::new(
                    "default",
                    &data_dir,
                    Media::Hdd,
                    Durability::Durable,
                    Tier::Warm,
                ),
            ]);
            let engine = coordinode_storage::engine::core::StorageEngine::open(&config)?;
            let disk = engine.disk_space()?;
            info!(disk_bytes = disk, "storage opened successfully");

            if deep {
                info!("deep verification: scanning all partitions...");
                for &part in coordinode_storage::engine::partition::Partition::all() {
                    let iter = engine.prefix_scan(part, b"")?;
                    let mut count = 0u64;
                    for guard in iter {
                        let _ = guard.into_inner()?;
                        count += 1;
                    }
                    info!(
                        partition = part.name(),
                        entries = count,
                        "partition verified"
                    );
                }
            }

            info!("verification complete");
        }

        cli::Command::Checkpoint { data_dir, output } => {
            logging::init_logging();
            info!(data_dir = %data_dir, output = %output, "creating checkpoint");

            let config = coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
                EndpointConfig::new(
                    "default",
                    &data_dir,
                    Media::Hdd,
                    Durability::Durable,
                    Tier::Warm,
                ),
            ]);
            let engine = coordinode_storage::engine::core::StorageEngine::open(&config)?;
            let summary = engine
                .create_checkpoint(std::path::Path::new(&output))
                .map_err(|e| format!("checkpoint failed: {e}"))?;
            info!(
                partitions = summary.partitions,
                copied_bytes = summary.total_bytes,
                oplog_bytes = summary.oplog_bytes,
                max_seqno = summary.max_seqno,
                output = %output,
                "checkpoint complete"
            );
        }

        cli::Command::Serve {
            mode,
            node_id,
            grpc_addr,
            advertise_addr,
            #[cfg(feature = "rest-proxy")]
            rest_addr,
            ops_addr,
            data_dir,
            peers,
        } => {
            logging::init_logging();

            let addr: SocketAddr = grpc_addr.parse()?;
            // Advertise address is what peers use to connect to this node.
            // Falls back to grpc_addr when not explicitly set.
            let effective_advertise = advertise_addr.unwrap_or_else(|| grpc_addr.clone());
            let cluster_mode = peers.is_some();
            info!(
                data_dir = %data_dir,
                mode = %mode,
                node_id = node_id,
                cluster = cluster_mode,
                advertise = %effective_advertise,
                "coordinode v{} starting on {addr}",
                env!("CARGO_PKG_VERSION")
            );

            coordinode_vector::metrics::log_simd_capabilities();

            // All modes use RaftProposalPipeline — unified write path.
            //
            // - Standalone (no --peers): single-node Raft (node_id=1, StubNetwork).
            //   Writes go through Raft → oplog always populated → CDC works in both modes.
            // - Cluster (--peers): multi-node Raft (GrpcNetwork, leader election).
            //   Writes replicated to followers before commit.
            //
            // `raft_node_shared` provides the read fence (R141), ClusterService
            // administration, and ensures consistent apply ordering via oracle.

            // Common setup: open storage engine + timestamp oracle.
            let storage_config =
                coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
                    EndpointConfig::new(
                        "default",
                        &data_dir,
                        Media::Hdd,
                        Durability::Durable,
                        Tier::Warm,
                    ),
                ]);
            let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
            let engine = coordinode_storage::engine::core::StorageEngine::open_with_oracle(
                &storage_config,
                oracle.clone(),
            )
            .map_err(|e| format!("failed to open storage: {e}"))?;
            let engine = Arc::new(engine);

            // Open Raft node and build database — both modes use RaftProposalPipeline.
            //
            // Three construction paths:
            //
            // 1. Standalone (no --peers): single-node Raft via StubNetworkFactory.
            //    No gRPC Raft handler needed — no peers can connect.
            //
            // 2. Cluster, node_id == 1 (bootstrap leader): open_cluster_embedded().
            //    Calls initialize(), returns a RaftGrpcHandler for the main router.
            //
            // 3. Cluster, node_id > 1 (joining node): open_joining_embedded().
            //    Does NOT call initialize(). Waits for leader to add it via
            //    `coordinode admin node join`. Returns a RaftGrpcHandler for the
            //    main router.
            //
            // In cases 2 and 3, RaftServiceServer is registered at the end of
            // router construction so inter-node Raft RPCs share the :7080 port.
            let (raft_node, raft_grpc_handler) = if let Some(ref peers_list) = peers {
                let peer_count = peers_list.len();
                if node_id == 1 {
                    info!(
                        peers = peer_count,
                        node_id, "cluster mode: bootstrap leader (open_cluster_embedded)"
                    );
                    let (rn, handler) = coordinode_raft::cluster::RaftNode::open_cluster_embedded(
                        node_id,
                        Arc::clone(&engine),
                        effective_advertise,
                    )
                    .await
                    .map_err(|e| format!("failed to open cluster Raft node: {e}"))?;
                    (rn, Some(handler))
                } else {
                    info!(
                        peers = peer_count,
                        node_id, "cluster mode: joining node (open_joining_embedded)"
                    );
                    let (rn, handler) = coordinode_raft::cluster::RaftNode::open_joining_embedded(
                        node_id,
                        Arc::clone(&engine),
                    )
                    .await
                    .map_err(|e| format!("failed to open joining Raft node: {e}"))?;
                    (rn, Some(handler))
                }
            } else {
                info!(node_id, "standalone mode: single-node Raft (StubNetwork)");
                let rn = coordinode_raft::cluster::RaftNode::open_with_oracle(
                    node_id,
                    Arc::clone(&engine),
                    Some(Arc::clone(&oracle)),
                )
                .await
                .map_err(|e| format!("failed to open Raft node: {e}"))?;
                (rn, None)
            };

            let raft_node = Arc::new(raft_node);

            let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> = Arc::new(
                coordinode_raft::proposal::RaftProposalPipeline::new(Arc::clone(raft_node.raft())),
            );

            // no-std: spin::RwLock (drop-in).
            let database = Arc::new(parking_lot::RwLock::new(
                coordinode_embed::Database::from_engine(
                    &data_dir,
                    Arc::clone(&engine),
                    oracle.clone(),
                    pipeline,
                )
                .map_err(|e| format!("failed to open database: {e}"))?,
            ));

            let raft_node_shared: Option<Arc<coordinode_raft::cluster::RaftNode>> =
                Some(Arc::clone(&raft_node));

            // Refresh node-local derived state when replicated entries
            // apply: property values are encoded against interner ids,
            // and a follower that never refreshes its in-memory interner
            // resolves every replicated property to null. The refresh is
            // a cheap length pre-check unless the mapping actually grew.
            if peers.is_some() {
                let mut applied_rx = raft_node.subscribe_applied();
                let db = Arc::clone(&database);
                tokio::spawn(async move {
                    while applied_rx.changed().await.is_ok() {
                        let guard = db.read();
                        if let Err(e) = guard.refresh_field_interner() {
                            tracing::warn!(%e, "field interner refresh failed");
                        }
                        // Replicated CREATE VECTOR INDEX definitions are
                        // brought live here: register + local HNSW rebuild
                        // (the graph itself is never replicated).
                        match guard.refresh_vector_indexes() {
                            Ok(0) => {}
                            Ok(n) => tracing::info!(n, "vector indexes brought live from apply"),
                            Err(e) => tracing::warn!(%e, "vector index refresh failed"),
                        }
                    }
                });
            }

            let query_registry = Arc::new(coordinode_query::advisor::QueryRegistry::new());
            let nplus1_detector =
                Arc::new(coordinode_query::advisor::nplus1::NPlus1Detector::new());

            let graph_service = services::graph::GraphServiceImpl::new(Arc::clone(&database));
            let schema_service = services::schema::SchemaServiceImpl::new(Arc::clone(&database));
            let cypher_service = {
                let svc = services::cypher::CypherServiceImpl::new(
                    Arc::clone(&database),
                    Arc::clone(&query_registry),
                    Arc::clone(&nplus1_detector),
                );
                if let Some(ref rn) = raft_node_shared {
                    svc.with_raft_node(Arc::clone(rn))
                } else {
                    svc
                }
            };
            let vector_service = services::vector::VectorServiceImpl::new(Arc::clone(&database));
            let text_service = services::text::TextServiceImpl::new(Arc::clone(&database));
            let health_service = services::health::HealthServiceImpl;
            // CDC service: tails oplog/<shard>/ dir. Empty stream in embedded mode
            // (no oplog); populated in Raft cluster mode (LogStore writes oplog).
            let cdc_service =
                services::cdc::ChangeEventServiceImpl::new(std::path::PathBuf::from(&data_dir));

            // ClusterService: cluster join/leave lifecycle.
            // Available only in cluster mode (requires a RaftNode).
            let cluster_service = raft_node_shared
                .as_ref()
                .map(|rn| services::cluster::ClusterServiceImpl::new(Arc::clone(rn)));

            // BlobService shares the same storage engine as the Database.
            // Read guard is dropped immediately — only need engine_shared().
            let blob_engine = database.read().engine_shared();
            let blob_service = services::blob::BlobServiceImpl::new(blob_engine);

            // Spawn operational HTTP server (default :7084, configurable via --ops-addr).
            let ops_sock: SocketAddr = ops_addr.parse()?;
            tokio::spawn(async move {
                if let Err(e) = ops::start_ops_server(ops_sock).await {
                    tracing::error!("ops server error: {e}");
                }
            });

            // Spawn embedded REST/JSON proxy (default :7081, configurable via --rest-addr).
            // Transcodes HTTP/JSON requests to gRPC via google.api.http annotations.
            // Compiled only when the `rest-proxy` feature is enabled (default).
            // Disable for embedded/mobile builds: --no-default-features --features vector,full-text
            #[cfg(feature = "rest-proxy")]
            {
                use structured_proxy::config::{
                    DescriptorSource, ListenConfig, ProxyConfig, ServiceConfig, UpstreamConfig,
                };
                static DESCRIPTOR_BYTES: &[u8] =
                    include_bytes!("../../../coordinode.descriptor.bin");
                let grpc_upstream = format!("http://127.0.0.1:{}", addr.port());
                let config = ProxyConfig {
                    upstream: UpstreamConfig {
                        default: grpc_upstream,
                    },
                    descriptors: vec![DescriptorSource::Embedded {
                        bytes: DESCRIPTOR_BYTES,
                    }],
                    listen: ListenConfig { http: rest_addr },
                    service: ServiceConfig {
                        name: "coordinode".into(),
                    },
                    aliases: vec![],
                    openapi: None,
                    auth: None,
                    shield: None,
                    oidc_discovery: None,
                    maintenance: Default::default(),
                    cors: Default::default(),
                    logging: Default::default(),
                    metrics_classes: vec![],
                    forwarded_headers: vec![
                        "authorization".into(),
                        "x-request-id".into(),
                        "x-forwarded-for".into(),
                        "x-real-ip".into(),
                        "user-agent".into(),
                        "accept-language".into(),
                    ],
                };
                let proxy = structured_proxy::ProxyServer::from_config(config);
                tokio::spawn(async move {
                    if let Err(e) = proxy.serve().await {
                        tracing::error!("REST proxy error: {e}");
                    }
                });
            }

            info!(
                port = addr.port(),
                node_id,
                mode = %mode,
                "gRPC server listening"
            );

            // Graceful shutdown: wait for SIGTERM (Docker / test harness) or Ctrl+C.
            // When the signal fires, `serve_with_shutdown` stops accepting new
            // connections and waits for in-flight RPCs to complete before returning.
            // The returned future resolves → all Arc<Database> / Arc<RaftNode> drop
            // → StorageEngine::Drop flushes all memtables to SST files.
            #[cfg(unix)]
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .map_err(|e| format!("failed to install SIGTERM handler: {e}"))?;

            let shutdown = async move {
                #[cfg(unix)]
                tokio::select! {
                    _ = sigterm.recv() => {
                        info!("SIGTERM received — initiating graceful shutdown");
                    }
                    _ = tokio::signal::ctrl_c() => {
                        info!("Ctrl+C received — initiating graceful shutdown");
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = tokio::signal::ctrl_c().await;
                    info!("Ctrl+C received — initiating graceful shutdown");
                }
            };

            // NodeInfoLayer: inject x-coordinode-node / x-coordinode-hops /
            // x-coordinode-load response headers on every gRPC response.
            let mut builder = Server::builder().layer(grpc::NodeInfoLayer::new(node_id));
            let mut router = builder
                .add_service(proto::graph::graph_service_server::GraphServiceServer::new(
                    graph_service,
                ))
                .add_service(
                    proto::graph::schema_service_server::SchemaServiceServer::new(schema_service),
                )
                .add_service(
                    proto::query::cypher_service_server::CypherServiceServer::new(cypher_service),
                )
                .add_service(
                    proto::query::vector_service_server::VectorServiceServer::new(vector_service),
                )
                .add_service(
                    proto::query::text_service_server::TextServiceServer::new(text_service),
                )
                .add_service(
                    proto::health::health_service_server::HealthServiceServer::new(health_service),
                )
                .add_service(proto::graph::blob_service_server::BlobServiceServer::new(
                    blob_service,
                ))
                .add_service(
                    proto::replication::cdc::change_stream_service_server::ChangeStreamServiceServer::new(
                        cdc_service,
                    ),
                );

            // Register ClusterService only in cluster mode (requires Raft node).
            if let Some(cs) = cluster_service {
                router = router.add_service(
                    proto::admin::cluster::cluster_service_server::ClusterServiceServer::new(cs),
                );
                info!("ClusterService registered — cluster join/leave management available");
            }

            // Register RaftService in cluster mode — embedded into :7080 so
            // inter-node Raft RPCs share the main gRPC port (no separate server).
            if let Some(handler) = raft_grpc_handler {
                use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
                router = router.add_service(RaftServiceServer::new(handler));
                info!(node_id, "RaftService registered on :7080 (shared port)");
            }

            router.serve_with_shutdown(addr, shutdown).await?;
        }

        cli::Command::Backup {
            data_dir,
            output,
            format,
            namespace: _namespace,
            since,
        } => {
            logging::init_logging();
            info!(
                data_dir = %data_dir,
                output = %output,
                format = ?format,
                "starting backup"
            );

            let db = coordinode_embed::Database::open(&data_dir)
                .map_err(|e| format!("failed to open database: {e}"))?;

            let snapshot = db.engine().snapshot();
            let shard_id = 1u16;

            let file = std::fs::File::create(&output)
                .map_err(|e| format!("failed to create output file '{output}': {e}"))?;
            let mut writer = std::io::BufWriter::new(file);

            let stats = match format {
                coordinode_embed::backup::BackupFormat::Json => {
                    coordinode_embed::backup::export::export_json(
                        db.engine(),
                        &db.interner(),
                        shard_id,
                        &snapshot,
                        &mut writer,
                    )
                    .map_err(|e| format!("backup failed: {e}"))?
                }
                coordinode_embed::backup::BackupFormat::Cypher => {
                    coordinode_embed::backup::export::export_cypher(
                        db.engine(),
                        &db.interner(),
                        shard_id,
                        &snapshot,
                        &mut writer,
                    )
                    .map_err(|e| format!("backup failed: {e}"))?
                }
                coordinode_embed::backup::BackupFormat::Binary => {
                    coordinode_embed::backup::export::export_binary(
                        db.engine(),
                        &db.interner(),
                        shard_id,
                        &snapshot,
                        &mut writer,
                    )
                    .map_err(|e| format!("backup failed: {e}"))?
                }
                coordinode_embed::backup::BackupFormat::ApocJson
                | coordinode_embed::backup::BackupFormat::ApocCypher
                | coordinode_embed::backup::BackupFormat::HetioJson => {
                    return Err("apoc-json, apoc-cypher and hetio-json are import-only \
                                formats; use them with restore, not backup"
                        .into());
                }
                coordinode_embed::backup::BackupFormat::RaftSnapshot => {
                    // Self-contained whole-database blob, not the entity-counted
                    // logical export. The Raft snapshot omits the `meta:` Schema
                    // keys (per-node config) including the field interner, so a
                    // standalone backup frames the interner and a mode byte ahead
                    // of it: [mode u8][u32 interner_len][interner][snapshot],
                    // where mode 0 = full, 1 = incremental (changes after a seqno).
                    use std::io::Write;
                    let interner_bytes = db.interner().to_bytes();
                    let current_seqno: u64 = db.engine().snapshot();
                    let (mode, snapshot): (u8, Vec<u8>) = match since {
                        Some(since_seqno) => {
                            let ts =
                                coordinode_core::txn::timestamp::Timestamp::from_raw(since_seqno);
                            match coordinode_raft::snapshot::build_incremental_snapshot(
                                db.engine(),
                                ts,
                            )
                            .map_err(|e| format!("backup failed: {e}"))?
                            {
                                Some(delta) => (1u8, delta),
                                None => {
                                    info!(
                                        since = since_seqno,
                                        "no changes since seqno; empty incremental backup"
                                    );
                                    (1u8, Vec::new())
                                }
                            }
                        }
                        None => {
                            let full = coordinode_raft::snapshot::build_full_snapshot(db.engine())
                                .map_err(|e| format!("backup failed: {e}"))?;
                            (0u8, full)
                        }
                    };
                    let interner_len = u32::try_from(interner_bytes.len())
                        .map_err(|_| "field interner too large to frame".to_string())?;
                    writer
                        .write_all(&[mode])
                        .and_then(|()| writer.write_all(&interner_len.to_be_bytes()))
                        .and_then(|()| writer.write_all(&interner_bytes))
                        .and_then(|()| writer.write_all(&snapshot))
                        .and_then(|()| writer.flush())
                        .map_err(|e| format!("backup write failed: {e}"))?;
                    info!(
                        mode = if mode == 1 { "incremental" } else { "full" },
                        seqno = current_seqno,
                        interner_bytes = interner_bytes.len(),
                        snapshot_bytes = snapshot.len(),
                        "backup complete (raft-snapshot); pass --since {current_seqno} \
                         for the next incremental"
                    );
                    return Ok(());
                }
            };

            info!(
                nodes = stats.nodes,
                edges = stats.edges,
                output = %output,
                "backup complete"
            );
        }

        cli::Command::Restore {
            data_dir,
            input,
            format,
            namespace: _namespace,
            only_labels,
        } => {
            logging::init_logging();
            info!(
                data_dir = %data_dir,
                input = %input,
                format = ?format,
                "starting restore"
            );

            // Selective restore label filter (json / apoc-json / hetio-json).
            let label_filter: Option<std::collections::HashSet<String>> = if only_labels.is_empty()
            {
                None
            } else {
                Some(only_labels.into_iter().collect())
            };

            let db = coordinode_embed::Database::open(&data_dir)
                .map_err(|e| format!("failed to open database: {e}"))?;

            let file = std::fs::File::open(&input)
                .map_err(|e| format!("failed to open input file '{input}': {e}"))?;
            // Transparently decompress a bzip2/gzip-compressed input (tool-side,
            // pure-Rust). Uncompressed input passes through unchanged.
            let mut reader = decompressing_reader(file)
                .map_err(|e| format!("failed to read input file '{input}': {e}"))?;

            match format {
                coordinode_embed::backup::BackupFormat::Json => {
                    let mut interner = db.interner().clone();
                    let shard_id = 1u16;
                    let stats = coordinode_embed::backup::restore::restore_json(
                        db.engine(),
                        &mut interner,
                        shard_id,
                        &mut reader,
                        label_filter.as_ref(),
                    )
                    .map_err(|e| format!("restore failed: {e}"))?;
                    info!(
                        nodes = stats.nodes,
                        edges = stats.edges,
                        schema = stats.schema_entries,
                        "restore complete (json)"
                    );
                }
                coordinode_embed::backup::BackupFormat::Binary => {
                    let (stats, _interner) =
                        coordinode_embed::backup::restore::restore_binary(db.engine(), &mut reader)
                            .map_err(|e| format!("restore failed: {e}"))?;
                    info!(
                        nodes = stats.nodes,
                        edges = stats.edges,
                        schema = stats.schema_entries,
                        "restore complete (binary)"
                    );
                }
                coordinode_embed::backup::BackupFormat::Cypher => {
                    let mut interner = db.interner().clone();
                    let shard_id = 1u16;
                    let stats = coordinode_embed::backup::restore::restore_cypher(
                        db.engine(),
                        &mut interner,
                        shard_id,
                        &mut reader,
                    )
                    .map_err(|e| format!("restore failed: {e}"))?;
                    *db.interner_arc().write() = interner;
                    info!(
                        nodes = stats.nodes,
                        edges = stats.edges,
                        schema = stats.schema_entries,
                        "restore complete (cypher)"
                    );
                }
                coordinode_embed::backup::BackupFormat::ApocJson => {
                    let mut interner = db.interner().clone();
                    let shard_id = 1u16;
                    let stats = coordinode_embed::backup::restore::restore_apoc_json(
                        db.engine(),
                        &mut interner,
                        shard_id,
                        &mut reader,
                        label_filter.as_ref(),
                    )
                    .map_err(|e| format!("restore failed: {e}"))?;
                    *db.interner_arc().write() = interner;
                    info!(
                        nodes = stats.nodes,
                        edges = stats.edges,
                        schema = stats.schema_entries,
                        "restore complete (apoc-json)"
                    );
                }
                coordinode_embed::backup::BackupFormat::ApocCypher => {
                    let mut interner = db.interner().clone();
                    let shard_id = 1u16;
                    let stats = coordinode_embed::backup::restore::restore_apoc_cypher(
                        db.engine(),
                        &mut interner,
                        shard_id,
                        &mut reader,
                    )
                    .map_err(|e| format!("restore failed: {e}"))?;
                    *db.interner_arc().write() = interner;
                    info!(
                        nodes = stats.nodes,
                        edges = stats.edges,
                        schema = stats.schema_entries,
                        "restore complete (apoc-cypher)"
                    );
                }
                coordinode_embed::backup::BackupFormat::HetioJson => {
                    let mut interner = db.interner().clone();
                    let shard_id = 1u16;
                    let stats = coordinode_embed::backup::restore::restore_hetio_json(
                        db.engine(),
                        &mut interner,
                        shard_id,
                        &mut reader,
                        label_filter.as_ref(),
                    )
                    .map_err(|e| format!("restore failed: {e}"))?;
                    *db.interner_arc().write() = interner;
                    info!(
                        nodes = stats.nodes,
                        edges = stats.edges,
                        schema = stats.schema_entries,
                        "restore complete (hetio-json)"
                    );
                }
                coordinode_embed::backup::BackupFormat::RaftSnapshot => {
                    use std::io::Read;
                    let mut data = Vec::new();
                    reader
                        .read_to_end(&mut data)
                        .map_err(|e| format!("restore read failed: {e}"))?;
                    // Frame: [mode u8][u32 interner_len][interner][snapshot].
                    // Restore the framed interner first (the snapshot omits it),
                    // then install per mode (0 full, 1 incremental).
                    if data.len() < 5 {
                        return Err("raft-snapshot file truncated (no frame header)".into());
                    }
                    let mode = data[0];
                    let interner_len =
                        u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as usize;
                    let body = &data[5..];
                    if body.len() < interner_len {
                        return Err("raft-snapshot file truncated (interner body)".into());
                    }
                    let (interner_bytes, snapshot) = body.split_at(interner_len);
                    db.persist_field_interner_bytes(interner_bytes)
                        .map_err(|e| format!("restore interner failed: {e}"))?;
                    match mode {
                        0 => {
                            coordinode_raft::snapshot::install_full_snapshot(db.engine(), snapshot)
                                .map_err(|e| format!("restore failed: {e}"))?
                        }
                        1 if snapshot.is_empty() => {
                            info!("incremental backup had no changes; nothing to apply");
                        }
                        1 => coordinode_raft::snapshot::install_incremental_snapshot(
                            db.engine(),
                            snapshot,
                        )
                        .map_err(|e| format!("restore failed: {e}"))?,
                        other => {
                            return Err(format!("unknown snapshot mode byte: {other}").into());
                        }
                    }
                    info!(
                        mode = if mode == 1 { "incremental" } else { "full" },
                        interner_bytes = interner_len,
                        snapshot_bytes = snapshot.len(),
                        "restore complete (raft-snapshot)"
                    );
                }
            }
        }

        cli::Command::AdminNodeJoin {
            cluster_addr,
            node_id,
            node_addr,
            pre_seeded,
            follow,
        } => {
            admin_node_join(cluster_addr, node_id, node_addr, pre_seeded, follow).await?;
        }

        cli::Command::AdminNodeDecommission {
            cluster_addr,
            node_id,
            pruning,
            force,
            skip_confirmation,
        } => {
            admin_node_decommission(cluster_addr, node_id, pruning, force, skip_confirmation)
                .await?;
        }
    }

    Ok(())
}

/// Wrap a restore input file so a bzip2- or gzip-compressed dump is
/// transparently decompressed before parsing. The leading magic bytes are
/// sniffed; an uncompressed file passes through unchanged. Decompression lives
/// tool-side (this binary) with pure-Rust decoders only (bzip2-rs decompress,
/// flate2/miniz_oxide) so the database runtime never links a compression
/// codec. A zstd-compressed input is rejected with guidance rather than
/// silently mishandled.
fn decompressing_reader<R: std::io::Read + 'static>(
    mut reader: R,
) -> std::io::Result<Box<dyn std::io::BufRead>> {
    use std::io::Read;
    let mut magic = [0u8; 4];
    let mut filled = 0;
    while filled < magic.len() {
        match reader.read(&mut magic[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    let head = magic[..filled].to_vec();
    let chained = std::io::Cursor::new(head.clone()).chain(reader);
    if head.starts_with(b"BZh") {
        Ok(Box::new(std::io::BufReader::new(
            bzip2_rs::DecoderReader::new(chained),
        )))
    } else if head.starts_with(&[0x1f, 0x8b]) {
        Ok(Box::new(std::io::BufReader::new(
            flate2::read::GzDecoder::new(chained),
        )))
    } else if head.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Err(std::io::Error::other(
            "zstd-compressed restore input is not yet supported; decompress it first",
        ))
    } else {
        Ok(Box::new(std::io::BufReader::new(chained)))
    }
}

/// Execute `coordinode admin node decommission` — connect to a running cluster and
/// gracefully decommission a node via the Phase 0-2 protocol.
///
/// Steps:
/// 1. Connect to any cluster member via gRPC.
/// 2. Call `ClusterService.DecommissionNode` — executes quorum gate, leadership
///    transfer (if target is leader), and membership remove.
/// 3. Print the result including any advisory cleanup message.
async fn admin_node_decommission(
    cluster_addr: String,
    node_id: u64,
    pruning: bool,
    force: bool,
    skip_confirmation: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use proto::admin::cluster::{
        cluster_service_client::ClusterServiceClient, DecommissionNodeRequest,
    };

    if force && !skip_confirmation {
        eprintln!(
            "error: --force requires --skip-confirmation to acknowledge potential data loss.\n\
             Emergency decommission may cause permanent data loss if the node held\n\
             the only copy of any data. Re-run with both --force --skip-confirmation."
        );
        std::process::exit(1);
    }

    let endpoint = if cluster_addr.starts_with("http://") || cluster_addr.starts_with("https://") {
        cluster_addr.clone()
    } else {
        format!("http://{cluster_addr}")
    };

    eprintln!("Connecting to cluster at {endpoint} ...");

    let mut client = ClusterServiceClient::connect(endpoint)
        .await
        .map_err(|e| format!("failed to connect to cluster: {e}"))?;

    eprintln!(
        "Decommissioning node {node_id}{}{}...",
        if pruning { " (--pruning)" } else { "" },
        if force { " [EMERGENCY --force]" } else { "" },
    );

    let resp = client
        .decommission_node(DecommissionNodeRequest {
            node_id,
            pruning,
            force,
            skip_confirmation,
        })
        .await
        .map_err(|e| format!("DecommissionNode failed: {e}"))?
        .into_inner();

    eprintln!("Decommission complete: {}", resp.message);

    if resp.operator_cleanup_required {
        eprintln!(
            "\nNOTE: Data cleanup required on node {node_id}.\n\
             CE does not automatically wipe decommissioned node data.\n\
             Operator must manually delete the data directory on node {node_id}\n\
             after verifying the node is no longer serving traffic."
        );
    }

    Ok(())
}

/// Execute `coordinode admin node join` — connect to a running cluster and initiate
/// the full join lifecycle for a new node.
///
/// Steps:
/// 1. Connect to any cluster member via gRPC.
/// 2. Call `ClusterService.JoinNode` — adds node as Learner, starts background promotion.
/// 3. If `--follow`, subscribe to `JoinProgress` stream until COMPLETE/FAILED.
async fn admin_node_join(
    cluster_addr: String,
    node_id: u64,
    node_addr: String,
    pre_seeded: bool,
    follow: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use proto::admin::cluster::{
        cluster_service_client::ClusterServiceClient, JoinNodeRequest, JoinPhase,
        JoinProgressRequest,
    };

    // Normalize cluster_addr to include http:// scheme for tonic.
    let endpoint = if cluster_addr.starts_with("http://") || cluster_addr.starts_with("https://") {
        cluster_addr.clone()
    } else {
        format!("http://{cluster_addr}")
    };

    eprintln!("Connecting to cluster at {endpoint} ...");

    let mut client = ClusterServiceClient::connect(endpoint)
        .await
        .map_err(|e| format!("failed to connect to cluster: {e}"))?;

    eprintln!("Initiating join for node {node_id} at {node_addr} ...");

    let resp = client
        .join_node(JoinNodeRequest {
            node_id,
            address: node_addr.clone(),
            pre_seeded,
        })
        .await
        .map_err(|e| format!("JoinNode failed: {e}"))?
        .into_inner();

    eprintln!("JoinNode: {} (node_id={})", resp.status, resp.node_id);

    if !follow {
        eprintln!(
            "Join initiated. Use `--follow` to stream progress, \
             or poll `GetClusterStatus` to monitor lag."
        );
        return Ok(());
    }

    // Stream JoinProgress until COMPLETE or FAILED.
    eprintln!("Streaming join progress (Ctrl+C to detach) ...");

    let mut stream = client
        .join_progress(JoinProgressRequest { node_id })
        .await
        .map_err(|e| format!("JoinProgress failed: {e}"))?
        .into_inner();

    use tokio_stream::StreamExt as _;

    while let Some(status) = stream.next().await {
        let s = status.map_err(|e| format!("stream error: {e}"))?;

        let phase_name = match s.phase {
            p if p == JoinPhase::Learner as i32 => "LEARNER",
            p if p == JoinPhase::ReadyCheck as i32 => "READY_CHECK",
            p if p == JoinPhase::Promoting as i32 => "PROMOTING",
            p if p == JoinPhase::Complete as i32 => "COMPLETE",
            p if p == JoinPhase::Failed as i32 => "FAILED",
            _ => "UNKNOWN",
        };

        if s.lag_entries == 0 && s.phase == JoinPhase::Learner as i32 {
            // lag_entries=0 in LEARNER phase means "not yet known"
            eprintln!("[{phase_name}] {}% — {}", s.percent, s.message);
        } else {
            eprintln!(
                "[{phase_name}] {}% lag={} — {}",
                s.percent, s.lag_entries, s.message
            );
        }

        match s.phase {
            p if p == JoinPhase::Complete as i32 => {
                eprintln!("Node {node_id} successfully joined as Voter.");
                break;
            }
            p if p == JoinPhase::Failed as i32 => {
                return Err(format!("Node {node_id} join failed: {}", s.message).into());
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod decompress_tests {
    use super::decompressing_reader;
    use std::io::{Read, Write};

    #[test]
    fn gzip_input_is_transparently_decompressed() {
        let plain = b"hello\nworld\n";
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(plain).unwrap();
        let gz = enc.finish().unwrap();
        let mut r = decompressing_reader(std::io::Cursor::new(gz)).unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, plain);
    }

    #[test]
    fn uncompressed_input_passes_through() {
        let plain = b"{\"type\":\"node\"}\n";
        let mut r = decompressing_reader(std::io::Cursor::new(plain.to_vec())).unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, plain);
    }

    #[test]
    fn zstd_magic_is_rejected_with_guidance() {
        let zstd = vec![0x28u8, 0xb5, 0x2f, 0xfd, 0, 0, 0, 0];
        match decompressing_reader(std::io::Cursor::new(zstd)) {
            Err(e) => assert!(e.to_string().contains("zstd"), "got: {e}"),
            Ok(_) => panic!("expected zstd rejection"),
        }
    }
}
