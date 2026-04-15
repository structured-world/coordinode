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
    }
    pub mod admin {
        pub mod cluster {
            tonic::include_proto!("coordinode.v1.admin");
        }
    }
}

mod cli;
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

            let config = coordinode_storage::engine::config::StorageConfig::new(&data_dir);
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

        cli::Command::Serve {
            grpc_addr,
            rest_addr,
            ops_addr,
            data_dir,
            peers,
        } => {
            logging::init_logging();

            let addr: SocketAddr = grpc_addr.parse()?;
            let cluster_mode = peers.is_some();
            info!(
                data_dir = %data_dir,
                cluster = cluster_mode,
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
            let storage_config = coordinode_storage::engine::config::StorageConfig::new(&data_dir);
            let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
            let engine = coordinode_storage::engine::core::StorageEngine::open_with_oracle(
                &storage_config,
                oracle.clone(),
            )
            .map_err(|e| format!("failed to open storage: {e}"))?;
            let engine = Arc::new(engine);

            // Open Raft node and build database — both modes use RaftProposalPipeline.
            let (database, raft_node_shared) = {
                let (mode_label, peer_count) = if peers.is_some() {
                    (
                        "cluster mode: RaftProposalPipeline (multi-node Raft)",
                        peers.as_ref().map(|p| p.len()).unwrap_or(0),
                    )
                } else {
                    (
                        "standalone mode: RaftProposalPipeline (single-node Raft)",
                        0,
                    )
                };
                if peer_count > 0 {
                    info!(peers = peer_count, "{mode_label}");
                } else {
                    info!("{mode_label}");
                }

                // Node ID 1 for single-node or bootstrap; peer discovery wiring
                // is deferred to R150 (monolithic binary --mode selection).
                let raft_node = coordinode_raft::cluster::RaftNode::open_with_oracle(
                    1,
                    Arc::clone(&engine),
                    Some(Arc::clone(&oracle)),
                )
                .await
                .map_err(|e| format!("failed to open Raft node: {e}"))?;
                let raft_node = Arc::new(raft_node);

                let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
                    Arc::new(coordinode_raft::proposal::RaftProposalPipeline::new(
                        Arc::clone(raft_node.raft()),
                    ));

                let db = Arc::new(std::sync::Mutex::new(
                    coordinode_embed::Database::from_engine(
                        &data_dir,
                        Arc::clone(&engine),
                        oracle.clone(),
                        pipeline,
                    )
                    .map_err(|e| format!("failed to open database: {e}"))?,
                ));
                (db, Arc::clone(&raft_node))
            };
            let raft_node_shared: Option<Arc<coordinode_raft::cluster::RaftNode>> =
                Some(raft_node_shared);

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
            let blob_engine = database
                .lock()
                .map_err(|e| format!("failed to lock database for blob engine: {e}"))?
                .engine_shared();
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
            // Uses an embedded proto descriptor compiled into the binary — no external
            // descriptor file or separate container required.
            static DESCRIPTOR_BYTES: &[u8] = include_bytes!("../../../coordinode.descriptor.bin");
            {
                use structured_proxy::config::{
                    DescriptorSource, ListenConfig, ProxyConfig, ServiceConfig, UpstreamConfig,
                };
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

            info!(port = addr.port(), "gRPC server listening");

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

            let mut builder = Server::builder();
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

            router.serve_with_shutdown(addr, shutdown).await?;
        }

        cli::Command::Backup {
            data_dir,
            output,
            format,
            namespace: _namespace,
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
                        db.interner(),
                        shard_id,
                        &snapshot,
                        &mut writer,
                    )
                    .map_err(|e| format!("backup failed: {e}"))?
                }
                coordinode_embed::backup::BackupFormat::Cypher => {
                    coordinode_embed::backup::export::export_cypher(
                        db.engine(),
                        db.interner(),
                        shard_id,
                        &snapshot,
                        &mut writer,
                    )
                    .map_err(|e| format!("backup failed: {e}"))?
                }
                coordinode_embed::backup::BackupFormat::Binary => {
                    coordinode_embed::backup::export::export_binary(
                        db.engine(),
                        db.interner(),
                        shard_id,
                        &snapshot,
                        &mut writer,
                    )
                    .map_err(|e| format!("backup failed: {e}"))?
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
        } => {
            logging::init_logging();
            info!(
                data_dir = %data_dir,
                input = %input,
                format = ?format,
                "starting restore"
            );

            let db = coordinode_embed::Database::open(&data_dir)
                .map_err(|e| format!("failed to open database: {e}"))?;

            let file = std::fs::File::open(&input)
                .map_err(|e| format!("failed to open input file '{input}': {e}"))?;

            match format {
                coordinode_embed::backup::BackupFormat::Json => {
                    let mut reader = std::io::BufReader::new(file);
                    let mut interner = db.interner().clone();
                    let shard_id = 1u16;
                    let stats = coordinode_embed::backup::restore::restore_json(
                        db.engine(),
                        &mut interner,
                        shard_id,
                        &mut reader,
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
                    let mut reader = std::io::BufReader::new(file);
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
                    eprintln!(
                        "error: Cypher restore not yet implemented. \
                         Use json or binary format for restore."
                    );
                    std::process::exit(1);
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
