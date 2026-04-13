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

            // Open database with appropriate proposal pipeline:
            // - Embedded (no --peers): OwnedLocalProposalPipeline (local writes)
            // - Cluster (--peers): RaftProposalPipeline (Raft-replicated writes)
            //
            // In cluster mode, DrainBuffer and TTL reaper submit mutations
            // through Raft for replication to followers (G063).
            let database = if let Some(ref peer_addrs) = peers {
                info!(
                    peers = peer_addrs.len(),
                    "cluster mode: DrainBuffer using RaftProposalPipeline"
                );
                // Create Raft node and proposal pipeline for cluster mode.
                // RaftNode handles leader election, log replication, and
                // state machine apply. The pipeline wraps client_write().
                let storage_config =
                    coordinode_storage::engine::config::StorageConfig::new(&data_dir);
                let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
                let engine = coordinode_storage::engine::core::StorageEngine::open_with_oracle(
                    &storage_config,
                    oracle.clone(),
                )
                .map_err(|e| format!("failed to open storage: {e}"))?;
                let engine = Arc::new(engine);

                // Node ID 1 for single-node bootstrap; peer discovery wiring
                // is deferred to R150 (monolithic binary --mode selection).
                let raft_node = coordinode_raft::cluster::RaftNode::open_with_oracle(
                    1, // node_id
                    Arc::clone(&engine),
                    Some(Arc::clone(&oracle)),
                )
                .await
                .map_err(|e| format!("failed to open Raft node: {e}"))?;

                let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
                    Arc::new(coordinode_raft::proposal::RaftProposalPipeline::new(
                        Arc::clone(raft_node.raft()),
                    ));

                // Share engine + oracle between RaftNode and Database.
                // RaftNode uses engine for state machine apply.
                // Database uses engine for reads + ExecutionContext.
                Arc::new(std::sync::Mutex::new(
                    coordinode_embed::Database::from_engine(&data_dir, engine, oracle, pipeline)
                        .map_err(|e| format!("failed to open database: {e}"))?,
                ))
            } else {
                // Embedded mode: OwnedLocalProposalPipeline (default).
                Arc::new(std::sync::Mutex::new(
                    coordinode_embed::Database::open(&data_dir)
                        .map_err(|e| format!("failed to open database: {e}"))?,
                ))
            };

            let query_registry = Arc::new(coordinode_query::advisor::QueryRegistry::new());
            let nplus1_detector =
                Arc::new(coordinode_query::advisor::nplus1::NPlus1Detector::new());

            let graph_service = services::graph::GraphServiceImpl::new(Arc::clone(&database));
            let schema_service = services::schema::SchemaServiceImpl::new(Arc::clone(&database));
            let cypher_service = services::cypher::CypherServiceImpl::new(
                Arc::clone(&database),
                Arc::clone(&query_registry),
                Arc::clone(&nplus1_detector),
            );
            let vector_service = services::vector::VectorServiceImpl::new(Arc::clone(&database));
            let text_service = services::text::TextServiceImpl::new(Arc::clone(&database));
            let health_service = services::health::HealthServiceImpl;
            // CDC service: tails oplog/<shard>/ dir. Empty stream in embedded mode
            // (no oplog); populated in Raft cluster mode (LogStore writes oplog).
            let cdc_service =
                services::cdc::ChangeEventServiceImpl::new(std::path::PathBuf::from(&data_dir));

            // BlobService shares the same storage engine as the Database.
            let blob_engine = database
                .lock()
                .map_err(|e| format!("failed to lock database for blob engine: {e}"))?
                .engine_shared();
            let blob_service = services::blob::BlobServiceImpl::new(blob_engine);

            // Spawn operational HTTP server on :7084
            let ops_addr: SocketAddr = "[::]:7084".parse()?;
            tokio::spawn(async move {
                if let Err(e) = ops::start_ops_server(ops_addr).await {
                    tracing::error!("ops server error: {e}");
                }
            });

            info!(port = addr.port(), "gRPC server listening");

            Server::builder()
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
                )
                .serve(addr)
                .await?;
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
    }

    Ok(())
}
