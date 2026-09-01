//! CoordiNode server library.
//!
//! The `coordinode` binary is a thin wrapper over this crate: it parses argv
//! and hands the resulting [`cli::Command`] to [`run`]. Everything else lives
//! here, so an enterprise distribution can link the same serving stack and
//! extend it through [`ServerBuilder`] instead of forking the binary.
//!
//! # Cluster-ready notes
//! - gRPC server is stateless: all state in CoordiNode storage.
//! - In CE 3-node HA: each node runs identical gRPC server.
//! - Inter-node communication uses the same :7080 port (distributed mode).

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
    pub mod session {
        tonic::include_proto!("coordinode.v1.session");
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

mod admin;
mod builder;
mod checkpoint;
pub mod cli;
pub mod config;
mod grpc;
mod logging;
mod metrics_catalog;
mod ops;
mod pg;
mod registry;
mod serve;
/// The gRPC services this binary serves.
///
/// Public so an integration test can stand a service up on a port and drive it
/// the way a client does. Nothing here is a stable API for other crates: the
/// binary is the product, and no crate may depend on it.
pub mod services;

use admin::{admin_node_decommission, admin_node_join, admin_storage_config};
use tracing::info;

pub use builder::{
    BackgroundTask, GrpcServiceProvider, ServeModeHandler, ServerBuilder, ServerContext,
};

/// Version of the server this binary is built from.
///
/// A distribution that wraps the server reports this number as its own: the
/// wrapper ships the same release with more of it enabled, so there is one
/// version to name in a support request, not two to reconcile.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Execute a parsed CLI command with an unextended server.
///
/// This is the whole of the `coordinode` binary's behaviour; `main` only
/// parses argv and calls here. To extend the server first, build a
/// [`ServerBuilder`] and call [`ServerBuilder::run`] instead.
pub async fn run(command: cli::Command) -> Result<(), Box<dyn std::error::Error>> {
    run_with(ServerBuilder::new(), command).await
}

/// Execute a parsed CLI command against an assembled [`ServerBuilder`].
///
/// Only `serve` consults the registrations; every other subcommand is
/// self-contained and ignores them.
pub(crate) async fn run_with(
    extensions: ServerBuilder,
    command: cli::Command,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        cli::Command::Version => {
            println!("coordinode v{}", env!("CARGO_PKG_VERSION"));
        }

        cli::Command::Verify {
            data_dir,
            config_path,
            deep,
        } => {
            logging::init_logging();
            info!(data_dir = %data_dir, deep = deep, "verifying storage integrity");

            let config = admin_storage_config(config_path.as_deref(), &data_dir)?;
            let engine = coordinode_storage::engine::core::StorageEngine::open(&config)?;
            let disk = engine.disk_space()?;
            info!(disk_bytes = disk, "storage opened successfully");

            if deep {
                info!("deep verification: scrubbing all on-disk blocks...");
                // Full-speed scrub (operator-invoked, not a background pass):
                // verify every block's checksum across every partition.
                let report = coordinode_storage::scrub::scrub_all(
                    &engine,
                    &coordinode_storage::scrub::ScrubConfig::default(),
                )?;
                info!(
                    blocks_checked = report.blocks_checked,
                    sst_files_checked = report.sst_files_checked,
                    errors = report.errors.len(),
                    duration_ms = report.duration.as_millis(),
                    "deep verification complete"
                );
                if report.has_errors() {
                    for err in &report.errors {
                        tracing::error!(
                            partition = err.partition.name(),
                            detail = %err.message,
                            "corruption detected"
                        );
                    }
                    return Err(format!(
                        "deep verification found {} corrupt block(s)",
                        report.errors.len()
                    )
                    .into());
                }
            }

            info!("verification complete");
        }

        cli::Command::Checkpoint {
            data_dir,
            config_path,
            output,
        } => {
            logging::init_logging();
            info!(data_dir = %data_dir, output = %output, "creating checkpoint");

            let config = admin_storage_config(config_path.as_deref(), &data_dir)?;
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

        cli::Command::Compact {
            data_dir,
            config_path,
        } => {
            logging::init_logging();
            info!(data_dir = %data_dir, "compacting database");

            let config = admin_storage_config(config_path.as_deref(), &data_dir)?;
            let engine = coordinode_storage::engine::core::StorageEngine::open(&config)?;
            for &part in coordinode_storage::engine::partition::Partition::all() {
                engine
                    .force_compaction(part)
                    .map_err(|e| format!("compaction failed for {part:?}: {e}"))?;
                info!(partition = ?part, "partition compacted");
            }
            info!(data_dir = %data_dir, "compaction complete");
        }

        cli::Command::Serve {
            config_path,
            overrides,
        } => {
            serve::serve(extensions, config_path, overrides).await?;
        }

        cli::Command::Backup {
            data_dir,
            config_path,
            output,
            format,
            namespace: _namespace,
            since,
        } => {
            admin::run_backup(data_dir, config_path, output, format, since)?;
        }

        cli::Command::Restore {
            data_dir,
            config_path,
            input,
            format,
            namespace: _namespace,
            only_labels,
            force,
        } => {
            admin::run_restore(data_dir, config_path, input, format, only_labels, force)?;
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
