//! Offline administrative subcommands: storage inspection helpers, logical
//! backup / restore, and the cluster membership client used by
//! `coordinode admin node join` / `... decommission`.
//!
//! Nothing here touches the serving path. Each entry point opens the database
//! on its own, does its work and returns; the server process is not involved.

use crate::config;
use crate::logging;
use crate::proto;

use tracing::info;

/// Resolve the storage config for an offline admin command (`verify`,
/// `checkpoint`, `compact`).
///
/// With `--config`, the storage topology (including a multi-endpoint layout)
/// is read from the same file the server uses, so an admin op opens the
/// database at the configured endpoint paths. Without it, the command operates
/// on a single endpoint rooted at `data_dir`. Both paths funnel through
/// [`config::ServerConfig::resolve_storage_config`] so the desugar matches the
/// server exactly.
pub(crate) fn admin_storage_config(
    config_path: Option<&str>,
    data_dir: &str,
) -> Result<coordinode_storage::engine::config::StorageConfig, Box<dyn std::error::Error>> {
    let cfg = match config_path {
        Some(p) => config::ServerConfig::load(Some(p)).map_err(|e| format!("config error: {e}"))?,
        None => config::ServerConfig {
            data_dir: data_dir.to_string(),
            ..config::ServerConfig::default()
        },
    };
    Ok(cfg.resolve_storage_config())
}

/// Export the database to a backup file in the requested format.
///
/// Takes a consistent MVCC snapshot up front, so writes are never blocked for
/// the duration of the dump.
pub(crate) fn run_backup(
    data_dir: String,
    config_path: Option<String>,
    output: String,
    format: coordinode_embed::backup::BackupFormat,
    since: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    logging::init_logging();
    info!(
        data_dir = %data_dir,
        output = %output,
        format = ?format,
        "starting backup"
    );

    let storage_config = admin_storage_config(config_path.as_deref(), &data_dir)?;
    let db = coordinode_embed::Database::open_with_config(storage_config)
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
                    let ts = coordinode_core::txn::timestamp::Timestamp::from_raw(since_seqno);
                    match coordinode_raft::snapshot::build_incremental_snapshot(db.engine(), ts)
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

    Ok(())
}

/// Import a database from a backup file, decompressing the input transparently.
///
/// `only_labels` narrows a json / apoc-json / hetio-json restore to nodes
/// carrying one of the listed labels (and the edges between kept nodes); an
/// empty list restores everything.
pub(crate) fn run_restore(
    data_dir: String,
    config_path: Option<String>,
    input: String,
    format: coordinode_embed::backup::BackupFormat,
    only_labels: Vec<String>,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    logging::init_logging();
    info!(
        data_dir = %data_dir,
        input = %input,
        format = ?format,
        force,
        "starting restore"
    );

    // Selective restore label filter (json / apoc-json / hetio-json).
    let label_filter: Option<std::collections::HashSet<String>> = if only_labels.is_empty() {
        None
    } else {
        Some(only_labels.into_iter().collect())
    };

    let storage_config = admin_storage_config(config_path.as_deref(), &data_dir)?;
    let db = coordinode_embed::Database::open_with_config(storage_config)
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
                coordinode_embed::backup::restore::restore_binary(db.engine(), &mut reader, force)
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
            let interner_len = u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as usize;
            let body = &data[5..];
            if body.len() < interner_len {
                return Err("raft-snapshot file truncated (interner body)".into());
            }
            let (interner_bytes, snapshot) = body.split_at(interner_len);
            db.persist_field_interner_bytes(interner_bytes)
                .map_err(|e| format!("restore interner failed: {e}"))?;
            match mode {
                0 => coordinode_raft::snapshot::install_full_snapshot(db.engine(), snapshot)
                    .map_err(|e| format!("restore failed: {e}"))?,
                1 if snapshot.is_empty() => {
                    info!("incremental backup had no changes; nothing to apply");
                }
                1 => coordinode_raft::snapshot::install_incremental_snapshot(db.engine(), snapshot)
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
pub(crate) async fn admin_node_decommission(
    cluster_addr: String,
    node_id: u64,
    pruning: bool,
    force: bool,
    skip_confirmation: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use proto::admin::cluster::{
        DecommissionNodeRequest, cluster_service_client::ClusterServiceClient,
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
pub(crate) async fn admin_node_join(
    cluster_addr: String,
    node_id: u64,
    node_addr: String,
    pre_seeded: bool,
    follow: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use proto::admin::cluster::{
        JoinNodeRequest, JoinPhase, JoinProgressRequest,
        cluster_service_client::ClusterServiceClient,
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
mod tests;
