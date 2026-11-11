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
    Ok(cfg
        .resolve_storage_config()
        .map_err(|e| format!("config error: {e}"))?)
}

/// Open the database for an offline admin command the way the server opens it.
///
/// The oracle is what tells the engine its seqnos are a clock, and only then
/// does the configured retention window hold the GC watermark back. Opened
/// with a plain counter instead, the window is inert and the watermark sits at
/// the newest seqno, so the first compaction folds every version underneath
/// it: an operator command would end time travel over the whole window without
/// saying so. A fresh oracle is correct here because the engine seeds it from
/// the highest seqno it recovers, which is where a server would have resumed.
pub(crate) fn admin_open_engine(
    config: &coordinode_storage::engine::config::StorageConfig,
) -> Result<coordinode_storage::engine::core::StorageEngine, Box<dyn std::error::Error>> {
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    Ok(coordinode_storage::engine::core::StorageEngine::open_with_oracle(config, oracle)?)
}

/// Open the store for an offline command that only reads it, as the store's
/// writer opens it: a store outside any group replays its journal, a group
/// member's store opens as the member does, so nothing is written outside the
/// group's log.
pub(crate) fn admin_open_for_reading(
    config: &coordinode_storage::engine::config::StorageConfig,
) -> Result<coordinode_storage::engine::core::StorageEngine, Box<dyn std::error::Error>> {
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    match coordinode_storage::engine::core::StorageEngine::open_embedded(config, oracle) {
        Err(coordinode_storage::error::StorageError::GroupMemberStore { .. }) => {
            admin_open_engine(config)
        }
        opened => Ok(opened?),
    }
}

/// The raft-snapshot backup frame's mode byte: a whole database, framed with
/// the deployment identity its operation records belong to.
const SNAPSHOT_MODE_FULL: u8 = 2;

/// The mode byte of a whole-database frame without the deployment identity,
/// written by earlier releases.
const SNAPSHOT_MODE_FULL_WITHOUT_DOMAIN: u8 = 0;

/// Export the database to a backup file in the requested format.
///
/// Takes a consistent MVCC snapshot up front, so writes are never blocked for
/// the duration of the dump.
pub(crate) fn run_backup(
    data_dir: String,
    config_path: Option<String>,
    output: String,
    format: coordinode_embed::backup::BackupFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let _log = logging::init_logging();
    info!(
        data_dir = %data_dir,
        output = %output,
        format = ?format,
        "starting backup"
    );

    let storage_config = admin_storage_config(config_path.as_deref(), &data_dir)?;
    let engine = std::sync::Arc::new(
        admin_open_for_reading(&storage_config)
            .map_err(|e| format!("failed to open database: {e}"))?,
    );

    // One pinned cut for everything the backup holds: the records, the
    // dictionary and the deployment identity framed with them.
    let (snapshot, _pin) = engine
        .pin_capture_cut()
        .map_err(|e| format!("backup refused: {e}"))?;
    let shard_id = 1u16;

    // Readable too: the raft-snapshot format reads its body back to checksum it.
    let file = std::fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output)
        .map_err(|e| format!("failed to create output file '{output}': {e}"))?;
    let mut writer = std::io::BufWriter::new(file);
    let dictionary =
        coordinode_storage::engine::metadata::load_field_dictionary_at(&engine, snapshot)
            .map_err(|e| format!("read the field dictionary: {e}"))?;

    let stats = match format {
        coordinode_embed::backup::BackupFormat::Json => {
            coordinode_embed::backup::export::export_json(
                &engine,
                &dictionary,
                shard_id,
                &snapshot,
                &mut writer,
            )
            .map_err(|e| format!("backup failed: {e}"))?
        }
        coordinode_embed::backup::BackupFormat::Cypher => {
            coordinode_embed::backup::export::export_cypher(
                &engine,
                &dictionary,
                shard_id,
                &snapshot,
                &mut writer,
            )
            .map_err(|e| format!("backup failed: {e}"))?
        }
        coordinode_embed::backup::BackupFormat::Binary => {
            coordinode_embed::backup::export::export_binary(
                &engine,
                &dictionary,
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
            // logical export. The frame is [mode u8][u32 interner_len]
            // [interner][u8 has_deployment][16-byte deployment][snapshot]:
            // the dictionary ahead of the records, so restore publishes its
            // bindings before it installs anything encoded with them, and the
            // deployment the operation records belong to, which the snapshot
            // leaves out with the rest of the node's own configuration.
            use std::io::Write;
            let interner_bytes = dictionary
                .to_bytes()
                .map_err(|e| format!("serialize the field dictionary: {e}"))?;
            let interner_len = u32::try_from(interner_bytes.len())
                .map_err(|_| "field interner too large to frame".to_string())?;
            let deployment =
                coordinode_storage::engine::metadata::bound_deployment_at(&engine, snapshot)
                    .map_err(|e| format!("read the deployment identity: {e}"))?;
            let mut domain = [0u8; 17];
            if let Some(id) = deployment {
                domain[0] = 1;
                domain[1..].copy_from_slice(id.as_bytes());
            }
            // The snapshot streams into the file, which it seeks and reads
            // back, so it takes the file itself rather than the buffer.
            let mut file = writer
                .into_inner()
                .map_err(|e| format!("backup write failed: {}", e.error()))?;
            file.write_all(&[SNAPSHOT_MODE_FULL])
                .and_then(|()| file.write_all(&interner_len.to_be_bytes()))
                .and_then(|()| file.write_all(&interner_bytes))
                .and_then(|()| file.write_all(&domain))
                .map_err(|e| format!("backup write failed: {e}"))?;
            // No inventory is sent: a restore checks every index copy
            // before using it.
            let snapshot_bytes = coordinode_raft::snapshot::write_full_snapshot_at(
                &engine, snapshot, None, &mut file,
            )
            .map_err(|e| format!("backup failed: {e}"))?;
            info!(
                interner_bytes = interner_bytes.len(),
                snapshot_bytes, "backup complete (raft-snapshot)"
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
    let _log = logging::init_logging();
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

    // A bzip2/gzip-compressed input is decompressed tool-side (pure Rust);
    // an uncompressed one passes through unchanged. A logical restore reads
    // it twice, so the file is opened once per pass.
    let source = InputFile {
        path: std::path::PathBuf::from(&input),
    };
    match format {
        coordinode_embed::backup::BackupFormat::RaftSnapshot => {
            let limit = admin_memory_limit(config_path.as_deref())?;
            let budget = coordinode_core::budget::QueryBudget::new(limit);
            // One handle for both passes: a file renamed over the path in
            // between is not the one installed.
            let file = std::fs::File::open(&input)
                .map_err(|e| format!("failed to read input file '{input}': {e}"))?;
            // Checked through before the target is looked at, holding no
            // more than the memory limit at once.
            let mut reader = read_pass(&file)?;
            let frame = read_frame(&mut reader, &budget)?;
            let check = coordinode_raft::snapshot::check_image(&mut reader, &budget)
                .map_err(|e| format!("the backup's image is not valid: {e}"))?;
            drop(reader);
            frame.describes(&check)?;
            restore_snapshot_backup(storage_config, &file, &frame, &check, &budget)?;
            info!(
                memory_peak_bytes = budget.memory_peak(),
                memory_limit_bytes = limit,
                "restore complete (raft-snapshot)"
            );
        }
        logical => {
            let db = coordinode_embed::Database::open_with_config(storage_config)
                .map_err(|e| format!("failed to open database: {e}"))?;
            let options = coordinode_embed::backup::restore::RestoreOptions {
                only_labels: label_filter.as_ref(),
                force,
                ..Default::default()
            };
            let stats = db
                .restore(logical, &source, &options)
                .map_err(|e| format!("restore failed: {e}"))?;
            info!(
                nodes = stats.nodes,
                edges = stats.edges,
                schema = stats.schema_entries,
                format = ?logical,
                "restore complete"
            );
        }
    }

    Ok(())
}

/// The memory limit an offline restore holds itself to: the configured
/// `query_memory_limit_mb`, or the default query limit.
fn admin_memory_limit(config_path: Option<&str>) -> Result<u64, Box<dyn std::error::Error>> {
    let mib = match config_path {
        Some(p) => {
            config::ServerConfig::load(Some(p))
                .map_err(|e| format!("config error: {e}"))?
                .query_memory_limit_mb
        }
        None => None,
    };
    match mib {
        None => Ok(coordinode_core::budget::DEFAULT_QUERY_MEMORY_LIMIT),
        Some(mib) => mib
            .checked_mul(1 << 20)
            .ok_or_else(|| format!("query_memory_limit_mb = {mib} does not fit in bytes").into()),
    }
}

/// A fresh pass over the restore input `file` from its start, decompressed
/// as it is read.
fn read_pass(
    file: &std::fs::File,
) -> Result<Box<dyn std::io::BufRead>, Box<dyn std::error::Error>> {
    use std::io::Seek;
    let mut handle = file
        .try_clone()
        .map_err(|e| format!("read the input file: {e}"))?;
    handle
        .seek(std::io::SeekFrom::Start(0))
        .map_err(|e| format!("read the input file: {e}"))?;
    Ok(decompressing_reader(handle).map_err(|e| format!("read the input file: {e}"))?)
}

/// The frame of a raft-snapshot backup, ahead of its image.
struct Frame<'b> {
    interner_bytes: Vec<u8>,
    deployment: Option<coordinode_core::group::DeploymentId>,
    _charge: coordinode_core::budget::MemoryCharge<'b>,
}

impl Frame<'_> {
    /// Whether `other` frames the same image as this.
    fn same_as(&self, other: &Frame<'_>) -> bool {
        self.interner_bytes == other.interner_bytes && self.deployment == other.deployment
    }

    /// Refuse a frame that does not describe `check`, the image it carries.
    fn describes(
        &self,
        check: &coordinode_raft::snapshot::ImageCheck<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.deployment != check.domain() {
            return Err(format!(
                "the backup's frame names deployment {:?}, but its image belongs to {:?}; the \
                 backup is inconsistent",
                self.deployment,
                check.domain()
            )
            .into());
        }
        // Read in place, in id order as both are kept: nothing is built.
        let invalid = |e| format!("the backup's field dictionary is not valid: {e}");
        let mut framed = coordinode_core::graph::intern::EncodedBindings::new(&self.interner_bytes)
            .map_err(invalid)?;
        let carried = check.field_bindings();
        let mut same = framed.len() == carried.len();
        for (name, id) in carried {
            match framed.next() {
                Some(Ok((framed_name, framed_id))) => {
                    same &= framed_name == name && framed_id == *id;
                }
                Some(Err(e)) => return Err(invalid(e).into()),
                None => same = false,
            }
        }
        if let Some(Err(e)) = framed.next() {
            return Err(invalid(e).into());
        }
        if !same {
            return Err(
                "the backup's frame and image carry different field dictionaries; the backup \
                 is inconsistent"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Read the frame `[mode u8][u32 interner_len][interner][u8 has_deployment]
/// [16-byte deployment]` ahead of a raft-snapshot backup's image, the
/// dictionary's bytes reserved from `budget` before they are read.
fn read_frame<'b>(
    reader: &mut impl std::io::Read,
    budget: &'b coordinode_core::budget::QueryBudget,
) -> Result<Frame<'b>, Box<dyn std::error::Error>> {
    let mut frame = [0u8; 5];
    reader
        .read_exact(&mut frame)
        .map_err(|e| format!("raft-snapshot file truncated (no frame header): {e}"))?;
    match frame[0] {
        SNAPSHOT_MODE_FULL => {}
        // Without the deployment its operation records belong to, a restored
        // store could not tell their descriptors from another domain's.
        SNAPSHOT_MODE_FULL_WITHOUT_DOMAIN => {
            return Err(
                "this raft-snapshot backup was taken by an earlier release and does not carry \
                 its deployment identity; take a new backup"
                    .into(),
            );
        }
        // 0.6 also wrote seqno-bounded deltas; a delta holds only what
        // changed after a seqno and cannot stand alone.
        1 => {
            return Err(
                "this is an incremental raft-snapshot backup; only whole-database backups \
                 restore. Take a full backup with the release that wrote it"
                    .into(),
            );
        }
        other => return Err(format!("unknown snapshot mode byte: {other}").into()),
    }
    let interner_len = u32::from_be_bytes([frame[1], frame[2], frame[3], frame[4]]);
    // Reserved before it is read: a length past the limit is refused, not
    // allocated. Held, and charged, as long as the frame.
    let charge = budget
        .reserve(u64::from(interner_len))
        .map_err(|e| format!("the backup's field dictionary does not fit the memory limit: {e}"))?;
    let mut interner_bytes = vec![0u8; interner_len as usize];
    reader
        .read_exact(&mut interner_bytes)
        .map_err(|e| format!("raft-snapshot file truncated (interner body): {e}"))?;
    let mut domain = [0u8; 17];
    reader
        .read_exact(&mut domain)
        .map_err(|e| format!("raft-snapshot file truncated (deployment): {e}"))?;
    let deployment = match domain[0] {
        0 => None,
        1 => coordinode_core::group::DeploymentId::from_slice(&domain[1..]),
        other => return Err(format!("unknown deployment flag: {other}").into()),
    };
    Ok(Frame {
        interner_bytes,
        deployment,
        _charge: charge,
    })
}

/// Install the backup in `file`, checked as `frame` and `check`, as the
/// whole database at `storage_config`'s directories, which must be empty or
/// hold the unfinished restore of this same image. The directories are
/// claimed before the store is opened and released only once everything
/// installed is durable and the claim is ended: a restore cut short leaves a
/// store no ordinary open serves, and running it again starts its install
/// over.
fn restore_snapshot_backup(
    storage_config: coordinode_storage::engine::config::StorageConfig,
    file: &std::fs::File,
    frame: &Frame<'_>,
    check: &coordinode_raft::snapshot::ImageCheck<'_>,
    budget: &coordinode_core::budget::QueryBudget,
) -> Result<(), Box<dyn std::error::Error>> {
    let claim = coordinode_storage::format::begin_restore(&storage_config, check.checksum())
        .map_err(|e| format!("restore refused: {e}"))?;
    abort_at("claimed");
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = coordinode_storage::engine::core::StorageEngine::open_restoring(
        &storage_config,
        oracle,
        claim,
    )
    .map_err(|e| format!("failed to open the target: {e}"))?;
    let mut reader = read_pass(file)?;
    if !read_frame(&mut reader, budget)?.same_as(frame) {
        return Err(
            "the input changed between its check and its install; nothing installed is \
             served, and the next attempt starts over"
                .into(),
        );
    }
    coordinode_raft::snapshot::install_image(&engine, &mut reader, check, budget)
        .map_err(|e| format!("restore failed: {e}"))?;
    abort_at("installed");
    // The store continues the image's domain, bound with its records. Every
    // incarnation the image holds is closed, so a descriptor issued before
    // the backup resolves to its recorded outcome and admits nothing new. The
    // target held no history of its own.
    coordinode_storage::engine::operations::fence_restored(&engine, &[])
        .map_err(|e| format!("fence the restored operation records: {e}"))?;
    engine
        .persist()
        .map_err(|e| format!("persist the restored database: {e}"))?;
    abort_at("persisted");
    // While the engine still holds the directories.
    coordinode_storage::format::finish_restore(&storage_config)
        .map_err(|e| format!("finish the restore: {e}"))?;
    drop(engine);
    Ok(())
}

/// Test builds only: COORDINODE_TEST_RESTORE_ABORT names a step of the
/// restore at which the process dies on the spot, so a test cuts a restore
/// short at that step through the process.
fn abort_at(step: &str) {
    #[cfg(feature = "test-faults")]
    if std::env::var("COORDINODE_TEST_RESTORE_ABORT").as_deref() == Ok(step) {
        std::process::abort();
    }
    #[cfg(not(feature = "test-faults"))]
    let _ = step;
}

/// A restore input file, decompressed as it is read.
struct InputFile {
    path: std::path::PathBuf,
}

impl coordinode_embed::backup::restore::RestoreSource for InputFile {
    fn open(&self) -> std::io::Result<Box<dyn std::io::BufRead + '_>> {
        let file = std::fs::File::open(&self.path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("open input file '{}': {e}", self.path.display()),
            )
        })?;
        decompressing_reader(file)
    }
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

/// The URI an admin client dials for `addr`: as given when it names a
/// scheme, plain HTTP/2 otherwise.
fn client_uri(addr: &str) -> String {
    if addr.contains("://") {
        addr.to_owned()
    } else {
        format!("http://{addr}")
    }
}

/// Execute `coordinode admin node decommission` — connect to a running cluster and
/// gracefully decommission a node (quorum check, leadership transfer,
/// membership removal).
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

    let endpoint = client_uri(&cluster_addr);

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

/// Execute `coordinode admin node replace`: put a new node, running on its
/// own empty data directory, in place of a member that can no longer take
/// part. Returns once the replacement is complete; run it again with the
/// same arguments to resume one that was cut short.
pub(crate) async fn admin_node_replace(
    cluster_addr: String,
    old_node_id: u64,
    new_node_id: u64,
    node_addr: String,
    node_identity: String,
) -> Result<(), Box<dyn std::error::Error>> {
    use proto::admin::cluster::{ReplaceNodeRequest, cluster_service_client::ClusterServiceClient};

    let endpoint = client_uri(&cluster_addr);
    eprintln!("Connecting to cluster at {endpoint} ...");
    let mut client = ClusterServiceClient::connect(endpoint)
        .await
        .map_err(|e| format!("failed to connect to cluster: {e}"))?;
    eprintln!("Replacing node {old_node_id} with node {new_node_id} at {node_addr} ...");
    let resp = client
        .replace_node(ReplaceNodeRequest {
            old_node_id,
            new_node_id,
            address: node_addr,
            node_identity,
        })
        .await
        .map_err(|e| format!("ReplaceNode failed: {e}"))?
        .into_inner();
    eprintln!(
        "Replacement complete: node {} is a voter, node {} left the group.\n\
         The old node's data directory was left as it was.",
        resp.new_node_id, resp.old_node_id
    );
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
    node_identity: String,
    pre_seeded: bool,
    follow: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use proto::admin::cluster::{
        JoinNodeRequest, JoinPhase, JoinProgressRequest,
        cluster_service_client::ClusterServiceClient,
    };

    let endpoint = client_uri(&cluster_addr);

    eprintln!("Connecting to cluster at {endpoint} ...");

    let mut client = ClusterServiceClient::connect(endpoint)
        .await
        .map_err(|e| format!("failed to connect to cluster: {e}"))?;

    eprintln!("Initiating join for node {node_id} ({node_identity}) at {node_addr} ...");

    let resp = client
        .join_node(JoinNodeRequest {
            node_id,
            address: node_addr.clone(),
            pre_seeded,
            node_identity,
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
