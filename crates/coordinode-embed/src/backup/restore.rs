//! Restore graph data from backup files into CoordiNode storage.
//!
//! Every format keeps the node identifiers of its input. A restore reads the
//! input twice: the first pass checks that none of its identifiers is issued
//! in the target and finds the highest one, and only then, with the target's
//! identifier lease raised above them, the second pass writes. A record of the
//! load, kept until it is complete, lets a rerun of the same input finish a
//! load a crash interrupted instead of refusing it as its own collision.

use std::collections::HashSet;
use std::io::{BufRead, Read};

use coordinode_core::graph::edge::EdgeProperties;
use coordinode_core::graph::intern::{FieldInterner, FieldRegistrar};
use coordinode_core::graph::node::{
    NODE_KEY_PREFIX, NODE_LEASE_TOKEN_LEN, NodeId, decode_node_key, decode_temporal_node_key,
    encode_node_key,
};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_modality::edge::{EdgeStore, LocalEdgeStore};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::metadata::{node_lease_ceiling, node_lease_holder};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use sha2::{Digest as _, Sha256};

use super::BackupFormat;
use super::export::BackupEntry;

/// Errors during restore.
#[derive(Debug, thiserror::Error)]
pub enum RestoreError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("deserialization error: {0}")]
    Deserialization(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid backup format: {0}")]
    InvalidFormat(String),

    #[error("incompatible dump: {0}")]
    IncompatibleVersion(String),

    #[error("schema fingerprint mismatch: {0}")]
    SchemaMismatch(String),

    /// Identifiers of the input the target already issued: writing them would
    /// replace a live node or hand out an identifier twice. A restore goes to
    /// a new instance.
    #[error(
        "{count} node identifier(s) of the input are already issued in the target \
         (first: {first:?}); restore into a new database"
    )]
    IdentifiersIssued { count: u64, first: Vec<u64> },

    /// An earlier restore of a different input did not finish; the target
    /// holds part of it.
    #[error(
        "the target holds an unfinished restore of a different input; rerun that input \
         to finish it, or restore into a new database"
    )]
    UnfinishedLoad,

    /// The target's identifier lease could not be raised above the input's
    /// identifiers.
    #[error("raise the node identifier lease: {0}")]
    Lease(String),

    /// The records are in, but an index over them could not be built, for
    /// instance a unique index the restored data breaks. The load stays
    /// recorded; a rerun of the same input after the cause is removed
    /// finishes it.
    #[error("the restored nodes are in, but their indexes could not be built: {0}")]
    Indexes(String),

    /// The records are in, but a node breaks a constraint of its label, or
    /// a restored constraint's name is held by another label. The load
    /// stays recorded; a rerun of the same input after the cause is removed
    /// finishes it.
    #[error("the restored nodes are in, but they break a constraint: {0}")]
    Constraints(String),

    /// The format has no logical restore through this path.
    #[error("{0:?} is not restored record by record; install it as a snapshot")]
    Unsupported(BackupFormat),
}

/// Statistics from a restore operation.
#[derive(Debug, Default, Clone)]
pub struct RestoreStats {
    pub nodes: u64,
    pub edges: u64,
    pub schema_entries: u64,
}

/// The input of a restore, opened once per pass.
pub trait RestoreSource {
    /// A reader positioned at the start of the input.
    ///
    /// # Errors
    ///
    /// The input cannot be opened.
    fn open(&self) -> std::io::Result<Box<dyn BufRead + '_>>;
}

impl RestoreSource for &[u8] {
    fn open(&self) -> std::io::Result<Box<dyn BufRead + '_>> {
        Ok(Box::new(*self))
    }
}

impl RestoreSource for Vec<u8> {
    fn open(&self) -> std::io::Result<Box<dyn BufRead + '_>> {
        Ok(Box::new(self.as_slice()))
    }
}

/// How a restore reads its input.
#[derive(Debug, Clone, Copy)]
pub struct RestoreOptions<'a> {
    /// The shard of the node keys a text format writes (CE: 1).
    pub shard_id: u16,
    /// Keep only nodes carrying one of these labels, and the edges between
    /// them (json, APOC json, Hetionet).
    pub only_labels: Option<&'a HashSet<String>>,
    /// Restore a binary dump whose manifest is missing, newer than this build,
    /// or made against a different schema. Never lets an issued identifier
    /// through.
    pub force: bool,
}

impl Default for RestoreOptions<'_> {
    fn default() -> Self {
        Self {
            shard_id: 1,
            only_labels: None,
            force: false,
        }
    }
}

/// What a restore writes into.
pub(crate) struct RestoreTarget<'a> {
    pub(crate) engine: &'a StorageEngine,
    pub(crate) fields: &'a dyn FieldRegistrar,
    /// Take the node identifier sequences `(base, target]` through the log
    /// under the given token, while the granted ceiling is still `base`, so
    /// no allocator of the group issues them. `false` when another grant
    /// moved the ceiling first.
    pub(crate) raise_lease: &'a RaiseLease<'a>,
    /// Build every declared index from the nodes in the store, once the
    /// records are in: the load writes them past index maintenance.
    pub(crate) build_indexes: &'a (dyn Fn() -> Result<(), String> + 'a),
    /// Check the nodes in the store against every constraint and record the
    /// name of each constraint the load brought, once the indexes are built:
    /// the load writes the records and schemas past every check.
    pub(crate) check_constraints: &'a (dyn Fn() -> Result<(), String> + 'a),
}

/// See [`RestoreTarget::raise_lease`].
pub(crate) type RaiseLease<'a> =
    dyn Fn(u64, u64, [u8; NODE_LEASE_TOKEN_LEN]) -> Result<bool, String> + 'a;

/// The record of a load in progress: the digest of its input and the
/// highest sequence of hint 0 it holds. A node-local `meta:` key, since the
/// load writes this member's store directly.
pub(crate) const LOAD_KEY: &[u8] = b"meta:restore:load";

/// The value of [`LOAD_KEY`] for a load of the input with `digest`.
pub(crate) fn load_record(digest: &[u8; 32], max_sequence: u64) -> [u8; 40] {
    let mut record = [0u8; 40];
    record[..32].copy_from_slice(digest);
    record[32..].copy_from_slice(&max_sequence.to_be_bytes());
    record
}

/// The token a load's lease grant carries: the grant record then tells a
/// rerun whether that load took its sequences, since it writes nothing before.
fn lease_token(digest: &[u8; 32]) -> [u8; NODE_LEASE_TOKEN_LEN] {
    let mut token = [0u8; NODE_LEASE_TOKEN_LEN];
    token.copy_from_slice(&digest[..NODE_LEASE_TOKEN_LEN]);
    token
}

/// Surveys lost to other allocators before a restore gives up: each loss
/// means a lease was granted between the survey and its own grant.
const MAX_SURVEYS: u32 = 8;

/// Restore `source` in `format` into `target`.
pub(crate) fn run(
    target: &RestoreTarget<'_>,
    format: BackupFormat,
    source: &dyn RestoreSource,
    options: &RestoreOptions<'_>,
) -> Result<RestoreStats, RestoreError> {
    let engine = target.engine;
    let storage = |e: coordinode_storage::error::StorageError| RestoreError::Storage(e.to_string());
    // A load may have written records once its sequences were taken (or at
    // once, when it holds no identifier of hint 0).
    let may_have_written = |digest: &[u8; 32], max_sequence: u64| {
        if max_sequence == 0 {
            return Ok(true);
        }
        node_lease_holder(engine, max_sequence)
            .map(|holder| holder == Some(lease_token(digest)))
            .map_err(storage)
    };
    let record_load = |value: &[u8]| {
        engine
            .put(Partition::Schema, LOAD_KEY, value)
            .and_then(|()| engine.persist_partition(Partition::Schema))
            .map_err(storage)
    };

    let mut surveys = 0;
    loop {
        surveys += 1;
        // Pass one: read only, so a refusal leaves the target as it was.
        let ceiling = node_lease_ceiling(engine).map_err(storage)?;
        let mut checked = Survey::new(engine, ceiling)?;
        let mut hasher = Sha256::new();
        read_input(
            format,
            source,
            options,
            &mut Load::survey(target, &mut checked),
            Some(&mut hasher),
        )?;
        let digest: [u8; 32] = hasher.finalize().into();

        let recorded = engine.get(Partition::Schema, LOAD_KEY).map_err(storage)?;
        let resume = match recorded.as_deref() {
            None => false,
            Some(value) => {
                let (recorded_digest, recorded_max) = value
                    .split_first_chunk::<32>()
                    .and_then(|(d, rest)| Some((*d, u64::from_be_bytes(rest.try_into().ok()?))))
                    .ok_or(RestoreError::UnfinishedLoad)?;
                let wrote = may_have_written(&recorded_digest, recorded_max)?;
                if recorded_digest == digest {
                    wrote
                } else if wrote {
                    return Err(RestoreError::UnfinishedLoad);
                } else {
                    // That load stopped before taking its sequences, so it
                    // wrote nothing and holds nothing.
                    false
                }
            }
        };
        if let Some(manifest) = &checked.manifest {
            // A rerun meets the schema its own interrupted load wrote.
            validate_manifest(engine, manifest, options.force || resume)?;
        }
        if resume {
            // Its own records are in the store and its sequences are taken.
            break;
        }
        if checked.issued_count > 0 {
            if recorded.is_some() {
                engine
                    .delete(Partition::Schema, LOAD_KEY)
                    .map_err(storage)?;
                engine
                    .persist_partition(Partition::Schema)
                    .map_err(storage)?;
            }
            return Err(RestoreError::IdentifiersIssued {
                count: checked.issued_count,
                first: checked.issued,
            });
        }
        // Durable before the grant: a crash after it must find the load, or
        // the rerun would take the load's own sequences for issued ones.
        record_load(&load_record(&digest, checked.max_sequence))?;
        if checked.max_sequence == 0
            || (target.raise_lease)(ceiling, checked.max_sequence, lease_token(&digest))
                .map_err(RestoreError::Lease)?
        {
            break;
        }
        // A lease was granted since the survey read the ceiling, and it may
        // hold identifiers of the input: check them again against it.
        if surveys >= MAX_SURVEYS {
            return Err(RestoreError::Lease(format!(
                "other allocators took a lease during each of {MAX_SURVEYS} surveys"
            )));
        }
    }

    // Pass two: write.
    let mut write = Load::write(target);
    read_input(format, source, options, &mut write, None)?;
    let stats = write.stats;

    // The rows went in directly, so the planner's counters never saw them,
    // and no index did. Both are built while the load is still recorded: a
    // crash before they are whole reruns it, which builds them again.
    coordinode_storage::engine::stats::rebuild_node_counters(engine).map_err(storage)?;
    (target.build_indexes)().map_err(RestoreError::Indexes)?;
    (target.check_constraints)().map_err(RestoreError::Constraints)?;
    // Every record durable before the load stops being unfinished.
    engine.persist().map_err(storage)?;
    engine
        .delete(Partition::Schema, LOAD_KEY)
        .map_err(storage)?;
    engine
        .persist_partition(Partition::Schema)
        .map_err(storage)?;
    Ok(stats)
}

/// Read `source` in `format` through `load`, hashing the input into `digest`.
fn read_input(
    format: BackupFormat,
    source: &dyn RestoreSource,
    options: &RestoreOptions<'_>,
    load: &mut Load<'_, '_>,
    digest: Option<&mut Sha256>,
) -> Result<(), RestoreError> {
    let reader = source.open()?;
    let mut reader = HashingReader::new(reader, digest);
    match format {
        BackupFormat::Binary => restore_binary(load, &mut reader, options.force),
        BackupFormat::Json => restore_json(load, options, &mut reader),
        BackupFormat::Cypher => restore_cypher(load, options, &mut reader),
        BackupFormat::ApocJson => restore_apoc_json(load, options, &mut reader),
        BackupFormat::ApocCypher => restore_apoc_cypher(load, options, &mut reader),
        BackupFormat::HetioJson => restore_hetio_json(load, options, &mut reader),
        BackupFormat::RaftSnapshot => Err(RestoreError::Unsupported(format)),
    }
}

/// A reader that hashes every byte it hands out.
struct HashingReader<'h, R> {
    inner: R,
    digest: Option<&'h mut Sha256>,
}

impl<'h, R> HashingReader<'h, R> {
    fn new(inner: R, digest: Option<&'h mut Sha256>) -> Self {
        Self { inner, digest }
    }
}

impl<R: BufRead> Read for HashingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let available = self.fill_buf()?;
        let n = available.len().min(buf.len());
        buf[..n].copy_from_slice(&available[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl<R: BufRead> BufRead for HashingReader<'_, R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        if let Some(digest) = self.digest.as_deref_mut() {
            // Consumed bytes come from the buffer the last fill_buf returned.
            if let Ok(buffered) = self.inner.fill_buf() {
                digest.update(&buffered[..amount.min(buffered.len())]);
            }
        }
        self.inner.consume(amount);
    }
}

/// What the first pass found.
struct Survey {
    /// The lease ceiling of the target: every sequence of hint 0 at or
    /// below it is issued.
    ceiling: u64,
    /// The highest sequence of hint 0 in the input.
    max_sequence: u64,
    /// The first identifiers found issued, for the error.
    issued: Vec<u64>,
    issued_count: u64,
    /// The node checked last: the versions of a temporal node follow one
    /// another in a dump and are one identifier.
    last_checked: Option<(u16, NodeId)>,
    /// Whether the target holds any node record: a restore into a new
    /// database then probes no key per node.
    holds_nodes: bool,
    manifest: Option<Manifest>,
}

impl Survey {
    fn new(engine: &StorageEngine, ceiling: u64) -> Result<Self, RestoreError> {
        use coordinode_storage::Guard as _;
        let storage = |e: &dyn std::fmt::Display| RestoreError::Storage(e.to_string());
        let holds_nodes = match engine
            .prefix_scan(Partition::Node, NODE_KEY_PREFIX)
            .map_err(|e| storage(&e))?
            .next()
        {
            None => false,
            Some(record) => record.into_inner().map(|_| true).map_err(|e| storage(&e))?,
        };
        Ok(Self {
            ceiling,
            max_sequence: 0,
            issued: Vec::new(),
            issued_count: 0,
            last_checked: None,
            holds_nodes,
            manifest: None,
        })
    }
}

/// A binary dump's manifest, checked once the pass knows whether it resumes.
struct Manifest {
    format_version: u32,
    producer: String,
    schema_fingerprint: u64,
}

/// How many issued identifiers the error names.
const ISSUED_REPORTED: usize = 10;

/// One pass over the input: a survey that only checks identifiers, or the
/// write. The format parsers report every record to it and do not know which.
struct Load<'t, 'a> {
    target: &'t RestoreTarget<'a>,
    /// What the survey pass records into; absent in the write pass.
    survey: Option<&'t mut Survey>,
    stats: RestoreStats,
}

impl<'t, 'a> Load<'t, 'a> {
    fn survey(target: &'t RestoreTarget<'a>, survey: &'t mut Survey) -> Self {
        Self {
            target,
            survey: Some(survey),
            stats: RestoreStats::default(),
        }
    }

    fn write(target: &'t RestoreTarget<'a>) -> Self {
        Self {
            target,
            survey: None,
            stats: RestoreStats::default(),
        }
    }

    fn writes(&self) -> bool {
        self.survey.is_none()
    }

    /// Check a node about to be written on `shard_id` (survey pass).
    fn check_node(&mut self, shard_id: u16, id: NodeId) -> Result<(), RestoreError> {
        let engine = self.target.engine;
        let Some(survey) = self.survey.as_mut() else {
            return Ok(());
        };
        if survey.last_checked == Some((shard_id, id)) {
            return Ok(());
        }
        survey.last_checked = Some((shard_id, id));
        // The plain key prefixes every temporal version of the node, so one
        // probe finds a record of either kind.
        let holds_nodes = survey.holds_nodes;
        let recorded = || -> Result<bool, RestoreError> {
            use coordinode_storage::Guard as _;
            if !holds_nodes {
                return Ok(false);
            }
            let storage = |e: &dyn std::fmt::Display| RestoreError::Storage(e.to_string());
            let mut records = engine
                .prefix_scan(Partition::Node, &encode_node_key(shard_id, id))
                .map_err(|e| storage(&e))?;
            match records.next() {
                None => Ok(false),
                Some(record) => record.into_inner().map(|_| true).map_err(|e| storage(&e)),
            }
        };
        let issued = if id.origin_shard_hint() == 0 {
            let sequence = id.sequence();
            survey.max_sequence = survey.max_sequence.max(sequence);
            // Sequence 0 is never allocated, so only a record can hold it.
            (1..=survey.ceiling).contains(&sequence) || recorded()?
        } else {
            // CE allocates from hint 0 only: an identifier of another hint is
            // issued only as a record here.
            recorded()?
        };
        if issued {
            survey.issued_count += 1;
            if survey.issued.len() < ISSUED_REPORTED {
                survey.issued.push(id.as_raw());
            }
        }
        Ok(())
    }

    /// A node of a text format: checked in the survey, written in the write
    /// pass.
    fn node(
        &mut self,
        shard_id: u16,
        id: u64,
        write: impl FnOnce(&RestoreTarget<'_>) -> Result<(), RestoreError>,
    ) -> Result<(), RestoreError> {
        let node_id = NodeId::from_raw(id);
        if self.writes() {
            write(self.target)?;
            self.stats.nodes += 1;
            Ok(())
        } else {
            self.check_node(shard_id, node_id)
        }
    }

    /// An edge: written in the write pass only.
    fn edge(
        &mut self,
        write: impl FnOnce(&RestoreTarget<'_>) -> Result<(), RestoreError>,
    ) -> Result<(), RestoreError> {
        if self.writes() {
            write(self.target)?;
            self.stats.edges += 1;
        }
        Ok(())
    }
}

/// Restore from a binary (MessagePack) backup dump.
///
/// Reads length-prefixed MessagePack entries and writes them
/// directly to storage partitions. Fastest restore method.
/// Writes use plain engine.put(); the oracle stamps each seqno.
///
/// The dump's records are encoded with its field dictionary, which it
/// carries ahead of them: those exact bindings are published through
/// `fields` before any record is written, and a dump whose bindings
/// contradict the target's is refused.
fn restore_binary<R: Read>(
    load: &mut Load<'_, '_>,
    reader: &mut R,
    force: bool,
) -> Result<(), RestoreError> {
    let storage = |e: coordinode_storage::error::StorageError| RestoreError::Storage(e.to_string());
    let engine = load.target.engine;
    let mut adopted = false;
    let mut len_buf = [0u8; 4];
    let mut entries = 0u64;

    loop {
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(RestoreError::Io(e)),
        }

        let entry_len = u32::from_le_bytes(len_buf) as usize;
        let mut entry_buf = vec![0u8; entry_len];
        reader.read_exact(&mut entry_buf)?;

        let entry: BackupEntry = rmp_serde::from_slice(&entry_buf)
            .map_err(|e| RestoreError::Deserialization(e.to_string()))?;

        // The manifest must lead the dump. Any other first entry means a
        // pre-versioned or corrupt file; reject unless forced.
        let first_entry = entries == 0;
        entries += 1;
        if first_entry && !matches!(entry, BackupEntry::Manifest { .. }) && !force {
            return Err(RestoreError::IncompatibleVersion(
                "binary dump has no leading manifest (pre-versioned or corrupt); \
                 re-export, or force to restore anyway"
                    .to_string(),
            ));
        }

        match entry {
            BackupEntry::Manifest {
                format_version,
                producer,
                schema_fingerprint,
            } => {
                // Checked after the survey, once the load knows whether it
                // resumes its own interrupted run.
                if let Some(survey) = load.survey.as_mut() {
                    survey.manifest = Some(Manifest {
                        format_version,
                        producer,
                        schema_fingerprint,
                    });
                }
            }
            BackupEntry::Interner(data) => {
                let bindings = FieldInterner::from_bytes(&data)
                    .map_err(|e| RestoreError::Deserialization(e.to_string()))?;
                if load.writes() {
                    load.target
                        .fields
                        .adopt(&bindings)
                        .map_err(|e| RestoreError::Storage(e.to_string()))?;
                }
                adopted = true;
            }
            BackupEntry::Node { .. } | BackupEntry::EdgeProp { .. } if !adopted => {
                return Err(RestoreError::InvalidFormat(
                    "the dump holds property records before the field dictionary \
                     that encodes them"
                        .into(),
                ));
            }
            BackupEntry::Node { key, value } => {
                if load.writes() {
                    engine.put(Partition::Node, &key, &value).map_err(storage)?;
                    load.stats.nodes += 1;
                } else {
                    let (shard, id) = decode_node_key(&key)
                        .or_else(|| {
                            decode_temporal_node_key(&key).map(|(shard, id, _)| (shard, id))
                        })
                        .ok_or_else(|| {
                            RestoreError::InvalidFormat(
                                "the dump holds a malformed node key".into(),
                            )
                        })?;
                    load.check_node(shard, id)?;
                }
            }
            BackupEntry::Adj { key, value } => {
                if load.writes() {
                    // Adj keys are raw (no MVCC timestamps) — write directly to engine.
                    engine.put(Partition::Adj, &key, &value).map_err(storage)?;
                    let key_str = std::str::from_utf8(&key).unwrap_or("");
                    if key_str.contains(":out:") {
                        load.stats.edges += 1;
                    }
                }
            }
            BackupEntry::EdgeProp { key, value } => {
                if load.writes() {
                    engine
                        .put(Partition::EdgeProp, &key, &value)
                        .map_err(storage)?;
                }
            }
            BackupEntry::Schema { key, value } => {
                if load.writes() {
                    // Schema is not MVCC-versioned — write directly.
                    engine
                        .put(Partition::Schema, &key, &value)
                        .map_err(storage)?;
                    load.stats.schema_entries += 1;
                }
            }
        }
    }
    Ok(())
}

/// Validate a binary dump manifest against the target engine.
///
/// Two gates, both overridable with `force`:
/// 1. **Format version**: refuse a dump whose `format_version` is newer
///    than this build understands (a newer producer may use encodings we
///    cannot decode). Older dumps are accepted (we stay backward-readable).
/// 2. **Schema fingerprint**: if the target database already holds schema
///    (non-empty fingerprint that differs from the dump's), merging risks
///    conflicting label / property-type definitions, so reject. Restoring
///    into a fresh database (fingerprint of an empty schema) always passes.
fn validate_manifest(
    engine: &StorageEngine,
    manifest: &Manifest,
    force: bool,
) -> Result<(), RestoreError> {
    use super::export::{BINARY_FORMAT_VERSION, schema_fingerprint};

    let Manifest {
        format_version,
        ref producer,
        schema_fingerprint: dump_fingerprint,
    } = *manifest;
    if format_version > BINARY_FORMAT_VERSION && !force {
        return Err(RestoreError::IncompatibleVersion(format!(
            "dump format v{format_version} from {producer} is newer than supported v{BINARY_FORMAT_VERSION}; \
             upgrade coordinode or force to attempt a best-effort restore"
        )));
    }

    let target_fingerprint =
        schema_fingerprint(engine).map_err(|e| RestoreError::Storage(e.to_string()))?;
    // Fingerprint of an empty schema partition is the FNV offset basis; a
    // fresh restore target matches it, so only a populated, differing target
    // trips the guard.
    let empty_fingerprint = schema_fingerprint_of_empty();
    let target_is_empty = target_fingerprint == empty_fingerprint;
    if !target_is_empty && target_fingerprint != dump_fingerprint && !force {
        return Err(RestoreError::SchemaMismatch(format!(
            "target schema fingerprint {target_fingerprint:#x} differs from dump {dump_fingerprint:#x}; \
             restoring would merge incompatible schemas, force to override"
        )));
    }

    Ok(())
}

/// FNV-1a fingerprint of an empty schema partition (no `(k,v)` pairs).
/// Equals the algorithm's offset basis since the mix loop never runs.
fn schema_fingerprint_of_empty() -> u64 {
    0xcbf2_9ce4_8422_2325
}

/// Restore from JSON Lines backup.
///
/// Each line is a JSON object with `"type": "node"` or `"type": "edge"`.
/// Nodes are created via direct storage writes; edges are created by
/// encoding the adjacency keys.
///
/// Property names are registered through `fields` before each record that
/// uses them. Writes go straight to the engine; the oracle stamps each seqno.
/// An edge instance of a `DISCRIMINATED BY` or temporal edge type carries its
/// encoded discriminator in hex under `"discriminator"`.
fn restore_json<R: BufRead>(
    load: &mut Load<'_, '_>,
    options: &RestoreOptions<'_>,
    reader: &mut R,
) -> Result<(), RestoreError> {
    let RestoreOptions {
        shard_id,
        only_labels,
        ..
    } = *options;
    // Selective restore: with `only_labels`, keep only nodes carrying a matching
    // label and drop edges whose endpoints were filtered out. Exports list nodes
    // before edges, so `kept` is complete by the time edges are read.
    let mut kept: HashSet<u64> = HashSet::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| RestoreError::Deserialization(format!("invalid JSON: {e}")))?;

        let entity_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RestoreError::InvalidFormat("missing 'type' field".into()))?;

        match entity_type {
            "node" => {
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| RestoreError::InvalidFormat("node missing 'id'".into()))?;
                let labels = json_labels(obj.get("labels"));
                if let Some(filter) = only_labels {
                    if !labels.iter().any(|l| filter.contains(l)) {
                        continue;
                    }
                    kept.insert(id);
                }
                let valid_from = match obj.get("valid_from") {
                    None | Some(serde_json::Value::Null) => None,
                    Some(v) => Some(v.as_i64().ok_or_else(|| {
                        RestoreError::InvalidFormat(format!(
                            "node valid_from is not an integer: {v}"
                        ))
                    })?),
                };
                let props = obj.get("properties").and_then(|v| v.as_object());
                load.node(shard_id, id, |t| {
                    write_node(t, shard_id, id, valid_from, labels, json_props(props))
                })?;
            }
            "edge" => {
                let source = obj
                    .get("source")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| RestoreError::InvalidFormat("edge missing 'source'".into()))?;
                let target = obj
                    .get("target")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| RestoreError::InvalidFormat("edge missing 'target'".into()))?;
                let edge_type = obj
                    .get("edge_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        RestoreError::InvalidFormat("edge missing 'edge_type'".into())
                    })?;
                let discriminator = match obj.get("discriminator") {
                    None | Some(serde_json::Value::Null) => None,
                    Some(serde_json::Value::String(hex)) => Some(decode_hex(hex)?),
                    Some(other) => {
                        return Err(RestoreError::InvalidFormat(format!(
                            "edge discriminator is not a hex string: {other}"
                        )));
                    }
                };
                if only_labels.is_some() && (!kept.contains(&source) || !kept.contains(&target)) {
                    continue;
                }
                let props = obj.get("properties").and_then(|v| v.as_object());
                load.edge(|t| {
                    write_edge_record(
                        t,
                        source,
                        target,
                        edge_type,
                        discriminator.as_deref(),
                        props,
                    )
                })?;
            }
            kind @ ("label_schema" | "edge_type_schema" | "index") => {
                json_schema(load, kind, &obj, only_labels)?;
            }
            other => {
                return Err(RestoreError::InvalidFormat(format!(
                    "unknown entity type: {other}"
                )));
            }
        }
    }
    Ok(())
}

/// One schema declaration of a `json` dump. A target that declares the same
/// name otherwise is refused, in the survey, before anything is written; a
/// target that lacks it gets it in the write pass, and one that declares it
/// the same way is left as it is.
fn json_schema(
    load: &mut Load<'_, '_>,
    kind: &str,
    obj: &serde_json::Value,
    only_labels: Option<&HashSet<String>>,
) -> Result<(), RestoreError> {
    use coordinode_core::schema::definition::{EdgeTypeSchema, LabelSchema};
    use coordinode_modality::{
        IndexStore as _, LocalIndexStore, LocalSchemaStore, SchemaStore as _,
    };
    use coordinode_query::index::IndexDescriptor;

    let engine = load.target.engine;
    let storage = |e: coordinode_modality::StoreError| RestoreError::Storage(e.to_string());
    let field = if kind == "index" {
        "definition"
    } else {
        "schema"
    };
    let body = obj
        .get(field)
        .cloned()
        .ok_or_else(|| RestoreError::InvalidFormat(format!("{kind} line without '{field}'")))?;
    let decode_error = |e: serde_json::Error| RestoreError::Deserialization(format!("{kind}: {e}"));
    let differs = |what: &str, name: &str| {
        RestoreError::SchemaMismatch(format!(
            "{what} '{name}' is declared differently in the target"
        ))
    };
    let kept = |label: &str| only_labels.is_none_or(|filter| filter.contains(label));
    let schemas = LocalSchemaStore::new(engine);
    let indexes = LocalIndexStore::new(engine);

    let written = match kind {
        "label_schema" => {
            let schema: LabelSchema = serde_json::from_value(body).map_err(decode_error)?;
            if !kept(&schema.name) {
                return Ok(());
            }
            match schemas.load_label(&schema.name).map_err(storage)? {
                Some(current) if current == schema => false,
                Some(_) => return Err(differs("label", &schema.name)),
                None if load.writes() => {
                    schemas.save_label(&schema).map_err(storage)?;
                    true
                }
                None => false,
            }
        }
        "edge_type_schema" => {
            let schema: EdgeTypeSchema = serde_json::from_value(body).map_err(decode_error)?;
            match schemas.load_edge_type(&schema.name).map_err(storage)? {
                Some(current) if current == schema => false,
                Some(_) => return Err(differs("edge type", &schema.name)),
                None if load.writes() => {
                    schemas.save_edge_type(&schema).map_err(storage)?;
                    true
                }
                None => false,
            }
        }
        _ => {
            let dumped: IndexDescriptor = serde_json::from_value(body).map_err(decode_error)?;
            if !kept(&dumped.label) {
                return Ok(());
            }
            // The index the target already has under the dumped one's name,
            // or, for an index without a name, one that declares the same.
            let current = match &dumped.name {
                Some(name) => match indexes.resolve_name(name).map_err(storage)? {
                    Some(id) => indexes.load_definition(id).map_err(storage)?,
                    None => None,
                },
                None => indexes
                    .list_definitions()
                    .map_err(storage)?
                    .into_iter()
                    .find(|current| same_index(current, &dumped)),
            };
            let shown = dumped.name.clone().unwrap_or_else(|| "(unnamed)".into());
            match current {
                Some(current) if same_index(&current, &dumped) => false,
                Some(_) => return Err(differs("index", &shown)),
                None if load.writes() => {
                    // An index resolves its label and properties through the
                    // field dictionary, as its DDL registered them; the
                    // entries are built from the loaded nodes once they are in.
                    let mut names: Vec<&str> = vec![dumped.label.as_str()];
                    names.extend(dumped.properties.iter().map(String::as_str));
                    field_ids(load.target.fields, &names)?;
                    // Recreated in the target catalog, which gives it
                    // identities of its own; the restore writes straight to
                    // storage, so the publication applies at once.
                    let mut txn = coordinode_storage::engine::transaction::Transaction::new(
                        engine,
                        None,
                        coordinode_core::txn::timestamp::Timestamp::ZERO,
                        None,
                    );
                    indexes
                        .publish_definition_txn(&mut txn, dumped)
                        .map_err(storage)?;
                    true
                }
                None => false,
            }
        }
    };
    if written {
        load.stats.schema_entries += 1;
    }
    Ok(())
}

/// Whether an index of the target and a dumped one declare the same index:
/// its identities, build state, entry layout and the multikey flag its data
/// set are not part of the declaration.
fn same_index(
    current: &coordinode_query::index::IndexDefinition,
    dumped: &coordinode_query::index::IndexDescriptor,
) -> bool {
    let mut dumped = dumped.clone();
    dumped.state = current.state.clone();
    dumped.layout = current.layout;
    dumped.multikey = current.multikey;
    dumped.maintenance.epoch = current.maintenance.epoch;
    dumped.maintenance.source = current.maintenance.source;
    current.descriptor == dumped
}

/// Decode a lowercase or uppercase hex string.
fn decode_hex(text: &str) -> Result<Vec<u8>, RestoreError> {
    super::export::hex::decode(text)
        .ok_or_else(|| RestoreError::InvalidFormat(format!("not a hex string: {text}")))
}

/// Restore from a Neo4j APOC json-export dump (`apoc.export.json.all`).
///
/// APOC emits JSON Lines of `{"type":"node"|"relationship", ...}` where ids
/// are stringified Neo4j internal ids. This is the same on-disk path our own
/// json restore uses, modulo three mechanical differences handled here:
/// string ids, a `relationship` record (vs our `edge`) carrying nested
/// `start`/`end` objects, and `label` (vs `edge_type`). We never execute
/// APOC; this reads its portable output and writes straight to storage, the
/// same way [`restore_json`] does. Records of other types (graph metadata)
/// are skipped.
fn restore_apoc_json<R: BufRead>(
    load: &mut Load<'_, '_>,
    options: &RestoreOptions<'_>,
    reader: &mut R,
) -> Result<(), RestoreError> {
    let RestoreOptions {
        shard_id,
        only_labels,
        ..
    } = *options;
    // Selective restore: keep only label-matching nodes; drop edges to dropped
    // nodes (APOC lists nodes before relationships).
    let mut kept: HashSet<u64> = HashSet::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| RestoreError::Deserialization(format!("invalid JSON: {e}")))?;

        match obj.get("type").and_then(|v| v.as_str()) {
            Some("node") => {
                let id = apoc_id(obj.get("id"), "node")?;
                let labels = json_labels(obj.get("labels"));
                if let Some(filter) = only_labels {
                    if !labels.iter().any(|l| filter.contains(l)) {
                        continue;
                    }
                    kept.insert(id);
                }
                let props = obj.get("properties").and_then(|v| v.as_object());
                load.node(shard_id, id, |t| {
                    write_node_record(t, shard_id, id, labels, props)
                })?;
            }
            Some("relationship") => {
                let source = apoc_id(
                    obj.get("start").and_then(|s| s.get("id")),
                    "relationship start",
                )?;
                let target = apoc_id(obj.get("end").and_then(|e| e.get("id")), "relationship end")?;
                let edge_type = obj.get("label").and_then(|v| v.as_str()).ok_or_else(|| {
                    RestoreError::InvalidFormat("relationship missing 'label'".into())
                })?;
                if only_labels.is_some() && (!kept.contains(&source) || !kept.contains(&target)) {
                    continue;
                }
                let props = obj.get("properties").and_then(|v| v.as_object());
                load.edge(|t| write_edge_record(t, source, target, edge_type, None, props))?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Collect a JSON labels array into owned strings.
fn json_labels(v: Option<&serde_json::Value>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse an APOC stringified internal id into a `u64`. Tolerates a raw
/// numeric id too, so non-standard exporters that skip the quoting still load.
fn apoc_id(v: Option<&serde_json::Value>, what: &str) -> Result<u64, RestoreError> {
    let v = v.ok_or_else(|| RestoreError::InvalidFormat(format!("{what} missing id")))?;
    if let Some(n) = v.as_u64() {
        return Ok(n);
    }
    v.as_str()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| RestoreError::InvalidFormat(format!("{what} id is not a numeric string")))
}

/// The ids of `names`, registering the missing ones in one batch, durable
/// before the record that uses them is written.
fn field_ids(fields: &dyn FieldRegistrar, names: &[&str]) -> Result<Vec<u32>, RestoreError> {
    fields
        .register(names)
        .map_err(|e| RestoreError::Storage(e.to_string()))
}

/// Write one node (labels + properties) to storage, registering its
/// property names first. Each record commits in its own MVCC transaction
/// (bulk restore is a stream of independent writes; per-record commit
/// bounds buffer growth).
fn write_node_record(
    target: &RestoreTarget<'_>,
    shard_id: u16,
    id: u64,
    labels: Vec<String>,
    props: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<(), RestoreError> {
    write_node(target, shard_id, id, None, labels, json_props(props))
}

/// A json property map as `(name, value)` pairs.
fn json_props(props: Option<&serde_json::Map<String, serde_json::Value>>) -> Vec<(String, Value)> {
    props
        .map(|props| {
            props
                .iter()
                .map(|(name, value)| (name.clone(), json_to_value(value)))
                .collect()
        })
        .unwrap_or_default()
}

/// Write a single fully-built node record straight to the engine (cheap
/// direct-mode transaction: bulk restore is non-transactional, so it skips
/// the MVCC buffer / OCC / snapshot / commit overhead — `put` lands
/// immediately). Shared by every text restore path. A `valid_from` makes it
/// the version of a temporal node stored under that instant.
fn write_node(
    target: &RestoreTarget<'_>,
    shard_id: u16,
    id: u64,
    valid_from: Option<i64>,
    labels: Vec<String>,
    props: Vec<(String, Value)>,
) -> Result<(), RestoreError> {
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};

    let mut record = NodeRecord::with_labels(labels);
    let names: Vec<&str> = props.iter().map(|(n, _)| n.as_str()).collect();
    let ids = field_ids(target.fields, &names)?;
    for ((_, value), field_id) in props.into_iter().zip(ids) {
        record.set(field_id, value);
    }
    let mut txn = Transaction::new(target.engine, None, Timestamp::ZERO, None);
    let node_id = NodeId::from_raw(id);
    match valid_from {
        None => LocalNodeStore.put(&mut txn, shard_id, node_id, &record),
        Some(valid_from) => {
            LocalNodeStore.put_temporal(&mut txn, shard_id, node_id, valid_from, &record)
        }
    }
    .map_err(|e| RestoreError::Storage(e.to_string()))
}

/// Write one edge from a json property map.
fn write_edge_record(
    target: &RestoreTarget<'_>,
    source: u64,
    target_id: u64,
    edge_type: &str,
    discriminator: Option<&[u8]>,
    props: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<(), RestoreError> {
    write_edge(
        target,
        source,
        target_id,
        edge_type,
        discriminator,
        json_props(props),
    )
}

/// Write one edge: both adjacency directions (merge operator, raw keys) plus
/// its property body in executor-native shape. An instance of a discriminated
/// or temporal edge type is its body under the discriminator, so that body is
/// written even when it holds no property; a plain edge without properties
/// has none.
fn write_edge(
    target: &RestoreTarget<'_>,
    source: u64,
    target_id: u64,
    edge_type: &str,
    discriminator: Option<&[u8]>,
    props: Vec<(String, Value)>,
) -> Result<(), RestoreError> {
    use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse};

    let engine = target.engine;
    let fwd_key = encode_adj_key_forward(edge_type, NodeId::from_raw(source));
    engine
        .merge(
            Partition::Adj,
            &fwd_key,
            &coordinode_storage::engine::merge::encode_add(target_id),
        )
        .map_err(|e| RestoreError::Storage(e.to_string()))?;

    let rev_key = encode_adj_key_reverse(edge_type, NodeId::from_raw(target_id));
    engine
        .merge(
            Partition::Adj,
            &rev_key,
            &coordinode_storage::engine::merge::encode_add(source),
        )
        .map_err(|e| RestoreError::Storage(e.to_string()))?;

    if props.is_empty() && discriminator.is_none() {
        return Ok(());
    }
    // Write the edge property body through the typed EdgeStore direct-write
    // helper so restore never hand-rolls the edge-prop key. The body uses the
    // single canonical codec, so restored bytes match a transactional write
    // exactly and stay readable by queries.
    let mut edge_props = EdgeProperties::new();
    let names: Vec<&str> = props.iter().map(|(n, _)| n.as_str()).collect();
    let ids = field_ids(target.fields, &names)?;
    for ((_, value), field_id) in props.into_iter().zip(ids) {
        edge_props.set(field_id, value);
    }
    LocalEdgeStore
        .put_props_direct(
            engine,
            edge_type,
            NodeId::from_raw(source),
            NodeId::from_raw(target_id),
            discriminator,
            &edge_props,
        )
        .map_err(|e| RestoreError::Storage(e.to_string()))
}

/// Restore from a Cypher (OpenCypher `CREATE` statements) backup dump.
///
/// Parses the statement form emitted by [`super::export::export_cypher`]:
///
/// ```text
/// CREATE (n<id>:<L1>:<L2> {<props>});
/// CREATE (n<src>)-[:<TYPE> {<props>}]->(n<tgt>);
/// ```
///
/// and writes directly to storage, preserving the original node ids so
/// edges link to their endpoints. Property values are JSON literals (see
/// `format_cypher_props`), parsed via the shared `json_to_value`.
///
/// This is the round-trip path for CoordiNode's own cypher dumps. It is
/// deliberately NOT a general OpenCypher importer: arbitrary external
/// cypher (foreign schemas, computed expressions, multi-statement scope)
/// must go through the query engine. An instance of a discriminated or
/// temporal edge type ends its line with [`CYPHER_DISCRIMINATOR`] and the
/// encoded discriminator in hex, a comment any other OpenCypher reader skips;
/// a version of a temporal node ends its line with [`CYPHER_VALID_FROM`] and
/// the instant it is stored under.
fn restore_cypher<R: BufRead>(
    load: &mut Load<'_, '_>,
    options: &RestoreOptions<'_>,
    reader: &mut R,
) -> Result<(), RestoreError> {
    let shard_id = options.shard_id;
    // One statement per line, each terminated by `;`. Export emits JSON
    // values, which escape newlines, so line-based splitting is safe.
    for line_result in reader.lines() {
        let line = line_result?;
        let (line, discriminator) = split_cypher_discriminator(line.trim())?;
        let (line, valid_from) = split_cypher_valid_from(line);
        let stmt = line.trim().trim_end_matches(';').trim();
        if stmt.is_empty() || stmt.starts_with("//") {
            continue;
        }
        let body = stmt
            .strip_prefix("CREATE ")
            .ok_or_else(|| {
                RestoreError::InvalidFormat(format!("expected CREATE statement: {stmt}"))
            })?
            .trim();

        if body.contains(")-[") {
            let (source, edge_type, target, props) = parse_cypher_edge(body)?;
            load.edge(|t| {
                write_edge(
                    t,
                    source,
                    target,
                    &edge_type,
                    discriminator.as_deref(),
                    props,
                )
            })?;
        } else {
            let (id, labels, props) = parse_cypher_node(body)?;
            load.node(shard_id, id, |t| {
                write_node(t, shard_id, id, valid_from, labels, props)
            })?;
        }
    }
    Ok(())
}

/// The comment that carries an edge instance's discriminator on its line,
/// followed by a space and the discriminator in hex (nothing at all for an
/// empty one).
pub(crate) const CYPHER_DISCRIMINATOR: &str = "; // discriminator";

/// The comment that carries the valid_from a temporal node version is stored
/// under on its line, followed by a space and the instant in milliseconds.
pub(crate) const CYPHER_VALID_FROM: &str = "; // valid_from";

/// Split a trailing `marker` comment off `line`, returning the statement
/// with its `;` and the comment's value. The last occurrence counts, and only
/// when it ends the line (an empty value, if `accept` takes one) or a space
/// and an accepted value follow it to the end: a string value that happens to
/// hold the text is followed by more of the statement.
fn split_cypher_comment<'l>(
    line: &'l str,
    marker: &str,
    accept: impl Fn(&str) -> bool,
) -> (&'l str, Option<&'l str>) {
    let Some(at) = line.rfind(marker) else {
        return (line, None);
    };
    let rest = &line[at + marker.len()..];
    let value = if rest.is_empty() {
        ""
    } else {
        match rest.strip_prefix(' ') {
            Some(value) if !value.is_empty() => value,
            _ => return (line, None),
        }
    };
    if !accept(value) {
        return (line, None);
    }
    (&line[..=at], Some(value))
}

/// Split a trailing [`CYPHER_DISCRIMINATOR`] comment off `line`.
fn split_cypher_discriminator(line: &str) -> Result<(&str, Option<Vec<u8>>), RestoreError> {
    match split_cypher_comment(line, CYPHER_DISCRIMINATOR, |hex| {
        hex.bytes().all(|b| b.is_ascii_hexdigit())
    }) {
        (line, None) => Ok((line, None)),
        (line, Some(hex)) => Ok((line, Some(decode_hex(hex)?))),
    }
}

/// Split a trailing [`CYPHER_VALID_FROM`] comment off `line`.
fn split_cypher_valid_from(line: &str) -> (&str, Option<i64>) {
    match split_cypher_comment(line, CYPHER_VALID_FROM, |ms| ms.parse::<i64>().is_ok()) {
        (line, Some(ms)) => (line, ms.parse().ok()),
        (line, None) => (line, None),
    }
}

/// Decoded `(node_id, labels, properties)` from a cypher node statement.
type CypherNode = (u64, Vec<String>, Vec<(String, Value)>);
/// Decoded `(source_id, edge_type, target_id, properties)` from a cypher edge.
type CypherEdge = (u64, String, u64, Vec<(String, Value)>);

/// Parse `(n<id>:<labels> {<props>})` or `(n<id>:<labels>)`.
fn parse_cypher_node(body: &str) -> Result<CypherNode, RestoreError> {
    let inner = body
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed node: {body}")))?;
    // Split the variable+labels head from the optional ` {props}` tail.
    let (head, props) = match inner.find(" {") {
        Some(brace) => {
            let head = inner[..brace].trim();
            let props_block = inner[brace + 2..]
                .trim_end()
                .strip_suffix('}')
                .ok_or_else(|| {
                    RestoreError::InvalidFormat(format!("unterminated props: {inner}"))
                })?;
            (head, parse_cypher_props(props_block)?)
        }
        None => (inner.trim(), Vec::new()),
    };
    // head = `n<id>:<L1>:<L2>` (labels may be empty).
    let head = head.strip_prefix('n').ok_or_else(|| {
        RestoreError::InvalidFormat(format!("node var must start with n: {head}"))
    })?;
    let (id_str, label_str) = match head.find(':') {
        Some(c) => (&head[..c], &head[c + 1..]),
        None => (head, ""),
    };
    let id: u64 = id_str
        .trim()
        .parse()
        .map_err(|_| RestoreError::InvalidFormat(format!("bad node id: {id_str}")))?;
    let labels: Vec<String> = label_str
        .split(':')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    Ok((id, labels, props))
}

/// Parse `(n<src>)-[:<TYPE> {<props>}]->(n<tgt>)`.
fn parse_cypher_edge(body: &str) -> Result<CypherEdge, RestoreError> {
    let rel_open = body
        .find(")-[")
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed edge: {body}")))?;
    let rel_close = body
        .find("]->(")
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed edge: {body}")))?;
    let src = parse_node_ref(&body[..rel_open + 1])?;
    let tgt_str = body[rel_close + 4..]
        .trim_end()
        .strip_suffix(')')
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed edge target: {body}")))?;
    let tgt = parse_pid(tgt_str)?;
    // rel = `:<TYPE>` or `:<TYPE> {props}`.
    let rel = body[rel_open + 3..rel_close]
        .trim()
        .strip_prefix(':')
        .ok_or_else(|| RestoreError::InvalidFormat(format!("edge missing type: {body}")))?
        .trim();
    let (edge_type, props) = match rel.find(" {") {
        Some(brace) => {
            let props_block = rel[brace + 2..]
                .trim_end()
                .strip_suffix('}')
                .ok_or_else(|| RestoreError::InvalidFormat(format!("unterminated props: {rel}")))?;
            (
                rel[..brace].trim().to_string(),
                parse_cypher_props(props_block)?,
            )
        }
        None => (rel.to_string(), Vec::new()),
    };
    Ok((src, edge_type, tgt, props))
}

/// Parse `(n<id>)` -> id.
fn parse_node_ref(s: &str) -> Result<u64, RestoreError> {
    let inner = s
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed node ref: {s}")))?;
    parse_pid(inner.trim())
}

/// Parse `n<id>` -> id.
fn parse_pid(s: &str) -> Result<u64, RestoreError> {
    s.strip_prefix('n')
        .and_then(|d| d.trim().parse().ok())
        .ok_or_else(|| RestoreError::InvalidFormat(format!("bad node ref: {s}")))
}

/// Parse a cypher property block `key: <json>, key2: <json>` (no braces)
/// as emitted by `format_cypher_props`: bare identifier keys, JSON values.
fn parse_cypher_props(s: &str) -> Result<Vec<(String, Value)>, RestoreError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for seg in split_top_level(s, ',') {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }
        let colon = find_top_level(seg, ':')
            .ok_or_else(|| RestoreError::InvalidFormat(format!("bad property: {seg}")))?;
        let key = seg[..colon].trim().to_string();
        let val_str = seg[colon + 1..].trim();
        let json: serde_json::Value = serde_json::from_str(val_str).map_err(|e| {
            RestoreError::Deserialization(format!("property value '{val_str}': {e}"))
        })?;
        out.push((key, json_to_value(&json)));
    }
    Ok(out)
}

/// Split `s` on `delim` at the top nesting level only (ignores delimiters
/// inside `"..."` strings and `[]` / `{}` brackets).
fn split_top_level(s: &str, delim: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            _ if c == delim && depth == 0 => {
                parts.push(s[start..i].to_string());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(s[start..].to_string());
    parts
}

/// Byte index of the first top-level `delim` in `s`, or None.
fn find_top_level(s: &str, delim: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            _ if c == delim && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Restore from a Neo4j APOC cypher-export dump (`apoc.export.cypher.all`).
///
/// This is a structural parser, NOT a Cypher engine: it recognises the
/// statement shapes APOC emits and writes node/edge records straight to
/// storage, the same way [`restore_cypher`] does for our own dumps. APOC is
/// never executed (it is a Neo4j plugin that does not run correctly on a
/// sharded cluster).
///
/// Both APOC export modes are handled:
/// - **plain** (`useOptimizations: {type: "NONE"}`): one `CREATE (...)`
///   statement per node and a `MATCH (...), (...) CREATE (a)-[:T]->(b)`
///   per relationship. Node ids come from the `UNIQUE IMPORT ID` property.
/// - **unwind-batch** (APOC's default): `UNWIND [ {...}, ... ] AS row CREATE
///   (n{...: row._id}) SET n += row.properties` and the relationship variant.
///
/// Schema statements (`CREATE CONSTRAINT` / `CREATE INDEX` / `DROP ...`),
/// transaction markers (`:begin` / `:commit` / `BEGIN` / `COMMIT`), and the
/// `UNIQUE IMPORT LABEL` cleanup pass are skipped: CoordiNode rebuilds those
/// natively. Cypher functions / temporal literals in values are a hard error
/// (we import data, not evaluate Cypher).
fn restore_apoc_cypher<R: BufRead>(
    load: &mut Load<'_, '_>,
    options: &RestoreOptions<'_>,
    reader: &mut R,
) -> Result<(), RestoreError> {
    let shard_id = options.shard_id;
    let mut text = String::new();
    reader.read_to_string(&mut text)?;

    // Transaction markers sit on their own line with no `;`, so they would
    // otherwise glue onto the following statement and hide it. Drop them
    // before splitting; they carry no graph data.
    let filtered = text
        .lines()
        .filter(|l| {
            let u = l.trim().to_uppercase();
            !(matches!(u.as_str(), "BEGIN" | "COMMIT" | ":BEGIN" | ":COMMIT")
                || u.starts_with("SCHEMA AWAIT"))
        })
        .collect::<Vec<_>>()
        .join("\n");

    for stmt in split_statements(&filtered) {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        apply_apoc_cypher_stmt(stmt, load, shard_id)?;
    }
    Ok(())
}

/// Classify and apply one APOC cypher statement.
fn apply_apoc_cypher_stmt(
    stmt: &str,
    load: &mut Load<'_, '_>,
    shard_id: u16,
) -> Result<(), RestoreError> {
    let upper = stmt.to_uppercase();

    // Transaction markers, schema DDL, and the import-label cleanup pass:
    // CoordiNode rebuilds schema natively, so these carry no graph data.
    let is_skippable = stmt.starts_with("//")
        || matches!(upper.as_str(), "BEGIN" | "COMMIT" | ":BEGIN" | ":COMMIT")
        || upper.starts_with("SCHEMA AWAIT")
        || upper.starts_with("CREATE CONSTRAINT")
        || upper.starts_with("DROP CONSTRAINT")
        || upper.starts_with("CREATE INDEX")
        || upper.starts_with("DROP INDEX")
        || upper.starts_with("CREATE RANGE INDEX")
        || upper.starts_with("CREATE TEXT INDEX")
        || upper.starts_with("CREATE POINT INDEX")
        || upper.starts_with("CREATE LOOKUP INDEX")
        || upper.starts_with("CREATE FULLTEXT INDEX")
        || upper.starts_with("CREATE VECTOR INDEX")
        || (upper.starts_with("MATCH (N:") && upper.contains("REMOVE"));
    if is_skippable {
        return Ok(());
    }

    if upper.starts_with("UNWIND") {
        apply_apoc_unwind(stmt, load, shard_id)
    } else if upper.starts_with("CREATE (") || upper.starts_with("CREATE(") {
        apply_apoc_plain_node(stmt, load, shard_id)
    } else if upper.starts_with("MATCH") && stmt.contains("]->") {
        apply_apoc_plain_rel(stmt, load)
    } else {
        // Unknown maintenance statement (e.g. a vendor-specific clause):
        // skip rather than fail; only CREATE/UNWIND carry graph data.
        Ok(())
    }
}

/// Apply an `UNWIND [...] AS row CREATE/MATCH ...` batch (node or relationship).
fn apply_apoc_unwind(
    stmt: &str,
    load: &mut Load<'_, '_>,
    shard_id: u16,
) -> Result<(), RestoreError> {
    let lb = stmt
        .find('[')
        .ok_or_else(|| RestoreError::InvalidFormat("UNWIND without a list".into()))?;
    let mut lit = CypherLit::new(&stmt[lb..]);
    let rows = lit.list()?;
    let rest = lit.rest();
    let rows = rows
        .as_array()
        .ok_or_else(|| RestoreError::InvalidFormat("UNWIND list is not an array".into()))?;

    if rest.contains("]->") {
        let edge_type = extract_reltype(&rest)?;
        for row in rows {
            let source = nested_id(row, "start")?;
            let target = nested_id(row, "end")?;
            let props = row.get("properties").and_then(|v| v.as_object());
            load.edge(|t| write_edge_record(t, source, target, &edge_type, None, props))?;
        }
    } else {
        let labels = extract_create_labels(&rest)?;
        for row in rows {
            let id = row
                .get("_id")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| RestoreError::InvalidFormat("UNWIND node row missing _id".into()))?;
            let props = row.get("properties").and_then(|v| v.as_object());
            load.node(shard_id, id, |t| {
                write_node_record(t, shard_id, id, labels.clone(), props)
            })?;
        }
    }
    Ok(())
}

/// `row["<side>"]["_id"]` as a `u64` (relationship endpoint).
fn nested_id(row: &serde_json::Value, side: &str) -> Result<u64, RestoreError> {
    row.get(side)
        .and_then(|s| s.get("_id"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| RestoreError::InvalidFormat(format!("UNWIND rel row missing {side}._id")))
}

/// Apply a plain `CREATE (:Labels {props, `UNIQUE IMPORT ID`: N});` node.
fn apply_apoc_plain_node(
    stmt: &str,
    load: &mut Load<'_, '_>,
    shard_id: u16,
) -> Result<(), RestoreError> {
    let body = stmt["CREATE".len()..].trim();
    let inner = body
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed node: {stmt}")))?;
    let brace = first_brace(inner).ok_or_else(|| {
        RestoreError::InvalidFormat(format!("node without UNIQUE IMPORT ID props: {stmt}"))
    })?;
    let labels = parse_label_list(&inner[..brace]);
    let map = CypherLit::new(&inner[brace..]).map()?;
    let mut obj = map
        .as_object()
        .cloned()
        .ok_or_else(|| RestoreError::InvalidFormat("node props not a map".into()))?;
    let id = obj
        .remove("UNIQUE IMPORT ID")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            RestoreError::InvalidFormat(
                "plain node missing `UNIQUE IMPORT ID` (constraint-keyed nodes are not supported)"
                    .into(),
            )
        })?;
    load.node(shard_id, id, |t| {
        write_node_record(t, shard_id, id, labels, Some(&obj))
    })
}

/// Apply a plain `MATCH (a{id:X}), (b{id:Y}) CREATE (a)-[:T {props}]->(b);`.
fn apply_apoc_plain_rel(stmt: &str, load: &mut Load<'_, '_>) -> Result<(), RestoreError> {
    let cpos = find_top_keyword(stmt, "CREATE")
        .ok_or_else(|| RestoreError::InvalidFormat(format!("rel without CREATE: {stmt}")))?;
    let (match_part, create_part) = stmt.split_at(cpos);
    let ids = extract_import_ids(match_part);
    if ids.len() != 2 {
        return Err(RestoreError::InvalidFormat(format!(
            "expected 2 UNIQUE IMPORT IDs in MATCH, got {}: {stmt}",
            ids.len()
        )));
    }
    let edge_type = extract_reltype(create_part)?;
    let props = extract_rel_props(create_part)?;
    let (source, target) = (ids[0], ids[1]);
    load.edge(|t| write_edge_record(t, source, target, &edge_type, None, props.as_ref()))
}

/// Pull the relationship type out of a `-[var:`TYPE` {props}]->` pattern.
fn extract_reltype(s: &str) -> Result<String, RestoreError> {
    let open = s
        .find("-[")
        .ok_or_else(|| RestoreError::InvalidFormat("missing relationship pattern".into()))?;
    let close = s[open..]
        .find("]->")
        .map(|i| open + i)
        .ok_or_else(|| RestoreError::InvalidFormat("missing `]->`".into()))?;
    let inner = &s[open + 2..close];
    let colon = inner
        .find(':')
        .ok_or_else(|| RestoreError::InvalidFormat("relationship missing type".into()))?;
    Ok(strip_label_token(&inner[colon + 1..]))
}

/// Properties of a `-[var:`TYPE` {props}]->` pattern, if any.
fn extract_rel_props(
    s: &str,
) -> Result<Option<serde_json::Map<String, serde_json::Value>>, RestoreError> {
    let open = s
        .find("-[")
        .ok_or_else(|| RestoreError::InvalidFormat("missing relationship pattern".into()))?;
    let close = s[open..]
        .find("]->")
        .map(|i| open + i)
        .ok_or_else(|| RestoreError::InvalidFormat("missing `]->`".into()))?;
    let inner = &s[open + 2..close];
    match first_brace(inner) {
        Some(b) => Ok(CypherLit::new(&inner[b..]).map()?.as_object().cloned()),
        None => Ok(None),
    }
}

/// Labels of a `CREATE (var:`L1`:`L2` ...)` clause inside a larger statement.
fn extract_create_labels(s: &str) -> Result<Vec<String>, RestoreError> {
    let open = s
        .find("CREATE (")
        .map(|i| i + "CREATE (".len())
        .or_else(|| s.find("CREATE(").map(|i| i + "CREATE(".len()))
        .ok_or_else(|| RestoreError::InvalidFormat("UNWIND batch without CREATE".into()))?;
    let head_end = s[open..]
        .find(['{', ')'])
        .map(|i| open + i)
        .unwrap_or(s.len());
    Ok(parse_label_list(&s[open..head_end]))
}

/// Every integer following a `UNIQUE IMPORT ID` occurrence, in order. Used to
/// read both relationship endpoints out of a plain `MATCH ..., ...` head.
fn extract_import_ids(s: &str) -> Vec<u64> {
    const NEEDLE: &str = "UNIQUE IMPORT ID";
    let mut ids = Vec::new();
    let mut from = 0;
    while let Some(p) = s[from..].find(NEEDLE) {
        let after = from + p + NEEDLE.len();
        if let Some(colon) = s[after..].find(':') {
            let num: String = s[after + colon + 1..]
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = num.parse::<u64>() {
                ids.push(n);
            }
        }
        from = after;
    }
    ids
}

/// Split `head` (`var:`L1`:`L2``) into labels, dropping the leading variable
/// and the synthetic `UNIQUE IMPORT LABEL`.
fn parse_label_list(head: &str) -> Vec<String> {
    let mut segs = Vec::new();
    let mut buf = String::new();
    let mut in_bt = false;
    for c in head.chars() {
        match c {
            '`' => {
                in_bt = !in_bt;
                buf.push(c);
            }
            ':' if !in_bt => segs.push(std::mem::take(&mut buf)),
            _ => buf.push(c),
        }
    }
    segs.push(buf);
    segs.into_iter()
        .skip(1) // first segment is the node variable, not a label
        .map(|s| s.trim().trim_matches('`').to_string())
        .filter(|s| !s.is_empty() && s != "UNIQUE IMPORT LABEL")
        .collect()
}

/// A single label/type token: backtick-quoted `` `Name` `` or bare, stopping
/// at whitespace or `{`.
fn strip_label_token(s: &str) -> String {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix('`') {
        if let Some(end) = rest.find('`') {
            return rest[..end].to_string();
        }
    }
    s.split(|c: char| c.is_whitespace() || c == '{')
        .next()
        .unwrap_or("")
        .to_string()
}

/// Byte index of the first `{` not inside a quoted run, or None.
fn first_brace(s: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => quote = Some(c),
            '{' => return Some(i),
            _ => {}
        }
    }
    None
}

/// Byte index of `kw` at bracket-depth 0 outside any quoted run, or None.
fn find_top_keyword(s: &str, kw: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => quote = Some(c),
            '[' | '{' | '(' => depth += 1,
            ']' | '}' | ')' => depth -= 1,
            _ if depth == 0 && s[i..].starts_with(kw) => return Some(i),
            _ => {}
        }
    }
    None
}

/// Split a cypher script into statements on top-level `;`, honoring quoted
/// runs (`"`, `'`, `` ` ``) and `()` / `[]` / `{}` nesting.
fn split_statements(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for c in text.chars() {
        if let Some(q) = quote {
            buf.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => {
                quote = Some(c);
                buf.push(c);
            }
            '[' | '{' | '(' => {
                depth += 1;
                buf.push(c);
            }
            ']' | '}' | ')' => {
                depth -= 1;
                buf.push(c);
            }
            ';' if depth == 0 => out.push(std::mem::take(&mut buf)),
            _ => buf.push(c),
        }
    }
    if !buf.trim().is_empty() {
        out.push(buf);
    }
    out
}

/// A minimal recursive parser for the Cypher literal subset APOC emits: maps
/// with backtick / bare / quoted keys, lists, double/single-quoted strings,
/// numbers, `true`/`false`/`null`, and arbitrary nesting. It does NOT evaluate
/// Cypher functions (`datetime(...)` etc.) — such a value is a hard error,
/// because this path imports data, it does not execute Cypher.
struct CypherLit {
    chars: Vec<char>,
    pos: usize,
}

impl CypherLit {
    fn new(s: &str) -> Self {
        Self {
            chars: s.chars().collect(),
            pos: 0,
        }
    }

    /// Remaining unconsumed input (after a `list`/`map` parse).
    fn rest(&self) -> String {
        self.chars[self.pos..].iter().collect()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn err(msg: impl Into<String>) -> RestoreError {
        RestoreError::InvalidFormat(format!("cypher literal: {}", msg.into()))
    }

    fn expect(&mut self, c: char) -> Result<(), RestoreError> {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(Self::err(format!("expected '{c}'")))
        }
    }

    fn value(&mut self) -> Result<serde_json::Value, RestoreError> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.map(),
            Some('[') => self.list(),
            Some('"') | Some('\'') => Ok(serde_json::Value::String(self.string()?)),
            Some(c) if c == '-' || c == '+' || c.is_ascii_digit() => self.number(),
            Some(_) => self.bareword(),
            None => Err(Self::err("unexpected end")),
        }
    }

    fn map(&mut self) -> Result<serde_json::Value, RestoreError> {
        self.expect('{')?;
        let mut obj = serde_json::Map::new();
        loop {
            self.skip_ws();
            match self.peek() {
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                None => return Err(Self::err("unterminated map")),
                _ => {}
            }
            let key = self.key()?;
            self.expect(':')?;
            let val = self.value()?;
            obj.insert(key, val);
            self.skip_ws();
            match self.advance() {
                Some(',') => {}
                Some('}') => break,
                _ => return Err(Self::err("expected ',' or '}'")),
            }
        }
        Ok(serde_json::Value::Object(obj))
    }

    fn list(&mut self) -> Result<serde_json::Value, RestoreError> {
        self.expect('[')?;
        let mut arr = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                None => return Err(Self::err("unterminated list")),
                _ => {}
            }
            arr.push(self.value()?);
            self.skip_ws();
            match self.advance() {
                Some(',') => {}
                Some(']') => break,
                _ => return Err(Self::err("expected ',' or ']'")),
            }
        }
        Ok(serde_json::Value::Array(arr))
    }

    /// A map key: backtick-quoted, string-quoted, or a bare identifier.
    fn key(&mut self) -> Result<String, RestoreError> {
        self.skip_ws();
        match self.peek() {
            Some('`') => self.backtick(),
            Some('"') | Some('\'') => self.string(),
            Some(_) => {
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if c.is_whitespace() || c == ':' {
                        break;
                    }
                    self.pos += 1;
                }
                if self.pos == start {
                    return Err(Self::err("empty key"));
                }
                Ok(self.chars[start..self.pos].iter().collect())
            }
            None => Err(Self::err("expected key")),
        }
    }

    fn backtick(&mut self) -> Result<String, RestoreError> {
        self.expect('`')?;
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == '`' {
                let s: String = self.chars[start..self.pos].iter().collect();
                self.pos += 1;
                return Ok(s);
            }
            self.pos += 1;
        }
        Err(Self::err("unterminated backtick"))
    }

    fn string(&mut self) -> Result<String, RestoreError> {
        let quote = self.advance().ok_or_else(|| Self::err("expected string"))?;
        let mut out = String::new();
        while let Some(c) = self.advance() {
            if c == '\\' {
                match self.advance() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('b') => out.push('\u{8}'),
                    Some('f') => out.push('\u{c}'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let h = self.advance().ok_or_else(|| Self::err("bad \\u escape"))?;
                            code = code * 16
                                + h.to_digit(16).ok_or_else(|| Self::err("bad \\u hex"))?;
                        }
                        out.push(char::from_u32(code).ok_or_else(|| Self::err("bad codepoint"))?);
                    }
                    Some(other) => out.push(other),
                    None => return Err(Self::err("dangling escape")),
                }
            } else if c == quote {
                return Ok(out);
            } else {
                out.push(c);
            }
        }
        Err(Self::err("unterminated string"))
    }

    fn number(&mut self) -> Result<serde_json::Value, RestoreError> {
        let start = self.pos;
        if matches!(self.peek(), Some('+') | Some('-')) {
            self.pos += 1;
        }
        let mut is_float = false;
        while let Some(c) = self.peek() {
            match c {
                '0'..='9' => self.pos += 1,
                '.' | 'e' | 'E' => {
                    is_float = true;
                    self.pos += 1;
                }
                '+' | '-' => self.pos += 1, // exponent sign
                _ => break,
            }
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            let f: f64 = s
                .parse()
                .map_err(|_| Self::err(format!("bad number '{s}'")))?;
            serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .ok_or_else(|| Self::err("non-finite number"))
        } else {
            let i: i64 = s
                .parse()
                .map_err(|_| Self::err(format!("bad integer '{s}'")))?;
            Ok(serde_json::Value::Number(i.into()))
        }
    }

    fn bareword(&mut self) -> Result<serde_json::Value, RestoreError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let word: String = self.chars[start..self.pos].iter().collect();
        match word.to_lowercase().as_str() {
            "true" => Ok(serde_json::Value::Bool(true)),
            "false" => Ok(serde_json::Value::Bool(false)),
            "null" => Ok(serde_json::Value::Null),
            _ => Err(Self::err(format!(
                "unsupported value '{word}' (cypher functions and temporals are not imported)"
            ))),
        }
    }
}

/// Convert JSON value to CoordiNode Value.
fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::Array(items)
        }
        serde_json::Value::Object(obj) => {
            // Check for special types
            if let Some(ts) = obj.get("_timestamp").and_then(|v| v.as_i64()) {
                return Value::Timestamp(ts);
            }
            if let Some(doc_val) = obj.get("_document") {
                return Value::Document(json_to_rmpv(doc_val));
            }
            if let Some(mv_val) = obj.get("_multi_vector") {
                if let Some(arr) = mv_val.as_array() {
                    let rows: Vec<Vec<f32>> = arr
                        .iter()
                        .filter_map(|row| row.as_array())
                        .map(|row| {
                            row.iter()
                                .filter_map(|x| x.as_f64().map(|f| f as f32))
                                .collect()
                        })
                        .collect();
                    if let Some(v) = Value::try_multi_vector(rows) {
                        return v;
                    }
                    return Value::Null;
                }
            }
            let map: std::collections::BTreeMap<String, Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect();
            Value::Map(map)
        }
    }
}

/// Convert a serde_json::Value to rmpv::Value for Document restore.
fn json_to_rmpv(v: &serde_json::Value) -> rmpv::Value {
    match v {
        serde_json::Value::Null => rmpv::Value::Nil,
        serde_json::Value::Bool(b) => rmpv::Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                rmpv::Value::Integer(i.into())
            } else if let Some(f) = n.as_f64() {
                rmpv::Value::F64(f)
            } else {
                rmpv::Value::Nil
            }
        }
        serde_json::Value::String(s) => rmpv::Value::String(s.clone().into()),
        serde_json::Value::Array(arr) => rmpv::Value::Array(arr.iter().map(json_to_rmpv).collect()),
        serde_json::Value::Object(obj) => {
            // Sort keys alphabetically to produce deterministic rmpv::Map order.
            // serde_json::Map is BTreeMap (sorted), so iterating already yields
            // sorted keys — but we sort explicitly to be safe.
            let mut entries: Vec<(rmpv::Value, rmpv::Value)> = obj
                .iter()
                .map(|(k, v)| (rmpv::Value::String(k.clone().into()), json_to_rmpv(v)))
                .collect();
            entries.sort_by(|(a, _), (b, _)| {
                let ak = a.as_str().unwrap_or("");
                let bk = b.as_str().unwrap_or("");
                ak.cmp(bk)
            });
            rmpv::Value::Map(entries)
        }
    }
}

/// Restore from a Hetionet "hetnet" JSON document (the dhimmel/hetio source
/// format that the `hetnetpy` library turns into a Neo4j graph). The document
/// is a single object: `{nodes: [{kind, identifier, name, data}], edges:
/// [{source_id: [kind, id], target_id: [kind, id], kind, data}]}`.
///
/// Node identity is a `(kind, identifier)` pair (identifier may be a string or
/// an integer), so this mints a sequential node id per node and resolves edge
/// endpoints through that map. `kind` becomes the node label / relationship
/// type; `name` and `data` become properties. This mirrors what hetnetpy does
/// when it writes the hetnet into Neo4j, so the dataset loads straight from the
/// JSON with no Neo4j round trip.
fn restore_hetio_json<R: BufRead>(
    load: &mut Load<'_, '_>,
    options: &RestoreOptions<'_>,
    reader: &mut R,
) -> Result<(), RestoreError> {
    use serde::Deserialize;

    let RestoreOptions {
        shard_id,
        only_labels,
        ..
    } = *options;

    #[derive(Deserialize)]
    struct HetNode {
        kind: String,
        identifier: serde_json::Value,
        #[serde(default)]
        name: Option<serde_json::Value>,
        #[serde(default)]
        data: serde_json::Map<String, serde_json::Value>,
    }
    #[derive(Deserialize)]
    struct HetEdge {
        source_id: (String, serde_json::Value),
        target_id: (String, serde_json::Value),
        kind: String,
        #[serde(default)]
        data: serde_json::Map<String, serde_json::Value>,
    }
    #[derive(Deserialize)]
    struct HetnetDoc {
        nodes: Vec<HetNode>,
        edges: Vec<HetEdge>,
    }

    let doc: HetnetDoc = serde_json::from_reader(reader)
        .map_err(|e| RestoreError::Deserialization(format!("hetnet json: {e}")))?;

    let mut id_map: std::collections::HashMap<(String, String), u64> =
        std::collections::HashMap::with_capacity(doc.nodes.len());

    for (idx, n) in doc.nodes.iter().enumerate() {
        let id = idx as u64;
        // Selective restore: a node whose kind is filtered out is never added to
        // the id map, so edges referencing it resolve to None and drop below.
        if let Some(filter) = only_labels {
            if !filter.contains(&n.kind) {
                continue;
            }
        }
        id_map.insert((n.kind.clone(), ident_key(&n.identifier)), id);
        let mut props = n.data.clone();
        if let Some(name) = &n.name {
            props.insert("name".to_string(), name.clone());
        }
        props.insert("identifier".to_string(), n.identifier.clone());
        load.node(shard_id, id, |t| {
            write_node_record(t, shard_id, id, vec![n.kind.clone()], Some(&props))
        })?;
    }

    for e in &doc.edges {
        let src = id_map.get(&(e.source_id.0.clone(), ident_key(&e.source_id.1)));
        let tgt = id_map.get(&(e.target_id.0.clone(), ident_key(&e.target_id.1)));
        let (Some(&src), Some(&tgt)) = (src, tgt) else {
            // Endpoint not present in the node set: skip the dangling edge.
            continue;
        };
        let props = if e.data.is_empty() {
            None
        } else {
            Some(&e.data)
        };
        load.edge(|t| write_edge_record(t, src, tgt, &e.kind, None, props))?;
    }
    Ok(())
}

/// Canonical string key for a hetnet identifier (string or integer) so a node
/// and the edge endpoints that reference it resolve to the same map entry.
fn ident_key(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
