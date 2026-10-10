//! Storage-backed segment export and install: the bridge between the
//! placement-segment primitive (`coordinode-storage`) and the swarm transport.
//!
//! A segment moves as a **portable key-value stream**, not raw SST bytes: the
//! source reads the segment's key range and serialises its entries; the target
//! re-ingests them under its own codec and tier policy. This is codec- and
//! disk-format-independent, which is required across heterogeneous tiers and
//! rolling upgrades (a cold zstd source and a hot uncompressed target never
//! share a byte format). Raw-SST shipping is a future same-tier/same-codec
//! opt-in; re-ingest is the default.
//!
//! - [`export_segment`] reads a [`SegmentDescriptor`]'s key range into a
//!   portable blob; hand it to a [`LocalPieceStore`](coordinode_swarm::LocalPieceStore)
//!   (`insert`) to serve it over the swarm transport.
//! - [`SegmentInstaller`] implements [`SegmentSink`]: it decodes a received
//!   self-describing blob and installs the entries into the partition named by
//!   the blob's leading wire tag.

use std::sync::Arc;

use std::collections::HashMap;
use std::sync::Mutex;

use coordinode_core::index::identity::GenerationId;
use coordinode_storage::engine::core::{PartitionCopy, RaftPosition, StorageEngine};
use coordinode_storage::engine::installation::{GenerationHistory, HistoryEntry};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::{StorageError, StorageResult};
use coordinode_storage::placement::{
    KeyRange, SegmentDescriptor, partition_from_wire_tag, partition_wire_tag,
};
use coordinode_swarm::{
    Freshness, NodeId, PieceEncoding, PieceSource, SourceCandidate, split_segment, swarm_download,
};

use crate::transfer::proto::SegmentDescriptorRef;
use crate::transfer::{BuiltSegment, GrpcPieceSource, SegmentSink, SegmentSource, SegmentSubject};

/// One key-value entry of a segment's portable representation.
type KvEntry = (Vec<u8>, Vec<u8>);

/// Serialise key-value entries into the portable segment blob.
///
/// Layout per entry: `u32 LE key_len | key | u32 LE val_len | value`, in scan
/// (sorted) order. Deterministic and re-ingestible by [`decode_kv_blob`].
fn encode_kv_blob(entries: &[KvEntry]) -> StorageResult<Vec<u8>> {
    let mut out = Vec::new();
    for (key, value) in entries {
        push_chunk(&mut out, key, "segment key")?;
        push_chunk(&mut out, value, "segment value")?;
    }
    Ok(out)
}

/// Append `bytes` with its `u32 LE` length prefix.
fn push_chunk(out: &mut Vec<u8>, bytes: &[u8], what: &str) -> StorageResult<()> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| StorageError::Serialization(format!("{what} exceeds u32")))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

/// Kinds of a history entry on the wire.
const HISTORY_PUT: u8 = 0;
const HISTORY_DELETE: u8 = 1;
const HISTORY_REMOVE_RANGE: u8 = 2;

/// Serialise history entries: per entry `u8 kind | u64 BE seqno` followed by
/// the key and value (a put), the key (a delete) or the start and end (a range
/// delete), each length-prefixed.
fn encode_history(out: &mut Vec<u8>, entries: &[HistoryEntry]) -> StorageResult<()> {
    for entry in entries {
        match entry {
            HistoryEntry::Put { key, value, seqno } => {
                out.push(HISTORY_PUT);
                out.extend_from_slice(&seqno.to_be_bytes());
                push_chunk(out, key, "history key")?;
                push_chunk(out, value, "history value")?;
            }
            HistoryEntry::Delete { key, seqno } => {
                out.push(HISTORY_DELETE);
                out.extend_from_slice(&seqno.to_be_bytes());
                push_chunk(out, key, "history key")?;
            }
            HistoryEntry::RemoveRange { start, end, seqno } => {
                out.push(HISTORY_REMOVE_RANGE);
                out.extend_from_slice(&seqno.to_be_bytes());
                push_chunk(out, start, "history range start")?;
                push_chunk(out, end, "history range end")?;
            }
        }
    }
    Ok(())
}

/// Parse the entries [`encode_history`] wrote.
fn decode_history(blob: &[u8]) -> Result<Vec<HistoryEntry>, String> {
    let mut entries = Vec::new();
    let mut pos = 0usize;
    while pos < blob.len() {
        let kind = blob[pos];
        let seqno = read_u64(blob, pos + 1, "history seqno")?;
        pos += 9;
        entries.push(match kind {
            HISTORY_PUT => HistoryEntry::Put {
                key: read_chunk(blob, &mut pos)?,
                value: read_chunk(blob, &mut pos)?,
                seqno,
            },
            HISTORY_DELETE => HistoryEntry::Delete {
                key: read_chunk(blob, &mut pos)?,
                seqno,
            },
            HISTORY_REMOVE_RANGE => HistoryEntry::RemoveRange {
                start: read_chunk(blob, &mut pos)?,
                end: read_chunk(blob, &mut pos)?,
                seqno,
            },
            other => return Err(format!("unknown history entry kind {other}")),
        });
    }
    Ok(entries)
}

/// The `u64 BE` at `at`.
fn read_u64(blob: &[u8], at: usize, what: &str) -> Result<u64, String> {
    at.checked_add(8)
        .and_then(|end| blob.get(at..end))
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
        .map(u64::from_be_bytes)
        .ok_or_else(|| format!("segment blob truncated in its {what}"))
}

/// Parse the portable segment blob produced by [`encode_kv_blob`] back into its
/// key-value entries.
fn decode_kv_blob(blob: &[u8]) -> Result<Vec<KvEntry>, String> {
    let mut entries = Vec::new();
    let mut pos = 0usize;
    while pos < blob.len() {
        let key = read_chunk(blob, &mut pos)?;
        let value = read_chunk(blob, &mut pos)?;
        entries.push((key, value));
    }
    Ok(entries)
}

/// Read a `u32 LE length` prefix then that many bytes, advancing `pos`.
fn read_chunk(blob: &[u8], pos: &mut usize) -> Result<Vec<u8>, String> {
    let len_end = pos
        .checked_add(4)
        .filter(|e| *e <= blob.len())
        .ok_or_else(|| "segment blob truncated in length prefix".to_string())?;
    let len = u32::from_le_bytes(
        blob[*pos..len_end]
            .try_into()
            .map_err(|_| "segment blob length prefix".to_string())?,
    ) as usize;
    let data_end = len_end
        .checked_add(len)
        .filter(|e| *e <= blob.len())
        .ok_or_else(|| "segment blob truncated in payload".to_string())?;
    let chunk = blob[len_end..data_end].to_vec();
    *pos = data_end;
    Ok(chunk)
}

/// Export the entries covered by `descriptor` from the engine into a portable,
/// self-describing blob, ready to be split into swarm pieces.
///
/// The blob is `[u8 partition tag] [length-prefixed key-value entries]`. The
/// leading tag lets [`SegmentInstaller`] route a received segment to the right
/// partition with no out-of-band state. Reads the descriptor's key range (the
/// whole partition when the range is unbounded above) and keeps only entries the
/// range actually contains (half-open `[start, end)`).
///
/// # Errors
///
/// Returns an error if the partition is unavailable, a scanned entry cannot be
/// read, or a key/value exceeds the `u32` length bound.
pub fn export_segment(
    engine: &StorageEngine,
    descriptor: &SegmentDescriptor,
) -> StorageResult<Vec<u8>> {
    export_range(engine, descriptor.partition, &descriptor.key_range)
}

/// Export the entries of `part` covered by `range` into the portable,
/// self-describing blob (the core of [`export_segment`], addressed by partition +
/// key range rather than a full descriptor — what the receiver-driven swarm pull
/// needs). The scan is deterministic (sorted key order), so every node exporting
/// the same `(part, range)` produces byte-identical output.
///
/// # Errors
/// Returns an error if the partition is unavailable, a scanned entry cannot be
/// read, or a key/value exceeds the `u32` length bound.
pub fn export_range(
    engine: &StorageEngine,
    part: Partition,
    range: &KeyRange,
) -> StorageResult<Vec<u8>> {
    // The partition's replicated rows, captured at the exact Raft position
    // they stand at when the engine runs Raft, narrowed to the half-open
    // `[start, end)` range (an unbounded range is the whole partition).
    let copy = engine.copy_partition(part)?;
    let entries: Vec<KvEntry> = copy
        .rows
        .into_iter()
        .filter(|(key, _)| range.contains(key))
        .collect();
    let whole = range.start.is_empty() && range.end.is_empty();

    let mut blob = vec![partition_wire_tag(part)];
    let mut flags = 0u8;
    if copy.position.is_some() {
        flags |= FLAG_POSITION;
    }
    if whole {
        flags |= FLAG_WHOLE;
    }
    blob.push(flags);
    if let Some(position) = &copy.position {
        let payload_len = u32::try_from(position.payload.len())
            .map_err(|_| StorageError::Serialization("position payload exceeds u32".into()))?;
        blob.extend_from_slice(&position.next.to_be_bytes());
        blob.extend_from_slice(&payload_len.to_le_bytes());
        blob.extend_from_slice(&position.payload);
    }
    blob.extend_from_slice(&encode_kv_blob(&entries)?);
    Ok(blob)
}

/// Export the retained history of index `generation` into a portable,
/// self-describing blob: `[u8 Idx tag] [u8 FLAG_HISTORY] [u64 BE generation]
/// [u64 BE covers_through] [u64 BE history_from] [history entries]`. Every
/// replica holds a generation's versions at the same commit timestamps, so
/// peers whose compactions kept the same versions build byte-identical blobs.
///
/// # Errors
///
/// The generation has no copy here or holds a version a history cannot
/// carry; a read failure; an entry exceeding the `u32` length bound.
pub fn export_generation(engine: &StorageEngine, generation: u64) -> StorageResult<Vec<u8>> {
    let history = engine.export_generation_history(GenerationId::from_raw(generation))?;
    let mut blob = vec![partition_wire_tag(Partition::Idx), FLAG_HISTORY];
    blob.extend_from_slice(&generation.to_be_bytes());
    blob.extend_from_slice(&history.covers_through.to_be_bytes());
    blob.extend_from_slice(&history.history_from.to_be_bytes());
    encode_history(&mut blob, &history.entries)?;
    Ok(blob)
}

/// Segment header flag: a Raft log position follows the flags.
const FLAG_POSITION: u8 = 1;
/// Segment header flag: the segment holds the whole partition.
const FLAG_WHOLE: u8 = 2;
/// Segment header flag: the segment is an index generation's history
/// ([`export_generation`]), alone among the flags.
const FLAG_HISTORY: u8 = 4;

/// A received segment, decoded.
enum Segment {
    /// Current rows of a partition.
    Rows {
        partition: Partition,
        whole: bool,
        copy: PartitionCopy,
    },
    /// The history of an index generation.
    History {
        generation: u64,
        history: GenerationHistory,
    },
}

/// Parse a segment blob produced by [`export_range`] or [`export_generation`].
fn decode_segment(data: &[u8]) -> Result<Segment, String> {
    let (&tag, rest) = data
        .split_first()
        .ok_or_else(|| "empty segment blob (missing partition tag)".to_string())?;
    let partition =
        partition_from_wire_tag(tag).ok_or_else(|| format!("unknown partition wire tag {tag}"))?;
    let (&flags, mut rest) = rest
        .split_first()
        .ok_or_else(|| "segment blob truncated in its flags".to_string())?;
    if flags == FLAG_HISTORY && partition == Partition::Idx {
        let generation = read_u64(rest, 0, "generation")?;
        let covers_through = read_u64(rest, 8, "history position")?;
        let history_from = read_u64(rest, 16, "history floor")?;
        // The reads above proved the 24 header bytes are there.
        let entries = decode_history(&rest[24..])?;
        return Ok(Segment::History {
            generation,
            history: GenerationHistory {
                covers_through,
                history_from,
                entries,
            },
        });
    }
    if flags & !(FLAG_POSITION | FLAG_WHOLE) != 0 {
        return Err(format!("unknown segment flags {flags:#04x}"));
    }
    let position = if flags & FLAG_POSITION == 0 {
        None
    } else {
        let next = rest
            .get(..8)
            .and_then(|b| <[u8; 8]>::try_from(b).ok())
            .map(u64::from_be_bytes)
            .ok_or_else(|| "segment blob truncated in its position".to_string())?;
        let mut pos = 8;
        let payload = read_chunk(rest, &mut pos)?;
        rest = &rest[pos..];
        Some(RaftPosition { next, payload })
    };
    Ok(Segment::Rows {
        partition,
        whole: flags & FLAG_WHOLE != 0,
        copy: PartitionCopy {
            rows: decode_kv_blob(rest)?,
            position,
        },
    })
}

/// Failure modes of [`drain_segment_to_peer`].
#[derive(Debug, thiserror::Error)]
pub enum DrainError {
    /// Reading the segment's key range out of local storage failed.
    #[error("export segment: {0}")]
    Export(#[from] StorageError),
    /// Splitting the exported blob into swarm pieces failed.
    #[error("split into pieces: {0}")]
    Pieces(String),
    /// Could not open the gRPC connection to the peer.
    #[error("connect to {endpoint}: {source}")]
    Connect {
        /// The peer endpoint that could not be reached.
        endpoint: String,
        /// The underlying tonic transport error.
        source: tonic::transport::Error,
    },
    /// The transfer stream failed at the transport level.
    #[error("transfer rpc: {0}")]
    Rpc(#[from] tonic::Status),
    /// The peer received the stream but rejected the segment (checksum, decode,
    /// or storage failure on the target — the segment was not installed).
    #[error("peer rejected segment {segment}: {error}")]
    Rejected {
        /// The segment id the peer rejected.
        segment: u64,
        /// The peer's reported reason.
        error: String,
    },
}

/// Drain (push) the segment described by `descriptor` from `engine` to the
/// [`SegmentTransferService`](crate::transfer) at `endpoint`
/// (e.g. `"http://10.0.0.2:7080"`).
///
/// Reads the segment's key range into a portable blob, splits it into
/// `piece_size`-byte swarm pieces under `encoding` (the in-flight wire encoding,
/// independent of the target's on-disk codec), and streams them to the peer,
/// which verifies each piece and installs the segment under its own tier/codec
/// policy. Returns the peer's ack on success.
///
/// This is the source side of the `full-storage → compute` segment drain and of
/// operator-commanded migration; the receive side is the registered
/// [`SegmentTransferHandler`](crate::transfer::SegmentTransferHandler).
///
/// # Errors
/// [`DrainError`] for an export failure, a piece-split failure, a connection
/// failure, a transport error, or a target rejection.
pub async fn drain_segment_to_peer(
    engine: &StorageEngine,
    descriptor: &SegmentDescriptor,
    endpoint: &str,
    piece_size: usize,
    encoding: coordinode_swarm::PieceEncoding,
) -> Result<crate::transfer::proto::TransferAck, DrainError> {
    use crate::transfer::proto::segment_transfer_service_client::SegmentTransferServiceClient;

    let blob = export_segment(engine, descriptor)?;
    let seg = coordinode_swarm::SegmentId(descriptor.id.0);
    let mut store = coordinode_swarm::LocalPieceStore::new();
    store
        .insert(seg, &blob, piece_size, encoding)
        .map_err(|e| DrainError::Pieces(e.to_string()))?;
    let frames =
        crate::transfer::frames_for(&store, seg).map_err(|e| DrainError::Pieces(e.to_string()))?;

    let ep = coordinode_wire::peer_endpoint(endpoint).map_err(|source| DrainError::Connect {
        endpoint: endpoint.to_string(),
        source,
    })?;
    let channel = ep.connect().await.map_err(|source| DrainError::Connect {
        endpoint: endpoint.to_string(),
        source,
    })?;
    let mut client = SegmentTransferServiceClient::new(channel);
    let ack = client
        .transfer_pieces(futures_util::stream::iter(frames))
        .await?
        .into_inner();
    if !ack.ok {
        return Err(DrainError::Rejected {
            segment: seg.0,
            error: ack.error,
        });
    }
    Ok(ack)
}

/// Installs a received, assembled segment into the engine, routing it to the
/// partition named by the blob's leading wire tag and writing each entry. The
/// target re-encodes locally per its own tier/codec policy (entries are plain
/// key-value bytes). Holds an `Arc<StorageEngine>` so it can be registered as a
/// long-lived transfer handler on the server.
///
/// Current install is upsert-per-entry (correct for repair fill and migration);
/// bulk ingestion and atomic replace-of-corrupt are deferred refinements.
pub struct SegmentInstaller {
    engine: Arc<StorageEngine>,
    // Build cache for the serve side: a recently exported+split segment keyed by
    // its build parameters, so the many GetPiece calls of one pull do not
    // re-export the partition per piece. Bounded crudely (cleared past a cap) —
    // repair/migration serving is a cold path, an LRU is a future refinement.
    // no-std: parking_lot::Mutex
    build_cache: Mutex<HashMap<BuildKey, Arc<BuiltSegment>>>,
}

/// Cache key for [`SegmentInstaller`]'s serve-side build cache: the subject,
/// piece size, and encoding discriminant, everything that makes the split
/// pieces byte-identical.
type BuildKey = (BuildSubject, usize, u32);

/// The hashable form of a [`SegmentSubject`].
#[derive(PartialEq, Eq, Hash)]
enum BuildSubject {
    Range(Partition, Vec<u8>, Vec<u8>),
    History(u64),
}

/// Max distinct segments held in the serve-side build cache before it is cleared.
const BUILD_CACHE_CAP: usize = 8;

impl SegmentInstaller {
    /// An installer over the shared engine.
    #[must_use]
    pub fn new(engine: Arc<StorageEngine>) -> Self {
        Self {
            engine,
            build_cache: Mutex::new(HashMap::new()),
        }
    }
}

impl SegmentSource for SegmentInstaller {
    fn build_segment(
        &self,
        subject: &SegmentSubject,
        piece_size: usize,
        encoding: PieceEncoding,
        fresh: bool,
    ) -> Result<Arc<BuiltSegment>, String> {
        let (enc_disc, _) = encoding.to_wire();
        let build = match subject {
            SegmentSubject::Range { partition, range } => {
                BuildSubject::Range(*partition, range.start.clone(), range.end.clone())
            }
            SegmentSubject::History { generation } => BuildSubject::History(*generation),
        };
        let key: BuildKey = (build, piece_size, enc_disc);

        // Tolerate a poisoned lock: a panic in a prior holder left the cache
        // readable; the data is a rebuildable cache, never corrupt-on-panic.
        if !fresh {
            let cache = self
                .build_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(hit) = cache.get(&key) {
                return Ok(Arc::clone(hit));
            }
        }

        let blob = match subject {
            SegmentSubject::Range { partition, range } => {
                export_range(&self.engine, *partition, range)
            }
            SegmentSubject::History { generation } => export_generation(&self.engine, *generation),
        }
        .map_err(|e| e.to_string())?;
        let (manifest, wire) =
            split_segment(&blob, piece_size, encoding).map_err(|e| e.to_string())?;
        let built = Arc::new(BuiltSegment { manifest, wire });

        let mut cache = self
            .build_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.len() >= BUILD_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, Arc::clone(&built));
        Ok(built)
    }
}

impl SegmentSink for SegmentInstaller {
    fn store_segment(
        &self,
        _segment: coordinode_swarm::SegmentId,
        data: &[u8],
    ) -> Result<(), String> {
        self.install(data).map_err(|e| e.to_string())
    }
}

impl SegmentInstaller {
    /// Install a received segment: rows through
    /// [`StorageEngine::install_partition`]; a generation's history into the
    /// replacement of it this node has staged, recorded complete when it
    /// reaches the replacement's registration.
    fn install(&self, data: &[u8]) -> Result<(), RepairError> {
        let behind_or_install = |e: StorageError| match e {
            StorageError::PositionBehind { .. } => RepairError::Behind(e.to_string()),
            e => RepairError::Install(e.to_string()),
        };
        match decode_segment(data).map_err(RepairError::Install)? {
            Segment::Rows {
                partition,
                whole,
                copy,
            } => self
                .engine
                .install_partition(partition, &copy, whole)
                .map_err(behind_or_install),
            Segment::History {
                generation,
                history,
            } => {
                let generation = GenerationId::from_raw(generation);
                self.engine
                    .import_generation_history(generation, &history.entries)
                    .map_err(behind_or_install)?;
                self.engine
                    .finish_generation_import(
                        generation,
                        history.covers_through,
                        history.history_from,
                    )
                    .map_err(behind_or_install)
            }
        }
    }
}

/// What `descriptor` asks for, for an error message.
fn describe(descriptor: &SegmentDescriptorRef) -> String {
    match descriptor.generation {
        Some(generation) => format!("index generation {generation}"),
        None => u8::try_from(descriptor.partition)
            .ok()
            .and_then(partition_from_wire_tag)
            .map_or_else(
                || format!("partition tag {}", descriptor.partition),
                |partition| format!("partition {partition:?}"),
            ),
    }
}

/// Failure modes of [`SegmentInstaller::repair_partition`].
#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    /// No reachable peer could serve the segment (all unreachable or none held
    /// it), so the local copy cannot be repaired from a replica.
    #[error("no healthy peer served {0}")]
    NoSource(String),
    /// The multi-source download failed (a piece checksum, assembly, or the
    /// whole-segment checksum did not verify).
    #[error("swarm download: {0}")]
    Download(String),
    /// Installing the reconstructed segment into local storage failed.
    #[error("install: {0}")]
    Install(String),
    /// Every copy served stood behind this node's Raft applies, so none of
    /// them holds the entries this node applied last.
    #[error("behind: {0}")]
    Behind(String),
}

/// Pulls of a partition copy tried before [`RepairError::Behind`] is given
/// up on. A healthy peer is at most a few entries behind the commit index,
/// so a pull a moment later normally stands past this node.
const REPAIR_ATTEMPTS: u32 = 5;

/// Wait before the second pull; doubled for each later one.
const REPAIR_BACKOFF: std::time::Duration = std::time::Duration::from_millis(100);

impl SegmentInstaller {
    /// Repair a (possibly corrupt) partition by pulling a fresh copy from healthy
    /// peers over the swarm transport and re-installing it locally.
    ///
    /// CE coarse repair: re-fetches the **whole** partition (Merkle page-level
    /// localization is EE). Connects a [`GrpcPieceSource`] to each reachable peer
    /// (TLS when inter-node TLS is configured), runs the rarest-first,
    /// multi-source [`swarm_download`], and installs the reconstructed segment,
    /// replacing the local partition. Peers that are unreachable or do not hold
    /// the segment are skipped; the pull proceeds from whoever answers. Returns
    /// the number of bytes installed.
    ///
    /// Each peer serves the partition at its own Raft position, so pieces are
    /// pulled only from peers whose copy is byte-identical to the first one's.
    /// A copy standing behind this node's applies is refused and the pull
    /// retried, up to `REPAIR_ATTEMPTS` (5) times.
    ///
    /// Must be called from within a tokio runtime; the synchronous download loop
    /// and the install run on blocking threads.
    ///
    /// # Errors
    /// [`RepairError`] if no peer serves the segment, the download fails its
    /// checksums, every copy stands behind this node, or the install fails.
    pub async fn repair_partition(
        self: &Arc<Self>,
        peers: &[String],
        partition: Partition,
        piece_size: usize,
        encoding: PieceEncoding,
    ) -> Result<usize, RepairError> {
        let (enc_disc, zstd_level) = encoding.to_wire();
        let tag = partition_wire_tag(partition);
        // Whole-partition descriptor: empty range is unbounded both ends, so the
        // serving peer exports the entire partition.
        let descriptor = SegmentDescriptorRef {
            segment_id: u64::from(tag),
            partition: u32::from(tag),
            range_start: Vec::new(),
            range_end: Vec::new(),
            piece_size: piece_size as u32,
            encoding: enc_disc,
            zstd_level,
            generation: None,
        };
        self.pull_until_current(peers, &descriptor).await
    }

    /// Fill a local replacement of index `generation` from `peers`: register
    /// it beside the published copy, pull a peer's history of the generation
    /// over the swarm transport and import it, recorded complete once it
    /// reaches the registration. The replacement is then ready for
    /// [`StorageEngine::publish_generation`]; on failure it is dropped and
    /// the published copy stays. Returns the number of bytes imported.
    ///
    /// A peer's history standing behind this node's applies at registration
    /// is refused and the pull retried, up to `REPAIR_ATTEMPTS` (5) times.
    ///
    /// Must be called from within a tokio runtime.
    ///
    /// # Errors
    /// [`RepairError`] if registering fails, no peer serves the history, the
    /// download fails its checksums, every history stands behind this node,
    /// or the import fails.
    pub async fn prepare_generation(
        self: &Arc<Self>,
        peers: &[String],
        generation: GenerationId,
        piece_size: usize,
        encoding: PieceEncoding,
    ) -> Result<usize, RepairError> {
        let engine = Arc::clone(&self.engine);
        tokio::task::spawn_blocking(move || engine.stage_generation(generation))
            .await
            .map_err(|e| RepairError::Install(e.to_string()))?
            .map_err(|e| RepairError::Install(e.to_string()))?;
        let (enc_disc, zstd_level) = encoding.to_wire();
        let tag = partition_wire_tag(Partition::Idx);
        let descriptor = SegmentDescriptorRef {
            segment_id: generation.as_raw(),
            partition: u32::from(tag),
            range_start: Vec::new(),
            range_end: Vec::new(),
            piece_size: piece_size as u32,
            encoding: enc_disc,
            zstd_level,
            generation: Some(generation.as_raw()),
        };
        let outcome = self.pull_until_current(peers, &descriptor).await;
        if outcome.is_err() {
            let engine = Arc::clone(&self.engine);
            // The replacement is unreferenced: dropping it is the cleanup,
            // and a failure here leaves it for the next open to drop.
            let dropped =
                tokio::task::spawn_blocking(move || engine.abandon_generation(generation)).await;
            if let Ok(Err(e)) | Err(e) = dropped.map_err(|e| StorageError::Io(e.to_string())) {
                tracing::warn!(generation = generation.as_raw(), error = %e, "replacement not dropped");
            }
        }
        outcome
    }

    /// Pull the segment `descriptor` names from `peers` and install it,
    /// pulling again while every copy stands behind this node.
    async fn pull_until_current(
        self: &Arc<Self>,
        peers: &[String],
        descriptor: &SegmentDescriptorRef,
    ) -> Result<usize, RepairError> {
        let mut backoff = REPAIR_BACKOFF;
        let mut attempt = 1;
        loop {
            match self.pull_and_install(peers, descriptor).await {
                Err(RepairError::Behind(reason)) if attempt < REPAIR_ATTEMPTS => {
                    tracing::debug!(segment = descriptor.segment_id, %reason, attempt, "copy behind; pulling again");
                    tokio::time::sleep(backoff).await;
                    backoff *= 2;
                    attempt += 1;
                }
                outcome => return outcome,
            }
        }
    }

    /// One pull of the segment `descriptor` names from `peers`, installed on
    /// success.
    async fn pull_and_install(
        self: &Arc<Self>,
        peers: &[String],
        descriptor: &SegmentDescriptorRef,
    ) -> Result<usize, RepairError> {
        let mut sources: Vec<GrpcPieceSource> = Vec::new();
        let mut manifest = None;
        for (i, endpoint) in peers.iter().enumerate() {
            let Ok(ep) = coordinode_wire::peer_endpoint(endpoint) else {
                continue;
            };
            let Ok(channel) = ep.connect().await else {
                continue;
            };
            // node id within the transfer mesh: peer index + 1 (0 is local).
            let node = NodeId(i as u64 + 1);
            let candidate = SourceCandidate {
                node,
                utilization: 0.0,
                bandwidth_to_target: 1.0,
                same_rack: false,
                tit_for_tat: 1.0,
                freshness: Freshness::Verified,
            };
            if let Ok((source, m)) =
                GrpcPieceSource::connect(node, channel, descriptor.clone(), candidate).await
            {
                // A peer at another position built other bytes; its pieces
                // would fail the chosen manifest's checksums.
                let chosen = manifest.get_or_insert_with(|| m.clone());
                if chosen.total_hash == m.total_hash && chosen.total_len == m.total_len {
                    sources.push(source);
                }
            }
        }

        let manifest = manifest.ok_or_else(|| RepairError::NoSource(describe(descriptor)))?;

        // The download loop is synchronous (it block_on's the gRPC client), so it
        // must not run on a runtime worker — hand it to a blocking thread.
        let assembled = tokio::task::spawn_blocking(move || {
            let refs: Vec<&dyn PieceSource> =
                sources.iter().map(|s| s as &dyn PieceSource).collect();
            swarm_download(NodeId(0), &manifest, &refs, Vec::new())
        })
        .await
        .map_err(|e| RepairError::Download(e.to_string()))?
        .map_err(|e| RepairError::Download(e.to_string()))?;

        let bytes = assembled.len();

        // The install replaces the partition only after the download
        // verified, so a failed fetch never leaves it empty; it clears the
        // tables without reading their blocks, so the corrupt SST goes
        // rather than being shadowed by newer versions.
        let installer = Arc::clone(self);
        tokio::task::spawn_blocking(move || installer.install(&assembled))
            .await
            .map_err(|e| RepairError::Install(e.to_string()))??;
        Ok(bytes)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
