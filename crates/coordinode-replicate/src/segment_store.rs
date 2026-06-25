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

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::{StorageError, StorageResult};
use coordinode_storage::oplog::OplogEntry;
use coordinode_storage::placement::{
    partition_from_wire_tag, partition_wire_tag, KeyRange, SegmentDescriptor,
};
use coordinode_swarm::{
    split_segment, swarm_download, Freshness, NodeId, PieceEncoding, PieceSource, SourceCandidate,
};
use tonic::transport::Channel;

use crate::transfer::proto::SegmentDescriptorRef;
use crate::transfer::{BuiltSegment, GrpcPieceSource, SegmentSink, SegmentSource};

/// One key-value entry of a segment's portable representation.
type KvEntry = (Vec<u8>, Vec<u8>);

/// Serialise key-value entries into the portable segment blob.
///
/// Layout per entry: `u32 LE key_len | key | u32 LE val_len | value`, in scan
/// (sorted) order. Deterministic and re-ingestible by [`decode_kv_blob`].
fn encode_kv_blob(entries: &[KvEntry]) -> StorageResult<Vec<u8>> {
    let mut out = Vec::new();
    for (key, value) in entries {
        let key_len = u32::try_from(key.len())
            .map_err(|_| StorageError::Serialization("segment key exceeds u32".into()))?;
        let val_len = u32::try_from(value.len())
            .map_err(|_| StorageError::Serialization("segment value exceeds u32".into()))?;
        out.extend_from_slice(&key_len.to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&val_len.to_le_bytes());
        out.extend_from_slice(value);
    }
    Ok(out)
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
    // Scan the partition at a stable snapshot and keep only entries the half-open
    // `[start, end)` range actually contains (an unbounded-above range is the
    // whole partition).
    let snapshot = engine.snapshot();
    let prefix = format!("{}:", part.name());
    let scanned = engine.snapshot_prefix_scan(&snapshot, part, prefix.as_bytes())?;

    let mut entries = Vec::with_capacity(scanned.len());
    for (key, value) in scanned {
        if range.contains(&key) {
            entries.push((key, value.to_vec()));
        }
    }
    let mut blob = Vec::new();
    blob.push(partition_wire_tag(part));
    blob.extend_from_slice(&encode_kv_blob(&entries)?);
    Ok(blob)
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

    let mut ep =
        tonic::transport::Endpoint::from_shared(endpoint.to_string()).map_err(|source| {
            DrainError::Connect {
                endpoint: endpoint.to_string(),
                source,
            }
        })?;
    // Encrypt the drain connection when inter-node TLS is configured
    // (process-global, set once at startup). Off = plaintext.
    if let Some(tls) = coordinode_wire::wire_client_tls() {
        ep = ep.tls_config(tls).map_err(|source| DrainError::Connect {
            endpoint: endpoint.to_string(),
            source,
        })?;
    }
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

/// Cache key for [`SegmentInstaller`]'s serve-side build cache: the partition,
/// half-open key range, piece size, and encoding discriminant — everything that
/// makes the split pieces byte-identical.
type BuildKey = (Partition, Vec<u8>, Vec<u8>, usize, u32);

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
        partition: Partition,
        range: &KeyRange,
        piece_size: usize,
        encoding: PieceEncoding,
    ) -> Result<Arc<BuiltSegment>, String> {
        let (enc_disc, _) = encoding.to_wire();
        let key: BuildKey = (
            partition,
            range.start.clone(),
            range.end.clone(),
            piece_size,
            enc_disc,
        );

        // Tolerate a poisoned lock: a panic in a prior holder left the cache
        // readable; the data is a rebuildable cache, never corrupt-on-panic.
        {
            let cache = self
                .build_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(hit) = cache.get(&key) {
                return Ok(Arc::clone(hit));
            }
        }

        let blob = export_range(&self.engine, partition, range).map_err(|e| e.to_string())?;
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
        let (&tag, rest) = data
            .split_first()
            .ok_or_else(|| "empty segment blob (missing partition tag)".to_string())?;
        let partition = partition_from_wire_tag(tag)
            .ok_or_else(|| format!("unknown partition wire tag {tag}"))?;
        let entries = decode_kv_blob(rest)?;
        for (key, value) in entries {
            self.engine
                .put(partition, &key, &value)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Failure modes of [`repair_partition`].
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
    /// Opening or reading the checkpoint base for WAL-replay repair failed.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
}

impl SegmentInstaller {
    /// Repair a (possibly corrupt) partition by pulling a fresh copy from healthy
    /// peers over the swarm transport and re-installing it locally.
    ///
    /// CE coarse repair: re-fetches the **whole** partition (Merkle page-level
    /// localization is EE). Connects a [`GrpcPieceSource`] to each reachable peer
    /// (TLS when inter-node TLS is configured), runs the rarest-first,
    /// multi-source [`swarm_download`], and installs the reconstructed segment,
    /// overwriting local data. Peers that are unreachable or do not hold the
    /// segment are skipped; the pull proceeds from whoever answers. Returns the
    /// number of bytes installed.
    ///
    /// Must be called from within a tokio runtime; the synchronous download loop
    /// runs on a blocking thread.
    ///
    /// # Errors
    /// [`RepairError`] if no peer serves the segment, the download fails its
    /// checksums, or the install fails.
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
        };

        let mut sources: Vec<GrpcPieceSource> = Vec::new();
        let mut manifest = None;
        for (i, endpoint) in peers.iter().enumerate() {
            let Ok(mut ep) = Channel::from_shared(endpoint.clone()) else {
                continue;
            };
            if let Some(tls) = coordinode_wire::wire_client_tls() {
                let Ok(with_tls) = ep.tls_config(tls) else {
                    continue;
                };
                ep = with_tls;
            }
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
                manifest.get_or_insert(m);
                sources.push(source);
            }
        }

        let manifest =
            manifest.ok_or_else(|| RepairError::NoSource(format!("partition {partition:?}")))?;

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

        // Physically drop the partition's existing tables before reinstalling.
        // Done only AFTER the download succeeded, so a fetch failure never leaves
        // the partition empty. This is a true replace, not an upsert over
        // corruption: the corrupt SST block is removed (table-level drop, which
        // does not read the block) rather than shadowed by newer versions — a
        // shadowed corrupt block would still break compaction and re-trip the
        // scrub. The full range `b""..` contains every table, so all are dropped.
        self.engine
            .drop_range(partition, b"".as_slice()..)
            .map_err(|e| RepairError::Install(format!("drop partition before reinstall: {e}")))?;

        self.store_segment(
            coordinode_swarm::SegmentId(descriptor.segment_id),
            &assembled,
        )
        .map_err(RepairError::Install)?;
        Ok(bytes)
    }

    /// Repair a corrupt partition with no healthy replica by rebuilding it from a
    /// local checkpoint plus oplog replay (the "WAL replay repair" fallback after
    /// [`repair_partition`] returns [`RepairError::NoSource`]).
    ///
    /// Opens `checkpoint_dir` read-only, exports the partition as of the
    /// checkpoint, drops the live (corrupt) tables, installs the checkpoint base,
    /// then replays the granular ops of `oplog_since` (the oplog entries with
    /// index after the checkpoint's cursor) for this partition to roll the data
    /// forward to current. `RaftEntry` / `Noop` / membership ops are skipped — the
    /// granular `Insert` / `Delete` / `Merge` / `RemoveRange` carry the mutations.
    /// The corrupt block is physically removed (table-level drop, not shadowed).
    ///
    /// # Errors
    /// [`RepairError::Checkpoint`] if the checkpoint cannot be opened/read,
    /// [`RepairError::Install`] if dropping, installing, or replaying fails.
    pub fn wal_replay_repair(
        &self,
        checkpoint_dir: &std::path::Path,
        oplog_since: &[OplogEntry],
        partition: Partition,
    ) -> Result<usize, RepairError> {
        // The rebuild-from-local-checkpoint-plus-oplog-replay logic is a
        // storage-level capability shared with the embedded (no-Raft) repair
        // path; the canonical implementation lives on the engine. The cluster
        // path reaches it here after a swarm-fetch source is unavailable
        // (`repair_partition` → `NoSource`).
        self.engine
            .repair_partition_from_checkpoint(checkpoint_dir, oplog_since, partition)
            .map_err(|e| RepairError::Checkpoint(e.to_string()))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
