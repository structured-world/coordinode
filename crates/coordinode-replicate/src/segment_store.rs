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
//! - [`StorageSegmentSink`] implements [`SegmentSink`]: it decodes a received
//!   blob and installs the entries into a partition.

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::{StorageError, StorageResult};
use coordinode_storage::placement::SegmentDescriptor;

use crate::transfer::SegmentSink;

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

/// Export the entries covered by `descriptor` from the engine into a portable
/// blob, ready to be split into swarm pieces.
///
/// Reads the descriptor's key range (the whole partition when the range is
/// unbounded above) and keeps only entries the range actually contains
/// (half-open `[start, end)`), so a sub-range segment exports exactly its keys.
///
/// # Errors
///
/// Returns an error if the partition is unavailable, a scanned entry cannot be
/// read, or a key/value exceeds the `u32` length bound.
pub fn export_segment(
    engine: &StorageEngine,
    descriptor: &SegmentDescriptor,
) -> StorageResult<Vec<u8>> {
    let part = descriptor.partition;
    let range = &descriptor.key_range;

    // Scan the partition at a stable snapshot and keep only entries the
    // descriptor's half-open `[start, end)` range actually contains (an
    // unbounded-above range is the whole partition).
    let snapshot = engine.snapshot();
    let prefix = format!("{}:", part.name());
    let scanned = engine.snapshot_prefix_scan(&snapshot, part, prefix.as_bytes())?;

    let mut entries = Vec::with_capacity(scanned.len());
    for (key, value) in scanned {
        if range.contains(&key) {
            entries.push((key, value.to_vec()));
        }
    }
    encode_kv_blob(&entries)
}

/// Installs a received, assembled segment into a single partition by decoding
/// the portable blob and writing each entry. The target re-encodes locally per
/// its own tier/codec policy (the entries are plain key-value bytes).
///
/// Bound to one partition: the gRPC handler that dispatches by segment maps a
/// segment to its partition above this sink.
pub struct StorageSegmentSink<'a> {
    engine: &'a StorageEngine,
    partition: Partition,
}

impl<'a> StorageSegmentSink<'a> {
    /// A sink that installs received segments into `partition`.
    #[must_use]
    pub fn new(engine: &'a StorageEngine, partition: Partition) -> Self {
        Self { engine, partition }
    }
}

impl SegmentSink for StorageSegmentSink<'_> {
    fn store_segment(
        &self,
        _segment: coordinode_swarm::SegmentId,
        data: &[u8],
    ) -> Result<(), String> {
        let entries = decode_kv_blob(data)?;
        for (key, value) in entries {
            self.engine
                .put(self.partition, &key, &value)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
