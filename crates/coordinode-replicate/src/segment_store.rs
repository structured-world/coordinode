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

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::error::{StorageError, StorageResult};
use coordinode_storage::placement::{
    partition_from_wire_tag, partition_wire_tag, SegmentDescriptor,
};

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
    let mut blob = Vec::new();
    blob.push(partition_wire_tag(part));
    blob.extend_from_slice(&encode_kv_blob(&entries)?);
    Ok(blob)
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
}

impl SegmentInstaller {
    /// An installer over the shared engine.
    #[must_use]
    pub fn new(engine: Arc<StorageEngine>) -> Self {
        Self { engine }
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

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
