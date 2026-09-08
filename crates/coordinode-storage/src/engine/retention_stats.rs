//! What the MVCC retention window holds on disk beyond the live version.
//!
//! A snapshot read inside the window is served by the tree version that was
//! current at that seqno, and every retained version keeps every table it
//! references alive. The bytes those tables occupy over and above the live
//! version's own files are the price of the window; this module measures it
//! per partition so the price can be sized and watched rather than assumed.

use lsm_tree::AbstractTree;
use lsm_tree::file::BLOBS_FOLDER;
use lsm_tree::fs::Fs;
use std::path::Path;

use crate::error::StorageResult;

/// On-disk footprint of one partition split into what the live version
/// references and what only retained history still holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetainedHistory {
    /// Physical bytes of every table and blob file the current version
    /// references: what the partition would occupy with no history at all.
    pub live_bytes: u64,
    /// Physical bytes of table and blob files present in the partition's
    /// folders but not referenced by the current version: the tables every
    /// compaction inside the window consumed, kept for the snapshots that
    /// still see them. Never reaches zero on a partition that has been
    /// compacted: the history also keeps the newest version below the
    /// watermark, so a read at exactly the watermark has a version to be
    /// served from, and that version's tables stay until the next install
    /// below the watermark replaces it. Transiently also counts an
    /// in-flight compaction's output before it is installed.
    pub retained_bytes: u64,
}

impl RetainedHistory {
    /// Retained bytes as a fraction of the live version, or `0.0` for an
    /// empty partition.
    #[must_use]
    pub fn retained_ratio(&self) -> f64 {
        if self.live_bytes == 0 {
            0.0
        } else {
            self.retained_bytes as f64 / self.live_bytes as f64
        }
    }
}

/// Measures `tree`'s live footprint against everything on disk in its
/// folders (primary and level-routed tables folders plus the blob folder).
///
/// The folders are scanned before the live version is read: a compaction
/// installing between the two reads then shows up as history that is a
/// little too small, never as a live figure larger than the disk holds.
pub fn retained_history(tree: &lsm_tree::AnyTree) -> StorageResult<RetainedHistory> {
    let config = match tree {
        lsm_tree::AnyTree::Standard(t) => &t.0.config,
        lsm_tree::AnyTree::Blob(b) => &b.index.0.config,
    };

    let mut physical = 0u64;
    for (folder, fs) in config.all_tables_folders() {
        physical += folder_file_bytes(&*fs, &folder)?;
    }
    physical += folder_file_bytes(&*config.fs, &config.path.join(BLOBS_FOLDER))?;

    let live_bytes = tree.storage_stats()?.used_bytes;
    // The scan precedes the live read, so `physical >= live` except when a
    // flush landed a new table between them; that race is the only way the
    // difference goes negative, and it means "no history observed", which
    // is what the clamp to zero reports.
    let retained_bytes = physical.saturating_sub(live_bytes);
    Ok(RetainedHistory {
        live_bytes,
        retained_bytes,
    })
}

/// Sum of the sizes of the regular files directly under `folder`. A folder
/// that does not exist contributes nothing: a tree without blobs never
/// creates its blob folder, and a routed level folder appears on first use.
fn folder_file_bytes(fs: &dyn Fs, folder: &Path) -> StorageResult<u64> {
    let entries = match fs.read_dir(folder) {
        Ok(entries) => entries,
        Err(e) if e.kind() == lsm_tree::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(lsm_tree::Error::from(e).into()),
    };
    let mut bytes = 0u64;
    for entry in entries {
        if entry.is_dir {
            continue;
        }
        let meta = match fs.metadata(&entry.path) {
            Ok(meta) => meta,
            // Deleted between the listing and the stat: a table the version
            // history just released, which is exactly not retained.
            Err(e) if e.kind() == lsm_tree::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(lsm_tree::Error::from(e).into()),
        };
        if meta.is_file {
            bytes += meta.len;
        }
    }
    Ok(bytes)
}
