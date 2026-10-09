//! Oplog segments of a data directory: finding, reading and replacing them.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use coordinode_storage::engine::config::SyncMethod;
use coordinode_storage::oplog::{OplogEntry, SegmentReader, SegmentWriter};

use crate::{BACKUP_DIR, Backup};

/// A segment as read from disk.
pub struct Segment {
    /// The shard its header names.
    pub shard_id: u32,
    /// The first log index its header names.
    pub first_index: u64,
    /// Its entries, in order.
    pub entries: Vec<OplogEntry>,
    /// Whether it carries a footer; a crash leaves the last one without.
    pub sealed: bool,
}

/// Every oplog segment under `data`: the store's own journal on every
/// endpoint below it, and the copies inside its checkpoints, which a repair
/// replays too. The backup directory of earlier runs is left out.
///
/// # Errors
///
/// A directory cannot be listed.
pub fn segment_files(data: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut dirs = vec![data.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let entries = std::fs::read_dir(&dir).with_context(|| format!("list {}", dir.display()))?;
        for entry in entries {
            let entry = entry.with_context(|| format!("list {}", dir.display()))?;
            let path = entry.path();
            let kind = entry
                .file_type()
                .with_context(|| format!("stat {}", path.display()))?;
            if kind.is_dir() {
                if dir == data && entry.file_name() == BACKUP_DIR {
                    continue;
                }
                dirs.push(path);
            } else if is_segment_name(&entry.file_name().to_string_lossy()) {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

/// `oplog-<index>.bin`, the name the journal gives a segment.
fn is_segment_name(name: &str) -> bool {
    name.strip_prefix("oplog-")
        .and_then(|rest| rest.strip_suffix(".bin"))
        .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))
}

/// Read the segment at `path`: whole and verified when sealed, or the
/// complete entries a crash left in one that was never sealed.
///
/// # Errors
///
/// The file is not a segment, or a sealed one fails its checks.
pub fn read(path: &Path) -> anyhow::Result<Segment> {
    let (reader, sealed) = match SegmentReader::open(path) {
        Ok(reader) => (reader, true),
        Err(sealed_error) => match SegmentReader::open_active(path) {
            Ok(reader) => (reader, false),
            Err(_) => {
                return Err(anyhow::Error::new(sealed_error)
                    .context(format!("read segment {}", path.display())));
            }
        },
    };
    Ok(Segment {
        shard_id: reader.header.shard_id,
        first_index: reader.header.first_index,
        entries: reader.into_entries(),
        sealed,
    })
}

/// Replace the segment at `path` with `segment`, sealed: written beside it,
/// made durable, then renamed over it, so a crash leaves either the old file
/// or the new one. The old file is kept under `backup` first.
///
/// A segment a crash left unsealed is written sealed, as the journal's own
/// recovery would leave it.
///
/// # Errors
///
/// The new file cannot be written, the old one kept, or the rename fails.
pub fn replace(path: &Path, segment: &Segment, backup: &Backup) -> anyhow::Result<()> {
    let staged = path.with_extension("bin.migrating");
    if staged.exists() {
        // Left by a run that stopped before its rename: the original is still
        // in place, so this one is redone.
        std::fs::remove_file(&staged)
            .with_context(|| format!("remove leftover {}", staged.display()))?;
    }
    let mut writer = SegmentWriter::create(
        &staged,
        segment.shard_id,
        segment.first_index,
        SyncMethod::Full,
    )
    .with_context(|| format!("create {}", staged.display()))?;
    for entry in &segment.entries {
        writer
            .append(entry)
            .with_context(|| format!("write entry {} to {}", entry.index, staged.display()))?;
    }
    writer
        .seal()
        .with_context(|| format!("seal {}", staged.display()))?;
    backup.keep(path)?;
    std::fs::rename(&staged, path)
        .with_context(|| format!("rename {} over {}", staged.display(), path.display()))?;
    sync_parent(path)
}

/// Make a rename in `path`'s directory durable. Windows has no directory
/// handle to sync; NTFS journals the rename itself.
fn sync_parent(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        std::fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .with_context(|| format!("sync {}", dir.display()))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
