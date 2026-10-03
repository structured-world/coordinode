//! The engine format of a data directory, and its migration on open.
//!
//! A directory records the engine format that wrote it in a marker file on
//! its primary endpoint. A release opens a directory of its own format, and
//! migrates one written by the format just before it, once and in one
//! direction: every step of [`MIGRATIONS`] that starts at the directory's
//! format runs, and the marker moves only once the whole open has succeeded
//! ([`settle`]), so a crash part way, or an open that refuses the directory
//! for another reason, leaves the old marker and repeats the step next time.
//! A directory of any other format is refused by name and left untouched.
//! A directory with data and no marker was written before the marker
//! existed, which is format 0.

use std::path::{Path, PathBuf};

use crate::engine::config::{Durability, StorageConfig};
use crate::error::{StorageError, StorageResult};

/// Name of the marker file in the primary endpoint's directory.
pub const MARKER_FILE: &str = "ENGINE_FORMAT";

/// One step of directory migration: turns a directory written by format
/// `from` into one of format `from + 1`. A step is idempotent, since an open
/// that ends before the marker moves runs it again.
pub struct MigrationStep {
    /// The format the step reads.
    pub from: u32,
    /// Rewrite the directory at the path in place.
    pub migrate: fn(&Path) -> StorageResult<()>,
}

/// Every migration this release ships, at most one per source format.
/// Format 0 (a directory from before the marker) holds nothing that format
/// 1 reads differently, so its step only records the format.
pub static MIGRATIONS: &[MigrationStep] = &[
    MigrationStep {
        from: 0,
        migrate: no_change,
    },
    // A test build runs up to two formats past this release's, so a suite
    // can take a directory through an intermediate version.
    #[cfg(feature = "test-format-bump")]
    MigrationStep {
        from: coordinode_core::version::ENGINE_FORMAT_VERSION,
        migrate: no_change,
    },
    #[cfg(feature = "test-format-bump")]
    MigrationStep {
        from: coordinode_core::version::ENGINE_FORMAT_VERSION + 1,
        migrate: no_change,
    },
];

fn no_change(_: &Path) -> StorageResult<()> {
    Ok(())
}

/// What opening a directory did to its format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatOpen {
    /// No durable directory to mark: every endpoint is volatile.
    Volatile,
    /// A new directory, now marked with the current format.
    Created,
    /// Already in the current format.
    Current,
    /// Migrated from the named format to the current one; the marker moves
    /// when [`settle`] is called after the open succeeds.
    Migrated {
        /// The directory migrated.
        dir: PathBuf,
        /// The format the directory was in.
        from: u32,
        /// The format it is in now.
        to: u32,
    },
}

/// Record the format a migration reached, once the open that ran it has
/// succeeded. Nothing to do for any other outcome.
pub fn settle(opened: &FormatOpen) -> StorageResult<()> {
    match opened {
        FormatOpen::Migrated { dir, to, .. } => write_marker(dir, *to),
        _ => Ok(()),
    }
}

/// Bring the primary endpoint's directory to the engine format this
/// process runs before anything opens it: mark a new one, migrate one of the
/// previous format, refuse any other. Called first thing by every engine
/// open.
pub fn prepare(config: &StorageConfig) -> StorageResult<FormatOpen> {
    let Some(primary) = config.endpoints.first() else {
        return Ok(FormatOpen::Volatile);
    };
    if primary.durability == Durability::Volatile {
        return Ok(FormatOpen::Volatile);
    }
    prepare_dir(
        &primary.path,
        coordinode_core::version::engine_format_version(),
        MIGRATIONS,
    )
}

/// [`prepare`] for one directory, a target format and a set of steps.
pub fn prepare_dir(dir: &Path, runs: u32, steps: &[MigrationStep]) -> StorageResult<FormatOpen> {
    let found = match read_marker(dir)? {
        Some(found) => found,
        None if is_empty(dir)? => {
            std::fs::create_dir_all(dir).map_err(|e| io_at("create", dir, e))?;
            write_marker(dir, runs)?;
            return Ok(FormatOpen::Created);
        }
        None => 0,
    };
    if found == runs {
        return Ok(FormatOpen::Current);
    }
    let refuse = || StorageError::UnsupportedFormat {
        path: dir.display().to_string(),
        found,
        runs,
    };
    // Format numbering starts at 1, so `runs - 1` exists.
    if found != runs - 1 {
        return Err(refuse());
    }
    let step = steps.iter().find(|s| s.from == found).ok_or_else(refuse)?;
    tracing::info!(dir = %dir.display(), from = found, to = runs, "migrating the data directory");
    (step.migrate)(dir)?;
    Ok(FormatOpen::Migrated {
        dir: dir.to_path_buf(),
        from: found,
        to: runs,
    })
}

/// The format a directory's marker names, `None` without a marker.
pub fn read_marker(dir: &Path) -> StorageResult<Option<u32>> {
    let path = dir.join(MARKER_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_at("read", &path, e)),
    };
    text.trim().parse().map(Some).map_err(|_| {
        StorageError::Io(format!(
            "{} does not name an engine format: {text:?}",
            path.display()
        ))
    })
}

/// Record `format` as the directory's format: written aside, made durable,
/// then renamed over the marker, so a crash leaves the old marker or the new
/// one and never a torn one.
pub fn write_marker(dir: &Path, format: u32) -> StorageResult<()> {
    use std::io::Write;
    let path = dir.join(MARKER_FILE);
    let staged: PathBuf = dir.join(format!("{MARKER_FILE}.tmp"));
    let mut file = std::fs::File::create(&staged).map_err(|e| io_at("create", &staged, e))?;
    file.write_all(format!("{format}\n").as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(|e| io_at("write", &staged, e))?;
    drop(file);
    std::fs::rename(&staged, &path).map_err(|e| io_at("rename", &path, e))?;
    sync_dir(dir)
}

/// Make a rename in `dir` durable. Windows has no directory handle to sync;
/// NTFS journals the rename itself.
fn sync_dir(dir: &Path) -> StorageResult<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| io_at("sync", dir, e))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

fn is_empty(dir: &Path) -> StorageResult<bool> {
    match std::fs::read_dir(dir) {
        Ok(mut entries) => Ok(entries.next().is_none()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(e) => Err(io_at("list", dir, e)),
    }
}

fn io_at(what: &str, path: &Path, e: std::io::Error) -> StorageError {
    StorageError::Io(format!("{what} {}: {e}", path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
