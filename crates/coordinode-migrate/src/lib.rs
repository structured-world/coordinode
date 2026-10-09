//! Offline migration of a data directory to the format this build reads.
//!
//! The engine opens only its own format and refuses anything else without
//! touching it. A directory written by an earlier build is brought forward
//! here instead, with the server stopped: every [`Migration`] recognises the
//! old shape it handles from the bytes themselves, leaves current data alone,
//! and rewrites what it recognises. Code that reads a retired encoding lives
//! in this crate, never in the engine.
//!
//! A run only reports by default; [`apply`] changes files. Every file it
//! replaces is kept first as a hard link under `migrate-backup/<run>/` in the
//! data directory, so a run costs no space for the copy and can be undone by
//! moving the files back.

pub mod check;
pub mod journal_stats;
pub mod migrations;
pub mod oplog;

use std::path::{Path, PathBuf};

pub use migrations::MIGRATIONS;

/// Directory, inside the data directory, that keeps the replaced files.
pub const BACKUP_DIR: &str = "migrate-backup";

/// One migration: what it recognises and how it rewrites it.
pub struct Migration {
    /// Stable name, shown in reports.
    pub name: &'static str,
    /// What the migration changes, one line.
    pub summary: &'static str,
    /// Find what in the data directory needs this migration.
    pub survey: fn(&Path) -> anyhow::Result<Vec<Finding>>,
    /// Rewrite one finding, keeping what it replaces under `backup`.
    pub apply: fn(&Finding, &Backup) -> anyhow::Result<()>,
}

/// One file a migration would rewrite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// The file.
    pub path: PathBuf,
    /// What was found in it.
    pub detail: String,
}

/// Where a run keeps the files it replaces.
#[derive(Debug, Clone)]
pub struct Backup {
    data: PathBuf,
    root: PathBuf,
}

impl Backup {
    /// The backup of a run over `data` named `run`.
    pub fn new(data: &Path, run: &str) -> Self {
        Self {
            data: data.to_path_buf(),
            root: data.join(BACKUP_DIR).join(run),
        }
    }

    /// Keep `file` (inside the data directory) under the backup, at its path
    /// relative to the data directory: a hard link, or a copy where the
    /// filesystem has none.
    ///
    /// # Errors
    ///
    /// `file` is outside the data directory, or the link and the copy fail.
    pub fn keep(&self, file: &Path) -> anyhow::Result<PathBuf> {
        use anyhow::Context as _;
        let relative = file
            .strip_prefix(&self.data)
            .with_context(|| format!("{} is outside {}", file.display(), self.data.display()))?;
        let kept = self.root.join(relative);
        if let Some(parent) = kept.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        if std::fs::hard_link(file, &kept).is_err() {
            std::fs::copy(file, &kept)
                .with_context(|| format!("keep {} as {}", file.display(), kept.display()))?;
        }
        Ok(kept)
    }
}

/// The store at `data` and each checkpoint inside it: the directories that
/// carry an engine format marker. A directory without one is never opened,
/// since opening it would create a store there.
///
/// # Errors
///
/// The checkpoint directory cannot be listed.
pub fn stores(data: &Path) -> anyhow::Result<Vec<PathBuf>> {
    use anyhow::Context as _;
    let marked = |dir: &Path| dir.join(coordinode_storage::format::MARKER_FILE).is_file();
    let mut stores = Vec::new();
    if marked(data) {
        stores.push(data.to_path_buf());
    }
    let checkpoints = data.join("checkpoints");
    if checkpoints.is_dir() {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&checkpoints)
            .with_context(|| format!("list {}", checkpoints.display()))?
        {
            let path = entry
                .with_context(|| format!("list {}", checkpoints.display()))?
                .path();
            if marked(&path) {
                found.push(path);
            }
        }
        found.sort();
        stores.extend(found);
    }
    Ok(stores)
}

/// Open the store at `dir` as a plain engine: no journal is replayed and
/// nothing of the server runs.
///
/// # Errors
///
/// The engine refuses the directory.
pub fn open_store(dir: &Path) -> anyhow::Result<coordinode_storage::engine::core::StorageEngine> {
    use anyhow::Context as _;
    coordinode_storage::engine::core::StorageEngine::open_checkpoint(dir)
        .with_context(|| format!("open the store {}", dir.display()))
}

/// Everything every migration finds in `data`, in migration order.
///
/// # Errors
///
/// A migration cannot read what it surveys; nothing is changed.
pub fn survey(data: &Path) -> anyhow::Result<Vec<(&'static Migration, Vec<Finding>)>> {
    MIGRATIONS
        .iter()
        .map(|m| Ok((m, (m.survey)(data)?)))
        .collect()
}

/// Survey `data` and rewrite every finding, keeping what is replaced under
/// the backup named `run`. Returns what was rewritten.
///
/// # Errors
///
/// The first finding that cannot be rewritten; the ones before it are done
/// and kept in the backup, the ones after it are untouched.
pub fn apply(data: &Path, run: &str) -> anyhow::Result<Vec<(&'static Migration, Vec<Finding>)>> {
    let backup = Backup::new(data, run);
    let found = survey(data)?;
    for (migration, findings) in &found {
        for finding in findings {
            (migration.apply)(finding, &backup)?;
        }
    }
    Ok(found)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod test_support;
#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
