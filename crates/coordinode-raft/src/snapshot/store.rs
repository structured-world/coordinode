//! Where a node keeps its current Raft snapshot: one file under the engine's
//! data directory, named by a Schema record written together with the
//! snapshot's metadata. The bytes never enter the store itself, so a snapshot
//! does not carry the one before it, and nothing holds it in memory.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::storage::SnapshotMeta;

/// The current snapshot's metadata.
pub(crate) const KEY_SNAPSHOT_META: &[u8] = b"raft:snapshot:meta";
/// The current snapshot's file name inside the snapshot directory.
const KEY_SNAPSHOT_FILE: &[u8] = b"raft:snapshot:file";
/// Where earlier builds kept the snapshot bytes themselves.
const KEY_LEGACY_DATA: &[u8] = b"raft:snapshot:data";

/// A published snapshot.
const FINISHED: &str = "cnsn";
/// A snapshot still being written or received.
const STAGED: &str = "part";

/// The directory holding `engine`'s snapshot files.
pub fn snapshot_dir(engine: &StorageEngine) -> PathBuf {
    engine.data_dir().join("raft-snapshot")
}

/// One snapshot's bytes in a file. Reads and writes continue where the last
/// one ended; [`Seek`] moves the position. A staged file, one not yet
/// published as the current snapshot, is removed when dropped.
#[derive(Debug)]
pub struct SnapshotFile {
    file: File,
    path: PathBuf,
    staged: bool,
}

impl SnapshotFile {
    /// A new, empty staged file in `dir`, for bytes still to be written.
    ///
    /// # Errors
    ///
    /// The directory or the file cannot be created.
    pub fn stage(dir: &Path) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}.{STAGED}", unique_stem()));
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            file,
            path,
            staged: true,
        })
    }

    fn open(path: PathBuf) -> io::Result<Self> {
        let file = File::open(&path)?;
        Ok(Self {
            file,
            path,
            staged: false,
        })
    }

    /// The number of bytes in the file.
    ///
    /// # Errors
    ///
    /// The file's metadata cannot be read.
    pub fn size(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// A second handle on the same file, sharing this one's position, for
    /// reading it from another task.
    ///
    /// # Errors
    ///
    /// The handle cannot be duplicated.
    pub fn try_clone_file(&self) -> io::Result<File> {
        self.file.try_clone()
    }

    fn name(&self) -> io::Result<&str> {
        self.path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| io::Error::other(format!("snapshot file name {:?}", self.path)))
    }

    /// Make a staged file durable under its published name.
    fn finish(&mut self) -> io::Result<()> {
        self.file.sync_all()?;
        let published = self.path.with_extension(FINISHED);
        fs::rename(&self.path, &published)?;
        self.path = published;
        self.staged = false;
        sync_dir(self.path.parent())
    }
}

impl Read for SnapshotFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.file.read(buf)
    }
}

impl Write for SnapshotFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl Seek for SnapshotFile {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

impl Drop for SnapshotFile {
    fn drop(&mut self) {
        if self.staged {
            remove_quietly(&self.path);
        }
    }
}

/// The current snapshot of one engine.
pub(crate) struct SnapshotStore {
    engine: Arc<StorageEngine>,
    dir: PathBuf,
    /// One publication at a time, so the record and the files agree.
    publish: Mutex<()>,
}

impl SnapshotStore {
    /// Open the store: a snapshot an earlier build kept inside the Schema
    /// partition moves to a file, and every file the record does not name
    /// (a crash's leftovers) is removed.
    pub(crate) fn open(engine: Arc<StorageEngine>) -> io::Result<Self> {
        let dir = snapshot_dir(&engine);
        fs::create_dir_all(&dir)?;
        let store = Self {
            engine,
            dir,
            publish: Mutex::new(()),
        };
        store.adopt_legacy()?;
        let current = store.record()?.map(|(_, name)| name);
        for entry in fs::read_dir(&store.dir)? {
            let entry = entry?;
            if current.as_deref() != entry.file_name().to_str() {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(store)
    }

    /// A new staged file for a snapshot about to be written.
    pub(crate) fn stage(&self) -> io::Result<SnapshotFile> {
        SnapshotFile::stage(&self.dir)
    }

    /// The current snapshot, opened for reading from its start.
    pub(crate) fn current(&self) -> io::Result<Option<(SnapshotMeta, SnapshotFile)>> {
        // Held so a publication cannot remove the file between the record
        // read and the open.
        let _publishing = self.lock()?;
        match self.record()? {
            Some((meta, name)) => Ok(Some((meta, SnapshotFile::open(self.dir.join(name))?))),
            None => Ok(None),
        }
    }

    /// Make `file` the current snapshot, described by `meta`, and remove the
    /// one it replaces. A snapshot older than the current one is not
    /// published: `false` is returned and a staged `file` is removed when its
    /// owner drops it.
    pub(crate) fn publish(&self, meta: &SnapshotMeta, file: &mut SnapshotFile) -> io::Result<bool> {
        let _publishing = self.lock()?;
        let previous = self.record()?;
        if let Some((current, name)) = &previous {
            if current.last_log_id > meta.last_log_id {
                return Ok(false);
            }
            if !file.staged && file.path == self.dir.join(name) {
                return Ok(true);
            }
        }
        if file.path.parent() != Some(self.dir.as_path()) {
            // The current snapshot lives in this store's directory; a file
            // from anywhere else is copied in, and a staged one goes.
            let mut local = self.stage()?;
            file.seek(SeekFrom::Start(0))?;
            io::copy(file, &mut local)?;
            *file = local;
        }
        if file.staged {
            file.finish()?;
        }
        let meta_bytes = rmp_serde::to_vec(meta).map_err(io::Error::other)?;
        let mut batch = WriteBatch::new(&self.engine);
        batch.put(Partition::Schema, KEY_SNAPSHOT_META, meta_bytes);
        batch.put(
            Partition::Schema,
            KEY_SNAPSHOT_FILE,
            file.name()?.as_bytes(),
        );
        batch.commit().map_err(io::Error::other)?;
        // The record must be durable before the file it named goes.
        self.engine.persist().map_err(io::Error::other)?;
        if let Some((_, name)) = previous {
            remove_quietly(&self.dir.join(name));
        }
        Ok(true)
    }

    fn lock(&self) -> io::Result<std::sync::MutexGuard<'_, ()>> {
        self.publish
            .lock()
            .map_err(|e| io::Error::other(format!("snapshot store mutex poisoned: {e}")))
    }

    /// The recorded metadata and file name, when both are there.
    fn record(&self) -> io::Result<Option<(SnapshotMeta, String)>> {
        let Some(meta) = self.get(KEY_SNAPSHOT_META)? else {
            return Ok(None);
        };
        let Some(name) = self.get(KEY_SNAPSHOT_FILE)? else {
            return Ok(None);
        };
        let meta = rmp_serde::from_slice(&meta).map_err(io::Error::other)?;
        let name = String::from_utf8(name)
            .map_err(|_| io::Error::other("the snapshot file record is not UTF-8"))?;
        Ok(Some((meta, name)))
    }

    fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        Ok(self
            .engine
            .get(Partition::Schema, key)
            .map_err(io::Error::other)?
            .map(|v| v.to_vec()))
    }

    /// Move a snapshot an earlier build stored as a Schema value into a file.
    fn adopt_legacy(&self) -> io::Result<()> {
        let Some(data) = self.get(KEY_LEGACY_DATA)? else {
            return Ok(());
        };
        if let Some(meta) = self.get(KEY_SNAPSHOT_META)? {
            let meta: SnapshotMeta = rmp_serde::from_slice(&meta).map_err(io::Error::other)?;
            let mut file = self.stage()?;
            file.write_all(&data)?;
            self.publish(&meta, &mut file)?;
        }
        self.engine
            .delete(Partition::Schema, KEY_LEGACY_DATA)
            .map_err(io::Error::other)
    }
}

/// A file name no other snapshot of this process or an earlier one has.
fn unique_stem() -> String {
    static NEXT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let seq = NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    format!("{nanos:032x}-{seq:016x}")
}

/// Make a rename in `dir` durable. Windows cannot open a directory for
/// syncing; there the rename is as durable as the file system makes it.
fn sync_dir(dir: Option<&Path>) -> io::Result<()> {
    #[cfg(unix)]
    if let Some(dir) = dir {
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Remove a file nothing needs any more; a failure only leaves it for the
/// next open to remove.
fn remove_quietly(path: &Path) {
    if let Err(e) = fs::remove_file(path) {
        if e.kind() != io::ErrorKind::NotFound {
            tracing::warn!(?path, %e, "could not remove a snapshot file");
        }
    }
}
