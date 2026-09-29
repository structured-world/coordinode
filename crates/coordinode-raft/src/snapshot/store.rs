//! Where a node keeps its current Raft snapshot, under the engine's data
//! directory, named by a Schema record written together with the snapshot's
//! metadata. A snapshot this node built stays the capture it was built from,
//! a hard-linked checkpoint of the store, and is serialized only when a peer
//! needs it sent; one received from the leader is the file it arrived as. The
//! bytes never enter the store itself, so a snapshot does not carry the one
//! before it, and nothing holds it in memory.

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
/// A published snapshot kept as the capture of the store it was built from.
const CAPTURED: &str = "capt";
/// A snapshot still being written or received.
const STAGED: &str = "part";
/// A reader's private hard-linked copy of a published capture.
const READING: &str = "read";

/// The directory holding `engine`'s snapshot files.
pub fn snapshot_dir(engine: &StorageEngine) -> PathBuf {
    engine.data_dir().join("raft-snapshot")
}

/// One snapshot's bytes in a file. Reads and writes continue where the last
/// one ended; [`Seek`] moves the position. A staged file, one not yet
/// published as the current snapshot, is removed when dropped.
///
/// A snapshot kept as a capture has no bytes until they are needed: the
/// first access serializes the capture into a staged file (see
/// [`Self::materialize`]), so a snapshot nobody reads is never serialized.
#[derive(Debug)]
pub struct SnapshotFile {
    /// `None` until a capture-backed snapshot is materialized.
    file: Option<File>,
    path: PathBuf,
    staged: bool,
    /// The private hard-linked copy of a capture the bytes are still to be
    /// written from; removed once they are, or when this is dropped.
    capture: Option<PathBuf>,
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
            file: Some(file),
            path,
            staged: true,
            capture: None,
        })
    }

    fn open(path: PathBuf) -> io::Result<Self> {
        let file = File::open(&path)?;
        Ok(Self {
            file: Some(file),
            path,
            staged: false,
            capture: None,
        })
    }

    /// The snapshot held by `capture`, a copy this value owns, serialized
    /// into a staged file in `dir` on first access.
    fn from_capture(capture: PathBuf, dir: &Path) -> Self {
        Self {
            file: None,
            path: dir.join(format!("{}.{STAGED}", unique_stem())),
            staged: true,
            capture: Some(capture),
        }
    }

    /// Serialize a capture-backed snapshot into its staged file, positioned
    /// at the start; a no-op once the bytes exist. Blocking: it reads the
    /// whole captured store, so async callers run it off the runtime.
    ///
    /// # Errors
    ///
    /// The capture cannot be opened or read, or the file cannot be written.
    pub fn materialize(&mut self) -> io::Result<()> {
        if self.file.is_some() {
            return Ok(());
        }
        let capture = self
            .capture
            .clone()
            .ok_or_else(|| io::Error::other("a snapshot with neither bytes nor a capture"))?;
        let captured = StorageEngine::open_checkpoint(&capture)
            .map_err(|e| io::Error::other(format!("open the snapshot capture: {e}")))?;
        let mut file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&self.path)?;
        crate::snapshot::write_full_snapshot(&captured, &mut file)?;
        // The captured engine holds the copy's files open until dropped.
        drop(captured);
        file.seek(SeekFrom::Start(0))?;
        self.file = Some(file);
        self.capture = None;
        remove_dir_quietly(&capture);
        Ok(())
    }

    fn opened(&mut self) -> io::Result<&mut File> {
        self.materialize()?;
        self.file
            .as_mut()
            .ok_or_else(|| io::Error::other("snapshot bytes missing after materialize"))
    }

    /// The number of bytes in the file.
    ///
    /// # Errors
    ///
    /// The bytes cannot be materialized, or the file's metadata cannot be
    /// read.
    pub fn size(&mut self) -> io::Result<u64> {
        Ok(self.opened()?.metadata()?.len())
    }

    /// A second handle on the same file, sharing this one's position, for
    /// reading it from another task.
    ///
    /// # Errors
    ///
    /// The bytes cannot be materialized, or the handle cannot be duplicated.
    pub fn try_clone_file(&mut self) -> io::Result<File> {
        self.opened()?.try_clone()
    }

    fn name(&self) -> io::Result<&str> {
        self.path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| io::Error::other(format!("snapshot file name {:?}", self.path)))
    }

    /// Make a staged file durable under its published name.
    fn finish(&mut self) -> io::Result<()> {
        self.opened()?.sync_all()?;
        let published = self.path.with_extension(FINISHED);
        fs::rename(&self.path, &published)?;
        self.path = published;
        self.staged = false;
        sync_dir(self.path.parent())
    }
}

impl Read for SnapshotFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.opened()?.read(buf)
    }
}

impl Write for SnapshotFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.opened()?.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.opened()?.flush()
    }
}

impl Seek for SnapshotFile {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.opened()?.seek(pos)
    }
}

impl Drop for SnapshotFile {
    fn drop(&mut self) {
        if self.staged && self.file.is_some() {
            remove_quietly(&self.path);
        }
        if let Some(capture) = &self.capture {
            remove_dir_quietly(capture);
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
                if entry.file_type()?.is_dir() {
                    fs::remove_dir_all(entry.path())?;
                } else {
                    fs::remove_file(entry.path())?;
                }
            }
        }
        Ok(store)
    }

    /// A new staged file for a snapshot about to be written.
    pub(crate) fn stage(&self) -> io::Result<SnapshotFile> {
        SnapshotFile::stage(&self.dir)
    }

    /// The current snapshot, opened for reading from its start. A capture is
    /// handed out as a private hard-linked copy, so a later publication can
    /// remove the capture while the copy is still being read.
    pub(crate) fn current(&self) -> io::Result<Option<(SnapshotMeta, SnapshotFile)>> {
        // Held so a publication cannot remove the file between the record
        // read and the open.
        let _publishing = self.lock()?;
        let Some((meta, name)) = self.record()? else {
            return Ok(None);
        };
        let path = self.dir.join(&name);
        if path.extension().and_then(|e| e.to_str()) != Some(CAPTURED) {
            return Ok(Some((meta, SnapshotFile::open(path)?)));
        }
        let copy = self.dir.join(format!("{}.{READING}", unique_stem()));
        match link_tree(&path, &copy) {
            Ok(()) => Ok(Some((meta, SnapshotFile::from_capture(copy, &self.dir)))),
            // A crash lost the capture the record names. Without a snapshot,
            // openraft builds a new one from the state machine, which holds
            // everything the lost one did.
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                remove_dir_quietly(&copy);
                tracing::warn!(?path, "the recorded snapshot capture is missing");
                Ok(None)
            }
            Err(e) => {
                remove_dir_quietly(&copy);
                Err(e)
            }
        }
    }

    /// Make the store capture at `capture` the current snapshot, described
    /// by `meta`, moving it into this store; the snapshot it replaces is
    /// removed. Nothing is serialized: a peer that needs the snapshot gets
    /// it from [`Self::current`]. A snapshot older than the current one is
    /// not published: `false` is returned and `capture` stays where it is.
    pub(crate) fn publish_capture(&self, meta: &SnapshotMeta, capture: &Path) -> io::Result<bool> {
        let _publishing = self.lock()?;
        let previous = self.record()?;
        if let Some((current, _)) = &previous {
            if current.last_log_id > meta.last_log_id {
                return Ok(false);
            }
        }
        let name = format!("{}.{CAPTURED}", unique_stem());
        let target = self.dir.join(&name);
        fs::rename(capture, &target)?;
        sync_dir(Some(&target))?;
        sync_dir(Some(&self.dir))?;
        self.record_current(meta, &name)?;
        if let Some((_, previous)) = previous {
            remove_entry_quietly(&self.dir.join(previous));
        }
        Ok(true)
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
        self.record_current(meta, file.name()?)?;
        if let Some((_, name)) = previous {
            remove_entry_quietly(&self.dir.join(name));
        }
        Ok(true)
    }

    /// Record `name` in this store's directory as the current snapshot,
    /// described by `meta`, durably: the snapshot it replaces may be removed
    /// only after this returns.
    fn record_current(&self, meta: &SnapshotMeta, name: &str) -> io::Result<()> {
        let meta_bytes = rmp_serde::to_vec(meta).map_err(io::Error::other)?;
        let mut batch = WriteBatch::new(&self.engine);
        batch.put(Partition::Schema, KEY_SNAPSHOT_META, meta_bytes);
        batch.put(Partition::Schema, KEY_SNAPSHOT_FILE, name.as_bytes());
        batch.commit().map_err(io::Error::other)?;
        self.engine.persist().map_err(io::Error::other)
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

/// Recreate the directory tree at `src` under `dst` (which must not exist)
/// with every file hard-linked, not copied: the store never rewrites a file
/// in place once written, so the copy reads what `src` held when linked.
fn link_tree(src: &Path, dst: &Path) -> io::Result<()> {
    let entries = fs::read_dir(src)?;
    fs::create_dir(dst)?;
    for entry in entries {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            link_tree(&entry.path(), &target)?;
        } else {
            fs::hard_link(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Remove a published snapshot, a file or a capture directory.
fn remove_entry_quietly(path: &Path) {
    if path.is_dir() {
        remove_dir_quietly(path);
    } else {
        remove_quietly(path);
    }
}

/// Remove a directory tree nothing needs any more; a failure only leaves it
/// for the next open to remove.
fn remove_dir_quietly(path: &Path) {
    if let Err(e) = fs::remove_dir_all(path) {
        if e.kind() != io::ErrorKind::NotFound {
            tracing::warn!(?path, %e, "could not remove a snapshot directory");
        }
    }
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
