//! A Tantivy directory for an index that is rebuilt rather than recovered.
//!
//! An index the engine derives from its own store and rebuilds from it every
//! time the index is opened gains nothing from durable files: a crash loses
//! nothing the rebuild does not restore. [`ScratchDirectory`] therefore writes
//! files without syncing them, so a commit costs no device flush, which on a
//! loaded host made one commit take a second and held every search waiting
//! for the index behind it. Reads go through the memory-mapped directory as
//! usual; written pages are in the page cache either way.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tantivy::directory::error::{DeleteError, LockError, OpenReadError, OpenWriteError};
use tantivy::directory::{
    AntiCallToken, Directory, DirectoryLock, FileHandle, Lock, MmapDirectory, TerminatingWrite,
    WatchCallback, WatchHandle, WritePtr,
};

/// A memory-mapped directory whose writes are never synced. Only for an
/// index rebuilt from its source whenever it is opened.
#[derive(Clone, Debug)]
pub struct ScratchDirectory {
    inner: MmapDirectory,
    root: PathBuf,
}

impl ScratchDirectory {
    /// A scratch directory at `root`, which must exist.
    ///
    /// # Errors
    ///
    /// `root` cannot be opened as a directory.
    pub fn open(root: &Path) -> Result<Self, tantivy::directory::error::OpenDirectoryError> {
        Ok(Self {
            inner: MmapDirectory::open(root)?,
            root: root.to_path_buf(),
        })
    }
}

/// A file writer that flushes to the page cache and never to the device.
struct UnsyncedWriter(File);

impl Write for UnsyncedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl TerminatingWrite for UnsyncedWriter {
    fn terminate_ref(&mut self, _: AntiCallToken) -> io::Result<()> {
        self.0.flush()
    }
}

impl Directory for ScratchDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.inner.get_file_handle(path)
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.inner.delete(path)
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.inner.exists(path)
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join(path))
            .map_err(|e| {
                if e.kind() == io::ErrorKind::AlreadyExists {
                    OpenWriteError::FileAlreadyExists(path.to_path_buf())
                } else {
                    OpenWriteError::wrap_io_error(e, path.to_path_buf())
                }
            })?;
        Ok(BufWriter::new(Box::new(UnsyncedWriter(file))))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.inner.atomic_read(path)
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        // A reader never sees a partly written file: the content goes to a
        // sibling first and replaces the target by rename. One writer per
        // index, so one fixed sibling name suffices.
        let target = self.root.join(path);
        let mut staged = target.clone().into_os_string();
        staged.push(".staged");
        let staged = PathBuf::from(staged);
        std::fs::write(&staged, data)?;
        std::fs::rename(&staged, &target)
    }

    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        self.inner.acquire_lock(lock)
    }

    fn watch(&self, watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.inner.watch(watch_callback)
    }

    fn sync_directory(&self) -> io::Result<()> {
        Ok(())
    }
}
