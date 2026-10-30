//! A Tantivy directory for an index that is rebuilt rather than recovered.
//!
//! An index the engine derives from its own store and rebuilds from it every
//! time the index is opened gains nothing from durable files: a crash loses
//! nothing the rebuild does not restore. [`ScratchDirectory`] therefore writes
//! files without syncing them, so a commit costs no device flush, which on a
//! loaded host made one commit take a second and held every search waiting
//! for the index behind it. Reads go through the memory-mapped directory as
//! usual; written pages are in the page cache either way.
//!
//! Its files are charged to the owned footprint of the endpoint holding them
//! before they grow: an index build that does not fit stops with
//! [`io::ErrorKind::StorageFull`] instead of passing the endpoint's limit.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use coordinode_storage::engine::footprint::{FileCharge, Footprints, allocated_at};
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
    footprints: Footprints,
}

impl ScratchDirectory {
    /// A scratch directory at `root`, which must exist, its files charged to
    /// `footprints`.
    ///
    /// # Errors
    ///
    /// `root` cannot be opened as a directory.
    pub fn open(
        root: &Path,
        footprints: Footprints,
    ) -> Result<Self, tantivy::directory::error::OpenDirectoryError> {
        Ok(Self {
            inner: MmapDirectory::open(root)?,
            root: root.to_path_buf(),
            footprints,
        })
    }
}

/// A file writer that flushes to the page cache and never to the device,
/// charging what it writes, and what the file system may allocate ahead of
/// it, before it writes it; the charge comes down to what the file system
/// allocated once the file is complete.
struct UnsyncedWriter {
    file: File,
    path: PathBuf,
    charge: Option<Arc<FileCharge>>,
    len: u64,
}

impl UnsyncedWriter {
    fn settle(&self) {
        if let Some(charge) = &self.charge
            && let Some(allocated) = allocated_at(&self.path)
        {
            charge.settle(allocated);
        }
    }
}

impl Write for UnsyncedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(charge) = &self.charge {
            // A file's length fits u64.
            charge.grow_appending(self.len + buf.len() as u64)?;
        }
        let n = self.file.write(buf)?;
        self.len += n as u64;
        if let Some(charge) = &self.charge
            && let Some(allocated) = allocated_at(&self.path)
        {
            charge.observe(allocated);
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl TerminatingWrite for UnsyncedWriter {
    fn terminate_ref(&mut self, _: AntiCallToken) -> io::Result<()> {
        self.file.flush()?;
        self.settle();
        Ok(())
    }
}

impl Drop for UnsyncedWriter {
    fn drop(&mut self) {
        self.settle();
    }
}

impl Directory for ScratchDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.inner.get_file_handle(path)
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.inner.delete(path)?;
        self.footprints.removed(&self.root.join(path));
        Ok(())
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.inner.exists(path)
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        let full = self.root.join(path);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&full)
            .map_err(|e| {
                if e.kind() == io::ErrorKind::AlreadyExists {
                    OpenWriteError::FileAlreadyExists(path.to_path_buf())
                } else {
                    OpenWriteError::wrap_io_error(e, path.to_path_buf())
                }
            })?;
        // A name created anew holds nothing, whatever a removed file under
        // it was charged.
        self.footprints.removed(&full);
        Ok(BufWriter::new(Box::new(UnsyncedWriter {
            file,
            charge: self.footprints.file(&full),
            path: full,
            len: 0,
        })))
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
        // The write replaces whatever a failed earlier one left there.
        self.footprints.removed(&staged);
        if let Some(charge) = self.footprints.file(&staged) {
            charge.grow_to(data.len() as u64)?;
        }
        std::fs::write(&staged, data)?;
        std::fs::rename(&staged, &target)?;
        self.footprints.renamed(&staged, &target)
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
