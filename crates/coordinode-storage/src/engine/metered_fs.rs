//! The engine's file system, charging every file it grows to the owned
//! footprint of the endpoint that holds it ([`crate::engine::footprint`]).
//!
//! The trees write their tables, value files and manifests through this
//! wrapper: a write is charged before it reaches the inner file system and
//! refused with [`std::io::ErrorKind::StorageFull`] when the endpoint has no
//! room for it, so a flush or compaction stops with its inputs intact rather
//! than passing the limit. Removal, truncation and hole punching give the
//! space back; a hard link is the same bytes under a second name. The free
//! space a compaction plans against is what the footprint leaves past the
//! flushes and writes it has promised room to.
//!
//! A file opened only for reading is the inner file itself: reads pay
//! nothing for the accounting.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lsm_tree::fs::{
    BlockRead, FileHint, Fs, FsCapabilities, FsDirEntry, FsFile, FsMetadata, FsOpenOptions,
    ReadQueue, SyncMode, WriteLifetime,
};
use lsm_tree::io;

use crate::engine::footprint::{FileCharge, Footprints};

/// A file system whose writes under capped endpoints are charged first.
pub struct MeteredFs {
    inner: Arc<dyn Fs>,
    footprints: Footprints,
}

impl std::fmt::Debug for MeteredFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeteredFs")
            .field("footprints", &self.footprints)
            .finish_non_exhaustive()
    }
}

impl MeteredFs {
    /// `inner`, charging growth under the endpoints of `footprints`.
    pub fn new(inner: Arc<dyn Fs>, footprints: Footprints) -> Self {
        Self { inner, footprints }
    }

    /// The footprints this file system charges.
    pub fn footprints(&self) -> &Footprints {
        &self.footprints
    }
}

fn storage_full(e: std::io::Error) -> io::Error {
    io::Error::from(e)
}

impl Fs for MeteredFs {
    fn open(&self, path: &Path, opts: &FsOpenOptions) -> io::Result<Box<dyn FsFile>> {
        let writes = opts.write || opts.append || opts.create || opts.create_new || opts.truncate;
        if !writes {
            return self.inner.open(path, opts);
        }
        let Some(footprint) = self.footprints.for_path(path) else {
            return self.inner.open(path, opts);
        };
        if opts.create_new {
            // A name created anew holds no bytes yet, whatever a stale entry
            // of a file removed outside this file system said.
            footprint.removed(path);
        }
        let file = self.inner.open(path, opts)?;
        let charge = footprint.file(path);
        let len = file.metadata()?.len;
        if opts.truncate {
            charge.truncate_to(0);
        } else if len > 0 {
            // A file reopened for writing keeps the charge it has; one found
            // larger than charged (written before the store was tracked) is
            // recorded as it is.
            let allocated = self
                .inner
                .allocated_size(path)?
                .unwrap_or_else(|| footprint.allocation(len));
            charge.observe(allocated);
        }
        Ok(Box::new(MeteredFile {
            inner: file,
            charge,
            pos: if opts.append { len } else { 0 },
            fs: Arc::clone(&self.inner),
            path: path.to_path_buf(),
            len: if opts.truncate { 0 } else { len },
            append: opts.append,
        }))
    }

    fn read_blocks_batched(&self, reqs: &mut [BlockRead<'_>]) -> io::Result<()> {
        self.inner.read_blocks_batched(reqs)
    }

    fn read_blocks_batched_each(
        &self,
        reqs: &mut [BlockRead<'_>],
        on_read: &mut dyn FnMut(usize, &BlockRead<'_>),
    ) -> io::Result<()> {
        self.inner.read_blocks_batched_each(reqs, on_read)
    }

    fn read_queue(&self) -> Box<dyn ReadQueue + '_> {
        self.inner.read_queue()
    }

    fn reads_by_descriptor(&self) -> bool {
        self.inner.reads_by_descriptor()
    }

    fn check_descriptor_batch(&self, reqs: &[BlockRead<'_>]) -> io::Result<()> {
        self.inner.check_descriptor_batch(reqs)
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir_all(path)
    }

    fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir(path)
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<FsDirEntry>> {
        self.inner.read_dir(path)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        let removed = self.inner.remove_file(path);
        match &removed {
            Ok(()) => self.footprints.removed(path),
            Err(e) if e.kind() == io::ErrorKind::NotFound => self.footprints.removed(path),
            Err(_) => {}
        }
        removed
    }

    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        let removed = self.inner.remove_dir_all(path);
        // Whatever was removed before a failure is gone; what is left is
        // charged again when it is next opened for writing.
        self.footprints.removed_tree(path);
        removed
    }

    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir(path)
    }

    fn same_file(&self, a: &Path, b: &Path) -> io::Result<bool> {
        self.inner.same_file(a, b)
    }

    fn read_link(&self, path: &Path) -> io::Result<Option<PathBuf>> {
        self.inner.read_link(path)
    }

    fn symlink(&self, target: &Path, link: &Path, to_dir: bool) -> io::Result<()> {
        self.inner.symlink(target, link, to_dir)
    }

    fn symlink_to_dir(&self, path: &Path) -> io::Result<bool> {
        self.inner.symlink_to_dir(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.inner.rename(from, to)?;
        self.footprints.renamed(from, to).map_err(storage_full)
    }

    fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.inner.metadata(path)
    }

    fn sync_directory(&self, path: &Path) -> io::Result<()> {
        self.inner.sync_directory(path)
    }

    fn sync_directory_with(&self, path: &Path, mode: SyncMode) -> io::Result<()> {
        self.inner.sync_directory_with(path, mode)
    }

    fn sync_device(&self, path: &Path) -> io::Result<()> {
        self.inner.sync_device(path)
    }

    fn exists(&self, path: &Path) -> io::Result<bool> {
        self.inner.exists(path)
    }

    fn hard_link(&self, src: &Path, dst: &Path) -> io::Result<()> {
        self.inner.hard_link(src, dst)?;
        self.footprints.linked(src, dst).map_err(storage_full)
    }

    fn backend_id(&self) -> Option<u64> {
        self.inner.backend_id()
    }

    fn volume_id(&self, path: &Path) -> Option<u64> {
        self.inner.volume_id(path)
    }

    fn capabilities(&self, path: &Path) -> FsCapabilities {
        self.inner.capabilities(path)
    }

    fn try_disable_cow(&self, path: &Path) -> io::Result<()> {
        self.inner.try_disable_cow(path)
    }

    fn punch_hole(&self, path: &Path, offset: u64, len: u64) -> io::Result<()> {
        self.inner.punch_hole(path, offset, len)?;
        if let Some(charge) = self.footprints.file(path)
            && let Some(allocated) = self.inner.allocated_size(path)?
        {
            charge.settle(allocated);
        }
        Ok(())
    }

    fn reflink_file(&self, src: &Path, dst: &Path) -> io::Result<()> {
        // A clone shares its blocks only until either side is written; it is
        // charged whole, before it is made.
        if let Some(charge) = self.footprints.file(dst) {
            let len = self.inner.metadata(src)?.len;
            charge.grow_to(len).map_err(storage_full)?;
        }
        let cloned = self.inner.reflink_file(src, dst);
        if cloned.is_err() {
            self.footprints.removed(dst);
        }
        cloned
    }

    fn truncate_file(&self, path: &Path) -> io::Result<()> {
        self.inner.truncate_file(path)?;
        if let Some(charge) = self.footprints.file(path) {
            charge.truncate_to(0);
        }
        Ok(())
    }

    fn hard_link_count(&self, path: &Path) -> io::Result<u64> {
        self.inner.hard_link_count(path)
    }

    fn available_space(&self, path: &Path) -> io::Result<u64> {
        let free = self.inner.available_space(path)?;
        Ok(match self.footprints.for_path(path) {
            Some(footprint) => free.min(footprint.room()),
            None => free,
        })
    }

    fn allocated_size(&self, path: &Path) -> io::Result<Option<u64>> {
        self.inner.allocated_size(path)
    }

    fn extent_is_hole(&self, path: &Path, offset: u64, len: u64) -> io::Result<Option<bool>> {
        self.inner.extent_is_hole(path, offset, len)
    }

    fn extent_contains_hole(&self, path: &Path, offset: u64, len: u64) -> io::Result<Option<bool>> {
        self.inner.extent_contains_hole(path, offset, len)
    }
}

/// A file open for writing under a capped endpoint: each write is charged up
/// to the end it reaches, with what the file system may allocate ahead of
/// it, before it is made; the charge follows what the file system reports
/// it allocated once the write is done, and is settled to it at each sync and
/// when the handle closes.
struct MeteredFile {
    inner: Box<dyn FsFile>,
    charge: Arc<FileCharge>,
    /// The file system the file is on, which reports what it allocated.
    fs: Arc<dyn Fs>,
    path: PathBuf,
    /// The position the next write starts at.
    pos: u64,
    /// The file's length as this handle has seen it.
    len: u64,
    /// Writes go to the end, wherever the position is.
    append: bool,
}

impl MeteredFile {
    /// What the file system allocated to the file: as the file system reports
    /// it, or where it reports none (Windows) as the host measures the file,
    /// the measure open charges existing files by.
    fn allocated(&self) -> Option<u64> {
        self.fs
            .allocated_size(&self.path)
            .ok()
            .flatten()
            .or_else(|| crate::engine::footprint::allocated_at(&self.path))
    }

    /// Bring the charge to what the file system allocated: the allowance
    /// charged ahead of the writer is given back, an allocation past it kept.
    fn settle(&self) {
        if let Some(allocated) = self.allocated() {
            self.charge.settle(allocated);
        }
    }
}

impl Drop for MeteredFile {
    fn drop(&mut self) {
        // Done writing: the charge comes down to what the file holds.
        if let Some(allocated) = self.allocated() {
            self.charge.settle(allocated);
        }
    }
}

impl std::io::Read for MeteredFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Write for MeteredFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let start = if self.append { self.len } else { self.pos };
        // A write lands within the file's addressable range: no overflow.
        self.charge.grow_appending(start + buf.len() as u64)?;
        let n = self.inner.write(buf)?;
        let end = start + n as u64;
        self.pos = end;
        self.len = self.len.max(end);
        // An allocation past the allowance is counted at once, so the next
        // growth anywhere on the endpoint is checked against it.
        if let Some(allocated) = self.allocated() {
            self.charge.observe(allocated);
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl std::io::Seek for MeteredFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let at = self.inner.seek(pos)?;
        self.pos = at;
        Ok(at)
    }
}

impl FsFile for MeteredFile {
    fn sync_all(&self) -> io::Result<()> {
        self.inner.sync_all()?;
        self.settle();
        Ok(())
    }

    fn sync_data(&self) -> io::Result<()> {
        self.inner.sync_data()?;
        self.settle();
        Ok(())
    }

    fn sync_all_with(&self, mode: SyncMode) -> io::Result<()> {
        self.inner.sync_all_with(mode)?;
        self.settle();
        Ok(())
    }

    fn sync_data_with(&self, mode: SyncMode) -> io::Result<()> {
        self.inner.sync_data_with(mode)?;
        self.settle();
        Ok(())
    }

    fn metadata(&self) -> io::Result<FsMetadata> {
        self.inner.metadata()
    }

    fn hard_link_count(&self) -> io::Result<u64> {
        self.inner.hard_link_count()
    }

    fn set_len(&self, size: u64) -> io::Result<()> {
        self.charge.grow_to(size).map_err(storage_full)?;
        self.inner.set_len(size)?;
        self.charge.truncate_to(size);
        self.settle();
        Ok(())
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        self.inner.read_at(buf, offset)
    }

    fn read_many(&self, regions: &mut [(u64, &mut [u8])]) -> io::Result<()> {
        self.inner.read_many(regions)
    }

    fn backing_fd(&self) -> Option<i32> {
        self.inner.backing_fd()
    }

    fn lock_exclusive(&self) -> io::Result<()> {
        self.inner.lock_exclusive()
    }

    fn try_lock_exclusive(&self) -> io::Result<bool> {
        self.inner.try_lock_exclusive()
    }

    fn hint(&self, hint: FileHint) -> io::Result<()> {
        self.inner.hint(hint)
    }

    fn start_writeback(&self, offset: u64, len: u64) -> io::Result<()> {
        self.inner.start_writeback(offset, len)
    }

    fn set_write_lifetime(&self, lifetime: WriteLifetime) -> io::Result<()> {
        self.inner.set_write_lifetime(lifetime)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
