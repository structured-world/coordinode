//! Write admission by free disk space.
//!
//! A write that reaches the disk with no room left fails inside an fsync, and
//! a failed fsync of the consensus log stops the node for good. So writes
//! are refused before anything is written while the filesystems under the
//! durable endpoints are below a reserve: below `min_free_bytes` free on any
//! of them, every new write is refused with [`StorageError::OutOfSpace`];
//! reads go on. Writes are admitted again once every filesystem has
//! `resume_free_bytes` free, so a write that frees a little does not flip
//! the state back and forth.
//!
//! Free space is read from the filesystem (statvfs), which counts everything
//! on it, the database's own files and anything else. A write never reads
//! it: it loads the paused state, one atomic. A background monitor
//! ([`SpaceGuard::start_monitor`]) reads it: every check interval
//! (`free_space_check_ms`, 10 s by default, since other programs fill the
//! disk too), every [`PAUSED_RECHECK`] while writes are paused, so freed
//! space admits them again promptly, and sooner when writes already sized
//! for a hard limit have spent half the room the last reading left above
//! the reserve. That last is only a reason to read earlier: no write is
//! sized for it, and it is no limit of its own.
//!
//! The reserve lowers the risk of a full disk; it does not rule it out.
//! Another program can take the space between two readings, and a write
//! that then meets a full disk fails as an I/O error.
//!
//! The guard also answers for the endpoints' hard limits
//! ([`Footprints`]): a write that finds no room under one is refused with
//! [`StorageError::CapacityExhausted`], at the same points and with the same
//! consequences as one below the free-space reserve. That limit counts this
//! store's own files exactly, which is what keeps a consensus log append from
//! meeting it inside its write.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use coordinode_core::txn::wake::Wake;

use crate::engine::config::{Durability, StorageConfig};
use crate::engine::footprint::{Admit, Admitted, Footprints};
use crate::error::{StorageError, StorageResult};

/// How often the monitor reads free space with nothing else waking it.
pub const DEFAULT_FREE_SPACE_CHECK_MS: u64 = 10_000;

/// How often the monitor reads free space while writes are paused.
pub const PAUSED_RECHECK: Duration = Duration::from_millis(250);

/// Default free space below which writes are refused.
pub const DEFAULT_MIN_FREE_BYTES: u64 = 1 << 30;

/// Default free space above which refused writes are admitted again.
pub const DEFAULT_RESUME_FREE_BYTES: u64 = 2 << 30;

/// Free space as a write sees it, and whether writes are paused.
pub struct SpaceGuard {
    /// Roots of the durable endpoints: the filesystems a write lands on.
    paths: Vec<PathBuf>,
    min_free: AtomicU64,
    resume_free: AtomicU64,
    /// The least free space over `paths` at the last reading.
    available: AtomicU64,
    /// The path that had the least free space at the last reading.
    tightest: parking_lot::Mutex<Option<PathBuf>>,
    paused: AtomicBool,
    /// Bytes sized writes may still add before free space is read again:
    /// half the room the last reading left above the reserve, the other half
    /// for what the writes cost again on disk (flushes, compactions). A
    /// write that takes it to zero wakes the monitor.
    headroom: AtomicI64,
    /// How often, in ms, the monitor reads free space with nothing waking it.
    check_ms: AtomicU64,
    /// The monitor's wakeup.
    wake: Arc<Wake>,
    /// The endpoints' hard limits, answered for with the free space.
    footprints: Footprints,
    /// Test builds only: how many times free space was read.
    #[cfg(test)]
    readings: AtomicU64,
}

/// The running free-space monitor; dropping it stops its thread.
pub struct SpaceMonitor {
    stop: Arc<AtomicBool>,
    wake: Arc<Wake>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for SpaceMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpaceMonitor").finish_non_exhaustive()
    }
}

impl Drop for SpaceMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.wake.interrupt();
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::error!("the free-space monitor panicked");
        }
    }
}

impl SpaceGuard {
    /// A guard over the durable endpoints of `config`, with its reserve.
    pub fn new(config: &StorageConfig) -> Self {
        let paths = config
            .endpoints
            .iter()
            .filter(|e| e.durability != Durability::Volatile)
            .map(|e| e.path.clone())
            .collect();
        let guard = Self {
            paths,
            min_free: AtomicU64::new(config.min_free_bytes),
            resume_free: AtomicU64::new(config.resume_free_bytes.max(config.min_free_bytes)),
            available: AtomicU64::new(u64::MAX),
            tightest: parking_lot::Mutex::new(None),
            paused: AtomicBool::new(false),
            headroom: AtomicI64::new(0),
            check_ms: AtomicU64::new(config.free_space_check_ms.max(1)),
            wake: Arc::new(Wake::default()),
            footprints: Footprints::default(),
            #[cfg(test)]
            readings: AtomicU64::new(0),
        };
        guard.refresh();
        guard
    }

    /// Start the thread that reads free space for this guard, until the
    /// returned monitor drops. The guard is held weakly: the monitor does
    /// not keep it alive.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] when the thread cannot be spawned.
    pub fn start_monitor(self: &Arc<Self>) -> StorageResult<SpaceMonitor> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let wake = Arc::clone(&self.wake);
        let guard: Weak<Self> = Arc::downgrade(self);
        let thread = std::thread::Builder::new()
            .name("coord-free-space".into())
            .spawn(move || {
                let Some(first) = guard.upgrade() else {
                    return;
                };
                first.wake.bind();
                let wake = Arc::clone(&first.wake);
                drop(first);
                loop {
                    let interval = match guard.upgrade() {
                        Some(guard) => guard.next_reading(),
                        None => return,
                    };
                    wake.wait(Some(interval));
                    if stopped.load(Ordering::Acquire) {
                        return;
                    }
                    match guard.upgrade() {
                        Some(guard) => guard.refresh(),
                        None => return,
                    }
                }
            })
            .map_err(|e| StorageError::InvalidConfig(format!("free-space monitor spawn: {e}")))?;
        Ok(SpaceMonitor {
            stop,
            wake,
            thread: Some(thread),
        })
    }

    /// How long the monitor sleeps before its next reading.
    fn next_reading(&self) -> Duration {
        if self.paused.load(Ordering::Acquire) {
            PAUSED_RECHECK
        } else {
            Duration::from_millis(self.check_ms.load(Ordering::Relaxed))
        }
    }

    /// Change how often the monitor reads free space with nothing waking
    /// it, in ms (at least 1). Takes effect from its next sleep.
    pub fn set_check_interval_ms(&self, ms: u64) {
        self.check_ms.store(ms.max(1), Ordering::Relaxed);
        self.wake.notify();
    }

    /// The monitor's check interval, in ms.
    pub fn check_interval_ms(&self) -> u64 {
        self.check_ms.load(Ordering::Relaxed)
    }

    /// Take `bytes` this store is about to write off the headroom, and wake
    /// the monitor to read free space again when they use the last of it.
    pub fn spend(&self, bytes: u64) {
        // A write's size is far below i64::MAX; the headroom is set again at
        // each reading, so the subtractions between two readings stay far
        // from overflow.
        let bytes = i64::try_from(bytes).unwrap_or(i64::MAX);
        let before = self.headroom.fetch_sub(bytes, Ordering::Relaxed);
        if before > 0 && before <= bytes {
            self.wake.notify();
        }
    }

    /// Answer for the hard limits of `footprints` as well.
    #[must_use]
    pub fn with_footprints(mut self, footprints: Footprints) -> Self {
        self.footprints = footprints;
        self
    }

    /// The endpoints' hard limits the guard answers for.
    pub fn footprints(&self) -> &Footprints {
        &self.footprints
    }

    /// Admit a write, or refuse it while free space is below the reserve or
    /// an endpoint's hard limit leaves client writes no room.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] while writes are paused;
    /// [`StorageError::CapacityExhausted`] for an endpoint without room.
    pub fn admit(&self) -> StorageResult<()> {
        self.admit_free_space()?;
        self.has_room(Admit::Client)
    }

    /// [`Self::admit`] by free space alone, for a write the engine then
    /// admits on the footprint itself (reclaiming before it refuses). By the
    /// monitor's last reading: never reads the filesystem.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] while writes are paused.
    pub fn admit_free_space(&self) -> StorageResult<()> {
        if self.paused.load(Ordering::Acquire) {
            metrics::counter!("coordinode_storage_out_of_space_refusals_total").increment(1);
            return Err(self.refusal());
        }
        Ok(())
    }

    /// [`Self::admit`], then reserve `bytes` of `class` on every endpoint
    /// with a hard limit, held until the returned admission drops: a write
    /// that cannot be checked by size later (a consensus log append) holds
    /// its room from before it is proposed until it is in the log. The bytes
    /// are taken off the headroom ([`Self::spend`]).
    ///
    /// # Errors
    ///
    /// As [`Self::admit`], and [`StorageError::CapacityExhausted`] when the
    /// write does not fit.
    pub fn reserve(&self, bytes: u64, class: Admit) -> StorageResult<Admitted> {
        let class = class.on_this_thread();
        match class {
            Admit::Client => self.admit()?,
            Admit::Recovery => self.admit_recovery()?,
        }
        let admitted = self.footprints.admit(bytes, class)?;
        self.spend(bytes);
        Ok(admitted)
    }

    /// Refuse a write of `class` that no endpoint's hard limit has room for.
    fn has_room(&self, class: Admit) -> StorageResult<()> {
        let room = self.footprints.has_room(class);
        if room.is_err() {
            metrics::counter!("coordinode_footprint_refusals_total").increment(1);
        }
        room
    }

    /// Admit a write of the store's own recovery records (index use guards,
    /// the consensus vote and log bounds), which go on while client writes
    /// are refused: out of the upper half of the reserve only. The lower half
    /// is left to the flushes and compactions that retire those records from
    /// their journal, so recovery never takes the room its own completion
    /// needs.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] while free space is below half the
    /// reserve; [`StorageError::CapacityExhausted`] for an endpoint whose
    /// hard limit leaves even the store's own work no room.
    pub fn admit_recovery(&self) -> StorageResult<()> {
        self.admit_free_space_recovery()?;
        self.has_room(Admit::Recovery)
    }

    /// Whether every endpoint's hard limit has room for the store's own
    /// work, free space aside.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] for an endpoint whose hard limit
    /// leaves even the store's own work no room.
    pub fn has_recovery_room(&self) -> StorageResult<()> {
        self.has_room(Admit::Recovery)
    }

    /// [`Self::admit_recovery`] by free space alone, for a write the engine
    /// then admits on the footprint itself, or a caller that checks free
    /// space before it learns whether the store's own work also needs the
    /// hard limits' room ([`Self::has_recovery_room`]). By the monitor's
    /// last reading.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] while free space is below half the
    /// reserve.
    pub fn admit_free_space_recovery(&self) -> StorageResult<()> {
        if self.available_bytes() < self.min_free_bytes() / 2 {
            metrics::counter!("coordinode_storage_out_of_space_refusals_total").increment(1);
            return Err(self.refusal());
        }
        Ok(())
    }

    /// Whether client writes are paused: free space below the reserve at the
    /// monitor's last reading, or an endpoint whose hard limit leaves them no
    /// room.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire) || self.has_room(Admit::Client).is_err()
    }

    /// Whether free space alone pauses writes, whatever the hard limits say:
    /// for the store's own work that frees room under a limit (a snapshot
    /// that lets the consensus log be compacted) yet must not run on a disk
    /// below its reserve.
    pub fn is_paused_by_free_space(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// The least free space over the durable endpoints at the last reading.
    pub fn available_bytes(&self) -> u64 {
        self.available.load(Ordering::Acquire)
    }

    /// The reserve: below this, writes are refused.
    pub fn min_free_bytes(&self) -> u64 {
        self.min_free.load(Ordering::Acquire)
    }

    /// Above this, refused writes are admitted again.
    pub fn resume_free_bytes(&self) -> u64 {
        self.resume_free.load(Ordering::Acquire)
    }

    /// Change the reserve without a restart. `resume_free_bytes` below
    /// `min_free_bytes` is taken as `min_free_bytes`. Takes effect at once.
    pub fn set_reserve(&self, min_free_bytes: u64, resume_free_bytes: u64) {
        self.min_free.store(min_free_bytes, Ordering::Release);
        self.resume_free
            .store(resume_free_bytes.max(min_free_bytes), Ordering::Release);
        self.refresh();
    }

    /// Read free space now and update the paused state and the headroom: the
    /// monitor's reading, also taken when the reserve changes.
    pub fn refresh(&self) {
        #[cfg(test)]
        self.readings.fetch_add(1, Ordering::Relaxed);
        let mut least: Option<(u64, &PathBuf)> = None;
        for path in &self.paths {
            match fs4::available_space(path) {
                Ok(free) if least.is_none_or(|(l, _)| free < l) => least = Some((free, path)),
                Ok(_) => {}
                // A directory not created yet holds nothing to measure; an
                // unreadable one cannot prove room, so it does not either.
                Err(e) => tracing::debug!(path = %path.display(), %e, "free space unreadable"),
            }
        }
        let available = least.map_or(u64::MAX, |(free, _)| free);
        self.available.store(available, Ordering::Release);
        *self.tightest.lock() = least.map(|(_, p)| p.clone());
        let was = self.paused.load(Ordering::Acquire);
        let paused = if was {
            available < self.resume_free.load(Ordering::Acquire)
        } else {
            available < self.min_free.load(Ordering::Acquire)
        };
        if paused != was {
            self.paused.store(paused, Ordering::Release);
            if paused {
                tracing::error!(
                    available_bytes = available,
                    min_free_bytes = self.min_free_bytes(),
                    path = ?self.tightest.lock().as_deref(),
                    "free disk space below the reserve: writes are refused, reads go on"
                );
            } else {
                tracing::warn!(
                    available_bytes = available,
                    "free disk space back above the resume level: writes are admitted"
                );
            }
        }
        // Half the room above the reserve; none below it. An unmeasurable
        // store (no durable endpoint read) gives the monitor no reason to
        // read sooner than its interval.
        let above = available.saturating_sub(self.min_free.load(Ordering::Acquire)) / 2;
        self.headroom
            .store(i64::try_from(above).unwrap_or(i64::MAX), Ordering::Relaxed);
        metrics::gauge!("coordinode_storage_free_bytes").set(available as f64);
        metrics::gauge!("coordinode_storage_writes_paused").set(if paused { 1.0 } else { 0.0 });
    }

    fn refusal(&self) -> StorageError {
        StorageError::OutOfSpace {
            path: self
                .tightest
                .lock()
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            available_bytes: self.available_bytes(),
            min_free_bytes: self.min_free_bytes(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
