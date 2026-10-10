//! CoordiNode LSM storage engine — primary KV layer (coordinode-storage).

pub mod applied;
pub mod batch;
pub mod capacity;
pub mod cardinality;
pub mod claims;
pub(crate) mod compaction;
pub mod config;
pub mod coordinator;
pub mod core;
pub(crate) mod coverage;
pub(crate) mod flush;
pub mod installation;
pub mod merge;
pub mod metadata;
pub mod open_txns;
pub mod oplog_journal;
pub mod partition;
pub mod pending;
pub mod retention_stats;
pub mod routing;
pub mod space;
pub mod stats;
pub mod tap;
pub mod transaction;
pub mod vector_keys;

/// Most index entry effects one unit's DERIVED work may derive, on apply and
/// on journal replay alike: a small unit cannot fan out without bound on
/// every member.
pub(crate) const MAX_DERIVED_EFFECTS: usize = 1 << 22;

/// Snapshot type: a sequence number used as a read visibility bound.
///
/// Reads with this seqno see all writes with seqno ≤ this value.
/// Obtain via `StorageEngine::snapshot()` for the current latest,
/// or `StorageEngine::snapshot_at(ts)` for a specific point-in-time.
pub type StorageSnapshot = lsm_tree::SeqNo;

/// Iterator over key-value guards from a prefix or range scan, keys in the
/// form callers address them.
pub type StorageIter = Box<dyn DoubleEndedIterator<Item = StorageGuard> + Send + 'static>;

/// One key-value pair of a scan. A key of an index generation is read from
/// this member's installation of it and given back with the generation in
/// its place (see [`installation`]); every other key is given back as stored.
pub struct StorageGuard {
    inner: lsm_tree::IterGuardImpl,
    /// The generation whose installation the key was read from.
    generation: Option<u64>,
}

impl StorageGuard {
    /// A pair stored as it is addressed.
    #[inline]
    pub(crate) fn raw(inner: lsm_tree::IterGuardImpl) -> Self {
        Self {
            inner,
            generation: None,
        }
    }

    /// A pair of `generation`, read from its installation.
    #[inline]
    pub(crate) fn of_generation(inner: lsm_tree::IterGuardImpl, generation: u64) -> Self {
        Self {
            inner,
            generation: Some(generation),
        }
    }

    /// The key and value.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn into_inner(self) -> lsm_tree::Result<(lsm_tree::UserKey, lsm_tree::UserValue)> {
        use lsm_tree::Guard as _;
        let translation = Translation {
            generation: self.generation,
        };
        let (key, value) = self.inner.into_inner()?;
        Ok((translation.logical_key(key), value))
    }

    /// The key and value, with the key copied into an owned buffer: what a
    /// caller collecting the pairs keeps. A translated key costs no more
    /// than the copy.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn into_owned(self) -> lsm_tree::Result<(Vec<u8>, lsm_tree::UserValue)> {
        use lsm_tree::Guard as _;
        let translation = Translation {
            generation: self.generation,
        };
        let (key, value) = self.inner.into_inner()?;
        Ok((translation.logical_vec(&key), value))
    }

    /// The key, and the value only when `pred` accepts the key (a separated
    /// value is then not read).
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn into_inner_if(
        self,
        pred: impl Fn(&lsm_tree::UserKey) -> bool,
    ) -> lsm_tree::Result<(lsm_tree::UserKey, Option<lsm_tree::UserValue>)> {
        use lsm_tree::Guard as _;
        if self.generation.is_none() {
            return self.inner.into_inner_if(pred);
        }
        let translation = Translation {
            generation: self.generation,
        };
        let logical = std::cell::OnceCell::new();
        let (_, value) = self.inner.into_inner_if(|key| {
            let key = logical.get_or_init(|| translation.logical_key(key.clone()));
            pred(key)
        })?;
        let key = logical.into_inner().ok_or_else(|| {
            lsm_tree::Error::InvalidHeader("a scan pair was read without its key")
        })?;
        Ok((key, value))
    }

    /// The key.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn key(self) -> lsm_tree::Result<lsm_tree::UserKey> {
        use lsm_tree::Guard as _;
        let translation = Translation {
            generation: self.generation,
        };
        let key = self.inner.key()?;
        Ok(translation.logical_key(key))
    }

    /// The value's size.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn size(self) -> lsm_tree::Result<u32> {
        use lsm_tree::Guard as _;
        self.inner.size()
    }

    /// The value.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn value(self) -> lsm_tree::Result<lsm_tree::UserValue> {
        use lsm_tree::Guard as _;
        self.inner.value()
    }
}

/// How a scan pair's key is given back: with the generation it was read for
/// in place of the installation it was read from, or as stored.
#[derive(Clone, Copy)]
struct Translation {
    generation: Option<u64>,
}

impl Translation {
    #[inline]
    fn logical_vec(self, key: &[u8]) -> Vec<u8> {
        let mut out = key.to_vec();
        if let Some(generation) = self.generation {
            installation::put_generation(&mut out, generation);
        }
        out
    }

    #[inline]
    fn logical_key(self, key: lsm_tree::UserKey) -> lsm_tree::UserKey {
        match self.generation {
            Some(_) => lsm_tree::UserKey::from(self.logical_vec(&key)),
            None => key,
        }
    }
}

/// Seekable range-scan iterator: like [`StorageIter`] but can reposition in
/// place via `seek_to` / `seek_to_for_prev` (RocksDB `Seek` / `SeekForPrev`)
/// without reopening per-SST readers. Lets a consumer drive one open iterator
/// across disjoint subranges (skip-scan) — e.g. spatial Z-curve dead-zone
/// skipping — and `peek_key` the current position for leapfrog joins.
pub type SeekableStorageIter = Box<dyn SeekableScan + 'static>;

/// A tree's own seekable iterator, keys as stored.
pub(crate) type RawSeekableIter = Box<dyn lsm_tree::SeekableGuardIter + 'static>;

/// A range scan that can reposition in place, with keys in the form callers
/// address them (see [`StorageGuard`]).
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a seekable storage scan",
    label = "not a seekable scan",
    note = "the engine's `range_seekable` returns one"
)]
pub trait SeekableScan: DoubleEndedIterator<Item = StorageGuard> + Send {
    /// Reposition so the next item is the first with key `>= key`.
    fn seek_to(&mut self, key: &[u8]);

    /// Reposition so the next item from the back is the last with key
    /// `<= key`.
    fn seek_to_for_prev(&mut self, key: &[u8]);

    /// The key of the next item, without consuming it; `None` once the range
    /// is exhausted.
    fn peek_key(&mut self) -> Option<lsm_tree::Result<lsm_tree::UserKey>>;
}

impl SeekableScan for installation::Seekable {
    fn seek_to(&mut self, key: &[u8]) {
        Self::seek_to(self, key);
    }

    fn seek_to_for_prev(&mut self, key: &[u8]) {
        Self::seek_to_for_prev(self, key);
    }

    fn peek_key(&mut self) -> Option<lsm_tree::Result<lsm_tree::UserKey>> {
        Self::peek_key(self)
    }
}
