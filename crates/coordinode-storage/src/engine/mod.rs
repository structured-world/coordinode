//! CoordiNode LSM storage engine — primary KV layer (coordinode-storage).

#[cfg(feature = "tenancy")]
pub mod access;
/// Without `tenancy` no grant catalog and no authorization step is
/// compiled: the two points where a unit applies get their unchecked form.
#[cfg(not(feature = "tenancy"))]
#[path = "access_unchecked.rs"]
pub mod access;
pub mod applied;
pub mod batch;
pub mod capacity;
pub mod cardinality;
pub mod change_feed;
pub mod claims;
pub(crate) mod closure_ledger;
pub(crate) mod compaction;
pub mod config;
pub mod coordinator;
pub mod core;
pub(crate) mod coverage;
pub mod decoded_log;
pub mod fixed_cut;
pub(crate) mod flush;
pub mod footprint;
pub mod index_copies;
pub mod installation;
pub mod journal_reclaim;
pub(crate) mod local_journal;
pub mod local_pipeline;
pub mod merge;
pub mod metadata;
pub mod metered_fs;
pub mod open_sets;
pub mod open_txns;
pub mod operations;
pub mod oplog_journal;
pub mod owned_metrics;
pub mod partition;
pub mod pending;
pub mod promise;
pub mod rebuild;
pub mod retention_stats;
pub(crate) mod revision;
pub mod routing;
pub mod scope;
pub mod space;
pub mod stats;
pub mod store_transfer;
pub mod tap;
#[cfg(test)]
pub(crate) mod test_labels;
pub mod transaction;
pub mod tree_set;
pub mod vector_keys;

pub use compaction::CompactionStrategyKind;

/// Most index entry effects one unit's DERIVED work may derive, counted by
/// `change_bound` before anything is derived, at sealing, on apply and on
/// journal replay alike: a small unit cannot fan out without bound on every
/// member. Room for one change between two memberships each at the most
/// combinations a value set may make, so a write admitted by that limit is
/// never refused by this one.
pub(crate) const MAX_DERIVED_EFFECTS: usize = 2 * coordinode_core::index::derive::MAX_ENTRY_FAN_OUT;

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
/// A value is given back without the revision its record carries
/// ([`revision`]).
pub struct StorageGuard {
    inner: lsm_tree::IterGuardImpl,
    /// The generation whose installation the key was read from.
    generation: Option<u64>,
    /// Whether the value ends in a revision stamp: a user key of a partition
    /// whose records carry one. A scan of the user keyspace yields no
    /// reserved key, so this is decided once per scan.
    stamped: bool,
}

impl StorageGuard {
    /// A pair of `part`'s user keyspace stored as it is addressed.
    #[inline]
    pub(crate) fn raw(part: partition::Partition, inner: lsm_tree::IterGuardImpl) -> Self {
        Self {
            inner,
            generation: None,
            stamped: part.carries_revision(),
        }
    }

    /// A pair of `generation` of `part`, read from its installation.
    #[inline]
    pub(crate) fn of_generation(
        part: partition::Partition,
        inner: lsm_tree::IterGuardImpl,
        generation: u64,
    ) -> Self {
        Self {
            inner,
            generation: Some(generation),
            stamped: part.carries_revision(),
        }
    }

    /// `value` without its stamp when `stamped`: a view, no copy.
    #[inline]
    fn payload(stamped: bool, value: lsm_tree::UserValue) -> lsm_tree::Result<Payload> {
        let len = if stamped {
            revision::payload_of(&value)?.len()
        } else {
            value.len()
        };
        Ok(Payload { value, len })
    }

    /// The key and value.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn into_inner(self) -> lsm_tree::Result<(lsm_tree::UserKey, Payload)> {
        use lsm_tree::Guard as _;
        let translation = Translation {
            generation: self.generation,
        };
        let stamped = self.stamped;
        let (key, value) = self.inner.into_inner()?;
        Ok((translation.logical_key(key), Self::payload(stamped, value)?))
    }

    /// The key, copied into an owned buffer, and the value as stored: its
    /// revision stamp included. What a copy of the row to another tree takes,
    /// so the copy keeps the revision.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub(crate) fn into_stored(self) -> lsm_tree::Result<(Vec<u8>, lsm_tree::UserValue)> {
        use lsm_tree::Guard as _;
        let translation = Translation {
            generation: self.generation,
        };
        let (key, value) = self.inner.into_inner()?;
        Ok((translation.logical_vec(&key), value))
    }

    /// The key and value, with the key copied into an owned buffer: what a
    /// caller collecting the pairs keeps. A translated key costs no more
    /// than the copy.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn into_owned(self) -> lsm_tree::Result<(Vec<u8>, Payload)> {
        use lsm_tree::Guard as _;
        let translation = Translation {
            generation: self.generation,
        };
        let stamped = self.stamped;
        let (key, value) = self.inner.into_inner()?;
        Ok((
            translation.logical_vec(&key),
            Self::payload(stamped, value)?,
        ))
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
    ) -> lsm_tree::Result<(lsm_tree::UserKey, Option<Payload>)> {
        use lsm_tree::Guard as _;
        let stamped = self.stamped;
        let unstamp = |value: Option<lsm_tree::UserValue>| {
            value.map(|value| Self::payload(stamped, value)).transpose()
        };
        if self.generation.is_none() {
            let (key, value) = self.inner.into_inner_if(pred)?;
            return Ok((key, unstamp(value)?));
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
        Ok((key, unstamp(value)?))
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
        let size = self.inner.size()?;
        if !self.stamped {
            return Ok(size);
        }
        // `STAMP_LEN` is 8 and fits a `u32`.
        size.checked_sub(revision::STAMP_LEN as u32)
            .ok_or(lsm_tree::Error::InvalidHeader(
                "a stored record is shorter than its revision stamp",
            ))
    }

    /// The value.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn value(self) -> lsm_tree::Result<Payload> {
        use lsm_tree::Guard as _;
        let stamped = self.stamped;
        Self::payload(stamped, self.inner.value()?)
    }
}

/// The value of a record as a scan hands it out: its payload, a view into
/// the bytes the tree read, without the revision stamp stored after it.
/// Reads as `&[u8]`; what a caller keeps of it, it copies.
#[derive(Clone)]
pub struct Payload {
    value: lsm_tree::UserValue,
    /// The payload's length: the value's, less a stamp it carries.
    len: usize,
}

impl Payload {
    /// The payload's bytes.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // `len` is at most the value's length, by construction.
        &self.value[..self.len]
    }

    /// The payload copied into an owned buffer.
    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }
}

impl ::core::ops::Deref for Payload {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for Payload {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl ::core::borrow::Borrow<[u8]> for Payload {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl ::core::fmt::Debug for Payload {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.debug_tuple("Payload").field(&self.as_slice()).finish()
    }
}

impl Eq for Payload {}

impl<T: AsRef<[u8]> + ?Sized> PartialEq<T> for Payload {
    fn eq(&self, other: &T) -> bool {
        self.as_slice() == other.as_ref()
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
