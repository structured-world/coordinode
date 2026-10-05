//! Tiered block cache: cascading volatile cache layers.
//!
//! Each layer is a file-backed cache on a specific device (NVMe, SSD, etc).
//! Layers are ordered fastest→slowest. Read misses cascade down through
//! layers to persistent storage. Eviction drains entries from faster
//! layers to slower ones.
//!
//! **All layers are volatile.** Power loss = cold cache startup, zero data loss.
//! Persistent storage (CoordiNode storage) is the sole source of truth.
//!
//! # File Format (per layer)
//!
//! Append-only file with variable-size entries:
//! ```text
//! [MAGIC: 4B]["CNDC"]
//! [Entry: key_hash(8B) | partition(1B) | key_len(2B) | value_len(4B) | key | value]
//! ...
//! ```
//!
//! # Cluster-ready
//! Per-node local cache. Each CE replica has its own tiered cache.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use coordinode_core::txn::wake::Wake;
use tracing::{debug, info, warn};

use crate::engine::partition::Partition;

use super::access::{CacheKey, compute_cache_key};
use super::config::{CacheLayerConfig, TieredCacheConfig};

const MAGIC: &[u8; 4] = b"CNDC";
const ENTRY_HEADER_SIZE: usize = 8 + 1 + 2 + 4; // key_hash + partition + key_len + value_len
const MAX_ENTRY_BYTES: usize = 16 * 1024 * 1024; // 16MB — skip blobs

/// Bounds of the number of key stripes the invalidation generations are kept
/// by: 32 KiB to 32 MiB of counters.
const MIN_STRIPES: usize = 1 << 12;
const MAX_STRIPES: usize = 1 << 22;

// ── Per-layer statistics ──────────────────────────────────────────

/// Statistics for a single cache layer.
#[derive(Debug, Clone, Default)]
pub struct LayerStats {
    pub hits: u64,
    pub misses: u64,
    pub puts: u64,
    pub evictions: u64,
    pub drains: u64,
    pub live_entries: usize,
    pub live_bytes: u64,
    pub file_bytes: u64,
}

/// Aggregated statistics for the entire tiered cache.
#[derive(Debug, Clone, Default)]
pub struct TieredCacheStats {
    pub layers: Vec<LayerStats>,
    pub total_hits: u64,
    pub total_misses: u64,
}

// ── Entry metadata ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct EntryMeta {
    offset: u64,
    total_size: u32,
    value_size: u32,
    key_len: u16,
    /// Eviction priority weight. Higher weight = stays in cache longer.
    /// Default 1.0. Labels configured as "hot" get higher weights.
    weight: f32,
    /// Partition discriminant (`part as u8`), mirrored from the entry header so
    /// a partition-scoped invalidation (`clear_partition`, used by range delete)
    /// can filter the index without reading each backing entry.
    partition: u8,
    /// The generations the value was read under; it is served only while
    /// they are current.
    stamp: FillTicket,
}

/// The invalidation generations of one key when its value was read from the
/// trees: the generation of the key's stripe and of its partition. Taken
/// before the read with [`TieredCache::ticket`] and stored with the cached
/// value, which is served only while both are unchanged, so a value read
/// before a write and cached after its invalidation is never served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillTicket {
    key: u64,
    partition: u64,
}

/// An entry evicted from a layer, on its way to the next one.
struct Drained {
    part: Partition,
    key: Vec<u8>,
    value: Vec<u8>,
    stamp: FillTicket,
}

// ── CacheLayer: single file-backed cache ──────────────────────────

/// A single cache layer backed by an append-only file.
struct CacheLayer {
    path: PathBuf,
    file: Mutex<File>,
    index: RwLock<HashMap<CacheKey, EntryMeta>>,
    max_bytes: u64,
    max_entries: usize,
    compaction_threshold: f64,
    file_bytes: AtomicU64,
    live_bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    puts: AtomicU64,
    evictions: AtomicU64,
    drains: AtomicU64,
}

impl CacheLayer {
    fn open(config: &CacheLayerConfig) -> Result<Self, std::io::Error> {
        if let Some(parent) = config.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // A layer starts empty: the file records puts but not the
        // invalidations of the run that wrote it, so an entry found in it may
        // be older than a write that followed, and nothing in the file says
        // which. The cache is volatile by contract; a restart is a cold start.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&config.path)?;
        file.write_all(MAGIC)?;
        file.flush()?;
        info!(path = %config.path.display(), "cache layer opened empty");

        Ok(Self {
            path: config.path.clone(),
            file: Mutex::new(file),
            index: RwLock::new(HashMap::new()),
            max_bytes: config.max_bytes,
            max_entries: config.max_entries,
            compaction_threshold: config.compaction_threshold,
            file_bytes: AtomicU64::new(MAGIC.len() as u64),
            live_bytes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            puts: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            drains: AtomicU64::new(0),
        })
    }

    /// Read from this layer: the value and the generations it was read under.
    fn get(&self, part: Partition, key: &[u8]) -> Option<(bytes::Bytes, FillTicket)> {
        let cache_key = compute_cache_key(part, key);

        let meta = {
            let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
            idx.get(&cache_key).copied()
        };

        let meta = match meta {
            Some(m) => m,
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        match Self::read_entry(&mut file, &meta) {
            Ok((read_part, read_key, value)) => {
                if read_part == part && read_key == key {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    Some((bytes::Bytes::from(value), meta.stamp))
                } else {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
            Err(e) => {
                debug!(error = %e, "cache layer read error");
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Write to this layer with a priority weight and the generations the
    /// value was read under. Evicts if necessary, returns evicted entries for
    /// drain-through to the next layer. Higher weight = stays in cache longer
    /// during eviction.
    fn put(
        &self,
        part: Partition,
        key: &[u8],
        value: &[u8],
        weight: f32,
        stamp: FillTicket,
    ) -> Vec<Drained> {
        if value.len() > MAX_ENTRY_BYTES {
            return Vec::new();
        }

        let cache_key = compute_cache_key(part, key);
        let total_size = ENTRY_HEADER_SIZE + key.len() + value.len();

        // Evict if over capacity — collect evicted entries for drain-through
        let drained = self.evict_if_needed(total_size as u64);

        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());

        if let Err(e) = file.seek(SeekFrom::End(0)) {
            debug!(error = %e, "cache seek error");
            return drained;
        }

        let offset = match file.stream_position() {
            Ok(pos) => pos,
            Err(e) => {
                debug!(error = %e, "cache position error");
                return drained;
            }
        };

        if Self::write_entry(&mut file, cache_key, part, key, value).is_err() {
            return drained;
        }

        let mut idx = self.index.write().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = idx.remove(&cache_key) {
            self.live_bytes
                .fetch_sub(u64::from(old.total_size), Ordering::Relaxed);
        }

        #[allow(clippy::cast_possible_truncation)]
        let meta = EntryMeta {
            offset,
            total_size: total_size as u32,
            value_size: value.len() as u32,
            key_len: key.len() as u16,
            weight,
            partition: part as u8,
            stamp,
        };
        idx.insert(cache_key, meta);

        self.file_bytes
            .fetch_add(total_size as u64, Ordering::Relaxed);
        self.live_bytes
            .fetch_add(total_size as u64, Ordering::Relaxed);
        self.puts.fetch_add(1, Ordering::Relaxed);

        drained
    }

    /// Remove entry from this layer.
    fn remove(&self, part: Partition, key: &[u8]) {
        let cache_key = compute_cache_key(part, key);
        let mut idx = self.index.write().unwrap_or_else(|e| e.into_inner());
        if let Some(meta) = idx.remove(&cache_key) {
            self.live_bytes
                .fetch_sub(u64::from(meta.total_size), Ordering::Relaxed);
        }
    }

    /// Remove the entry of `cache_key` if it still carries `stamp`: a stale
    /// entry found by a read, and not one a later fill put in its place.
    fn remove_stale(&self, cache_key: CacheKey, stamp: FillTicket) {
        let mut idx = self.index.write().unwrap_or_else(|e| e.into_inner());
        if idx.get(&cache_key).is_some_and(|meta| meta.stamp == stamp) {
            if let Some(meta) = idx.remove(&cache_key) {
                self.live_bytes
                    .fetch_sub(u64::from(meta.total_size), Ordering::Relaxed);
            }
        }
    }

    /// Drop every index entry for `part`. Used by a range delete to invalidate
    /// all cached keys of the affected partition: the cache cannot range-query,
    /// and a range tombstone leaves the shadowed keys physically present, so a
    /// stale cache hit would otherwise return a deleted value. Orphaned file
    /// bytes are reclaimed by the next compaction, mirroring [`remove`].
    fn clear_partition(&self, part: Partition) {
        let target = part as u8;
        let mut idx = self.index.write().unwrap_or_else(|e| e.into_inner());
        let mut freed = 0u64;
        idx.retain(|_, meta| {
            if meta.partition == target {
                freed += u64::from(meta.total_size);
                false
            } else {
                true
            }
        });
        self.live_bytes.fetch_sub(freed, Ordering::Relaxed);
    }

    fn len(&self) -> usize {
        self.index.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn stats(&self) -> LayerStats {
        let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
        LayerStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            puts: self.puts.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            drains: self.drains.load(Ordering::Relaxed),
            live_entries: idx.len(),
            live_bytes: self.live_bytes.load(Ordering::Relaxed),
            file_bytes: self.file_bytes.load(Ordering::Relaxed),
        }
    }

    /// Compact the layer file by rewriting only live entries.
    fn compact(&self) -> Result<(), std::io::Error> {
        let file_bytes = self.file_bytes.load(Ordering::Relaxed);
        let live_bytes = self.live_bytes.load(Ordering::Relaxed);

        if file_bytes == 0 || live_bytes == 0 {
            return Ok(());
        }

        let dead_ratio = 1.0 - (live_bytes as f64 / file_bytes as f64);
        if dead_ratio < self.compaction_threshold {
            return Ok(());
        }

        let tmp_path = self.path.with_extension("compact.tmp");
        let mut tmp_file = File::create(&tmp_path)?;
        tmp_file.write_all(MAGIC)?;

        let mut new_index = HashMap::new();
        let mut new_file_bytes = MAGIC.len() as u64;

        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let idx = self.index.read().unwrap_or_else(|e| e.into_inner());

        for (&cache_key, &meta) in idx.iter() {
            if let Ok(entry_bytes) = Self::read_entry_raw(&mut file, &meta) {
                let new_offset = new_file_bytes;
                tmp_file.write_all(&entry_bytes)?;
                new_index.insert(
                    cache_key,
                    EntryMeta {
                        offset: new_offset,
                        ..meta
                    },
                );
                new_file_bytes += u64::from(meta.total_size);
            }
        }

        drop(idx);
        tmp_file.flush()?;
        std::fs::rename(&tmp_path, &self.path)?;

        *file = OpenOptions::new().read(true).write(true).open(&self.path)?;

        let mut idx = self.index.write().unwrap_or_else(|e| e.into_inner());
        *idx = new_index;

        let new_live = new_file_bytes.saturating_sub(MAGIC.len() as u64);
        self.file_bytes.store(new_file_bytes, Ordering::Relaxed);
        self.live_bytes.store(new_live, Ordering::Relaxed);

        Ok(())
    }

    // ── Internal ──────────────────────────────────────────────────

    /// Evict entries if over capacity. Returns the evicted entries, each with
    /// the generations it was read under, for drain-through.
    fn evict_if_needed(&self, incoming_bytes: u64) -> Vec<Drained> {
        let current = self.live_bytes.load(Ordering::Relaxed);
        let entry_count = {
            let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
            idx.len()
        };

        if current + incoming_bytes <= self.max_bytes && entry_count < self.max_entries {
            return Vec::new();
        }

        let target = self.max_bytes * 9 / 10;
        let to_free = (current + incoming_bytes).saturating_sub(target);

        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let mut idx = self.index.write().unwrap_or_else(|e| e.into_inner());

        // Collect candidates sorted by weight ascending (lowest priority evicted first).
        // This ensures "hot" labels with high weights stay in cache longer.
        let mut candidates: Vec<(CacheKey, EntryMeta)> =
            idx.iter().map(|(&k, &m)| (k, m)).collect();
        candidates.sort_by(|a, b| {
            a.1.weight
                .partial_cmp(&b.1.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut freed: u64 = 0;
        let mut keys_to_remove = Vec::new();
        for (k, meta) in candidates {
            if freed >= to_free {
                break;
            }
            freed += u64::from(meta.total_size);
            keys_to_remove.push((k, meta));
        }

        let mut drained = Vec::new();

        for (key, meta) in &keys_to_remove {
            // Read entry data for drain-through before removing
            if let Ok((part, key, value)) = Self::read_entry(&mut file, meta) {
                drained.push(Drained {
                    part,
                    key,
                    value,
                    stamp: meta.stamp,
                });
                self.drains.fetch_add(1, Ordering::Relaxed);
            }

            if let Some(removed) = idx.remove(key) {
                self.live_bytes
                    .fetch_sub(u64::from(removed.total_size), Ordering::Relaxed);
            }
        }

        self.evictions
            .fetch_add(keys_to_remove.len() as u64, Ordering::Relaxed);

        debug!(
            evicted = keys_to_remove.len(),
            freed,
            drained = drained.len(),
            "cache layer eviction"
        );

        drained
    }

    fn write_entry(
        file: &mut File,
        cache_key: CacheKey,
        part: Partition,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), std::io::Error> {
        file.write_all(&cache_key.to_le_bytes())?;
        file.write_all(&[part as u8])?;
        #[allow(clippy::cast_possible_truncation)]
        {
            file.write_all(&(key.len() as u16).to_le_bytes())?;
            file.write_all(&(value.len() as u32).to_le_bytes())?;
        }
        file.write_all(key)?;
        file.write_all(value)?;
        Ok(())
    }

    fn read_entry(
        file: &mut File,
        meta: &EntryMeta,
    ) -> Result<(Partition, Vec<u8>, Vec<u8>), std::io::Error> {
        file.seek(SeekFrom::Start(meta.offset))?;

        let mut header = [0u8; ENTRY_HEADER_SIZE];
        file.read_exact(&mut header)?;

        let partition = partition_from_byte(header[8]);

        let mut key_buf = vec![0u8; meta.key_len as usize];
        file.read_exact(&mut key_buf)?;

        let mut value_buf = vec![0u8; meta.value_size as usize];
        file.read_exact(&mut value_buf)?;

        Ok((partition, key_buf, value_buf))
    }

    fn read_entry_raw(file: &mut File, meta: &EntryMeta) -> Result<Vec<u8>, std::io::Error> {
        file.seek(SeekFrom::Start(meta.offset))?;
        let mut buf = vec![0u8; meta.total_size as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }
}

// ── TieredCache: cascading layers ─────────────────────────────────

/// Tiered block cache with drain-through eviction.
///
/// Layers are ordered fastest→slowest. On read miss, each layer is
/// tried in order before falling through to persistent storage.
/// On eviction from layer N, entries drain to layer N+1.
///
/// A background thread periodically compacts cache files when
/// dead-space ratio exceeds the configured threshold. The thread
/// is stopped automatically when the cache is dropped.
///
/// # Cluster-ready
/// Background compaction is local to each node. In distributed mode
/// (3-node HA), each replica runs its own compaction independently.
/// No coordination needed — cache is per-node volatile state.
pub struct TieredCache {
    layers: Arc<Vec<CacheLayer>>,
    /// Shutdown flag for background compaction thread.
    shutdown: Arc<AtomicBool>,
    /// The compaction thread sleeps on this between passes; the drop
    /// interrupts it.
    wake: Arc<Wake>,
    /// Background compaction thread handle.
    compaction_thread: Option<std::thread::JoinHandle<()>>,
    /// Per-label eviction priority weights. Used by `resolve_weight()`.
    label_weights: Arc<HashMap<String, f32>>,
    /// Invalidation generations by key stripe: bumped by every
    /// [`remove`](Self::remove) of a key of the stripe, after its write.
    key_generations: Box<[AtomicU64]>,
    /// Invalidation generations by partition discriminant: bumped by
    /// [`clear_partition`](Self::clear_partition).
    partition_generations: Box<[AtomicU64]>,
}

impl Drop for TieredCache {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.wake.interrupt();
        if let Some(handle) = self.compaction_thread.take() {
            let _ = handle.join();
        }
    }
}

impl TieredCache {
    /// Open a tiered cache from config. Each layer is opened/created independently.
    /// Spawns a background compaction thread if `compaction_interval_secs > 0`.
    pub fn open(config: &TieredCacheConfig) -> Result<Self, std::io::Error> {
        let label_weights = Arc::new(config.label_weights.clone());
        let mut layers = Vec::with_capacity(config.layers.len());
        for layer_config in &config.layers {
            layers.push(CacheLayer::open(layer_config)?);
        }

        let layer_count = layers.len();
        let layers = Arc::new(layers);
        let shutdown = Arc::new(AtomicBool::new(false));
        let wake = Arc::new(Wake::default());

        let compaction_thread = if config.compaction_interval_secs > 0 && !layers.is_empty() {
            let layers_ref = Arc::clone(&layers);
            let shutdown_ref = Arc::clone(&shutdown);
            let wake_ref = Arc::clone(&wake);
            let interval = Duration::from_secs(config.compaction_interval_secs);

            Some(
                std::thread::Builder::new()
                    .name("cache-compaction".to_string())
                    .spawn(move || {
                        Self::compaction_loop(&layers_ref, &shutdown_ref, &wake_ref, interval);
                    })
                    .map_err(std::io::Error::other)?,
            )
        } else {
            None
        };

        info!(
            layers = layer_count,
            background_compaction = compaction_thread.is_some(),
            "tiered cache opened"
        );

        // About one stripe per entry the layers hold, so a write invalidates
        // about one unrelated entry; bounded both ways.
        let entries: usize = config.layers.iter().map(|l| l.max_entries).sum();
        let stripes = entries.next_power_of_two().clamp(MIN_STRIPES, MAX_STRIPES);
        Ok(Self {
            layers,
            shutdown,
            wake,
            compaction_thread,
            label_weights,
            key_generations: (0..stripes).map(|_| AtomicU64::new(0)).collect(),
            partition_generations: (0..=usize::from(u8::MAX))
                .map(|_| AtomicU64::new(0))
                .collect(),
        })
    }

    /// The generations of `part`/`key` now. A value read from the trees is
    /// cached with the ticket taken BEFORE the read: a write whose
    /// invalidation follows the ticket then makes the cached value stale.
    pub fn ticket(&self, part: Partition, key: &[u8]) -> FillTicket {
        self.ticket_of(compute_cache_key(part, key), part)
    }

    fn ticket_of(&self, cache_key: CacheKey, part: Partition) -> FillTicket {
        FillTicket {
            key: self.key_generation(cache_key).load(Ordering::SeqCst),
            partition: self.partition_generations[usize::from(part as u8)].load(Ordering::SeqCst),
        }
    }

    fn key_generation(&self, cache_key: CacheKey) -> &AtomicU64 {
        // The stripe count is a power of two, and the cache key a hash.
        #[allow(clippy::cast_possible_truncation)]
        let stripe = (cache_key as usize) & (self.key_generations.len() - 1);
        &self.key_generations[stripe]
    }

    /// Read from the tiered cache, cascading through layers. An entry read
    /// before an invalidation that followed it is a miss.
    ///
    /// On hit at layer N, the value is promoted to layer 0 (if N > 0)
    /// for faster future access.
    pub fn get(&self, part: Partition, key: &[u8]) -> Option<bytes::Bytes> {
        let cache_key = compute_cache_key(part, key);
        for (i, layer) in self.layers.iter().enumerate() {
            let Some((value, stamp)) = layer.get(part, key) else {
                continue;
            };
            // Checked after the read: an invalidation the entry predates is
            // seen here even when it landed after the entry was found.
            if stamp != self.ticket_of(cache_key, part) {
                layer.remove_stale(cache_key, stamp);
                continue;
            }
            // Promote to faster layer on hit at deeper layer, still under
            // the generations it was read under.
            if i > 0 {
                let drained = self.layers[0].put(part, key, &value, 1.0, stamp);
                self.drain_to_next(0, drained);
            }
            return Some(value);
        }
        None
    }

    /// Cache `value` as the current value of `part`/`key`, with default
    /// weight (1.0). Evicted entries drain to slower layers. A value read
    /// from the trees goes through [`fill`](Self::fill) instead.
    pub fn put(&self, part: Partition, key: &[u8], value: &[u8]) {
        self.put_weighted(part, key, value, 1.0);
    }

    /// [`put`](Self::put) with a specific eviction weight. Higher weight =
    /// entry stays in cache longer during eviction.
    pub fn put_weighted(&self, part: Partition, key: &[u8], value: &[u8], weight: f32) {
        self.fill(part, key, value, weight, self.ticket(part, key));
    }

    /// Cache `value`, read from the trees after `ticket` was taken, with an
    /// eviction weight. A value whose ticket is already stale is not stored,
    /// and one that turns stale later is not served.
    pub fn fill(&self, part: Partition, key: &[u8], value: &[u8], weight: f32, ticket: FillTicket) {
        if self.layers.is_empty() || ticket != self.ticket(part, key) {
            return;
        }

        let drained = self.layers[0].put(part, key, value, weight, ticket);
        self.drain_to_next(0, drained);
    }

    /// Look up the eviction weight for a label.
    /// Returns the configured weight, or 1.0 if the label is not in the map.
    pub fn resolve_weight(&self, label: &str) -> f32 {
        self.label_weights.get(label).copied().unwrap_or(1.0)
    }

    /// Whether any label weights are configured.
    /// Fast-path check to skip value deserialization when no weights are set.
    pub fn label_weights_empty(&self) -> bool {
        self.label_weights.is_empty()
    }

    /// Invalidate `part`/`key` once a write to it is visible to reads: every
    /// value of it cached, or read before now and cached later, is stale.
    pub fn remove(&self, part: Partition, key: &[u8]) {
        let cache_key = compute_cache_key(part, key);
        // The generation moves first: an entry a concurrent fill or drain
        // puts back after the removal below still carries the old one.
        self.key_generation(cache_key)
            .fetch_add(1, Ordering::SeqCst);
        for layer in self.layers.iter() {
            layer.remove(part, key);
        }
    }

    /// Invalidate every cached entry for `part` across all layers, once a
    /// change to the whole partition is visible to reads. Called by a range
    /// delete, which cannot enumerate the affected keys cheaply and must not
    /// leave a stale cache hit shadowing a range-tombstoned key.
    pub fn clear_partition(&self, part: Partition) {
        self.partition_generations[usize::from(part as u8)].fetch_add(1, Ordering::SeqCst);
        for layer in self.layers.iter() {
            layer.clear_partition(part);
        }
    }

    /// Number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Total entries across all layers.
    pub fn total_entries(&self) -> usize {
        self.layers.iter().map(|l| l.len()).sum()
    }

    /// Whether all layers are empty.
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(|l| l.is_empty())
    }

    /// Per-layer and aggregate statistics.
    pub fn stats(&self) -> TieredCacheStats {
        let layer_stats: Vec<LayerStats> = self.layers.iter().map(|l| l.stats()).collect();
        let total_hits: u64 = layer_stats.iter().map(|s| s.hits).sum();
        let total_misses = if let Some(last) = layer_stats.last() {
            last.misses
        } else {
            0
        };
        TieredCacheStats {
            layers: layer_stats,
            total_hits,
            total_misses,
        }
    }

    /// Compact all layers manually.
    pub fn compact(&self) -> Result<(), std::io::Error> {
        for layer in self.layers.iter() {
            layer.compact()?;
        }
        Ok(())
    }

    /// Background compaction loop. Runs on a dedicated thread.
    fn compaction_loop(
        layers: &[CacheLayer],
        shutdown: &AtomicBool,
        wake: &Wake,
        interval: Duration,
    ) {
        debug!(
            interval_secs = interval.as_secs(),
            "background cache compaction started"
        );
        wake.bind();

        while !shutdown.load(Ordering::Relaxed) {
            // Asleep for the whole interval; the drop interrupts it.
            let deadline = std::time::Instant::now() + interval;
            loop {
                if shutdown.load(Ordering::Relaxed) {
                    return;
                }
                let now = std::time::Instant::now();
                if now >= deadline {
                    break;
                }
                wake.wait(Some(deadline - now));
            }

            if shutdown.load(Ordering::Relaxed) {
                return;
            }

            for (i, layer) in layers.iter().enumerate() {
                if let Err(e) = layer.compact() {
                    warn!(layer = i, error = %e, "background compaction failed");
                }
            }
        }

        debug!("background cache compaction stopped");
    }

    /// Drain evicted entries to the next layer in the cascade, each under
    /// the generations it was read under.
    fn drain_to_next(&self, from_layer: usize, entries: Vec<Drained>) {
        let next = from_layer + 1;
        if next >= self.layers.len() || entries.is_empty() {
            return; // Bottom layer — drop (lossy cache)
        }

        for entry in entries {
            // Drained entries get default weight (evicted = already "cooler").
            let further_drained =
                self.layers[next].put(entry.part, &entry.key, &entry.value, 1.0, entry.stamp);
            // Recursively drain deeper
            if !further_drained.is_empty() {
                self.drain_to_next(next, further_drained);
            }
        }
    }
}

fn partition_from_byte(b: u8) -> Partition {
    match b {
        0 => Partition::Node,
        1 => Partition::Adj,
        2 => Partition::EdgeProp,
        3 => Partition::Blob,
        4 => Partition::BlobRef,
        5 => Partition::Schema,
        6 => Partition::Idx,
        7 => Partition::Raft,
        8 => Partition::Counter,
        9 => Partition::VectorF32,
        10 => Partition::Registry,
        _ => Partition::Node,
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
