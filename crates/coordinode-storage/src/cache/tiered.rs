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

use tracing::{debug, info, warn};

use crate::engine::partition::Partition;

use super::access::{compute_cache_key, CacheKey};
use super::config::{CacheLayerConfig, TieredCacheConfig};

const MAGIC: &[u8; 4] = b"CNDC";
const ENTRY_HEADER_SIZE: usize = 8 + 1 + 2 + 4; // key_hash + partition + key_len + value_len
const MAX_ENTRY_BYTES: usize = 16 * 1024 * 1024; // 16MB — skip blobs

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

        let exists = config.path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&config.path)?;

        let mut index = HashMap::new();

        let (file_bytes, live_bytes) = if exists && file.metadata()?.len() >= MAGIC.len() as u64 {
            match Self::scan_file(&mut file, &mut index) {
                Ok(result) => {
                    info!(
                        path = %config.path.display(),
                        entries = index.len(),
                        "cache layer recovered"
                    );
                    result
                }
                Err(e) => {
                    warn!(path = %config.path.display(), error = %e, "cache layer corrupt, resetting");
                    Self::reset_file(&mut file)?;
                    index.clear();
                    (MAGIC.len() as u64, 0)
                }
            }
        } else {
            file.write_all(MAGIC)?;
            file.flush()?;
            info!(path = %config.path.display(), "cache layer created");
            (MAGIC.len() as u64, 0)
        };

        Ok(Self {
            path: config.path.clone(),
            file: Mutex::new(file),
            index: RwLock::new(index),
            max_bytes: config.max_bytes,
            max_entries: config.max_entries,
            compaction_threshold: config.compaction_threshold,
            file_bytes: AtomicU64::new(file_bytes),
            live_bytes: AtomicU64::new(live_bytes),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            puts: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            drains: AtomicU64::new(0),
        })
    }

    /// Read from this layer. Returns `Some(value)` on hit.
    fn get(&self, part: Partition, key: &[u8]) -> Option<bytes::Bytes> {
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
                    Some(bytes::Bytes::from(value))
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

    /// Write to this layer with a priority weight. Evicts if necessary,
    /// returns evicted entries for drain-through to the next layer.
    /// Higher weight = stays in cache longer during eviction.
    fn put(
        &self,
        part: Partition,
        key: &[u8],
        value: &[u8],
        weight: f32,
    ) -> Vec<(Partition, Vec<u8>, Vec<u8>)> {
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

    /// Evict entries if over capacity. Returns evicted (part, key, value) for drain-through.
    fn evict_if_needed(&self, incoming_bytes: u64) -> Vec<(Partition, Vec<u8>, Vec<u8>)> {
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
            if let Ok((part, entry_key, value)) = Self::read_entry(&mut file, meta) {
                drained.push((part, entry_key, value));
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

    fn scan_file(
        file: &mut File,
        index: &mut HashMap<CacheKey, EntryMeta>,
    ) -> Result<(u64, u64), std::io::Error> {
        file.seek(SeekFrom::Start(0))?;

        let mut magic_buf = [0u8; 4];
        file.read_exact(&mut magic_buf)?;
        if &magic_buf != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid cache magic",
            ));
        }

        let file_len = file.metadata()?.len();
        let mut offset = MAGIC.len() as u64;
        let mut live_bytes: u64 = 0;

        while offset + ENTRY_HEADER_SIZE as u64 <= file_len {
            file.seek(SeekFrom::Start(offset))?;

            let mut header = [0u8; ENTRY_HEADER_SIZE];
            if file.read_exact(&mut header).is_err() {
                break;
            }

            let cache_key = u64::from_le_bytes(header[0..8].try_into().unwrap_or([0; 8]));
            let key_len = u16::from_le_bytes(header[9..11].try_into().unwrap_or([0; 2]));
            let value_len = u32::from_le_bytes(header[11..15].try_into().unwrap_or([0; 4]));

            let total_size = ENTRY_HEADER_SIZE as u32 + u32::from(key_len) + value_len;
            let entry_end = offset + u64::from(total_size);

            if entry_end > file_len {
                break;
            }

            let meta = EntryMeta {
                offset,
                total_size,
                value_size: value_len,
                key_len,
                weight: 1.0, // Recovered entries use default weight
            };

            if let Some(old) = index.insert(cache_key, meta) {
                live_bytes -= u64::from(old.total_size);
            }
            live_bytes += u64::from(total_size);
            offset = entry_end;
        }

        Ok((offset, live_bytes))
    }

    fn reset_file(file: &mut File) -> Result<(), std::io::Error> {
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(MAGIC)?;
        file.flush()
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
    /// Background compaction thread handle.
    compaction_thread: Option<std::thread::JoinHandle<()>>,
    /// Per-label eviction priority weights. Used by `resolve_weight()`.
    label_weights: Arc<HashMap<String, f32>>,
}

impl Drop for TieredCache {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
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

        let compaction_thread = if config.compaction_interval_secs > 0 && !layers.is_empty() {
            let layers_ref = Arc::clone(&layers);
            let shutdown_ref = Arc::clone(&shutdown);
            let interval = Duration::from_secs(config.compaction_interval_secs);

            Some(
                std::thread::Builder::new()
                    .name("cache-compaction".to_string())
                    .spawn(move || {
                        Self::compaction_loop(&layers_ref, &shutdown_ref, interval);
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

        Ok(Self {
            layers,
            shutdown,
            compaction_thread,
            label_weights,
        })
    }

    /// Read from the tiered cache, cascading through layers.
    ///
    /// On hit at layer N, the value is promoted to layer 0 (if N > 0)
    /// for faster future access.
    pub fn get(&self, part: Partition, key: &[u8]) -> Option<bytes::Bytes> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(value) = layer.get(part, key) {
                // Promote to faster layer on hit at deeper layer
                if i > 0 {
                    self.layers[0].put(part, key, &value, 1.0);
                }
                return Some(value);
            }
        }
        None
    }

    /// Write to the fastest cache layer with default weight (1.0).
    /// Evicted entries drain to slower layers.
    pub fn put(&self, part: Partition, key: &[u8], value: &[u8]) {
        self.put_weighted(part, key, value, 1.0);
    }

    /// Write to the fastest cache layer with a specific eviction weight.
    /// Higher weight = entry stays in cache longer during eviction.
    pub fn put_weighted(&self, part: Partition, key: &[u8], value: &[u8], weight: f32) {
        if self.layers.is_empty() {
            return;
        }

        let drained = self.layers[0].put(part, key, value, weight);
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

    /// Remove entry from all layers.
    pub fn remove(&self, part: Partition, key: &[u8]) {
        for layer in self.layers.iter() {
            layer.remove(part, key);
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
    fn compaction_loop(layers: &[CacheLayer], shutdown: &AtomicBool, interval: Duration) {
        debug!(
            interval_secs = interval.as_secs(),
            "background cache compaction started"
        );

        while !shutdown.load(Ordering::Relaxed) {
            // Sleep in small increments to respond to shutdown quickly
            let mut elapsed = Duration::ZERO;
            let tick = Duration::from_millis(500);
            while elapsed < interval {
                if shutdown.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(tick.min(interval - elapsed));
                elapsed += tick;
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

    /// Drain evicted entries to the next layer in the cascade.
    fn drain_to_next(&self, from_layer: usize, entries: Vec<(Partition, Vec<u8>, Vec<u8>)>) {
        let next = from_layer + 1;
        if next >= self.layers.len() || entries.is_empty() {
            return; // Bottom layer — drop (lossy cache)
        }

        for (part, key, value) in entries {
            // Drained entries get default weight (evicted = already "cooler").
            let further_drained = self.layers[next].put(part, &key, &value, 1.0);
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
