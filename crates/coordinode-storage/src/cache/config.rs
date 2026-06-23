//! Tiered block cache configuration.

use std::collections::HashMap;
use std::path::PathBuf;

/// Configuration for a single cache layer (one file on one device).
#[derive(Debug, Clone)]
pub struct CacheLayerConfig {
    /// Path to the cache file on this device.
    pub path: PathBuf,

    /// Maximum size of this cache layer in bytes.
    pub max_bytes: u64,

    /// Maximum number of entries in this layer. Prevents unbounded
    /// in-memory index growth. Default: 1,000,000.
    pub max_entries: usize,

    /// Dead-space ratio that triggers compaction. Default: 0.5.
    pub compaction_threshold: f64,
}

impl CacheLayerConfig {
    /// Create a layer config with the given path and size.
    pub fn new(path: impl Into<PathBuf>, max_bytes: u64) -> Self {
        Self {
            path: path.into(),
            max_bytes,
            max_entries: 1_000_000,
            compaction_threshold: 0.5,
        }
    }
}

/// Configuration for the tiered block cache.
///
/// Layers are ordered fastest-first. On eviction from layer N,
/// the entry drains to layer N+1. On read miss at layer N,
/// layer N+1 is checked before going to persistent storage.
///
/// Each layer is volatile — power loss means cold cache, not data loss.
#[derive(Debug, Clone)]
pub struct TieredCacheConfig {
    /// Cache layers, ordered fastest → slowest.
    /// Empty = tiered cache disabled (only the storage engine's built-in DRAM cache).
    ///
    /// Typical CE deployment: one NVMe layer.
    /// ```toml
    /// [[cache.layers]]
    /// path = "/mnt/nvme-cache/coordinode.cache"
    /// max_bytes = 107374182400  # 100GB
    /// ```
    pub layers: Vec<CacheLayerConfig>,

    /// Interval between background compaction checks (seconds). Default: 60.
    /// The background thread checks each layer's dead-space ratio and
    /// compacts when it exceeds `compaction_threshold`.
    /// Set to 0 to disable background compaction.
    pub compaction_interval_secs: u64,

    /// Per-label eviction priority weights. Labels with higher weight stay
    /// in cache longer; labels with lower weight are evicted first.
    ///
    /// Default weight for unlisted labels is 1.0.
    ///
    /// ```toml
    /// [cache.label_weights]
    /// User = 2.0     # hot — stays in cache 2x longer
    /// Log = 0.1      # cold — evicted 10x sooner
    /// ```
    pub label_weights: HashMap<String, f32>,
}

impl Default for TieredCacheConfig {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            compaction_interval_secs: 60,
            label_weights: HashMap::new(),
        }
    }
}

impl TieredCacheConfig {
    /// Whether the tiered cache is enabled (at least one layer configured).
    pub fn is_enabled(&self) -> bool {
        !self.layers.is_empty()
    }
}

#[cfg(test)]
mod tests;
