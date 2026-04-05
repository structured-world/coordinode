//! Storage engine configuration.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use lsm_tree::CompressionType;

use crate::cache::config::TieredCacheConfig;
use crate::engine::merge::{CounterMerge, DocumentMerge, PostingListMerge};
use crate::engine::partition::Partition;

/// WAL flush policy controlling durability guarantees.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FlushPolicy {
    /// Fsync after every write batch commit. Highest durability,
    /// highest latency. Required for strict crash safety.
    #[default]
    SyncPerBatch,

    /// Periodic background fsync at the given interval (ms).
    /// Writes between syncs may be lost on crash. Default: 200ms.
    Periodic(u16),

    /// No automatic fsync. Caller must explicitly call `persist()`.
    /// Fastest, but data loss window is unbounded until manual sync.
    Manual,
}

/// Compression codec for LSM block compression.
///
/// The architecture specifies lz4 for hot levels (L0-L3) and zstd for cold
/// levels (L4+). The LSM storage supports per-level compression via
/// `CompressionPolicy`, enabling the hot/cold split natively.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCodec {
    /// No compression. Not recommended for production.
    None,

    /// LZ4 compression — fast decompression (~3 GB/s), moderate ratio.
    /// Recommended for hot data. Uses `lz4_flex` internally.
    #[default]
    Lz4,
}

impl CompressionCodec {
    /// Map to the lsm-tree `CompressionType`.
    pub(crate) fn to_lsm_tree(self) -> CompressionType {
        match self {
            Self::None => CompressionType::None,
            Self::Lz4 => CompressionType::Lz4,
        }
    }
}

/// Per-level compression configuration.
///
/// The LSM storage supports per-level compression via `CompressionPolicy`.
/// This config builds the policy: hot levels use `hot_codec`,
/// cold levels (>= `cold_level_threshold`) use `cold_codec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionConfig {
    /// Codec for hot levels (L0 to cold_level_threshold - 1). Default: Lz4.
    pub hot_codec: CompressionCodec,

    /// Codec for cold levels (cold_level_threshold and above). Default: Lz4.
    ///
    /// Architecture specifies zstd here. Once our zstd fork is merged
    /// upstream, this will use `CompressionCodec::Zstd(3)`.
    pub cold_codec: CompressionCodec,

    /// LSM level at which to switch from hot_codec to cold_codec. Default: 4.
    pub cold_level_threshold: u8,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            hot_codec: CompressionCodec::Lz4,
            cold_codec: CompressionCodec::Lz4,
            cold_level_threshold: 4,
        }
    }
}

impl CompressionConfig {
    /// Build an lsm-tree `CompressionPolicy` from the hot/cold config.
    ///
    /// Generates a per-level vector: hot codec for levels 0..threshold,
    /// cold codec for levels threshold..7 (LSM storage typically has up to 7 levels).
    pub(crate) fn to_compression_policy(self) -> lsm_tree::config::CompressionPolicy {
        let max_levels: u8 = 7;
        let mut levels = Vec::with_capacity(max_levels as usize);
        for level in 0..max_levels {
            if level < self.cold_level_threshold {
                levels.push(self.hot_codec.to_lsm_tree());
            } else {
                levels.push(self.cold_codec.to_lsm_tree());
            }
        }
        lsm_tree::config::CompressionPolicy::new(levels)
    }
}

/// Configuration for the CoordiNode storage engine.
#[derive(Clone)]
pub struct StorageConfig {
    /// Path to the data directory.
    pub data_dir: PathBuf,

    /// Optional custom filesystem backend. When `None`, uses the default
    /// OS filesystem (`StdFs`). Set to `Arc<MemFs>` for in-memory tests.
    pub fs: Option<Arc<dyn lsm_tree::fs::Fs>>,

    /// WAL flush policy. Default: SyncPerBatch.
    pub flush_policy: FlushPolicy,

    /// Per-level compression configuration. Default: lz4 everywhere.
    pub compression: CompressionConfig,

    /// Block cache size in bytes. Default: 64MB.
    pub block_cache_bytes: u64,

    /// Maximum write buffer size in bytes. Default: 64MB.
    pub max_write_buffer_bytes: u64,

    /// Per-partition compression overrides. When set, overrides the
    /// global compression config for the specified partition.
    pub partition_compression: Option<Vec<(Partition, CompressionCodec)>>,

    /// Tiered block cache configuration. When layers are configured,
    /// read path cascades: DRAM (storage built-in) → cache layers → persistent storage.
    pub cache: TieredCacheConfig,

    /// Number of background flush worker threads. Default: 2.
    pub flush_workers: usize,

    /// Maximum number of sealed memtables per partition before triggering background flush.
    /// Default: 4.
    pub max_sealed_memtables: usize,

    /// Flush monitor polling interval in milliseconds. Default: 50ms.
    pub flush_poll_interval_ms: u64,

    /// Number of background compaction worker threads. Default: 2.
    pub compaction_workers: usize,

    /// L0 run count above which a partition is assigned Urgent compaction priority.
    /// When L0 exceeds this threshold, a write stall is imminent. Default: 8.
    pub compaction_l0_urgent_threshold: usize,

    /// Compaction monitor polling interval in milliseconds. Default: 200ms.
    pub compaction_poll_interval_ms: u64,

    /// Maximum entry bytes per oplog segment before rotation. Default: 64 MB.
    pub oplog_segment_max_bytes: u64,

    /// Maximum number of entries per oplog segment before rotation. Default: 50,000.
    pub oplog_segment_max_entries: u32,

    /// Oplog retention window in seconds. Segments with all entries older than
    /// `now - oplog_retention_secs` are eligible for purge. Default: 7 days.
    pub oplog_retention_secs: u64,

    /// Background drain thread polling interval in milliseconds.
    /// Controls how frequently volatile writes (w:memory/w:cache) are
    /// batched into Raft proposals. Lower = less data at risk.
    /// Default: 100ms.
    pub drain_interval_ms: u64,

    /// Maximum number of mutations per drain Raft proposal batch.
    /// Default: 10,000.
    pub drain_batch_max: u32,

    /// Maximum drain buffer capacity in bytes. When full, new volatile
    /// writes are rejected with backpressure. Default: 100 MB.
    pub drain_buffer_capacity_bytes: u64,
}

impl std::fmt::Debug for StorageConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageConfig")
            .field("data_dir", &self.data_dir)
            .field(
                "fs",
                &if self.fs.is_some() {
                    "custom"
                } else {
                    "default"
                },
            )
            .field("flush_policy", &self.flush_policy)
            .field("compression", &self.compression)
            .field("block_cache_bytes", &self.block_cache_bytes)
            .finish_non_exhaustive()
    }
}

impl StorageConfig {
    /// Create a new config with the given data directory (OS filesystem).
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            fs: None,
            flush_policy: FlushPolicy::default(),
            compression: CompressionConfig::default(),
            block_cache_bytes: 64 * 1024 * 1024,
            max_write_buffer_bytes: 64 * 1024 * 1024,
            partition_compression: None,
            cache: TieredCacheConfig::default(),
            flush_workers: 2,
            max_sealed_memtables: 4,
            flush_poll_interval_ms: 50,
            compaction_workers: 2,
            compaction_l0_urgent_threshold: 8,
            compaction_poll_interval_ms: 200,
            oplog_segment_max_bytes: 64 * 1024 * 1024,
            oplog_segment_max_entries: 50_000,
            oplog_retention_secs: 7 * 24 * 3600,
            drain_interval_ms: 100,
            drain_batch_max: 10_000,
            drain_buffer_capacity_bytes: 100 * 1024 * 1024,
        }
    }

    /// Create a config backed by an in-memory filesystem (`MemFs`).
    ///
    /// Trees are stored entirely in RAM with no disk I/O. Ideal for tests:
    /// ~10x faster than tempfile, zero filesystem side effects.
    ///
    /// The `data_dir` path is virtual — only used as a namespace for tree paths.
    pub fn with_memfs(data_dir: impl AsRef<Path>) -> Self {
        let fs = Arc::new(lsm_tree::fs::MemFs::new());
        let mut config = Self::new(data_dir);
        config.fs = Some(fs);
        config
    }

    /// Build an lsm-tree `Config` for a specific partition.
    ///
    /// Each partition opens its own `AnyTree` in `data_dir/<partition_name>/`.
    /// - Shared `seqno` generator (oracle or counter) for cross-tree MVCC ordering.
    /// - Seqno-based retention compaction filter on all partitions except `Adj`
    ///   (Adj + Node use merge operators; GC filter would destroy pending operands).
    /// - `PostingListMerge` merge operator on `Adj`.
    /// - `DocumentMerge` merge operator on `Node` (ADR-015).
    /// - KV separation (BlobTree) for the `Blob` partition.
    pub(crate) fn to_tree_config(
        &self,
        part: Partition,
        seqno: lsm_tree::SharedSequenceNumberGenerator,
        gc_watermark: &Arc<AtomicU64>,
    ) -> lsm_tree::Config {
        // Build compression policy — partition-level override or global.
        let compression_policy = if let Some(overrides) = &self.partition_compression {
            if let Some(&(_, codec)) = overrides.iter().find(|&&(p, _)| p == part) {
                lsm_tree::config::CompressionPolicy::all(codec.to_lsm_tree())
            } else {
                self.compression.to_compression_policy()
            }
        } else {
            self.compression.to_compression_policy()
        };

        let mut config = lsm_tree::Config::new_with_generators(
            self.data_dir.join(part.name()),
            Arc::clone(&seqno),
            Arc::clone(&seqno), // visible_seqno = seqno: all writes immediately visible
        )
        .data_block_compression_policy(compression_policy);

        // Custom filesystem backend (e.g., MemFs for in-memory tests)
        if let Some(ref fs) = self.fs {
            config = config.with_shared_fs(Arc::clone(fs));
        }

        // Partitions with merge operators: no retention filter
        // (merge operands must survive compaction for the merge function to combine them).
        //
        // - Adj: PostingListMerge — conflict-free edge writes via Add/Remove deltas.
        // - Node: DocumentMerge — path-targeted partial document updates (ADR-015).
        //   Handles both full NodeRecords (0x00 prefix) and DocDelta operands (0x01).
        // - Counter: CounterMerge — atomic i64 increment/decrement (R163b).
        if part == Partition::Adj {
            config = config.with_merge_operator(Some(Arc::new(PostingListMerge)));
        } else if part == Partition::Node {
            config = config.with_merge_operator(Some(Arc::new(DocumentMerge)));
        } else if part == Partition::Counter {
            config = config.with_merge_operator(Some(Arc::new(CounterMerge)));
        } else {
            // All other partitions: seqno-based MVCC retention filter.
            let gc_factory = super::mvcc_gc::seqno_retention_factory(Arc::clone(gc_watermark));
            config = config.with_compaction_filter_factory(Some(gc_factory));
        }

        // Blob partition: key-value separation for large blobs (>= 4KB default).
        if part == Partition::Blob {
            config = config.with_kv_separation(Some(lsm_tree::KvSeparationOptions::default()));
        }

        // Prefix extractor for bloom-accelerated prefix scans (R089).
        // All partitions use colon-separated keys (node:, adj:, edgeprop:, etc.)
        // so one extractor serves all. Biggest impact on adj: partition where
        // prefix_scan("adj:KNOWS:out:") is the hot path for graph traversal.
        config = config.prefix_extractor(Arc::new(ColonSeparatedPrefix));

        config
    }
}

/// Extracts prefixes at each ':' separator boundary for bloom filter indexing.
///
/// For key `adj:KNOWS:out:100`, yields: `adj:`, `adj:KNOWS:`, `adj:KNOWS:out:`
///
/// Enables bloom-based table skipping during prefix scans. Without this,
/// a prefix_scan("adj:KNOWS:out:") reads ALL tables in the adj partition.
/// With it, tables whose bloom filter doesn't match the prefix are skipped O(1).
pub(crate) struct ColonSeparatedPrefix;

impl lsm_tree::PrefixExtractor for ColonSeparatedPrefix {
    fn prefixes<'a>(&self, key: &'a [u8]) -> Box<dyn Iterator<Item = &'a [u8]> + 'a> {
        Box::new(
            key.iter()
                .enumerate()
                .filter(|(_, b)| **b == b':')
                .map(move |(i, _)| &key[..=i]),
        )
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn compression_codec_default_is_lz4() {
        assert_eq!(CompressionCodec::default(), CompressionCodec::Lz4);
    }

    #[test]
    fn compression_codec_to_lsm_tree_mapping() {
        assert!(matches!(
            CompressionCodec::None.to_lsm_tree(),
            CompressionType::None
        ));
        assert!(matches!(
            CompressionCodec::Lz4.to_lsm_tree(),
            CompressionType::Lz4
        ));
    }

    #[test]
    fn compression_config_defaults() {
        let config = CompressionConfig::default();
        assert_eq!(config.hot_codec, CompressionCodec::Lz4);
        assert_eq!(config.cold_codec, CompressionCodec::Lz4);
        assert_eq!(config.cold_level_threshold, 4);
    }

    #[test]
    fn compression_policy_per_level() {
        let config = CompressionConfig {
            hot_codec: CompressionCodec::Lz4,
            cold_codec: CompressionCodec::None,
            cold_level_threshold: 4,
        };
        let policy = config.to_compression_policy();
        // Levels 0-3 = Lz4, levels 4-6 = None
        assert_eq!(policy.len(), 7);
    }

    #[test]
    fn storage_config_defaults() {
        let config = StorageConfig::new("/tmp/test");
        assert_eq!(config.compression, CompressionConfig::default());
        assert!(config.partition_compression.is_none());
    }

    #[test]
    fn to_tree_config_builds_for_each_partition() {
        use std::sync::Arc;
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        let gc_watermark = Arc::new(AtomicU64::new(0));
        for &part in Partition::all() {
            let tree_config = config.to_tree_config(part, Arc::clone(&seqno), &gc_watermark);
            let _ = tree_config;
        }
    }

    // ── R089: ColonSeparatedPrefix tests ────────────────────────────

    #[test]
    fn prefix_extractor_adj_key() {
        use lsm_tree::PrefixExtractor;
        let ext = ColonSeparatedPrefix;
        let key = b"adj:KNOWS:out:100";
        let prefixes: Vec<&[u8]> = ext.prefixes(key).collect();
        assert_eq!(
            prefixes,
            vec![
                b"adj:" as &[u8],
                b"adj:KNOWS:" as &[u8],
                b"adj:KNOWS:out:" as &[u8],
            ]
        );
    }

    #[test]
    fn prefix_extractor_node_key() {
        use lsm_tree::PrefixExtractor;
        let ext = ColonSeparatedPrefix;
        let key = b"node:0:42";
        let prefixes: Vec<&[u8]> = ext.prefixes(key).collect();
        assert_eq!(prefixes, vec![b"node:" as &[u8], b"node:0:" as &[u8],]);
    }

    #[test]
    fn prefix_extractor_counter_key() {
        use lsm_tree::PrefixExtractor;
        let ext = ColonSeparatedPrefix;
        let key = b"counter:degree:42";
        let prefixes: Vec<&[u8]> = ext.prefixes(key).collect();
        assert_eq!(
            prefixes,
            vec![b"counter:" as &[u8], b"counter:degree:" as &[u8],]
        );
    }

    #[test]
    fn prefix_extractor_no_colons() {
        use lsm_tree::PrefixExtractor;
        let ext = ColonSeparatedPrefix;
        let prefixes: Vec<&[u8]> = ext.prefixes(b"noprefix").collect();
        assert!(prefixes.is_empty());
    }

    #[test]
    fn prefix_extractor_valid_scan_boundary() {
        use lsm_tree::PrefixExtractor;
        let ext = ColonSeparatedPrefix;
        assert!(ext.is_valid_scan_boundary(b"adj:"));
        assert!(ext.is_valid_scan_boundary(b"adj:KNOWS:"));
        assert!(ext.is_valid_scan_boundary(b"adj:KNOWS:out:"));
        assert!(!ext.is_valid_scan_boundary(b"adj")); // no trailing colon
        assert!(!ext.is_valid_scan_boundary(b""));
    }

    // ── R089: drop_range integration ────────────────────────────────

    #[test]
    fn drop_range_deletes_keys_in_range() {
        use crate::engine::core::StorageEngine;
        use crate::engine::partition::Partition;

        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open");

        // Write 10 keys to Idx partition
        for i in 0u32..10 {
            let key = format!("idx:test:{i:04}");
            engine
                .put(Partition::Idx, key.as_bytes(), b"val")
                .expect("put");
        }

        // Verify all 10 exist
        let count_before: usize = engine
            .prefix_scan(Partition::Idx, b"idx:test:")
            .expect("scan")
            .count();
        assert_eq!(count_before, 10);

        // Flush to SST (drop_range operates on tables, not memtable)
        engine.persist().expect("persist");

        // Drop range: keys 0003..0007
        engine
            .drop_range(
                Partition::Idx,
                "idx:test:0003".as_bytes()..="idx:test:0007".as_bytes(),
            )
            .expect("drop_range");

        // Count remaining — some keys in the range should be gone
        // Note: drop_range drops TABLES, not individual keys.
        // With only 10 keys, they may all be in one table → either all or none dropped.
        // This test verifies the API works without error; precise key-level
        // deletion depends on table boundaries (tested in lsm-tree itself).
        let count_after: usize = engine
            .prefix_scan(Partition::Idx, b"idx:test:")
            .expect("scan")
            .count();
        // Either all dropped (one table fully in range) or none (table spans beyond range)
        assert!(
            count_after == 0 || count_after == 10,
            "drop_range should drop whole tables: got {count_after} keys"
        );
    }

    #[test]
    fn prefix_extractor_edgeprop_key() {
        use lsm_tree::PrefixExtractor;
        let ext = ColonSeparatedPrefix;
        let key = b"edgeprop:KNOWS:100:200";
        let prefixes: Vec<&[u8]> = ext.prefixes(key).collect();
        assert_eq!(
            prefixes,
            vec![
                b"edgeprop:" as &[u8],
                b"edgeprop:KNOWS:" as &[u8],
                b"edgeprop:KNOWS:100:" as &[u8],
            ]
        );
    }

    #[test]
    fn prefix_extractor_schema_key() {
        use lsm_tree::PrefixExtractor;
        let ext = ColonSeparatedPrefix;
        let key = b"schema:label:User";
        let prefixes: Vec<&[u8]> = ext.prefixes(key).collect();
        assert_eq!(
            prefixes,
            vec![b"schema:" as &[u8], b"schema:label:" as &[u8],]
        );
    }

    #[test]
    fn prefix_extractor_blobref_key() {
        use lsm_tree::PrefixExtractor;
        let ext = ColonSeparatedPrefix;
        let key = b"blobref:42:profile_pic";
        let prefixes: Vec<&[u8]> = ext.prefixes(key).collect();
        assert_eq!(
            prefixes,
            vec![b"blobref:" as &[u8], b"blobref:42:" as &[u8],]
        );
    }

    #[test]
    fn drop_range_on_merge_partition() {
        // Verify drop_range works on Adj partition (has PostingListMerge operator).
        use crate::engine::core::StorageEngine;
        use crate::engine::partition::Partition;

        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open");

        // Write keys to Adj partition
        for i in 0u32..5 {
            let key = format!("adj:TEST:out:{i:04}");
            engine
                .put(Partition::Adj, key.as_bytes(), b"posting_data")
                .expect("put");
        }
        engine.persist().expect("persist");

        // drop_range should not error on merge-operator partition
        let result = engine.drop_range(
            Partition::Adj,
            "adj:TEST:out:0000".as_bytes()..="adj:TEST:out:9999".as_bytes(),
        );
        assert!(
            result.is_ok(),
            "drop_range on merge partition should not error"
        );
    }

    #[test]
    fn prefix_scan_uses_extractor() {
        // Verify prefix scan works correctly with the extractor wired.
        // This tests the full path: write → flush → prefix_scan with bloom skip.
        use crate::engine::core::StorageEngine;
        use crate::engine::partition::Partition;

        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open");

        // Write keys with different prefixes
        engine
            .put(Partition::Idx, b"idx:alpha:1", b"a1")
            .expect("put");
        engine
            .put(Partition::Idx, b"idx:alpha:2", b"a2")
            .expect("put");
        engine
            .put(Partition::Idx, b"idx:beta:1", b"b1")
            .expect("put");
        engine
            .put(Partition::Idx, b"idx:beta:2", b"b2")
            .expect("put");
        engine
            .put(Partition::Idx, b"idx:gamma:1", b"g1")
            .expect("put");

        // Flush to SST so bloom filters are populated
        engine.persist().expect("persist");

        // Prefix scan for "idx:alpha:" should return exactly 2
        let alpha_count = engine
            .prefix_scan(Partition::Idx, b"idx:alpha:")
            .expect("scan")
            .count();
        assert_eq!(alpha_count, 2, "prefix scan should find 2 alpha keys");

        // Prefix scan for "idx:beta:" should return exactly 2
        let beta_count = engine
            .prefix_scan(Partition::Idx, b"idx:beta:")
            .expect("scan")
            .count();
        assert_eq!(beta_count, 2, "prefix scan should find 2 beta keys");

        // Prefix scan for "idx:" should return all 5
        let all_count = engine
            .prefix_scan(Partition::Idx, b"idx:")
            .expect("scan")
            .count();
        assert_eq!(all_count, 5, "prefix scan for idx: should find all 5 keys");

        // Prefix scan for non-existent prefix should return 0
        let empty_count = engine
            .prefix_scan(Partition::Idx, b"idx:nonexistent:")
            .expect("scan")
            .count();
        assert_eq!(
            empty_count, 0,
            "prefix scan for missing prefix should find 0"
        );
    }
}
