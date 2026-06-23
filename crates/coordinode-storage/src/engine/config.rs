//! Storage engine configuration.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use lsm_tree::CompressionType;
use serde::{Deserialize, Serialize};

use crate::cache::config::TieredCacheConfig;
use crate::engine::merge::{CounterMerge, DocumentMerge, PostingListMerge};
use crate::engine::partition::Partition;

// ── Storage endpoint types (arch/core/storage-stack.md Layer 1 + ────────
//                            arch/placement/storage-endpoints.md)        ──

/// Physical media type backing a storage endpoint.
///
/// Metadata only — does not determine durability or redundancy semantics.
/// The same media can be used as `Durable` (RAID), `Degraded` (single drive),
/// or `Volatile` (cache file). The operator marks `durability` explicitly per
/// endpoint at config time.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Media {
    /// Spinning HDD.
    #[default]
    Hdd,
    /// SATA / SAS SSD.
    Ssd,
    /// NVMe SSD.
    Nvme,
    /// In-process RAM (memory shard, in-memory cache file).
    Ram,
}

/// Durability class of a storage endpoint — declares what redundancy the
/// underlying media provides and what cluster-level invariants apply to data
/// stored here.
///
/// Operator-marked at config time. NEVER inferred from media kind — the same
/// SSD can be `Durable` (in RAID-1), `Degraded` (single drive primary
/// storage), or `Volatile` (cache file). The placement engine reads this
/// flag to enforce the invariants below.
///
/// See [storage-stack.md](../../arch/core/storage-stack.md) §Cross-cutting
/// axis: Durability tri-state for the full invariant set (INV-D1..D4) and
/// the redundancy-mechanism mapping table.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Durability {
    /// Hardware-redundant media (RAID-1/5/6/10/60, replicated SAN, iSCSI
    /// with backend mirror). Crash / reboot leaves data intact. Can be the
    /// sole copy of data in the cluster (RF=1 allowed). Page-level ECC is
    /// OPTIONAL on `Durable` (RAID's own ECC already covers bit-rot).
    #[default]
    Durable,
    /// Single drive without hardware redundancy. Drive death = data loss
    /// unless protected by another mechanism. INV-D2: MUST have EITHER
    /// segment EC OR cluster replica configured. Page-level ECC is enabled
    /// by default to catch media read errors when no array-level recovery
    /// is available.
    Degraded,
    /// Volatile storage — any failure (power loss, process restart) erases
    /// the data. RAM, NVMe-as-cache, write-buffer NVMe. INV-D1: MUST have
    /// a `Durable` copy at cluster level. INV-D4: cluster survives loss of
    /// ALL `Volatile` endpoints without data loss. Segment EC NEVER applied
    /// to `Volatile` (chunks all on volatile = useless: power-off erases
    /// all chunks simultaneously).
    Volatile,
}

/// Storage tier within a server — drives placement preference between
/// endpoints of differing media + access cost.
///
/// Tier hierarchy (fastest first): `Memory > HotCache > Hot > Warm > Cold`.
/// Per-LSM-level routing maps L0-L1 → `Hot`-tier endpoint, L2-L3 →
/// `Warm`, L4+ → `Cold`. Cache layers (DRAM, NVMe cache file) are
/// orthogonal — they live in-process, not as `Volatile` endpoints.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// In-process RAM. Only `Volatile` endpoints belong here.
    Memory,
    /// NVMe-as-cache (`Volatile`). Drain-through cache files live here.
    HotCache,
    /// Hot data (NVMe or SSD typically, `Durable` or `Degraded`).
    Hot,
    /// Warm data (SSD or fast HDD, `Durable` or `Degraded`).
    #[default]
    Warm,
    /// Cold archival data (HDD typically, `Durable`).
    Cold,
}

/// Configuration of one storage endpoint — one mount point on the local
/// server, with its own media, durability profile, capacity, and tier.
///
/// This is the **Layer 1** primitive in the storage stack
/// ([storage-stack.md](../../arch/core/storage-stack.md)). Multiple
/// endpoints per node are the normal case (CoordiNode runs against 40-disk
/// JBODs routinely); the single-endpoint case is just a one-element
/// `endpoints` vec passed to [`StorageConfig::with_endpoints`].
///
/// The full physical layout — per-LSM-level placement (L0-L1 → NVMe, L4+ →
/// HDD), cascade eviction across endpoints, WAL/oplog segment routing — is
/// driven by the placement and capacity-tracking layers that consume this
/// type. This module establishes only the configuration surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointConfig {
    /// Unique identifier for this endpoint (operator-chosen; appears in
    /// metrics, logs, placement rules). Example: `"ep-3b"`, `"nvme-cache-0"`.
    /// MUST be non-empty and unique within a `StorageConfig.endpoints` list
    /// — validated in [`StorageConfig::with_endpoints`].
    pub id: String,
    /// Server identifier the endpoint physically lives on. `None` in CE
    /// single-node deployments (server identity is implicit — the process
    /// itself). EE multi-server deployments populate this from the cluster
    /// topology layer (R163) so endpoints in the same `StorageConfig` can
    /// be distinguished by physical host when topology-aware placement
    /// rules apply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server: Option<String>,
    /// Mount-point path on the local filesystem. The endpoint owns this
    /// directory tree exclusively — multiple endpoints MUST NOT share a
    /// path (validated in [`StorageConfig::with_endpoints`]). Special
    /// value `"memory"` (combined with `media: Ram`) marks an in-process
    /// endpoint with no filesystem backing.
    pub path: PathBuf,
    /// Physical media kind — metadata only, does not determine durability.
    pub media: Media,
    /// Durability class — operator-marked, drives invariants D1..D4.
    pub durability: Durability,
    /// Storage tier — drives placement preference.
    pub tier: Tier,
    /// Physical capacity in bytes. `0` means "untracked" — the placement
    /// engine cannot enforce INV-D3 without this. Optional in config:
    /// omitting it leaves the endpoint untracked.
    #[serde(default)]
    pub capacity_bytes: u64,
    /// Hard-limit in bytes — placement engine NEVER writes past this point
    /// (INV-D3). `0` means "no hard limit" (placement engine still observes
    /// filesystem capacity if known). When both `capacity_bytes` and
    /// `hard_limit_bytes` are non-zero, `hard_limit_bytes` MUST be
    /// `<= capacity_bytes` — validated in [`StorageConfig::with_endpoints`].
    /// Cascade eviction triggers at 95% of `hard_limit_bytes` when non-zero.
    /// Behaviour implemented by the hard-limit enforcement layer. Optional in
    /// config: omitting it means no hard limit.
    #[serde(default)]
    pub hard_limit_bytes: u64,
    /// Per-block Reed-Solomon ECC policy for SST blocks on this endpoint.
    ///
    /// `Auto` (the default) derives the effective policy from the
    /// [`Durability`] class — see
    /// [`PageEccPolicy::effective_for_durability`]:
    /// * `Durability::Degraded` → ECC **on** (single drive without RAID
    ///   has no array-level redundancy; ECC is the only recovery
    ///   mechanism for an unrecoverable read).
    /// * `Durability::Durable` → ECC **off** (RAID covers bit-rot at the
    ///   array level; disabling Page ECC saves ~5-15% CPU on the read
    ///   path).
    /// * `Durability::Volatile` → ECC **off** (entire endpoint can
    ///   vanish; per-block ECC pointless).
    ///
    /// `ForceOn` / `ForceOff` override the auto rule (operator decision).
    /// `ForceOn` on a `Volatile` endpoint is legal but wasteful — the
    /// engine does NOT reject it; the operator is assumed to know why.
    ///
    /// The encoder/decoder lives in `coordinode-lsm-tree` and is gated
    /// behind a build-time feature flag there. Until the upstream flag
    /// lands, this field is the **config surface only** — flipping it
    /// has no on-disk effect.
    #[serde(default)]
    pub page_ecc: PageEccPolicy,
    /// Strategy for handling writes that would exceed
    /// [`hard_limit_bytes`](Self::hard_limit_bytes). See
    /// [`HardLimitStrategy`]. Default `Reject`.
    #[serde(default)]
    pub hard_limit_strategy: HardLimitStrategy,
    /// Free-form tags for placement rules (`{"zone": "eu-west-1a",
    /// "rack": "r42"}`). Consumed by the cluster-topology / CRUSH layer.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub tags: std::collections::BTreeMap<String, String>,
}

/// Per-block Reed-Solomon ECC policy on the SST block format.
///
/// The effective policy for an endpoint with `PageEccPolicy::Auto` is
/// derived from [`Durability`] at engine open time — see
/// [`Self::effective_for_durability`].
///
/// SST block layout when ECC is enabled (sketch):
///
/// ```text
/// [4 KB user payload] [4 B page xxh3 checksum] [N B Reed-Solomon ECC trailer]
/// ```
///
/// The page xxh3 checksum is **always present** regardless of ECC
/// policy — its byte position is fixed so a reader can verify
/// corruption before deciding whether to attempt ECC recovery.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageEccPolicy {
    /// Derive the effective policy from [`Durability`] at open time.
    /// This is the recommended default — operators rarely need to
    /// override.
    #[default]
    Auto,
    /// ECC always written + verified on this endpoint, regardless of
    /// durability class. Use when a durability-`durable` endpoint backs
    /// hardware known to drop bits silently (operator judgment).
    ForceOn,
    /// ECC never written or expected on this endpoint, regardless of
    /// durability class. Use when CPU is the bottleneck and the
    /// operator accepts the risk on a `degraded` endpoint, or when a
    /// `durable` endpoint is on certified-corruption-proof media and
    /// the auto-on rule changes in a future revision.
    ForceOff,
}

impl PageEccPolicy {
    /// Resolve the effective on/off decision for the given durability.
    ///
    /// The Auto-derive table:
    ///
    /// | durability  | Auto effective | rationale                       |
    /// |-------------|----------------|---------------------------------|
    /// | Durable     | off            | RAID covers bit-rot; CPU savings |
    /// | Degraded    | on             | only recovery mechanism         |
    /// | Volatile    | off            | endpoint can vanish; ECC pointless |
    ///
    /// `ForceOn` and `ForceOff` short-circuit the durability check.
    #[must_use]
    pub fn effective_for_durability(self, durability: Durability) -> bool {
        match self {
            Self::ForceOn => true,
            Self::ForceOff => false,
            Self::Auto => matches!(durability, Durability::Degraded),
        }
    }
}

/// Per-endpoint strategy when a write would push `used_bytes` past the
/// configured `hard_limit_bytes`.
///
/// `Reject` is the default — a write that cannot land here surfaces as
/// a `CapacityExhausted` error so the coordinator can try another
/// endpoint or fail back to the client. `CascadeEvict` instead asks
/// the tier-aware layer to demote SSTs from this endpoint to the next
/// cooler endpoint (per the per-LSM-level routing) and then retry.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardLimitStrategy {
    /// Fail the write with `StorageError::CapacityExhausted`.
    /// Coordinator may retry on a different endpoint per the
    /// tier-fallback policy. Safe default — never moves data on
    /// behalf of the operator.
    #[default]
    Reject,
    /// Trigger a cascade eviction targeting this endpoint when it
    /// crosses the auto-cascade threshold (95% of `hard_limit_bytes`),
    /// then continue with the write. Eviction runs in the background
    /// via the per-LSM-level routing's bottom-level compaction;
    /// individual writes do NOT block on it.
    CascadeEvict,
}

impl EndpointConfig {
    /// Construct an endpoint with required fields. `server` defaults to
    /// `None` (CE single-node); set via [`Self::with_server`] for EE
    /// multi-server. `capacity_bytes` / `hard_limit_bytes` default to `0`
    /// (untracked / no limit). Tags default to empty.
    pub fn new(
        id: impl Into<String>,
        path: impl AsRef<Path>,
        media: Media,
        durability: Durability,
        tier: Tier,
    ) -> Self {
        Self {
            id: id.into(),
            server: None,
            path: path.as_ref().to_path_buf(),
            media,
            durability,
            tier,
            capacity_bytes: 0,
            hard_limit_bytes: 0,
            page_ecc: PageEccPolicy::default(),
            hard_limit_strategy: HardLimitStrategy::default(),
            tags: std::collections::BTreeMap::new(),
        }
    }

    /// Effective ECC on/off decision for this endpoint, resolving
    /// [`PageEccPolicy::Auto`] against the configured [`Durability`].
    /// See [`PageEccPolicy::effective_for_durability`] for the table.
    #[must_use]
    pub fn is_page_ecc_enabled(&self) -> bool {
        self.page_ecc.effective_for_durability(self.durability)
    }

    /// Set the server identifier (EE multi-server topology).
    pub fn with_server(mut self, server: impl Into<String>) -> Self {
        self.server = Some(server.into());
        self
    }

    /// Set the physical capacity. `0` means untracked.
    pub fn with_capacity_bytes(mut self, capacity_bytes: u64) -> Self {
        self.capacity_bytes = capacity_bytes;
        self
    }

    /// Set the hard limit. `0` means no hard limit.
    pub fn with_hard_limit_bytes(mut self, hard_limit_bytes: u64) -> Self {
        self.hard_limit_bytes = hard_limit_bytes;
        self
    }

    /// Insert a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Override the per-block ECC policy. Default is
    /// [`PageEccPolicy::Auto`] — see [`PageEccPolicy`] for the
    /// durability-derived effective decision.
    #[must_use]
    pub fn with_page_ecc(mut self, policy: PageEccPolicy) -> Self {
        self.page_ecc = policy;
        self
    }

    /// Override the hard-limit strategy. Default is
    /// [`HardLimitStrategy::Reject`].
    #[must_use]
    pub fn with_hard_limit_strategy(mut self, strategy: HardLimitStrategy) -> Self {
        self.hard_limit_strategy = strategy;
        self
    }

    /// WAL eligibility predicate ([storage-stack.md](../../arch/core/storage-stack.md) Layer 1↔2):
    /// endpoint is eligible to host the standalone WAL iff it offers fast
    /// sequential writes (NVMe/SSD media OR `Hot` tier) AND it is non-volatile.
    /// `HotCache` is **not** eligible — it is a volatile-by-design RAM/NVMe
    /// cache tier and would have failed the non-volatile check anyway; we
    /// keep it out of the "fast" set so the predicate's intent stays clear
    /// (HotCache is for read acceleration, not durable WAL persistence).
    /// Volatile endpoints (RAM, NVMe-as-cache) are rejected because the WAL
    /// must survive process restart by definition.
    pub fn is_wal_eligible(&self) -> bool {
        let fast = self.tier == Tier::Hot || self.media == Media::Nvme || self.media == Media::Ssd;
        let non_volatile = self.durability != Durability::Volatile;
        fast && non_volatile
    }

    /// Oplog eligibility predicate ([storage-stack.md](../../arch/core/storage-stack.md) Layer 1↔2,
    /// [storage-endpoints.md](../../arch/placement/storage-endpoints.md) INV-D1):
    /// endpoint is eligible to host oplog segments iff its durability class
    /// guarantees survival of a process restart. `Volatile` is rejected
    /// (segments lost on restart = consensus log lost = data loss). Both
    /// `Durable` and `Degraded` are eligible — `Degraded` is acceptable
    /// here because oplog segments are consensus-replicated across nodes
    /// (single-disk failure on one replica is recoverable from peers).
    pub fn is_oplog_eligible(&self) -> bool {
        matches!(self.durability, Durability::Durable | Durability::Degraded)
    }
}

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
    /// Architecture specifies zstd here. lsm-tree's `zstd` feature now provides
    /// zstd via pure-Rust `structured-zstd` (no C FFI); wiring a
    /// `CompressionCodec::Zstd(3)` variant over it is the follow-up.
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
    /// Storage endpoints — one per physical mount point on the local
    /// server. Multi-endpoint is the normal case (40-disk JBODs); the
    /// single-endpoint case is just a one-element vec passed to
    /// [`StorageConfig::with_endpoints`].
    ///
    /// **Configuration surface only:** this field carries the endpoint
    /// list. Concrete placement decisions — per-LSM-level routing
    /// (L0-L1 → NVMe, L4+ → HDD), cascade eviction, WAL/oplog endpoint
    /// routing, page ECC — are made by the placement, capacity, and
    /// redundancy layers consuming this type.
    ///
    /// **Invariant** (checked at engine open): MUST be non-empty.
    pub endpoints: Vec<EndpointConfig>,

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

    /// Maximum age (in seconds) of a non-empty active memtable before it is
    /// rotated regardless of size. Bounds the worst-case window during which
    /// committed mutations may live in volatile memory only — critical at low
    /// or bursty load where the size threshold (`max_write_buffer_bytes`) may
    /// never fire on the larger partitions. Pairs with the cross-partition
    /// oplog purge gate (see `OplogManager::purge_before`) to keep oplog
    /// retention bounded while preserving crash safety.
    ///
    /// Set to `0` to disable the time-based trigger (size-based only — the
    /// pre-R076b behavior). Default: 30.
    pub max_memtable_age_secs: u64,

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

    /// Directory for the NVMe-backed write buffer used by `w:cache` write concern.
    ///
    /// When set, `w:cache` writes are persisted to this directory before ACK,
    /// enabling recovery of undraiend entries after a process crash (not power
    /// failure). Two files are managed in this directory:
    /// - `write_buffer_current.bin` — active write buffer being appended
    /// - `write_buffer_draining.NNN.bin` — checkpoint in progress (drain thread)
    ///
    /// When `None` (default), `w:cache` is treated identically to `w:memory`.
    pub nvme_write_buffer_path: Option<PathBuf>,
}

impl std::fmt::Debug for StorageConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageConfig")
            .field("endpoints", &self.endpoints)
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
    /// Construct from an explicit list of endpoints. This is the **only**
    /// `StorageConfig` constructor — pre-public-release-window storage
    /// format forces every caller to declare endpoints explicitly. No
    /// `::new(path)` convenience shim, no `::with_memfs(path)` shortcut:
    /// tests live below this layer, exercising the real API; production
    /// code declares production endpoints.
    ///
    /// Validation (panics on violation — config errors are caller bugs):
    /// - `endpoints` MUST be non-empty.
    /// - Endpoint `id` strings MUST be non-empty and unique within the
    ///   list. Duplicate IDs reject because metrics and placement rules
    ///   key by `id`.
    /// - Endpoint `path`s MUST be unique within the list — endpoints own
    ///   their directory trees exclusively.
    /// - When both `capacity_bytes` and `hard_limit_bytes` are non-zero on
    ///   the same endpoint, `hard_limit_bytes <= capacity_bytes` MUST
    ///   hold.
    ///
    /// All other config fields receive their defaults; chain `.with_*`
    /// builders or assign fields directly to customise.
    pub fn with_endpoints(endpoints: Vec<EndpointConfig>) -> Self {
        Self::validate_common(&endpoints);
        // INV-D1 (config-time): at least one oplog-eligible endpoint.
        // Oplog = Raft log = WAL = CDC source (ADR-017) MUST survive
        // process restart, so a config with no durable/degraded endpoints
        // cannot host a production storage engine. Tests that genuinely
        // need an all-volatile (MemFs) config must use
        // [`Self::with_endpoints_no_persistence`].
        assert!(
            endpoints.iter().any(EndpointConfig::is_oplog_eligible),
            "StorageConfig requires at least one oplog-eligible endpoint \
             (durability ∈ {{Durable, Degraded}}) — got only Volatile \
             endpoints. Use `with_endpoints_no_persistence` for in-memory \
             test configs that don't open Raft/WAL."
        );
        Self::build(endpoints)
    }

    /// Construct a StorageConfig **without** the INV-D1 persistence
    /// check — caller asserts the engine will NOT open Raft (`LogStore`)
    /// or the standalone WAL (`StorageEngine::open_with_wal`).
    ///
    /// Intended for `MemFs`-backed in-memory engines used in unit tests,
    /// migration tools, and pure read-path benchmarks. All other config
    /// validation (id/path uniqueness, hard_limit ≤ capacity) still runs.
    ///
    /// If a caller created via this constructor later opens `LogStore`
    /// or `open_with_wal`, those calls return an
    /// [`EndpointSelectionError`] — the runtime guard catches misuse.
    pub fn with_endpoints_no_persistence(endpoints: Vec<EndpointConfig>) -> Self {
        Self::validate_common(&endpoints);
        Self::build(endpoints)
    }

    fn validate_common(endpoints: &[EndpointConfig]) {
        assert!(
            !endpoints.is_empty(),
            "StorageConfig requires at least one endpoint"
        );
        // ID uniqueness + non-empty.
        let mut seen_ids: std::collections::HashSet<&str> =
            std::collections::HashSet::with_capacity(endpoints.len());
        for ep in endpoints {
            assert!(
                !ep.id.is_empty(),
                "EndpointConfig.id must be non-empty (path: {:?})",
                ep.path,
            );
            assert!(
                seen_ids.insert(ep.id.as_str()),
                "duplicate EndpointConfig.id: {:?}",
                ep.id,
            );
        }
        // Path uniqueness.
        let mut seen_paths: std::collections::HashSet<&Path> =
            std::collections::HashSet::with_capacity(endpoints.len());
        for ep in endpoints {
            assert!(
                seen_paths.insert(ep.path.as_path()),
                "duplicate EndpointConfig.path: {:?} (id: {:?})",
                ep.path,
                ep.id,
            );
        }
        // hard_limit ≤ capacity when both > 0.
        for ep in endpoints {
            if ep.capacity_bytes > 0 && ep.hard_limit_bytes > 0 {
                assert!(
                    ep.hard_limit_bytes <= ep.capacity_bytes,
                    "EndpointConfig.hard_limit_bytes ({}) > capacity_bytes ({}) on {:?}",
                    ep.hard_limit_bytes,
                    ep.capacity_bytes,
                    ep.id,
                );
            }
        }
    }

    fn build(endpoints: Vec<EndpointConfig>) -> Self {
        Self {
            endpoints,
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
            max_memtable_age_secs: 30,
            compaction_workers: 2,
            compaction_l0_urgent_threshold: 8,
            compaction_poll_interval_ms: 200,
            oplog_segment_max_bytes: 64 * 1024 * 1024,
            oplog_segment_max_entries: 50_000,
            oplog_retention_secs: 7 * 24 * 3600,
            drain_interval_ms: 100,
            drain_batch_max: 10_000,
            drain_buffer_capacity_bytes: 100 * 1024 * 1024,
            nvme_write_buffer_path: None,
        }
    }

    /// Builder: attach a custom filesystem backend (e.g. `MemFs` for
    /// in-memory testing). Default backend is the OS filesystem.
    pub fn with_fs(mut self, fs: Arc<dyn lsm_tree::fs::Fs>) -> Self {
        self.fs = Some(fs);
        self
    }

    /// Primary data directory — the path of the first configured endpoint.
    ///
    /// Kept as a convenience accessor for diagnostics and the single-
    /// endpoint case. Subsystems that need a specific endpoint per their
    /// placement contract (WAL → [`Self::select_wal_endpoint`], oplog →
    /// [`Self::select_oplog_endpoint`], partition trees → per-LSM-level
    /// routing) MUST use those methods rather than `data_dir`.
    pub fn data_dir(&self) -> &Path {
        // SAFETY: with_endpoints asserts non-empty.
        &self.endpoints[0].path
    }

    /// Select the endpoint that hosts the **standalone WAL** for this
    /// instance ([storage-stack.md](../../arch/core/storage-stack.md) Layer 1↔2).
    ///
    /// Picks the first endpoint matching [`EndpointConfig::is_wal_eligible`].
    /// WAL is non-replicated single-file storage; spreading it across
    /// endpoints does not improve throughput (sequential append + fsync
    /// is the bottleneck, not disk parallelism), so deterministic
    /// first-match keeps recovery simple.
    ///
    /// Returns `Err(WalNoEligibleEndpoint)` if no endpoint qualifies —
    /// the caller (typically standalone deployment) must reconfigure
    /// with at least one non-volatile NVMe/SSD/Hot-tier endpoint, or
    /// run without a standalone WAL (Raft-only mode).
    pub fn select_wal_endpoint(&self) -> Result<&EndpointConfig, EndpointSelectionError> {
        self.endpoints
            .iter()
            .find(|ep| ep.is_wal_eligible())
            .ok_or(EndpointSelectionError::NoWalEligible)
    }

    /// Select the endpoint that hosts oplog segments for `shard_id`
    /// ([storage-stack.md](../../arch/core/storage-stack.md) Layer 1↔2,
    /// [storage-endpoints.md](../../arch/placement/storage-endpoints.md) INV-D1).
    ///
    /// Round-robin within the oplog-eligible set, keyed by `shard_id`.
    /// CE single-shard with a single oplog-eligible endpoint always
    /// returns that endpoint. Multi-shard EE distributes shards' oplogs
    /// across eligible endpoints by `shard_id % count` — deterministic
    /// per-shard choice survives engine restarts without re-routing.
    ///
    /// Returns `Err(NoOplogEligible)` if no endpoint qualifies. This
    /// condition is rejected at config time by [`Self::with_endpoints`],
    /// so encountering it at runtime indicates a programming error
    /// (e.g., a caller mutated the endpoint list outside the
    /// constructor).
    pub fn select_oplog_endpoint(
        &self,
        shard_id: u32,
    ) -> Result<&EndpointConfig, EndpointSelectionError> {
        let eligible: Vec<&EndpointConfig> = self
            .endpoints
            .iter()
            .filter(|ep| ep.is_oplog_eligible())
            .collect();
        if eligible.is_empty() {
            return Err(EndpointSelectionError::NoOplogEligible);
        }
        // SAFETY: eligible non-empty checked above.
        Ok(eligible[(shard_id as usize) % eligible.len()])
    }

    /// All WAL-eligible endpoints — for recovery scan after a config
    /// change. The active WAL is at [`Self::select_wal_endpoint`]; on
    /// restart, the engine inspects every WAL-eligible endpoint for
    /// orphaned `wal/` directories left by a previous config that
    /// selected a different active endpoint.
    pub fn all_wal_eligible_endpoints(&self) -> Vec<&EndpointConfig> {
        self.endpoints
            .iter()
            .filter(|ep| ep.is_wal_eligible())
            .collect()
    }

    /// All oplog-eligible endpoints — for recovery scan. Active oplog
    /// segments are written to [`Self::select_oplog_endpoint`]; on
    /// restart, the engine scans every oplog-eligible endpoint's
    /// `oplog/<shard_id>/` directory for sealed segments left there
    /// before a config-driven endpoint re-route.
    pub fn all_oplog_eligible_endpoints(&self) -> Vec<&EndpointConfig> {
        self.endpoints
            .iter()
            .filter(|ep| ep.is_oplog_eligible())
            .collect()
    }
}

/// Errors returned by [`StorageConfig::select_wal_endpoint`] /
/// [`StorageConfig::select_oplog_endpoint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointSelectionError {
    /// No endpoint matches WAL eligibility criteria — non-volatile
    /// AND (Hot tier OR NVMe/SSD media).
    NoWalEligible,
    /// No endpoint matches oplog eligibility criteria — durability
    /// must be `Durable` or `Degraded` (rejected at config time, so
    /// runtime occurrence implies a programming bug).
    NoOplogEligible,
}

impl std::fmt::Display for EndpointSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoWalEligible => write!(
                f,
                "no WAL-eligible endpoint in config (need non-volatile + Hot tier OR Nvme/Ssd media)"
            ),
            Self::NoOplogEligible => write!(
                f,
                "no oplog-eligible endpoint in config (need Durable or Degraded durability)"
            ),
        }
    }
}

impl std::error::Error for EndpointSelectionError {}

impl StorageConfig {
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
        self.to_tree_config_with_routing(part, seqno, gc_watermark, None)
    }

    /// As [`Self::to_tree_config`] but with explicit per-LSM-level
    /// endpoint routing. When `routing` is `Some`, the lsm-tree `Config`
    /// is wired with `LevelRoute`s built from the routing — SST files
    /// for each LSM level land at the endpoint chosen for that level.
    /// When `routing` is `None`, the partition uses single-tier
    /// behaviour (all SSTs at the primary endpoint) — the bootstrap
    /// path for the Schema partition and the test path for engines
    /// opened without persistence.
    pub(crate) fn to_tree_config_with_routing(
        &self,
        part: Partition,
        seqno: lsm_tree::SharedSequenceNumberGenerator,
        gc_watermark: &Arc<AtomicU64>,
        routing: Option<&crate::engine::routing::PartitionRouting>,
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

        // The lsm-tree primary `Config.path` lives at the first
        // configured endpoint — that endpoint hosts the partition's
        // manifest. SST files land at per-level paths derived from
        // `routing`: L0-L1 on the hot endpoint, L2-L3 on warm, L4-L6
        // on cold. Levels routed to the primary endpoint fall through
        // to `Config.path` (no explicit `LevelRoute` emitted).
        //
        // When `routing` is `None` — the bootstrap case (Schema
        // partition, tests using `with_endpoints_no_persistence`) —
        // the partition stays single-tier on the primary endpoint.
        let primary_endpoint_id = self.endpoints[0].id.clone();
        let partition_dir = self.endpoints[0].path.join(part.name());
        let mut config = lsm_tree::Config::new_with_generators(
            partition_dir,
            Arc::clone(&seqno),
            Arc::clone(&seqno), // visible_seqno = seqno: all writes immediately visible
        )
        .data_block_compression_policy(compression_policy);

        // Custom filesystem backend (e.g., MemFs for in-memory tests)
        let fs_for_routes: Arc<dyn lsm_tree::fs::Fs> = if let Some(ref fs) = self.fs {
            config = config.with_shared_fs(Arc::clone(fs));
            Arc::clone(fs)
        } else {
            Arc::new(lsm_tree::fs::StdFs)
        };

        // Wire per-LSM-level endpoint routing into lsm-tree config.
        if let Some(routing) = routing {
            let routes = routing.to_level_routes(
                &self.endpoints,
                part.name(),
                &primary_endpoint_id,
                Arc::clone(&fs_for_routes),
            );
            if !routes.is_empty() {
                config = config.level_routes(routes);
            }
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
        // VectorF32 truth tier: 4 KB per 1024d vector fits the
        // "value-dominated row" pattern that KV separation is designed
        // for (segregates large values into a separate file, keeps the
        // main LSM bloom-able and seek-efficient).
        if matches!(part, Partition::Blob | Partition::VectorF32) {
            config = config.with_kv_separation(Some(lsm_tree::KvSeparationOptions::default()));
        }

        // Prefix extractor for bloom-accelerated prefix scans (R089).
        // All partitions use colon-separated keys (node:, adj:, edgeprop:, etc.)
        // so one extractor serves all. Biggest impact on adj: partition where
        // prefix_scan("adj:KNOWS:out:") is the hot path for graph traversal.
        config = config.prefix_extractor(Arc::new(ColonSeparatedPrefix));

        // Per-block Reed-Solomon page ECC (R159 Part C). The lsm-tree
        // flag is per-tree, but the ECC policy is per-endpoint, and a
        // tree's levels can span endpoints of different durability via
        // routing. Take the most-protective decision: enable ECC for the
        // tree if ANY endpoint it spans requires it (`Degraded` under
        // Auto, or `ForceOn`). Over-protecting a `Durable` level is just
        // spare parity bytes; under-protecting a `Degraded` one would
        // lose the only in-place recovery mechanism.
        //
        // Build-time gated: without the `page_ecc` feature the codec is
        // absent from lsm-tree, so we must not request it (open would
        // return `PageEccUnsupported`). Off-by-default is the accepted
        // pre-freeze state — the page xxh3 checksum is always present
        // regardless, only the Reed-Solomon trailer is gated.
        #[cfg(feature = "page_ecc")]
        {
            let ecc_required = self
                .endpoints
                .iter()
                .any(EndpointConfig::is_page_ecc_enabled);
            config = config.page_ecc(ecc_required);
        }

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
mod tests;
