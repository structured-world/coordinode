//! Unified server configuration facade.
//!
//! `ServerConfig` is the single in-code source and gate for every tunable: the
//! rest of the binary reads settings only from a resolved `ServerConfig`, never
//! from raw CLI flags or env directly. It is resolved by layering, lowest to
//! highest precedence:
//!
//! 1. Built-in defaults ([`ServerConfig::default`]).
//! 2. The YAML config file (`--config <path>`), if given.
//! 3. Command-line flag overrides ([`CliOverrides`]).
//!
//! So the command line overrides the config file, which overrides the defaults.
//!
//! Not every knob gets a CLI flag. The CLI carries only bootstrap-critical
//! settings (bind addresses, node id, data
//! dir, mode, peers, `--config`) — the argv length is OS-bounded (`ARG_MAX`).
//! Fine tunables live in the YAML config file only: add such a knob to
//! `ServerConfig` (+ its default) and the packaged `coordinode.conf`, and skip
//! `CliOverrides` / `apply_overrides`. A knob that also gets a CLI flag (the
//! bootstrap-critical ones) is the one added in all three places.

use std::collections::BTreeMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

use coordinode_storage::engine::config::{
    CompressionCodec, Durability, EndpointConfig, EndpointConfigError, Media, StorageConfig,
    SyncMethod, Tier,
};
use coordinode_storage::engine::partition::Partition;
use serde::Deserialize;

/// Storage topology: the physical endpoints this node manages.
///
/// An endpoint is one mount point with its own media, durability class, tier,
/// capacity, and per-block ECC policy (see [`EndpointConfig`]).
/// Declaring more than one endpoint is the multi-disk case (CoordiNode runs
/// against 40-disk JBODs routinely); the per-LSM-level placement, cascade
/// eviction, and WAL/oplog routing across them are driven by the storage layer
/// that consumes this list.
///
/// Empty (the default) means "derive a single endpoint from `data_dir`": a
/// durable HDD warm-tier endpoint named `default` rooted at the configured
/// data directory. This keeps the common single-disk deployment a one-liner
/// (`data_dir`) while letting a production operator declare the full topology
/// explicitly in the config file. Because a topology is a list of structured
/// records, it lives in the YAML config file only; the `--data` command-line
/// flag configures the single-endpoint case (and overrides any file topology,
/// see [`ServerConfig::apply_overrides`]).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageTopology {
    /// Explicit endpoint list. Empty = derive a single durable HDD warm-tier
    /// endpoint rooted at [`ServerConfig::data_dir`].
    pub endpoints: Vec<EndpointConfig>,
    /// Write-backpressure thresholds (L0 height and pending-compaction byte
    /// debt, each with a slowdown and a stop level). Slowdown raises
    /// compaction priority and metrics only; Stop rejects new client writes
    /// at commit with a retryable error until compaction catches up. Nothing
    /// ever sleeps on the verdict.
    #[serde(default)]
    pub backpressure: coordinode_storage::engine::config::BackpressureLimits,
    /// The oplog (Raft log, or the embedded journal of a standalone node):
    /// segment rotation, retention and how an append is made durable.
    #[serde(default)]
    pub oplog: OplogSettings,
    /// Block compression of the storage partitions' tables.
    #[serde(default)]
    pub compression: CompressionSettings,
}

/// Block compression of the storage partitions, from the config file; an
/// unset key keeps the engine default (lz4 on the hot levels, zstd level 3
/// from level 4 down). A table keeps the codec it was written with, so a
/// change reaches existing data as compaction rewrites it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(try_from = "CompressionFile")]
pub struct CompressionSettings {
    /// Codec of the levels above `cold_level_threshold`.
    pub hot: Option<CompressionCodec>,
    /// Codec of `cold_level_threshold` and the levels below it.
    pub cold: Option<CompressionCodec>,
    /// First level that takes the cold codec (`0..=7`).
    pub cold_level_threshold: Option<u8>,
    /// Partitions compressed with one codec at every level instead.
    pub partitions: Vec<(Partition, CompressionCodec)>,
}

/// `storage.compression` as written in the file.
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct CompressionFile {
    hot: Option<CodecFile>,
    cold: Option<CodecFile>,
    cold_level_threshold: Option<u8>,
    partitions: BTreeMap<String, CodecFile>,
}

/// One codec as written in the file: `{ codec: zstd, level: 19 }`.
#[derive(Deserialize, Clone, Copy)]
#[serde(deny_unknown_fields)]
struct CodecFile {
    codec: CodecName,
    level: Option<i32>,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum CodecName {
    None,
    Lz4,
    Zstd,
}

/// Zstd level when the file names zstd without one: the library's default.
const DEFAULT_ZSTD_LEVEL: i32 = 3;

impl CodecFile {
    fn resolve(self, key: &str) -> Result<CompressionCodec, String> {
        match (self.codec, self.level) {
            (CodecName::None, None) => Ok(CompressionCodec::None),
            (CodecName::Lz4, None) => Ok(CompressionCodec::Lz4),
            (CodecName::Zstd, level) => CompressionCodec::zstd(level.unwrap_or(DEFAULT_ZSTD_LEVEL))
                .map_err(|e| format!("{key}: {e}")),
            (CodecName::None | CodecName::Lz4, Some(_)) => {
                Err(format!("{key}: a level is only meaningful for zstd"))
            }
        }
    }
}

impl TryFrom<CompressionFile> for CompressionSettings {
    type Error = String;

    fn try_from(file: CompressionFile) -> Result<Self, String> {
        // Levels run 0..=6, so 7 puts every level on the hot codec.
        if let Some(threshold) = file.cold_level_threshold {
            if threshold > 7 {
                return Err(format!(
                    "storage.compression.cold_level_threshold: {threshold} is past the last level (7)"
                ));
            }
        }
        let mut partitions = Vec::with_capacity(file.partitions.len());
        for (name, codec) in file.partitions {
            let partition = Partition::all()
                .iter()
                .copied()
                .find(|p| p.name() == name)
                .ok_or_else(|| {
                    let known: Vec<&str> = Partition::all().iter().map(|p| p.name()).collect();
                    format!(
                        "storage.compression.partitions: unknown partition '{name}' (known: {})",
                        known.join(", ")
                    )
                })?;
            partitions.push((
                partition,
                codec.resolve(&format!("storage.compression.partitions.{name}"))?,
            ));
        }
        Ok(Self {
            hot: file
                .hot
                .map(|c| c.resolve("storage.compression.hot"))
                .transpose()?,
            cold: file
                .cold
                .map(|c| c.resolve("storage.compression.cold"))
                .transpose()?,
            cold_level_threshold: file.cold_level_threshold,
            partitions,
        })
    }
}

/// Oplog settings from the config file; an unset key keeps the engine default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OplogSettings {
    /// Bytes of entries per segment before it rotates (`None` = 64 MiB).
    pub segment_max_bytes: Option<NonZeroU64>,
    /// Entries per segment before it rotates (`None` = 50000).
    pub segment_max_entries: Option<NonZeroU32>,
    /// Age in seconds past which a segment may be purged, when nothing else
    /// still needs it (`None` = 7 days).
    pub retention_secs: Option<u64>,
    /// How an append is made durable: `full`, `fsync` or `open_datasync`
    /// (`None` = `full`).
    pub sync_method: Option<SyncMethod>,
}

/// Errors from loading the YAML config file.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file path could not be read.
    #[error("failed to read config file '{0}': {1}")]
    Read(String, std::io::Error),
    /// The config file contents are not valid YAML / have unknown keys.
    #[error("failed to parse config file '{0}': {1}")]
    Parse(String, String),
    /// A size in MiB whose byte count does not fit the machine's integers.
    #[error("{key} = {mib} MiB is too large to express in bytes")]
    SizeTooLarge { key: &'static str, mib: u64 },
}

/// The MiB-denominated settings, converted to bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSizes {
    /// `cache_size_mb` in bytes (`None` = engine default).
    pub cache_bytes: Option<u64>,
    /// `write_buffer_mb` in bytes (`None` = engine default).
    pub write_buffer_bytes: Option<u64>,
    /// `max_request_size_mb` in bytes.
    pub max_request_bytes: usize,
}

const MIB: u64 = 1024 * 1024;

fn mib_to_bytes(key: &'static str, mib: u64) -> Result<u64, ConfigError> {
    mib.checked_mul(MIB)
        .ok_or(ConfigError::SizeTooLarge { key, mib })
}

/// Resolved server configuration — the single gate every subsystem reads from.
///
/// Deserialized from YAML with per-field defaults, so a partial config file
/// only overrides the keys it sets. Unknown keys are rejected so typos surface
/// instead of being silently ignored.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Operational mode (`full`; `compute` / `storage` require coordinode-ee).
    pub mode: String,
    /// Numeric node id for this instance.
    pub node_id: u64,
    /// gRPC listen address (native API + inter-node Raft).
    pub grpc_addr: String,
    /// Address advertised to peers (defaults to `grpc_addr` when unset).
    pub advertise_addr: Option<String>,
    /// HTTP/REST listen address.
    pub rest_addr: String,
    /// Ops/metrics listen address.
    pub ops_addr: String,
    /// PostgreSQL wire-protocol listen address. `None` (default) disables the
    /// Postgres frontend; set it (e.g. `127.0.0.1:7085`) to accept SQL over the
    /// Postgres wire protocol. Trust authentication only — bind to a trusted
    /// interface.
    pub pg_addr: Option<String>,
    /// Data directory. Used by the consensus log, CDC, and the single-endpoint
    /// storage desugar when `storage.endpoints` is empty.
    pub data_dir: String,
    /// Physical storage topology (multi-endpoint). Empty = single endpoint
    /// derived from `data_dir`. See [`StorageTopology`].
    pub storage: StorageTopology,
    /// Cluster peer addresses (empty = standalone single-node).
    pub peers: Vec<String>,
    /// How long a membership change (join, promotion, decommission) waits for
    /// the previous one to commit before it is refused, in seconds (`None` = 30).
    pub membership_change_timeout_secs: Option<u64>,
    /// How many log entries a joining member may still lack when it is
    /// promoted from learner to voter (`None` = 1000). A member that has not
    /// answered at all is never promoted, whatever this says.
    pub join_readiness_lag_entries: Option<u64>,
    /// How long a join may take to catch the new member up before it fails,
    /// in seconds (`None` = 1800).
    pub join_timeout_secs: Option<NonZeroU64>,
    /// Entries applied since the last Raft snapshot that trigger the next
    /// (`None` = 10000).
    pub raft_snapshot_entries: Option<NonZeroU64>,
    /// Bytes the Raft log grows by since the last snapshot that trigger the
    /// next (`None` = 256 MiB).
    pub raft_snapshot_log_bytes: Option<NonZeroU64>,
    /// Longest time between Raft snapshots while entries are applied, in
    /// seconds (`None` = 60).
    pub raft_snapshot_interval_secs: Option<NonZeroU64>,
    /// How long the planner's storage statistics are reused before they are
    /// read again, in seconds (`None` = 60). A read that fails is remembered
    /// for the same time.
    pub planner_stats_ttl_secs: Option<u64>,
    /// How long a query waits for a vector index still being built, under
    /// the `block` policy, when it names no bound of its own, in milliseconds
    /// (`None` = 30000). A query's `vector_build_wait` hint overrides it.
    pub vector_build_wait_ms: Option<u64>,
    /// Bytes of replaced neighbour lists each vector index lets wait for
    /// reclamation before its writers hold off until running searches finish
    /// (`None` = 256 MiB).
    pub vector_retired_bytes_budget: Option<u64>,
    /// Open-file-descriptor target (`None` = raise soft limit to hard limit).
    pub nofile: Option<u64>,
    /// Max concurrent connections (`None` = unbounded).
    pub max_connections: Option<usize>,
    /// Max decoded request size in MiB.
    pub max_request_size_mb: usize,
    /// Per-request timeout in seconds (`None` = none).
    pub request_timeout_secs: Option<u64>,
    /// HTTP/2 keepalive ping interval in seconds (`None` = disabled).
    pub http2_keepalive_secs: Option<u64>,
    /// Block-cache size in MiB (`None` = engine default).
    pub cache_size_mb: Option<u64>,
    /// Memtable size in MiB (`None` = engine default).
    pub write_buffer_mb: Option<u64>,
    /// MVCC time-travel / `AS OF TIMESTAMP` horizon in seconds (`None` = 7 days).
    pub retention_window_secs: Option<u64>,
    /// Ceiling on the invariant claims held by all in-flight write attempts on
    /// this node, together (`None` = 100000).
    pub max_invariant_claims: Option<usize>,
    /// Ceiling on the commits admitted and not yet applied on this node
    /// (`None` = 10000).
    pub max_commits_in_flight: Option<usize>,
    /// How long a read waits, in ms, for commits still landing before it is
    /// answered from a view that stops behind them (`None` = 5).
    pub snapshot_wait_ms: Option<u64>,
    /// The shard whose node rows this engine holds (`None` = 0). The invariant
    /// guard resolves a node from its id alone and needs it to build the key.
    pub node_shard: Option<u16>,
    /// Consumer-registry heartbeat coalescing window in ms (`None` = 1000).
    pub registry_heartbeat_ms: Option<u64>,
    /// Shortest gap between consumer-registry TTL-eviction sweeps in ms
    /// (`None` = 1000).
    pub registry_eviction_ms: Option<u64>,
    /// How often a waiting CDC change stream heartbeats its registration, in
    /// ms (`None` = 10000). A BOUNDED consumer's liveness timeout must exceed
    /// it, which registration checks. Delivery does not wait on it: a
    /// caught-up stream wakes when an entry applies.
    pub cdc_heartbeat_interval_ms: Option<NonZeroU64>,
    /// Most entries a CDC change stream reads and sends per poll (`None` = 256).
    pub cdc_batch_size: Option<NonZeroUsize>,
    /// Interactive-transaction idle timeout in seconds.
    pub interactive_txn_idle_timeout_secs: u64,
    /// Max buffered (uncommitted) bytes per interactive transaction.
    pub interactive_txn_max_bytes: u64,
    /// Inter-node gRPC transport zstd compression level (C-zstd numbering:
    /// positive 1..=22 trade speed for ratio). Applied to inter-node wire
    /// traffic. Default 3 — zstd's standard speed/ratio default and the lowest
    /// panic-safe level (levels 1-2 use the Fast strategy whose huffman build is
    /// unguarded for sub-128 KiB messages); raise on a bandwidth-constrained link
    /// (db4 geo).
    pub wire_compression_level: i32,
    /// Path to the node's TLS certificate (PEM). Set together with [`Self::tls_key`]
    /// to serve inter-node + client gRPC over TLS; unset = plaintext (dev).
    pub tls_cert: Option<String>,
    /// Path to the node's TLS private key (PEM). Required when `tls_cert` is set.
    pub tls_key: Option<String>,
    /// Path to the CA certificate (PEM) verifying peers — trusted by clients to
    /// verify the server and (with `tls_require_client_auth`) by the server to
    /// verify connecting nodes for mTLS.
    pub tls_ca: Option<String>,
    /// Require + verify a client certificate (mutual TLS) on incoming
    /// connections. Needs `tls_ca`. Default false.
    pub tls_require_client_auth: bool,
    /// Whether the background integrity scrub runs. Each node scrubs its own
    /// local storage independently (no leader election). Default false: a
    /// full pass reads every block, which an operator schedules deliberately.
    pub scrub_enabled: bool,
    /// Interval between background scrub cycles, in seconds. Default 7 days.
    pub scrub_interval_secs: u64,
    /// Pause between consecutive SST scans during a background scrub, in
    /// milliseconds, so the scrub yields I/O to production traffic. `None` or 0
    /// runs at full speed. Default 50.
    pub scrub_throttle_ms: Option<u64>,
    /// Whether periodic local checkpoints are taken. A checkpoint is the base for
    /// WAL-replay repair (rebuild a corrupt partition from the last checkpoint +
    /// oplog when no healthy replica can serve it). Per-node, no leader election.
    /// Default true.
    pub checkpoint_enabled: bool,
    /// Interval between periodic checkpoints, in seconds. Default 1 hour.
    pub checkpoint_interval_secs: u64,
    /// Directory checkpoints are written under. `None` derives `<data_dir>/checkpoints`.
    pub checkpoint_dir: Option<String>,
    /// Number of recent checkpoints to retain; older ones are pruned. Default 3.
    pub checkpoint_keep: usize,
    // ── AFTER COMMIT trigger dispatch ────────────────────────────────────────
    // Fine tunables: config-file only, no CLI flag. The first three also have a runtime seam
    // (`Database::set_trigger_dispatch_config`) the future `setParameters` admin
    // command drives; `trigger_dispatch_interval_ms` is restart-only.
    /// AFTER COMMIT trigger cascade-depth cap (the async side of L1). An
    /// async trigger chain deeper than this is dead-lettered as a cascade
    /// overflow rather than executed. Per-trigger `CASCADE_LIMIT` overrides it.
    /// `None` = 10.
    pub trigger_max_cascade_depth: Option<u32>,
    /// Default total execution attempts for an AFTER COMMIT trigger that
    /// declares no `ON ERROR` policy, before dead-lettering. Per-trigger
    /// `ON ERROR RETRY n` overrides it. `None` = 3.
    pub trigger_default_retry_attempts: Option<u32>,
    /// Default base retry backoff in ms for AFTER COMMIT triggers with no
    /// `ON ERROR` policy (per-attempt wait = `backoff * 2^attempt`). Per-trigger
    /// `WITH BACKOFF ms` overrides it. `None` = 1000.
    pub trigger_default_backoff_ms: Option<u64>,
    /// Shortest gap between two passes of the leader-gated AFTER COMMIT
    /// dispatch worker, in ms. The worker wakes on each applied entry and at
    /// the earliest scheduled retry, never on a timer of its own. Restart to
    /// change. `None` = 1000.
    pub trigger_dispatch_interval_ms: Option<u64>,
    /// Keys belonging to whatever was registered on the [`crate::ServerBuilder`].
    ///
    /// The base server never reads this table; it exists so a registered
    /// extension can carry its own settings in the same config file without
    /// every key having to be declared here. Its contents are the extension's
    /// business, so they escape the `deny_unknown_fields` above: a typo inside
    /// this table surfaces where the extension parses it, not here.
    ///
    /// Read it through [`crate::ServerContext::extension_config`].
    pub extensions: BTreeMap<String, serde_yaml_ng::Value>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            mode: "full".to_string(),
            node_id: 1,
            grpc_addr: "[::]:7080".to_string(),
            advertise_addr: None,
            rest_addr: "[::]:7081".to_string(),
            ops_addr: "[::]:7084".to_string(),
            pg_addr: None,
            data_dir: "./data".to_string(),
            storage: StorageTopology::default(),
            peers: Vec::new(),
            membership_change_timeout_secs: None,
            join_readiness_lag_entries: None,
            join_timeout_secs: None,
            raft_snapshot_entries: None,
            raft_snapshot_log_bytes: None,
            raft_snapshot_interval_secs: None,
            planner_stats_ttl_secs: None,
            vector_build_wait_ms: None,
            vector_retired_bytes_budget: None,
            nofile: None,
            max_connections: None,
            max_request_size_mb: 16,
            request_timeout_secs: None,
            http2_keepalive_secs: None,
            cache_size_mb: None,
            write_buffer_mb: None,
            retention_window_secs: None,
            max_invariant_claims: None,
            max_commits_in_flight: None,
            snapshot_wait_ms: None,
            node_shard: None,
            registry_heartbeat_ms: None,
            registry_eviction_ms: None,
            cdc_heartbeat_interval_ms: None,
            cdc_batch_size: None,
            interactive_txn_idle_timeout_secs: 30,
            interactive_txn_max_bytes: 256 * 1024 * 1024,
            wire_compression_level: 3,
            tls_cert: None,
            tls_key: None,
            tls_ca: None,
            tls_require_client_auth: false,
            scrub_enabled: false,
            scrub_interval_secs: 7 * 24 * 3600,
            scrub_throttle_ms: Some(50),
            checkpoint_enabled: true,
            checkpoint_interval_secs: 3600,
            checkpoint_dir: None,
            checkpoint_keep: 3,
            trigger_max_cascade_depth: None,
            trigger_default_retry_attempts: None,
            trigger_default_backoff_ms: None,
            trigger_dispatch_interval_ms: None,
            extensions: BTreeMap::new(),
        }
    }
}

/// Command-line overrides: every knob is optional (`None` = not given on the
/// command line, so the config-file / default value stands). The CLI parser
/// fills this; [`ServerConfig::apply_overrides`] folds it in last so the
/// command line wins over the config file.
#[derive(Debug, Default, Clone)]
pub struct CliOverrides {
    pub mode: Option<String>,
    pub node_id: Option<u64>,
    pub grpc_addr: Option<String>,
    pub advertise_addr: Option<String>,
    pub rest_addr: Option<String>,
    pub ops_addr: Option<String>,
    pub pg_addr: Option<String>,
    pub data_dir: Option<String>,
    pub peers: Option<Vec<String>>,
    pub nofile: Option<u64>,
    pub tls_cert: Option<String>,
    pub tls_key: Option<String>,
    pub tls_ca: Option<String>,
    pub tls_require_client_auth: Option<bool>,
}

impl ServerConfig {
    /// Load the config from an optional YAML file path. `None` (no `--config`)
    /// returns the built-in defaults. A given path that cannot be read or
    /// parsed is an error (fail loud rather than silently fall back).
    pub fn load(path: Option<&str>) -> Result<Self, ConfigError> {
        match path {
            None => Ok(Self::default()),
            Some(p) => {
                let text =
                    std::fs::read_to_string(p).map_err(|e| ConfigError::Read(p.to_string(), e))?;
                serde_yaml_ng::from_str(&text)
                    .map_err(|e| ConfigError::Parse(p.to_string(), e.to_string()))
            }
        }
    }

    /// Convert the MiB settings to bytes, refusing a value whose byte count
    /// does not fit rather than clamping it to the integer maximum.
    pub fn byte_sizes(&self) -> Result<ByteSizes, ConfigError> {
        let max_request_mib =
            u64::try_from(self.max_request_size_mb).map_err(|_| ConfigError::SizeTooLarge {
                key: "max_request_size_mb",
                mib: u64::MAX,
            })?;
        let max_request_bytes =
            usize::try_from(mib_to_bytes("max_request_size_mb", max_request_mib)?).map_err(
                |_| ConfigError::SizeTooLarge {
                    key: "max_request_size_mb",
                    mib: max_request_mib,
                },
            )?;
        Ok(ByteSizes {
            cache_bytes: self
                .cache_size_mb
                .map(|mib| mib_to_bytes("cache_size_mb", mib))
                .transpose()?,
            write_buffer_bytes: self
                .write_buffer_mb
                .map(|mib| mib_to_bytes("write_buffer_mb", mib))
                .transpose()?,
            max_request_bytes,
        })
    }

    /// Fold command-line overrides in last: any field the CLI set (`Some`)
    /// wins over the config-file / default value; fields the CLI left `None`
    /// keep the resolved value.
    pub fn apply_overrides(&mut self, o: &CliOverrides) {
        if let Some(v) = &o.mode {
            self.mode = v.clone();
        }
        if let Some(v) = o.node_id {
            self.node_id = v;
        }
        if let Some(v) = &o.grpc_addr {
            self.grpc_addr = v.clone();
        }
        if o.advertise_addr.is_some() {
            self.advertise_addr = o.advertise_addr.clone();
        }
        if let Some(v) = &o.rest_addr {
            self.rest_addr = v.clone();
        }
        if let Some(v) = &o.ops_addr {
            self.ops_addr = v.clone();
        }
        if o.pg_addr.is_some() {
            self.pg_addr = o.pg_addr.clone();
        }
        if let Some(v) = &o.data_dir {
            self.data_dir = v.clone();
            // The command line wins over the file: an explicit `--data` names
            // a single directory, which cannot express a multi-endpoint
            // topology, so it overrides any `storage.endpoints` the file set
            // and collapses to the single-endpoint desugar at this path.
            self.storage.endpoints.clear();
        }
        if let Some(v) = &o.peers {
            self.peers = v.clone();
        }
        if o.nofile.is_some() {
            self.nofile = o.nofile;
        }
        if o.tls_cert.is_some() {
            self.tls_cert = o.tls_cert.clone();
        }
        if o.tls_key.is_some() {
            self.tls_key = o.tls_key.clone();
        }
        if o.tls_ca.is_some() {
            self.tls_ca = o.tls_ca.clone();
        }
        if let Some(v) = o.tls_require_client_auth {
            self.tls_require_client_auth = v;
        }
    }

    /// The directory periodic checkpoints are written under: the configured
    /// `checkpoint_dir`, or `<data_dir>/checkpoints` when unset.
    #[must_use]
    pub fn checkpoint_directory(&self) -> std::path::PathBuf {
        match &self.checkpoint_dir {
            Some(d) => std::path::PathBuf::from(d),
            None => std::path::Path::new(&self.data_dir).join("checkpoints"),
        }
    }

    /// Build the background-scrub config from the resolved server settings.
    /// A `scrub_throttle_ms` of 0 maps to "no throttle" (full speed).
    #[must_use]
    pub fn scrub_config(&self) -> coordinode_storage::scrub::ScrubConfig {
        coordinode_storage::scrub::ScrubConfig {
            enabled: self.scrub_enabled,
            interval: std::time::Duration::from_secs(self.scrub_interval_secs),
            throttle: self
                .scrub_throttle_ms
                .filter(|&ms| ms > 0)
                .map(std::time::Duration::from_millis),
            parallelism: 1,
        }
    }

    /// Build the runtime-tunable AFTER COMMIT trigger dispatch config
    /// from the resolved file settings, applying the built-in defaults for unset
    /// knobs. Applied to the `Database` at startup via
    /// `set_trigger_dispatch_config`; the same setter is the future
    /// `setParameters` seam.
    #[must_use]
    pub fn trigger_dispatch_config(&self) -> coordinode_embed::TriggerDispatchConfig {
        let d = coordinode_embed::TriggerDispatchConfig::default();
        coordinode_embed::TriggerDispatchConfig {
            max_cascade_depth: self
                .trigger_max_cascade_depth
                .unwrap_or(d.max_cascade_depth),
            default_retry_attempts: self
                .trigger_default_retry_attempts
                .unwrap_or(d.default_retry_attempts),
            default_backoff_ms: self
                .trigger_default_backoff_ms
                .unwrap_or(d.default_backoff_ms),
        }
    }

    /// Shortest gap between AFTER COMMIT dispatch passes (default 1s).
    #[must_use]
    pub fn trigger_dispatch_interval(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.trigger_dispatch_interval_ms.unwrap_or(1000))
    }

    /// Resolve the storage endpoints for this node.
    ///
    /// With an explicit `storage.endpoints` list, that list is the topology
    /// verbatim. With no endpoints configured (the common single-disk case),
    /// desugar `data_dir` into one durable HDD warm-tier endpoint named
    /// `default` — the historical single-endpoint behaviour.
    #[must_use]
    pub fn storage_endpoints(&self) -> Vec<EndpointConfig> {
        if self.storage.endpoints.is_empty() {
            vec![EndpointConfig::new(
                "default",
                &self.data_dir,
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]
        } else {
            self.storage.endpoints.clone()
        }
    }

    /// Build the [`StorageConfig`] for this node from the resolved endpoint
    /// topology ([`Self::storage_endpoints`]). This is the single place the
    /// server turns operator config into a storage-engine config; every
    /// subcommand that opens the engine routes through it.
    ///
    /// # Errors
    /// When the endpoint topology cannot host the engine (see
    /// [`StorageConfig::try_with_endpoints`]).
    pub fn resolve_storage_config(&self) -> Result<StorageConfig, EndpointConfigError> {
        let mut cfg = StorageConfig::try_with_endpoints(self.storage_endpoints())?;
        cfg.backpressure = self.storage.backpressure;
        let compression = &self.storage.compression;
        if let Some(codec) = compression.hot {
            cfg.compression.hot_codec = codec;
        }
        if let Some(codec) = compression.cold {
            cfg.compression.cold_codec = codec;
        }
        if let Some(threshold) = compression.cold_level_threshold {
            cfg.compression.cold_level_threshold = threshold;
        }
        if !compression.partitions.is_empty() {
            cfg.partition_compression = Some(compression.partitions.clone());
        }
        let oplog = self.storage.oplog;
        if let Some(bytes) = oplog.segment_max_bytes {
            cfg.oplog_segment_max_bytes = bytes.get();
        }
        if let Some(entries) = oplog.segment_max_entries {
            cfg.oplog_segment_max_entries = entries.get();
        }
        if let Some(secs) = oplog.retention_secs {
            cfg.oplog_retention_secs = secs;
        }
        if let Some(sync) = oplog.sync_method {
            cfg.oplog_sync_method = sync;
        }
        // The MVCC time-travel window is an engine setting: the engine holds
        // its GC watermark back by it on every node, registry or not.
        if let Some(secs) = self.retention_window_secs {
            cfg.retention_window_secs = secs;
        }
        // The invariant guard lives in the engine, so both of its settings are
        // engine settings: what it may hold, and where it looks for a node.
        if let Some(limit) = self.max_invariant_claims {
            cfg.max_invariant_claims = limit;
        }
        if let Some(limit) = self.max_commits_in_flight {
            cfg.max_commits_in_flight = limit;
        }
        if let Some(ms) = self.snapshot_wait_ms {
            cfg.snapshot_wait_ms = ms;
        }
        if let Some(shard) = self.node_shard {
            cfg.node_shard = shard;
        }
        Ok(cfg)
    }

    /// Whether any configured endpoint's effective ECC policy is "on".
    ///
    /// Used at startup to warn when an operator requested per-block ECC
    /// (`page_ecc: force_on`, or a `degraded` endpoint under the `auto` rule)
    /// but the binary was built without the `page_ecc` feature, so the request
    /// has no on-disk effect.
    #[must_use]
    pub fn page_ecc_requested(&self) -> bool {
        self.storage_endpoints()
            .iter()
            .any(EndpointConfig::is_page_ecc_enabled)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
