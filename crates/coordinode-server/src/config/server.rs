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
//! Not every knob gets a CLI flag. Per CLAUDE.md "Configuration Surface", the
//! CLI carries only bootstrap-critical settings (bind addresses, node id, data
//! dir, mode, peers, `--config`) — the argv length is OS-bounded (`ARG_MAX`).
//! Fine tunables live in the YAML config file only: add such a knob to
//! `ServerConfig` (+ its default) and the packaged `coordinode.conf`, and skip
//! `CliOverrides` / `apply_overrides`. A knob that also gets a CLI flag (the
//! bootstrap-critical ones) is the one added in all three places.

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
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
    /// Data directory. Used by the consensus log, CDC, and the single-endpoint
    /// storage desugar when `storage.endpoints` is empty.
    pub data_dir: String,
    /// Physical storage topology (multi-endpoint). Empty = single endpoint
    /// derived from `data_dir`. See [`StorageTopology`].
    pub storage: StorageTopology,
    /// Cluster peer addresses (empty = standalone single-node).
    pub peers: Vec<String>,
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
    /// Consumer-registry heartbeat coalescing window in ms (`None` = 100).
    pub registry_heartbeat_ms: Option<u64>,
    /// Consumer-registry TTL-eviction sweep interval in ms (`None` = 1000).
    pub registry_eviction_ms: Option<u64>,
    /// CDC change-stream consumer TTL in seconds (`None` = 30). How long a
    /// disconnected/crashed CDC reader's registration holds the oplog retention
    /// floor before it is TTL-reclaimed; connected readers heartbeat each poll
    /// and are never evicted.
    pub cdc_consumer_ttl_secs: Option<u64>,
    /// Interactive-transaction idle timeout in seconds (ADR-042).
    pub interactive_txn_idle_timeout_secs: u64,
    /// Max buffered (uncommitted) bytes per interactive transaction (ADR-042).
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
    /// local storage independently (no leader election). Default true.
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
    // ── AFTER COMMIT trigger dispatch (R192) ─────────────────────────────────
    // Fine tunables: config-file only (NOT CLI — see CLAUDE.md "Configuration
    // Surface"). The first three also have a runtime seam
    // (`Database::set_trigger_dispatch_config`) the future `setParameters` admin
    // command drives; `trigger_dispatch_interval_ms` is restart-only.
    /// AFTER COMMIT trigger cascade-depth cap (the trigger architecture L1). An
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
    /// How often the leader-gated AFTER COMMIT dispatch worker wakes to fire due
    /// retries, in ms (it also wakes immediately on each applied entry). Restart
    /// to change. `None` = 500.
    pub trigger_dispatch_interval_ms: Option<u64>,
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
            data_dir: "./data".to_string(),
            storage: StorageTopology::default(),
            peers: Vec::new(),
            nofile: None,
            max_connections: None,
            max_request_size_mb: 16,
            request_timeout_secs: None,
            http2_keepalive_secs: None,
            cache_size_mb: None,
            write_buffer_mb: None,
            retention_window_secs: None,
            registry_heartbeat_ms: None,
            registry_eviction_ms: None,
            cdc_consumer_ttl_secs: None,
            interactive_txn_idle_timeout_secs: 30,
            interactive_txn_max_bytes: 256 * 1024 * 1024,
            wire_compression_level: 3,
            tls_cert: None,
            tls_key: None,
            tls_ca: None,
            tls_require_client_auth: false,
            scrub_enabled: true,
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

    /// Build the runtime-tunable AFTER COMMIT trigger dispatch config (R192)
    /// from the resolved file settings, applying ADR-026 defaults for unset
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

    /// Leader-gated AFTER COMMIT dispatch worker poll interval (default 500ms).
    #[must_use]
    pub fn trigger_dispatch_interval(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.trigger_dispatch_interval_ms.unwrap_or(500))
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
    #[must_use]
    pub fn resolve_storage_config(&self) -> StorageConfig {
        StorageConfig::with_endpoints(self.storage_endpoints())
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
