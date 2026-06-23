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
//! Each parameter has exactly one field here; adding a knob means adding it once
//! to `ServerConfig`, once to `CliOverrides`, and one line to `apply_overrides`.

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
    /// Interactive-transaction idle timeout in seconds (ADR-042).
    pub interactive_txn_idle_timeout_secs: u64,
    /// Max buffered (uncommitted) bytes per interactive transaction (ADR-042).
    pub interactive_txn_max_bytes: u64,
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
            interactive_txn_idle_timeout_secs: 30,
            interactive_txn_max_bytes: 256 * 1024 * 1024,
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
    pub max_connections: Option<usize>,
    pub max_request_size_mb: Option<usize>,
    pub request_timeout_secs: Option<u64>,
    pub http2_keepalive_secs: Option<u64>,
    pub cache_size_mb: Option<u64>,
    pub write_buffer_mb: Option<u64>,
    pub retention_window_secs: Option<u64>,
    pub registry_heartbeat_ms: Option<u64>,
    pub registry_eviction_ms: Option<u64>,
    pub interactive_txn_idle_timeout_secs: Option<u64>,
    pub interactive_txn_max_bytes: Option<u64>,
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
        if o.max_connections.is_some() {
            self.max_connections = o.max_connections;
        }
        if let Some(v) = o.max_request_size_mb {
            self.max_request_size_mb = v;
        }
        if o.request_timeout_secs.is_some() {
            self.request_timeout_secs = o.request_timeout_secs;
        }
        if o.http2_keepalive_secs.is_some() {
            self.http2_keepalive_secs = o.http2_keepalive_secs;
        }
        if o.cache_size_mb.is_some() {
            self.cache_size_mb = o.cache_size_mb;
        }
        if o.write_buffer_mb.is_some() {
            self.write_buffer_mb = o.write_buffer_mb;
        }
        if o.retention_window_secs.is_some() {
            self.retention_window_secs = o.retention_window_secs;
        }
        if o.registry_heartbeat_ms.is_some() {
            self.registry_heartbeat_ms = o.registry_heartbeat_ms;
        }
        if o.registry_eviction_ms.is_some() {
            self.registry_eviction_ms = o.registry_eviction_ms;
        }
        if let Some(v) = o.interactive_txn_idle_timeout_secs {
            self.interactive_txn_idle_timeout_secs = v;
        }
        if let Some(v) = o.interactive_txn_max_bytes {
            self.interactive_txn_max_bytes = v;
        }
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
