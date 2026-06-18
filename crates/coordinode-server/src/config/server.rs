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
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_values() {
        let c = ServerConfig::default();
        assert_eq!(c.mode, "full");
        assert_eq!(c.node_id, 1);
        assert_eq!(c.max_request_size_mb, 16);
        assert_eq!(c.interactive_txn_idle_timeout_secs, 30);
        assert_eq!(c.interactive_txn_max_bytes, 256 * 1024 * 1024);
        assert!(c.cache_size_mb.is_none());
        assert!(c.peers.is_empty());
    }

    #[test]
    fn load_none_returns_defaults() {
        let c = ServerConfig::load(None).unwrap();
        assert_eq!(c.node_id, 1);
        assert_eq!(c.grpc_addr, "[::]:7080");
    }

    #[test]
    fn partial_yaml_overlays_only_its_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.yaml");
        std::fs::write(
            &path,
            "node_id: 7\ngrpc_addr: \"0.0.0.0:9999\"\ninteractive_txn_idle_timeout_secs: 120\n",
        )
        .unwrap();
        let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
        // Set keys come from the file.
        assert_eq!(c.node_id, 7);
        assert_eq!(c.grpc_addr, "0.0.0.0:9999");
        assert_eq!(c.interactive_txn_idle_timeout_secs, 120);
        // Unset keys keep defaults.
        assert_eq!(c.max_request_size_mb, 16);
        assert_eq!(c.ops_addr, "[::]:7084");
    }

    #[test]
    fn cli_overrides_beat_the_config_file() {
        let mut c = ServerConfig {
            node_id: 7,
            interactive_txn_idle_timeout_secs: 120,
            max_request_size_mb: 16,
            ..ServerConfig::default()
        };
        let cli = CliOverrides {
            // CLI sets node_id + idle timeout → wins.
            node_id: Some(99),
            interactive_txn_idle_timeout_secs: Some(5),
            // max_request_size_mb left None on CLI → file/default stands.
            ..CliOverrides::default()
        };
        c.apply_overrides(&cli);
        assert_eq!(c.node_id, 99, "CLI overrides the file value");
        assert_eq!(c.interactive_txn_idle_timeout_secs, 5, "CLI wins");
        assert_eq!(c.max_request_size_mb, 16, "unset CLI keeps file/default");
    }

    #[test]
    fn unknown_key_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.yaml");
        std::fs::write(&path, "not_a_real_knob: 1\n").unwrap();
        assert!(ServerConfig::load(Some(path.to_str().unwrap())).is_err());
    }

    #[test]
    fn malformed_yaml_is_an_error_not_silent_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.yaml");
        std::fs::write(&path, "node_id: \"not a number\"\n").unwrap();
        assert!(ServerConfig::load(Some(path.to_str().unwrap())).is_err());
    }

    // ── Storage topology ────────────────────────────────────────────────

    #[test]
    fn default_storage_desugars_to_single_endpoint_at_data_dir() {
        let c = ServerConfig {
            data_dir: "/var/lib/coordinode/data".to_string(),
            ..ServerConfig::default()
        };
        assert!(
            c.storage.endpoints.is_empty(),
            "default has no explicit list"
        );
        let eps = c.storage_endpoints();
        assert_eq!(eps.len(), 1, "desugar yields exactly one endpoint");
        assert_eq!(eps[0].id, "default");
        assert_eq!(eps[0].path.to_str().unwrap(), "/var/lib/coordinode/data");
        assert_eq!(eps[0].media, Media::Hdd);
        assert_eq!(eps[0].durability, Durability::Durable);
        assert_eq!(eps[0].tier, Tier::Warm);
    }

    #[test]
    fn explicit_multi_endpoint_topology_parses_from_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.yaml");
        std::fs::write(
            &path,
            "data_dir: /var/lib/coordinode/data\n\
             storage:\n\
             \x20 endpoints:\n\
             \x20   - id: nvme-hot\n\
             \x20     path: /mnt/nvme0\n\
             \x20     media: nvme\n\
             \x20     durability: durable\n\
             \x20     tier: hot\n\
             \x20   - id: hdd-cold\n\
             \x20     path: /mnt/hdd0\n\
             \x20     media: hdd\n\
             \x20     durability: degraded\n\
             \x20     tier: cold\n\
             \x20     page_ecc: force_on\n\
             \x20     capacity_bytes: 16000000000000\n",
        )
        .unwrap();
        let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
        let eps = c.storage_endpoints();
        assert_eq!(eps.len(), 2, "both endpoints parsed");
        assert_eq!(eps[0].id, "nvme-hot");
        assert_eq!(eps[0].media, Media::Nvme);
        assert_eq!(eps[0].tier, Tier::Hot);
        // Omitted capacity/hard_limit default to 0 (untracked / no limit).
        assert_eq!(eps[0].capacity_bytes, 0);
        assert_eq!(eps[0].hard_limit_bytes, 0);
        assert_eq!(eps[1].id, "hdd-cold");
        assert_eq!(eps[1].durability, Durability::Degraded);
        assert_eq!(eps[1].capacity_bytes, 16_000_000_000_000);
    }

    #[test]
    fn cli_data_flag_overrides_file_topology() {
        // File declares a two-endpoint topology...
        let mut c = ServerConfig {
            storage: StorageTopology {
                endpoints: vec![
                    EndpointConfig::new("a", "/mnt/a", Media::Nvme, Durability::Durable, Tier::Hot),
                    EndpointConfig::new("b", "/mnt/b", Media::Hdd, Durability::Durable, Tier::Cold),
                ],
            },
            ..ServerConfig::default()
        };
        // ...but the operator passes --data on the CLI for a one-off.
        c.apply_overrides(&CliOverrides {
            data_dir: Some("/tmp/oneoff".to_string()),
            ..CliOverrides::default()
        });
        assert!(
            c.storage.endpoints.is_empty(),
            "CLI --data collapses the file topology"
        );
        let eps = c.storage_endpoints();
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].path.to_str().unwrap(), "/tmp/oneoff");
    }

    #[test]
    fn unknown_endpoint_key_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.yaml");
        std::fs::write(
            &path,
            "storage:\n\
             \x20 endpoints:\n\
             \x20   - id: ep0\n\
             \x20     path: /mnt/ep0\n\
             \x20     media: ssd\n\
             \x20     durability: durable\n\
             \x20     tier: warm\n\
             \x20     typo_field: 1\n",
        )
        .unwrap();
        assert!(
            ServerConfig::load(Some(path.to_str().unwrap())).is_err(),
            "a typo in an endpoint key must fail loud"
        );
    }

    #[test]
    fn page_ecc_requested_tracks_endpoint_policy() {
        // Auto + durable → off.
        let durable = ServerConfig {
            storage: StorageTopology {
                endpoints: vec![EndpointConfig::new(
                    "d",
                    "/mnt/d",
                    Media::Ssd,
                    Durability::Durable,
                    Tier::Warm,
                )],
            },
            ..ServerConfig::default()
        };
        assert!(!durable.page_ecc_requested());

        // Auto + degraded → on.
        let degraded = ServerConfig {
            storage: StorageTopology {
                endpoints: vec![EndpointConfig::new(
                    "g",
                    "/mnt/g",
                    Media::Ssd,
                    Durability::Degraded,
                    Tier::Warm,
                )],
            },
            ..ServerConfig::default()
        };
        assert!(degraded.page_ecc_requested());
    }

    #[test]
    fn packaged_conf_parses_against_current_schema() {
        // The shipped /etc/coordinode/coordinode.conf must stay valid against
        // ServerConfig (deny_unknown_fields): a key removed here but left in the
        // packaged file, or vice versa, is a release regression. Resolve the
        // file relative to this crate's manifest dir.
        let conf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../packaging/coordinode.conf");
        let c = ServerConfig::load(Some(conf.to_str().unwrap()))
            .expect("packaged coordinode.conf must parse against ServerConfig");
        // It ships the single-endpoint default (storage.endpoints empty).
        assert!(c.storage.endpoints.is_empty());
        assert_eq!(c.storage_endpoints().len(), 1);
    }

    #[test]
    fn resolve_storage_config_carries_endpoints() {
        let c = ServerConfig {
            storage: StorageTopology {
                endpoints: vec![
                    EndpointConfig::new("a", "/mnt/a", Media::Nvme, Durability::Durable, Tier::Hot),
                    EndpointConfig::new("b", "/mnt/b", Media::Hdd, Durability::Durable, Tier::Cold),
                ],
            },
            ..ServerConfig::default()
        };
        let sc = c.resolve_storage_config();
        assert_eq!(sc.endpoints.len(), 2);
        assert_eq!(sc.endpoints[0].id, "a");
        assert_eq!(sc.endpoints[1].id, "b");
    }
}
