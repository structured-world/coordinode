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
    let conf =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packaging/coordinode.conf");
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
