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

/// MiB settings convert to bytes exactly, and a value whose byte count does
/// not fit is an error naming the key, not a silent `u64::MAX` cache.
#[test]
fn mib_settings_convert_to_bytes_or_name_the_key_that_overflows() {
    let c = ServerConfig {
        cache_size_mb: Some(64),
        write_buffer_mb: Some(8),
        max_request_size_mb: 16,
        ..ServerConfig::default()
    };
    let sizes = c.byte_sizes().expect("in-range sizes");
    assert_eq!(sizes.cache_bytes, Some(64 * 1024 * 1024));
    assert_eq!(sizes.write_buffer_bytes, Some(8 * 1024 * 1024));
    assert_eq!(sizes.max_request_bytes, 16 * 1024 * 1024);

    for (key, c) in [
        (
            "cache_size_mb",
            ServerConfig {
                cache_size_mb: Some(u64::MAX),
                ..ServerConfig::default()
            },
        ),
        (
            "write_buffer_mb",
            ServerConfig {
                write_buffer_mb: Some(u64::MAX),
                ..ServerConfig::default()
            },
        ),
        (
            "max_request_size_mb",
            ServerConfig {
                max_request_size_mb: usize::MAX,
                ..ServerConfig::default()
            },
        ),
    ] {
        let err = c
            .byte_sizes()
            .expect_err("an overflowing size must be refused");
        assert!(err.to_string().contains(key), "{key}: {err}");
    }
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

/// The membership-change wait is a config-file setting: unset it leaves the
/// node's default, set it carries the seconds given.
#[test]
fn membership_change_timeout_parses_from_the_config_file() {
    assert!(
        ServerConfig::default()
            .membership_change_timeout_secs
            .is_none()
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(&path, "membership_change_timeout_secs: 90\n").unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(c.membership_change_timeout_secs, Some(90));
}

/// The join settings are config-file settings: unset they leave the node's
/// defaults, set they carry the values given, and a zero timeout is refused.
#[test]
fn join_settings_parse_from_the_config_file() {
    let d = ServerConfig::default();
    assert!(d.join_readiness_lag_entries.is_none() && d.join_timeout_secs.is_none());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "join_readiness_lag_entries: 0\njoin_timeout_secs: 120\n",
    )
    .unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(c.join_readiness_lag_entries, Some(0));
    assert_eq!(c.join_timeout_secs.map(|v| v.get()), Some(120));

    std::fs::write(&path, "join_timeout_secs: 0\n").unwrap();
    assert!(ServerConfig::load(Some(path.to_str().unwrap())).is_err());
}

/// The Raft snapshot triggers are config-file settings: unset they leave the
/// node's defaults, set they carry the values given, and a zero, which would
/// snapshot on every entry, byte or moment, is refused.
#[test]
fn raft_snapshot_settings_parse_from_the_config_file() {
    let d = ServerConfig::default();
    assert!(
        d.raft_snapshot_entries.is_none()
            && d.raft_snapshot_log_bytes.is_none()
            && d.raft_snapshot_min_interval_secs.is_none()
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "raft_snapshot_entries: 500000\nraft_snapshot_log_bytes: 1073741824\nraft_snapshot_min_interval_secs: 300\n",
    )
    .unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(c.raft_snapshot_entries.map(|v| v.get()), Some(500_000));
    assert_eq!(
        c.raft_snapshot_log_bytes.map(|v| v.get()),
        Some(1_073_741_824)
    );
    assert_eq!(
        c.raft_snapshot_min_interval_secs.map(|v| v.get()),
        Some(300)
    );

    for zero in [
        "raft_snapshot_entries: 0\n",
        "raft_snapshot_log_bytes: 0\n",
        "raft_snapshot_min_interval_secs: 0\n",
    ] {
        std::fs::write(&path, zero).unwrap();
        assert!(
            ServerConfig::load(Some(path.to_str().unwrap())).is_err(),
            "{zero:?} is refused"
        );
    }
}

/// The planner-statistics reuse window is a config-file setting: unset it
/// leaves the database default, set it carries the seconds given.
#[test]
fn planner_stats_ttl_parses_from_the_config_file() {
    assert!(ServerConfig::default().planner_stats_ttl_secs.is_none());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(&path, "planner_stats_ttl_secs: 5\n").unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(c.planner_stats_ttl_secs, Some(5));
}

/// The default bound on waiting for a building vector index is a config-file
/// setting: unset it leaves the database default, set it carries the
/// milliseconds given.
#[test]
fn vector_build_wait_parses_from_the_config_file() {
    assert!(ServerConfig::default().vector_build_wait_ms.is_none());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(&path, "vector_build_wait_ms: 1500\n").unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(c.vector_build_wait_ms, Some(1500));
}

/// The statement defaults: a file that names none keeps the built-in local
/// read from the leader and the majority journaled write.
#[test]
fn statement_defaults_are_the_builtins_when_unset() {
    use coordinode_core::txn::read_concern::ReadConcernLevel;
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_raft::read_fence::ReadPreference;

    let d = ServerConfig::default().statement_defaults();
    assert_eq!(d.read_concern, ReadConcernLevel::Local);
    assert_eq!(d.read_preference, ReadPreference::Primary);
    assert_eq!(d.write_concern, WriteConcern::majority());
}

/// Every statement default parses from the file, each key on its own: the
/// write concern's unnamed fields keep their built-in values.
#[test]
fn statement_defaults_parse_from_the_config_file() {
    use coordinode_core::txn::read_concern::ReadConcernLevel;
    use coordinode_core::txn::write_concern::{Journal, WriteAck, WriteConcern};
    use coordinode_raft::read_fence::ReadPreference;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "default_read_concern: majority\n\
         default_read_preference: secondary_preferred\n\
         default_write_concern:\n  w: 2\n  timeout_ms: 500\n",
    )
    .unwrap();
    let d = ServerConfig::load(Some(path.to_str().unwrap()))
        .unwrap()
        .statement_defaults();
    assert_eq!(d.read_concern, ReadConcernLevel::Majority);
    assert_eq!(d.read_preference, ReadPreference::SecondaryPreferred);
    assert_eq!(
        d.write_concern,
        WriteConcern {
            w: WriteAck::Acks(2),
            journal: Journal::Journal,
            timeout_ms: 500,
        }
    );

    std::fs::write(
        &path,
        "default_write_concern: { w: majority, journal: journal }\n",
    )
    .unwrap();
    let d = ServerConfig::load(Some(path.to_str().unwrap()))
        .unwrap()
        .statement_defaults();
    assert_eq!(d.write_concern, WriteConcern::majority());

    std::fs::write(&path, "default_write_concern: { w: 1, journal: memory }\n").unwrap();
    let d = ServerConfig::load(Some(path.to_str().unwrap()))
        .unwrap()
        .statement_defaults();
    assert_eq!(d.write_concern, WriteConcern::memory());
}

/// A default the engine cannot honour fails the load, naming the key, rather
/// than failing every write that later relies on it; so does a level the
/// server does not know.
#[test]
fn statement_defaults_refuse_what_the_engine_cannot_honour() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    for (yaml, needle) in [
        (
            "default_write_concern: { w: majority, journal: memory }\n",
            "default_write_concern",
        ),
        ("default_read_concern: eventual\n", "eventual"),
        ("default_read_preference: fastest\n", "fastest"),
        ("default_write_concern: { w: 1, fsync: true }\n", "fsync"),
    ] {
        std::fs::write(&path, yaml).unwrap();
        let err = ServerConfig::load(Some(path.to_str().unwrap()))
            .expect_err(yaml)
            .to_string();
        assert!(err.contains(needle), "{yaml}: {err}");
    }
}

/// The vector indexes' retired-memory budget is a config-file setting: unset
/// it leaves the index default, set it carries the bytes given.
#[test]
fn vector_retired_bytes_budget_parses_from_the_config_file() {
    assert!(
        ServerConfig::default()
            .vector_retired_bytes_budget
            .is_none()
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(&path, "vector_retired_bytes_budget: 67108864\n").unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(c.vector_retired_bytes_budget, Some(64 << 20));
}

/// The change-stream pacing is a config-file setting; zero is refused at
/// parse, since a zero batch never reads and a zero heartbeat interval spins.
#[test]
fn cdc_stream_pacing_parses_from_the_config_file() {
    let d = ServerConfig::default();
    assert!(d.cdc_heartbeat_interval_ms.is_none() && d.cdc_batch_size.is_none());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "cdc_heartbeat_interval_ms: 2500\ncdc_batch_size: 64\n",
    )
    .unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(c.cdc_heartbeat_interval_ms.map(|v| v.get()), Some(2500));
    assert_eq!(c.cdc_batch_size.map(|v| v.get()), Some(64));

    for zero in ["cdc_heartbeat_interval_ms: 0\n", "cdc_batch_size: 0\n"] {
        std::fs::write(&path, zero).unwrap();
        assert!(
            ServerConfig::load(Some(path.to_str().unwrap())).is_err(),
            "accepted {zero:?}"
        );
    }
}

#[test]
fn cli_overrides_beat_the_config_file() {
    // The CLI carries only bootstrap-critical knobs now; a fine tunable
    // (interactive_txn_idle_timeout_secs) is config-file-only and is unaffected
    // by CLI parsing.
    let mut c = ServerConfig {
        node_id: 7,
        nofile: Some(1024),
        interactive_txn_idle_timeout_secs: 120,
        ..ServerConfig::default()
    };
    let cli = CliOverrides {
        // CLI sets node_id → wins; nofile left None → file value stands.
        node_id: Some(99),
        ..CliOverrides::default()
    };
    c.apply_overrides(&cli);
    assert_eq!(c.node_id, 99, "CLI overrides the file value");
    assert_eq!(c.nofile, Some(1024), "unset CLI keeps file/default");
    assert_eq!(
        c.interactive_txn_idle_timeout_secs, 120,
        "config-file-only tunable is untouched by the CLI"
    );
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
            ..StorageTopology::default()
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
            ..StorageTopology::default()
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
            ..StorageTopology::default()
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
            ..StorageTopology::default()
        },
        ..ServerConfig::default()
    };
    let sc = c.resolve_storage_config().expect("valid topology");
    assert_eq!(sc.endpoints.len(), 2);
    assert_eq!(sc.endpoints[0].id, "a");
    assert_eq!(sc.endpoints[1].id, "b");
}

/// An endpoint topology the engine cannot run on comes from the operator's
/// file, so it is reported as a configuration error, never a panic.
#[test]
fn an_unusable_endpoint_topology_is_an_error_not_a_panic() {
    let volatile_only = ServerConfig {
        storage: StorageTopology {
            endpoints: vec![EndpointConfig::new(
                "ram",
                "/mnt/ram",
                Media::Nvme,
                Durability::Volatile,
                Tier::Hot,
            )],
            ..StorageTopology::default()
        },
        ..ServerConfig::default()
    };
    let outcome = std::panic::catch_unwind(|| volatile_only.resolve_storage_config());
    assert!(
        matches!(outcome, Ok(Err(EndpointConfigError::NoOplogEndpoint))),
        "a volatile-only topology must be a NoOplogEndpoint error, got {outcome:?}"
    );
}

#[test]
fn trigger_dispatch_defaults_when_unset() {
    let c = ServerConfig::default();
    let d = c.trigger_dispatch_config();
    assert_eq!(d.max_cascade_depth, 10);
    assert_eq!(d.default_retry_attempts, 3);
    assert_eq!(d.default_backoff_ms, 1000);
    assert_eq!(
        c.trigger_dispatch_interval(),
        std::time::Duration::from_millis(1000)
    );
}

#[test]
fn trigger_dispatch_knobs_parse_from_config_file_only() {
    // Config-file keys (no CLI flag) deserialize and resolve into the dispatch
    // config; unset keys keep the built-in defaults.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "trigger_max_cascade_depth: 4\n\
         trigger_default_backoff_ms: 250\n\
         trigger_dispatch_interval_ms: 200\n",
    )
    .unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    let d = c.trigger_dispatch_config();
    assert_eq!(d.max_cascade_depth, 4);
    assert_eq!(d.default_backoff_ms, 250);
    assert_eq!(d.default_retry_attempts, 3, "unset knob keeps the default");
    assert_eq!(
        c.trigger_dispatch_interval(),
        std::time::Duration::from_millis(200)
    );
}

#[test]
fn extension_settings_survive_the_unknown_key_check() {
    // Keys the base server does not know are rejected, which is what makes a
    // typo visible. An extension's own settings must still get through, so
    // they live under `extensions` and are carried verbatim.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "node_id: 7\n\
         extensions:\n\
         \x20 pitr:\n\
         \x20   enabled: true\n\
         \x20   interval_secs: 900\n",
    )
    .unwrap();

    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    assert_eq!(c.node_id, 7);

    let pitr = c.extensions.get("pitr").expect("extension key preserved");
    assert_eq!(pitr.get("enabled").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        pitr.get("interval_secs").and_then(|v| v.as_u64()),
        Some(900)
    );

    // The same key outside the table is still a typo, not a setting.
    let stray = dir.path().join("stray.yaml");
    std::fs::write(&stray, "pitr:\n  enabled: true\n").unwrap();
    assert!(ServerConfig::load(Some(stray.to_str().unwrap())).is_err());
}

#[test]
fn backpressure_thresholds_parse_and_reach_the_storage_config() {
    // Partial override: only the set keys change, the rest keep defaults,
    // and the resolved StorageConfig carries the merged limits.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "storage:\n  backpressure:\n    l0_stop: 12\n    bytes_stop: 1024\n",
    )
    .unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    let sc = c.resolve_storage_config().expect("valid topology");
    assert_eq!(sc.backpressure.l0_stop, 12);
    assert_eq!(sc.backpressure.bytes_stop, 1024);
    // Unset keys stay at their defaults.
    let defaults = coordinode_storage::engine::config::BackpressureLimits::default();
    assert_eq!(sc.backpressure.l0_slowdown, defaults.l0_slowdown);
    assert_eq!(sc.backpressure.bytes_slowdown, defaults.bytes_slowdown);

    // An unknown key inside the section is a config error, not a silent no-op.
    std::fs::write(&path, "storage:\n  backpressure:\n    l0_sotp: 12\n").unwrap();
    assert!(
        ServerConfig::load(Some(path.to_str().unwrap())).is_err(),
        "a typoed backpressure key must be rejected"
    );
}

/// The oplog section reaches the storage config; unset keys keep the engine
/// defaults, which are the full flush and the 64 MiB / 50000 rotation.
#[test]
fn oplog_settings_parse_and_reach_the_storage_config() {
    use coordinode_storage::engine::config::SyncMethod;

    let defaults = ServerConfig::default()
        .resolve_storage_config()
        .expect("valid");
    assert_eq!(defaults.oplog_sync_method, SyncMethod::Full);
    assert_eq!(defaults.oplog_segment_max_bytes, 64 * 1024 * 1024);
    assert_eq!(defaults.oplog_segment_max_entries, 50_000);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "storage:\n  oplog:\n    sync_method: open_datasync\n    segment_max_entries: 1000\n",
    )
    .unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    let sc = c.resolve_storage_config().expect("valid topology");
    assert_eq!(sc.oplog_sync_method, SyncMethod::OpenDatasync);
    assert_eq!(sc.oplog_segment_max_entries, 1000);
    assert_eq!(sc.oplog_segment_max_bytes, defaults.oplog_segment_max_bytes);
    assert_eq!(sc.oplog_retention_secs, defaults.oplog_retention_secs);

    for method in ["full", "fsync"] {
        std::fs::write(
            &path,
            format!("storage:\n  oplog:\n    sync_method: {method}\n"),
        )
        .unwrap();
        assert!(
            ServerConfig::load(Some(path.to_str().unwrap())).is_ok(),
            "{method}"
        );
    }
}

/// The compression section reaches the storage config: the hot and cold
/// codecs with their zstd levels, the threshold and per-partition overrides;
/// unset keys keep the engine defaults.
#[test]
fn compression_settings_parse_and_reach_the_storage_config() {
    use coordinode_storage::engine::config::{CompressionCodec, CompressionConfig};
    use coordinode_storage::engine::partition::Partition;

    let defaults = ServerConfig::default()
        .resolve_storage_config()
        .expect("valid");
    assert_eq!(defaults.compression, CompressionConfig::default());
    assert!(defaults.partition_compression.is_none());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    std::fs::write(
        &path,
        "storage:\n  compression:\n    hot: { codec: zstd, level: 22 }\n    \
         cold: { codec: zstd, level: 22 }\n    cold_level_threshold: 2\n    \
         partitions:\n      idx: { codec: lz4 }\n      blob: { codec: none }\n",
    )
    .unwrap();
    let c = ServerConfig::load(Some(path.to_str().unwrap())).unwrap();
    let sc = c.resolve_storage_config().expect("valid topology");
    assert_eq!(sc.compression.hot_codec, CompressionCodec::Zstd(22));
    assert_eq!(sc.compression.cold_codec, CompressionCodec::Zstd(22));
    assert_eq!(sc.compression.cold_level_threshold, 2);
    let mut overrides = sc.partition_compression.expect("overrides");
    overrides.sort_by_key(|(p, _)| p.name());
    assert_eq!(
        overrides,
        [
            (Partition::Blob, CompressionCodec::None),
            (Partition::Idx, CompressionCodec::Lz4),
        ]
    );

    // A partial section changes only what it names; zstd without a level
    // takes the library default.
    std::fs::write(
        &path,
        "storage:\n  compression:\n    cold: { codec: zstd }\n",
    )
    .unwrap();
    let sc = ServerConfig::load(Some(path.to_str().unwrap()))
        .unwrap()
        .resolve_storage_config()
        .expect("valid");
    assert_eq!(sc.compression.cold_codec, CompressionCodec::Zstd(3));
    assert_eq!(sc.compression.hot_codec, defaults.compression.hot_codec);
    assert_eq!(
        sc.compression.cold_level_threshold,
        defaults.compression.cold_level_threshold
    );
}

/// A codec, level, threshold or partition the engine cannot honour, and an
/// unknown key, are refused when the file is read, not at engine open.
#[test]
fn bad_compression_settings_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    for body in [
        "storage:\n  compression:\n    hot: { codec: brotli }\n",
        "storage:\n  compression:\n    hot: { codec: zstd, level: 23 }\n",
        "storage:\n  compression:\n    hot: { codec: lz4, level: 1 }\n",
        "storage:\n  compression:\n    cold_level_threshold: 8\n",
        "storage:\n  compression:\n    partitions:\n      nodes: { codec: lz4 }\n",
        "storage:\n  compression:\n    hot: { codec: lz4, lvl: 1 }\n",
        "storage:\n  compression:\n    warm: { codec: lz4 }\n",
    ] {
        std::fs::write(&path, body).unwrap();
        assert!(
            ServerConfig::load(Some(path.to_str().unwrap())).is_err(),
            "accepted: {body}"
        );
    }
}

/// A sync method the engine does not have, a zero-sized segment and an
/// unknown key are refused when the file is read, not met at the first write.
#[test]
fn bad_oplog_settings_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    for body in [
        "storage:\n  oplog:\n    sync_method: sometimes\n",
        "storage:\n  oplog:\n    segment_max_bytes: 0\n",
        "storage:\n  oplog:\n    segment_max_entries: 0\n",
        "storage:\n  oplog:\n    sync_mode: full\n",
    ] {
        std::fs::write(&path, body).unwrap();
        assert!(
            ServerConfig::load(Some(path.to_str().unwrap())).is_err(),
            "accepted: {body}"
        );
    }
}
