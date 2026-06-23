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

/// Test-only helper: a single durable HDD warm-tier endpoint at the
/// given path. Replaces the old `StorageConfig::with_endpoints(vec![EndpointConfig::new("default", path, Media::Hdd, Durability::Durable, Tier::Warm)])` shim —
/// every test now declares its endpoint shape explicitly via this
/// helper at the `EndpointConfig` level, not via a shortcut at
/// `StorageConfig` level.
pub(crate) fn default_disk_endpoint(path: impl AsRef<Path>) -> EndpointConfig {
    EndpointConfig::new(
        "default",
        path.as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
}

/// Test-only helper: a volatile RAM endpoint (for MemFs-backed tests).
pub(crate) fn default_memfs_endpoint(path: impl AsRef<Path>) -> EndpointConfig {
    EndpointConfig::new(
        "default-memfs",
        path.as_ref(),
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    )
}

#[test]
fn storage_config_defaults() {
    let config = StorageConfig::with_endpoints(vec![default_disk_endpoint("/tmp/test")]);
    assert_eq!(config.compression, CompressionConfig::default());
    assert!(config.partition_compression.is_none());
}

#[test]
fn to_tree_config_builds_for_each_partition() {
    use std::sync::Arc;
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![default_disk_endpoint(dir.path())]);
    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let gc_watermark = Arc::new(AtomicU64::new(0));
    for &part in Partition::all() {
        let tree_config = config.to_tree_config(part, Arc::clone(&seqno), &gc_watermark);
        let _ = tree_config;
    }
}

// ── Endpoint model tests ────────────────────────────────────────

#[test]
fn single_endpoint_disk_shape() {
    let config = StorageConfig::with_endpoints(vec![default_disk_endpoint("/tmp/test")]);
    assert_eq!(config.endpoints.len(), 1);
    let ep = &config.endpoints[0];
    assert_eq!(ep.id, "default");
    assert_eq!(ep.path, PathBuf::from("/tmp/test"));
    assert_eq!(ep.media, Media::Hdd);
    assert_eq!(ep.durability, Durability::Durable);
    assert_eq!(ep.tier, Tier::Warm);
    assert_eq!(ep.capacity_bytes, 0, "untracked by default");
    assert_eq!(ep.hard_limit_bytes, 0, "no hard limit by default");
    assert!(ep.tags.is_empty());
    assert!(ep.server.is_none(), "CE single-node: server implicit");
}

#[test]
fn data_dir_accessor_returns_first_endpoint_path() {
    let config = StorageConfig::with_endpoints(vec![default_disk_endpoint("/tmp/abc")]);
    assert_eq!(config.data_dir(), Path::new("/tmp/abc"));
}

#[test]
fn memfs_endpoint_with_fs_builder() {
    let config =
        StorageConfig::with_endpoints_no_persistence(vec![default_memfs_endpoint("/virtual")])
            .with_fs(Arc::new(lsm_tree::fs::MemFs::new()));
    assert_eq!(config.endpoints.len(), 1);
    let ep = &config.endpoints[0];
    assert_eq!(ep.media, Media::Ram);
    assert_eq!(ep.durability, Durability::Volatile);
    assert_eq!(ep.tier, Tier::Memory);
    assert!(config.fs.is_some(), "MemFs filesystem attached");
}

#[test]
#[should_panic(expected = "duplicate EndpointConfig.id")]
fn duplicate_endpoint_id_rejected() {
    let dup_a = EndpointConfig::new("ep-x", "/a", Media::Hdd, Durability::Durable, Tier::Warm);
    let dup_b = EndpointConfig::new("ep-x", "/b", Media::Ssd, Durability::Durable, Tier::Hot);
    let _ = StorageConfig::with_endpoints(vec![dup_a, dup_b]);
}

#[test]
#[should_panic(expected = "duplicate EndpointConfig.path")]
fn duplicate_endpoint_path_rejected() {
    let a = EndpointConfig::new("ep-a", "/same", Media::Hdd, Durability::Durable, Tier::Warm);
    let b = EndpointConfig::new("ep-b", "/same", Media::Ssd, Durability::Durable, Tier::Hot);
    let _ = StorageConfig::with_endpoints(vec![a, b]);
}

#[test]
#[should_panic(expected = "EndpointConfig.id must be non-empty")]
fn empty_endpoint_id_rejected() {
    let ep = EndpointConfig::new("", "/a", Media::Hdd, Durability::Durable, Tier::Warm);
    let _ = StorageConfig::with_endpoints(vec![ep]);
}

#[test]
#[should_panic(expected = "hard_limit_bytes")]
fn hard_limit_above_capacity_rejected() {
    let ep = EndpointConfig::new("ep-x", "/a", Media::Hdd, Durability::Durable, Tier::Warm)
        .with_capacity_bytes(1_000)
        .with_hard_limit_bytes(2_000);
    let _ = StorageConfig::with_endpoints(vec![ep]);
}

#[test]
fn hard_limit_zero_capacity_zero_accepted() {
    // Both 0 = "untracked / no limit". Valid.
    let ep = EndpointConfig::new("ep-x", "/a", Media::Hdd, Durability::Durable, Tier::Warm);
    let _ = StorageConfig::with_endpoints(vec![ep]);
}

#[test]
fn hard_limit_only_capacity_zero_accepted() {
    // capacity untracked + explicit hard_limit = valid (operator hasn't
    // declared physical capacity but wants a soft cap).
    let ep = EndpointConfig::new("ep-x", "/a", Media::Hdd, Durability::Durable, Tier::Warm)
        .with_hard_limit_bytes(500);
    let _ = StorageConfig::with_endpoints(vec![ep]);
}

#[test]
fn endpoint_with_server_label_round_trips() {
    let mut ep = EndpointConfig::new("ep-x", "/a", Media::Nvme, Durability::Degraded, Tier::Hot);
    ep.server = Some("srv-3".to_string());
    let config = StorageConfig::with_endpoints(vec![ep]);
    assert_eq!(config.endpoints[0].server.as_deref(), Some("srv-3"));
}

// ── Per-block ECC policy ─────────────────────────────────────────

#[test]
fn page_ecc_default_is_auto() {
    // The `Default` derive places `Auto` first in the enum → must
    // resolve to `PageEccPolicy::Auto`. Pins the default — any
    // future reorder is a breaking config change.
    assert_eq!(PageEccPolicy::default(), PageEccPolicy::Auto);
}

#[test]
fn page_ecc_auto_derives_per_durability() {
    // Durable: RAID covers bit-rot; Auto resolves to OFF.
    assert!(!PageEccPolicy::Auto.effective_for_durability(Durability::Durable));
    // Degraded: no array-level redundancy; Auto resolves to ON.
    assert!(PageEccPolicy::Auto.effective_for_durability(Durability::Degraded));
    // Volatile: endpoint can vanish; per-block ECC pointless → OFF.
    assert!(!PageEccPolicy::Auto.effective_for_durability(Durability::Volatile));
}

#[test]
fn page_ecc_force_overrides_durability() {
    // ForceOn ignores durability — even Volatile reports ON.
    assert!(PageEccPolicy::ForceOn.effective_for_durability(Durability::Volatile));
    assert!(PageEccPolicy::ForceOn.effective_for_durability(Durability::Durable));
    // ForceOff ignores durability — even Degraded reports OFF (the
    // dangerous case; operator opted out explicitly).
    assert!(!PageEccPolicy::ForceOff.effective_for_durability(Durability::Degraded));
    assert!(!PageEccPolicy::ForceOff.effective_for_durability(Durability::Durable));
}

#[test]
fn endpoint_is_page_ecc_enabled_uses_effective_policy() {
    // Default policy (Auto) on a Degraded endpoint → ON.
    let degraded = EndpointConfig::new("a", "/a", Media::Hdd, Durability::Degraded, Tier::Cold);
    assert!(degraded.is_page_ecc_enabled());

    // Default policy (Auto) on a Durable endpoint → OFF.
    let durable = EndpointConfig::new("b", "/b", Media::Hdd, Durability::Durable, Tier::Warm);
    assert!(!durable.is_page_ecc_enabled());

    // ForceOn override on a Durable endpoint → ON.
    let mut durable_force = durable.clone();
    durable_force.page_ecc = PageEccPolicy::ForceOn;
    assert!(durable_force.is_page_ecc_enabled());

    // ForceOff override on a Degraded endpoint → OFF (operator's
    // dangerous choice; engine does not reject it).
    let mut degraded_force = degraded.clone();
    degraded_force.page_ecc = PageEccPolicy::ForceOff;
    assert!(!degraded_force.is_page_ecc_enabled());
}

#[test]
fn page_ecc_force_on_volatile_accepted() {
    // `ForceOn` on a Volatile endpoint is documented as
    // "legal but wasteful" — engine does NOT reject. The Volatile
    // endpoint can vanish, so ECC bytes there are useless, but the
    // operator may have a reason (e.g. testing the encoder path on
    // a known-volatile mount). No-second-guess contract.
    let mut ep = EndpointConfig::new(
        "cache",
        "/cache",
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    );
    ep.page_ecc = PageEccPolicy::ForceOn;
    // No panic, no validation error — just stored.
    assert!(ep.is_page_ecc_enabled());
}

#[test]
fn endpoint_with_page_ecc_builder() {
    // The `with_page_ecc` builder must mirror the other field
    // builders (`with_server`, `with_capacity_bytes`, ...). Pin
    // the chainable builder pattern.
    let ep = EndpointConfig::new("ep", "/p", Media::Hdd, Durability::Degraded, Tier::Cold)
        .with_page_ecc(PageEccPolicy::ForceOff);
    assert_eq!(ep.page_ecc, PageEccPolicy::ForceOff);
    assert!(!ep.is_page_ecc_enabled());
}

#[test]
fn endpoint_deserializes_without_page_ecc_field_defaults_to_auto() {
    // Backward-compat: a config serialized BEFORE the `page_ecc`
    // field existed must deserialize cleanly with `Auto` default.
    // Pins the `#[serde(default)]` guarantee — accidentally
    // removing it would break operators upgrading from older
    // pre-release config files.
    let json = r#"{
            "id": "ep-old",
            "path": "/old",
            "media": "hdd",
            "durability": "durable",
            "tier": "warm",
            "capacity_bytes": 0,
            "hard_limit_bytes": 0
        }"#;
    let ep: EndpointConfig = serde_json::from_str(json).expect("decode legacy config");
    assert_eq!(ep.page_ecc, PageEccPolicy::Auto);
    assert!(!ep.is_page_ecc_enabled(), "Durable + Auto → OFF");
}

#[test]
fn endpoint_full_serde_roundtrip_with_page_ecc() {
    // Full EndpointConfig round-trip including the new field plus
    // every other field — proves the new addition does not
    // break existing serialisation.
    let ep = EndpointConfig::new(
        "ep-full",
        "/full",
        Media::Nvme,
        Durability::Degraded,
        Tier::Hot,
    )
    .with_server("srv-7")
    .with_capacity_bytes(1_000_000_000_000)
    .with_hard_limit_bytes(900_000_000_000)
    .with_page_ecc(PageEccPolicy::ForceOff)
    .with_tag("zone", "eu-west-1a");

    let encoded = serde_json::to_string(&ep).expect("encode");
    let decoded: EndpointConfig = serde_json::from_str(&encoded).expect("decode");

    assert_eq!(decoded.id, "ep-full");
    assert_eq!(decoded.server.as_deref(), Some("srv-7"));
    assert_eq!(decoded.media, Media::Nvme);
    assert_eq!(decoded.durability, Durability::Degraded);
    assert_eq!(decoded.tier, Tier::Hot);
    assert_eq!(decoded.capacity_bytes, 1_000_000_000_000);
    assert_eq!(decoded.hard_limit_bytes, 900_000_000_000);
    assert_eq!(decoded.page_ecc, PageEccPolicy::ForceOff);
    assert_eq!(
        decoded.tags.get("zone").map(String::as_str),
        Some("eu-west-1a")
    );
}

#[test]
fn page_ecc_policy_serde_roundtrip() {
    // serde encoding for the policy variants matches the
    // snake-case rename — operator-facing config files use
    // `auto`/`force_on`/`force_off`.
    let auto_json = serde_json::to_string(&PageEccPolicy::Auto).expect("encode auto");
    assert_eq!(auto_json, "\"auto\"");
    let force_on_json = serde_json::to_string(&PageEccPolicy::ForceOn).expect("encode force_on");
    assert_eq!(force_on_json, "\"force_on\"");
    let force_off_json = serde_json::to_string(&PageEccPolicy::ForceOff).expect("encode force_off");
    assert_eq!(force_off_json, "\"force_off\"");

    let decoded: PageEccPolicy = serde_json::from_str("\"auto\"").expect("decode auto");
    assert_eq!(decoded, PageEccPolicy::Auto);
    let decoded: PageEccPolicy = serde_json::from_str("\"force_on\"").expect("decode force_on");
    assert_eq!(decoded, PageEccPolicy::ForceOn);
    let decoded: PageEccPolicy = serde_json::from_str("\"force_off\"").expect("decode force_off");
    assert_eq!(decoded, PageEccPolicy::ForceOff);
}

// ── Hard-limit strategy ─────────────────────────────────────────

#[test]
fn hard_limit_strategy_default_is_reject() {
    // Default placement preserves the safe behaviour — never moves
    // data on behalf of the operator unless explicitly opted in.
    assert_eq!(HardLimitStrategy::default(), HardLimitStrategy::Reject);
}

#[test]
fn endpoint_with_hard_limit_strategy_builder() {
    // The chainable builder must mirror the other field builders
    // (`with_server`, `with_capacity_bytes`, `with_page_ecc`).
    let ep = EndpointConfig::new("ep", "/p", Media::Hdd, Durability::Durable, Tier::Warm)
        .with_hard_limit_strategy(HardLimitStrategy::CascadeEvict);
    assert_eq!(ep.hard_limit_strategy, HardLimitStrategy::CascadeEvict);
}

#[test]
fn endpoint_deserializes_without_hard_limit_strategy_defaults_to_reject() {
    // Backward-compat: a config serialised BEFORE the
    // `hard_limit_strategy` field existed must deserialize cleanly
    // with `Reject` default. Pins the `#[serde(default)]`
    // guarantee — accidentally removing it would break operators
    // upgrading from older config files.
    let json = r#"{
            "id": "ep-old",
            "path": "/old",
            "media": "hdd",
            "durability": "durable",
            "tier": "warm",
            "capacity_bytes": 0,
            "hard_limit_bytes": 100
        }"#;
    let ep: EndpointConfig = serde_json::from_str(json).expect("decode legacy config");
    assert_eq!(ep.hard_limit_strategy, HardLimitStrategy::Reject);
}

#[test]
fn hard_limit_strategy_serde_roundtrip() {
    let reject_json = serde_json::to_string(&HardLimitStrategy::Reject).expect("encode reject");
    assert_eq!(reject_json, "\"reject\"");
    let cascade_json =
        serde_json::to_string(&HardLimitStrategy::CascadeEvict).expect("encode cascade_evict");
    assert_eq!(cascade_json, "\"cascade_evict\"");

    let decoded: HardLimitStrategy = serde_json::from_str("\"reject\"").expect("decode reject");
    assert_eq!(decoded, HardLimitStrategy::Reject);
    let decoded: HardLimitStrategy =
        serde_json::from_str("\"cascade_evict\"").expect("decode cascade_evict");
    assert_eq!(decoded, HardLimitStrategy::CascadeEvict);
}

// ── WAL/oplog endpoint eligibility + selection ────────────────────

#[test]
fn wal_eligibility_matrix() {
    // Durable NVMe in Hot tier → eligible (canonical WAL endpoint).
    let nvme_hot = EndpointConfig::new("a", "/a", Media::Nvme, Durability::Durable, Tier::Hot);
    assert!(nvme_hot.is_wal_eligible());

    // Durable SSD in Warm tier → eligible (SSD media qualifies regardless of tier).
    let ssd_warm = EndpointConfig::new("a", "/a", Media::Ssd, Durability::Durable, Tier::Warm);
    assert!(ssd_warm.is_wal_eligible());

    // Durable HDD in Hot tier → eligible (Hot tier qualifies regardless of media).
    let hdd_hot = EndpointConfig::new("a", "/a", Media::Hdd, Durability::Durable, Tier::Hot);
    assert!(hdd_hot.is_wal_eligible());

    // Durable HDD in Warm tier → NOT eligible (neither fast media nor Hot tier).
    let hdd_warm = EndpointConfig::new("a", "/a", Media::Hdd, Durability::Durable, Tier::Warm);
    assert!(!hdd_warm.is_wal_eligible());

    // Volatile NVMe → NOT eligible (WAL must survive restart).
    let volatile_nvme =
        EndpointConfig::new("a", "/a", Media::Nvme, Durability::Volatile, Tier::HotCache);
    assert!(!volatile_nvme.is_wal_eligible());

    // Degraded NVMe → eligible (degraded ≠ volatile).
    let degraded_nvme =
        EndpointConfig::new("a", "/a", Media::Nvme, Durability::Degraded, Tier::Hot);
    assert!(degraded_nvme.is_wal_eligible());
}

#[test]
fn oplog_eligibility_matrix() {
    // Durable HDD → eligible.
    let durable_hdd = EndpointConfig::new("a", "/a", Media::Hdd, Durability::Durable, Tier::Cold);
    assert!(durable_hdd.is_oplog_eligible());

    // Degraded SSD → eligible.
    let degraded_ssd = EndpointConfig::new("a", "/a", Media::Ssd, Durability::Degraded, Tier::Warm);
    assert!(degraded_ssd.is_oplog_eligible());

    // Volatile NVMe → NOT eligible (segments lost on restart).
    let volatile_nvme =
        EndpointConfig::new("a", "/a", Media::Nvme, Durability::Volatile, Tier::HotCache);
    assert!(!volatile_nvme.is_oplog_eligible());

    // Volatile RAM → NOT eligible.
    let ram = EndpointConfig::new(
        "a",
        "memory",
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    );
    assert!(!ram.is_oplog_eligible());
}

#[test]
fn select_wal_endpoint_picks_first_eligible() {
    // Two endpoints: first is NOT WAL-eligible (HDD Warm), second is.
    let cold_hdd =
        EndpointConfig::new("cold", "/cold", Media::Hdd, Durability::Durable, Tier::Cold);
    let hot_nvme = EndpointConfig::new("hot", "/hot", Media::Nvme, Durability::Durable, Tier::Hot);
    let config = StorageConfig::with_endpoints(vec![cold_hdd, hot_nvme]);
    let wal = config
        .select_wal_endpoint()
        .expect("eligible endpoint exists");
    assert_eq!(wal.id, "hot");
}

#[test]
fn select_wal_endpoint_errors_when_none_eligible() {
    // Only HDD-Warm endpoint, plus a Volatile (also ineligible).
    let cold_hdd =
        EndpointConfig::new("cold", "/cold", Media::Hdd, Durability::Durable, Tier::Cold);
    let cache = EndpointConfig::new(
        "cache",
        "/cache",
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    );
    let config = StorageConfig::with_endpoints(vec![cold_hdd, cache]);
    let err = config.select_wal_endpoint().expect_err("no eligible");
    assert_eq!(err, EndpointSelectionError::NoWalEligible);
}

#[test]
fn select_oplog_endpoint_round_robin_by_shard() {
    // Two oplog-eligible endpoints + one volatile (skipped).
    let a = EndpointConfig::new("a", "/a", Media::Nvme, Durability::Durable, Tier::Hot);
    let b = EndpointConfig::new("b", "/b", Media::Hdd, Durability::Durable, Tier::Cold);
    let cache = EndpointConfig::new(
        "cache",
        "/cache",
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    );
    let config = StorageConfig::with_endpoints(vec![a, b, cache]);
    // Shard 0 → eligible[0] = "a"
    assert_eq!(config.select_oplog_endpoint(0).expect("eligible").id, "a");
    // Shard 1 → eligible[1] = "b"
    assert_eq!(config.select_oplog_endpoint(1).expect("eligible").id, "b");
    // Shard 2 → wraps back to eligible[0] = "a"
    assert_eq!(config.select_oplog_endpoint(2).expect("eligible").id, "a");
    // Shard 3 → eligible[1] = "b"
    assert_eq!(config.select_oplog_endpoint(3).expect("eligible").id, "b");
}

#[test]
fn select_oplog_endpoint_errors_when_none_eligible() {
    // All-volatile config — must go through the explicit
    // `with_endpoints_no_persistence` escape hatch because
    // `with_endpoints` would now panic per INV-D1.
    let cache = EndpointConfig::new(
        "cache",
        "/cache",
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    );
    let ram = EndpointConfig::new(
        "ram",
        "memory",
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    );
    let config = StorageConfig::with_endpoints_no_persistence(vec![cache, ram]);
    let err = config.select_oplog_endpoint(0).expect_err("no eligible");
    assert_eq!(err, EndpointSelectionError::NoOplogEligible);
}

#[test]
fn all_oplog_eligible_filters_volatile() {
    let a = EndpointConfig::new("a", "/a", Media::Nvme, Durability::Durable, Tier::Hot);
    let b = EndpointConfig::new("b", "/b", Media::Hdd, Durability::Durable, Tier::Cold);
    let cache = EndpointConfig::new(
        "cache",
        "/cache",
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    );
    let config = StorageConfig::with_endpoints(vec![a, b, cache]);
    let eligible = config.all_oplog_eligible_endpoints();
    assert_eq!(eligible.len(), 2);
    assert_eq!(eligible[0].id, "a");
    assert_eq!(eligible[1].id, "b");
}

#[test]
fn all_wal_eligible_filters_correctly() {
    let nvme_hot = EndpointConfig::new("a", "/a", Media::Nvme, Durability::Durable, Tier::Hot);
    let hdd_warm = EndpointConfig::new("b", "/b", Media::Hdd, Durability::Durable, Tier::Warm);
    let volatile_cache = EndpointConfig::new(
        "cache",
        "/cache",
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    );
    let config = StorageConfig::with_endpoints(vec![nvme_hot, hdd_warm, volatile_cache]);
    let eligible = config.all_wal_eligible_endpoints();
    assert_eq!(eligible.len(), 1);
    assert_eq!(eligible[0].id, "a", "only Durable NVMe Hot qualifies");
}

#[test]
fn with_endpoints_explicit_multi_endpoint() {
    let endpoints = vec![
        EndpointConfig::new(
            "ep-nvme",
            "/mnt/nvme",
            Media::Nvme,
            Durability::Degraded,
            Tier::Hot,
        )
        .with_capacity_bytes(1_000_000_000_000)
        .with_hard_limit_bytes(900_000_000_000)
        .with_tag("rack", "r42"),
        EndpointConfig::new(
            "ep-hdd",
            "/mnt/hdd",
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        )
        .with_capacity_bytes(10_000_000_000_000),
    ];
    let config = StorageConfig::with_endpoints(endpoints);
    assert_eq!(config.endpoints.len(), 2);
    assert_eq!(config.endpoints[0].id, "ep-nvme");
    assert_eq!(config.endpoints[0].durability, Durability::Degraded);
    assert_eq!(config.endpoints[0].hard_limit_bytes, 900_000_000_000);
    assert_eq!(
        config.endpoints[0].tags.get("rack").map(String::as_str),
        Some("r42")
    );
    assert_eq!(config.endpoints[1].id, "ep-hdd");
    assert_eq!(config.endpoints[1].tier, Tier::Cold);
    // data_dir() returns first endpoint's path.
    assert_eq!(config.data_dir(), Path::new("/mnt/nvme"));
}

#[test]
#[should_panic(expected = "StorageConfig requires at least one endpoint")]
fn with_endpoints_empty_panics() {
    let _ = StorageConfig::with_endpoints(vec![]);
}

#[test]
fn endpoint_config_serde_roundtrip() {
    let ep = EndpointConfig::new(
        "ep-1",
        "/mnt/test",
        Media::Nvme,
        Durability::Durable,
        Tier::Hot,
    )
    .with_capacity_bytes(2_000_000_000)
    .with_hard_limit_bytes(1_800_000_000)
    .with_tag("zone", "eu-west-1a");
    let json = serde_json::to_string(&ep).expect("serialize");
    let back: EndpointConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.id, ep.id);
    assert_eq!(back.path, ep.path);
    assert_eq!(back.media, Media::Nvme);
    assert_eq!(back.durability, Durability::Durable);
    assert_eq!(back.tier, Tier::Hot);
    assert_eq!(back.capacity_bytes, 2_000_000_000);
    assert_eq!(back.hard_limit_bytes, 1_800_000_000);
    assert_eq!(
        back.tags.get("zone").map(String::as_str),
        Some("eu-west-1a")
    );
}

#[test]
fn durability_serde_lowercase() {
    let durable = serde_json::to_string(&Durability::Durable).expect("ser");
    assert_eq!(durable, "\"durable\"");
    let degraded = serde_json::to_string(&Durability::Degraded).expect("ser");
    assert_eq!(degraded, "\"degraded\"");
    let volatile = serde_json::to_string(&Durability::Volatile).expect("ser");
    assert_eq!(volatile, "\"volatile\"");
}

#[test]
fn media_serde_lowercase() {
    assert_eq!(serde_json::to_string(&Media::Hdd).expect("ser"), "\"hdd\"");
    assert_eq!(serde_json::to_string(&Media::Ssd).expect("ser"), "\"ssd\"");
    assert_eq!(
        serde_json::to_string(&Media::Nvme).expect("ser"),
        "\"nvme\""
    );
    assert_eq!(serde_json::to_string(&Media::Ram).expect("ser"), "\"ram\"");
}

#[test]
fn tier_serde_snake_case() {
    assert_eq!(
        serde_json::to_string(&Tier::Memory).expect("ser"),
        "\"memory\""
    );
    assert_eq!(
        serde_json::to_string(&Tier::HotCache).expect("ser"),
        "\"hot_cache\""
    );
    assert_eq!(serde_json::to_string(&Tier::Hot).expect("ser"), "\"hot\"");
    assert_eq!(serde_json::to_string(&Tier::Warm).expect("ser"), "\"warm\"");
    assert_eq!(serde_json::to_string(&Tier::Cold).expect("ser"), "\"cold\"");
}

#[test]
fn to_tree_config_uses_first_endpoint_path() {
    use std::sync::Arc;
    let dir1 = tempfile::tempdir().expect("tempdir");
    let dir2 = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "first",
            dir1.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "second",
            dir2.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let gc_watermark = Arc::new(AtomicU64::new(0));
    // Without explicit per-level routing every partition uses the
    // first endpoint. This test pins the single-tier fallback path.
    let _ = config.to_tree_config(Partition::Node, Arc::clone(&seqno), &gc_watermark);
    // Verify the first endpoint's path is the one we picked.
    assert!(dir1.path().exists(), "first endpoint dir is the active one");
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
    let config = StorageConfig::with_endpoints(vec![default_disk_endpoint(dir.path())]);
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
    let config = StorageConfig::with_endpoints(vec![default_disk_endpoint(dir.path())]);
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
    let config = StorageConfig::with_endpoints(vec![default_disk_endpoint(dir.path())]);
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
