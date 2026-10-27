//! Hard-limit enforcement + per-endpoint capacity tracking
//! (`used ≤ hard_limit` at all times).
//!
//! Covers:
//! - the capacity report follows the owned footprint, which open measures
//!   from the files already on the endpoint,
//! - severity transitions (Normal → Warning → Critical → Emergency → Full)
//!   against the part of the limit client writes may take,
//! - every write path (put, delete, merge, `WriteBatch`, Cypher) is refused
//!   with `CapacityExhausted` once it does not fit, before it lands,
//! - `CascadeEvict` strategy auto-fires at Emergency severity,
//! - `Reject` strategy does NOT auto-fire,
//! - the store's own metadata (Schema) writes into the recovery reserve
//!   client writes leave, never past the limit.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::capacity::CapacitySeverity;
use coordinode_storage::engine::config::{
    Durability, EndpointConfig, HardLimitStrategy, MIN_HARD_LIMIT_BYTES, Media, StorageConfig, Tier,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::StorageError;
use tempfile::TempDir;

/// The smallest supported limit: room for a filled store above what an
/// empty one takes at open, with the engine's own bounded structures.
const ROOMY: u64 = MIN_HARD_LIMIT_BYTES;

/// Tables in the tables folder `dir`, 0 when it does not exist. Empty files
/// are the engine's reserve of files for tables yet to be written, not
/// tables, and their number moves with no data moving.
fn tables_in(dir: &std::path::Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(dir)
        .expect("read tables dir")
        .filter(|entry| {
            entry
                .as_ref()
                .expect("dir entry")
                .metadata()
                .expect("metadata")
                .len()
                > 0
        })
        .count()
}

/// 64 bytes no codec shrinks, distinct per `i`: rows of them take their size
/// on disk, so a test filling an endpoint does not depend on how well the
/// storage format compresses or how compact its filters are.
fn incompressible(i: u32) -> [u8; 64] {
    let mut state = u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = [0u8; 64];
    for chunk in out.chunks_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        chunk.copy_from_slice(&state.to_le_bytes());
    }
    out
}

/// One endpoint at `dir` with a hard limit of `limit`.
fn capped(id: &str, dir: &std::path::Path, limit: u64) -> EndpointConfig {
    EndpointConfig::new(id, dir, Media::Hdd, Durability::Durable, Tier::Warm)
        .with_hard_limit_bytes(limit)
}

/// Put incompressible rows into `part`, numbered from `from`, until the
/// footprint refuses one; returns how many were accepted. Any other error
/// fails the test.
fn fill_from(engine: &StorageEngine, part: Partition, prefix: &str, from: u32) -> u32 {
    for i in from..from + 1_000_000 {
        let key = format!("{prefix}{i:010}");
        match engine.put(part, key.as_bytes(), &incompressible(i)) {
            Ok(()) => {}
            Err(StorageError::CapacityExhausted { .. }) => return i - from,
            Err(other) => panic!("unexpected refusal: {other:?}"),
        }
    }
    panic!("the footprint never refused a write");
}

/// [`fill_from`] from the first row.
fn fill(engine: &StorageEngine, part: Partition, prefix: &str) -> u32 {
    fill_from(engine, part, prefix, 0)
}

/// Fill `part` until even the room the store frees by itself (a refusal
/// flushes the memtables and removes the journal they cover; a flush can
/// start a compaction) takes no more rows: the next write of the same size
/// is refused. Returns how many rows were accepted in all.
fn saturate(engine: &StorageEngine, part: Partition, prefix: &str) -> u32 {
    let mut total = 0;
    loop {
        let accepted = fill_from(engine, part, prefix, total);
        total += accepted;
        if accepted == 0 {
            return total;
        }
        engine
            .persist()
            .expect("a filled store flushes into its reserve");
    }
}

/// What the file system allocated to the files under `root`, each file once
/// however many names it has.
fn allocated_under(root: &std::path::Path) -> u64 {
    let mut seen = std::collections::HashSet::new();
    let mut total = 0u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = std::fs::symlink_metadata(entry.path()) else {
                continue;
            };
            if meta.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if !meta.is_file() {
                continue;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if seen.insert((meta.dev(), meta.ino())) {
                    total += meta.blocks() * 512;
                }
            }
            // No inode to tell links apart: each name counts, by its length
            // in whole 4 KiB blocks.
            #[cfg(not(unix))]
            if seen.insert(entry.path()) {
                total += meta.len().div_ceil(4096) * 4096;
            }
        }
    }
    total
}

/// Writing data + persisting + refreshing the capacity tracker MUST
/// produce a non-zero `used_bytes` reading for the target endpoint.
/// Smoke check on the report path.
#[test]
fn refresh_picks_up_on_disk_sst_bytes() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("only", dir.path(), 10_000_000)]);
    let engine = StorageEngine::open(&config).expect("open");

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();

    let usage = engine.capacity().get("only").expect("endpoint tracked");
    assert!(
        usage.used() > 0,
        "the report must carry the footprint, got 0"
    );
    assert_eq!(usage.severity(), CapacitySeverity::Normal);
}

/// Filling an endpoint MUST end in a write refused with
/// [`StorageError::CapacityExhausted`] naming the endpoint and its limit,
/// before the files under the endpoint pass the limit.
#[test]
fn full_endpoint_rejects_subsequent_writes() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("small", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");

    let accepted = saturate(&engine, Partition::Node, "node:0:");
    assert!(accepted > 0, "an empty store refused its first write");
    assert!(allocated_under(dir.path()) <= ROOMY);

    let result = engine.put(Partition::Node, b"node:0:after-full", &incompressible(0));
    match result {
        Err(StorageError::CapacityExhausted {
            endpoint_id,
            hard_limit_bytes,
            ..
        }) => {
            assert_eq!(endpoint_id, "small");
            assert_eq!(hard_limit_bytes, ROOMY);
        }
        other => panic!("expected CapacityExhausted after endpoint filled, got: {other:?}"),
    }
    engine
        .persist()
        .expect("a full store still flushes into its reserve");
    assert!(allocated_under(dir.path()) <= ROOMY);
}

/// The store's own metadata MUST stay writable once client writes are
/// refused: a Schema write uses the recovery reserve the client writes
/// leave. Client writes stay refused.
#[test]
fn schema_writes_use_the_recovery_reserve_client_writes_leave() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("small", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");

    saturate(&engine, Partition::Node, "node:0:");

    engine
        .put(Partition::Schema, b"schema:label:Test", b"{}")
        .expect("Schema must remain writable when client writes are refused");

    let denied = engine.put(Partition::Node, b"node:0:denied", b"x");
    assert!(
        matches!(denied, Err(StorageError::CapacityExhausted { .. })),
        "user partition must still be refused",
    );
    assert!(allocated_under(dir.path()) <= ROOMY);
}

/// Files removed behind the engine's back are not seen while it runs: their
/// charge stays, so the account never falls below what the engine itself
/// holds. The next open measures the files as they are.
#[test]
fn files_removed_behind_the_engine_are_released_at_the_next_open() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("rec", dir.path(), ROOMY)]);
    let foreign = dir.path().join("operator-copy.bin");
    {
        // A first open lays the store out; a file an operator leaves on the
        // endpoint is measured by the next one.
        drop(StorageEngine::open(&config).expect("first open"));
    }
    std::fs::write(&foreign, vec![7u8; 256 * 1024]).expect("write foreign file");
    {
        let engine = StorageEngine::open(&config).expect("open");
        let footprint = engine.footprints().get("rec").expect("capped");
        let charge = footprint.file(&foreign);
        assert!(charge.bytes() >= 256 * 1024, "open measured the file");
        std::fs::remove_file(&foreign).expect("remove behind the engine");
        assert!(
            charge.bytes() >= 256 * 1024,
            "the running engine still counts the removed file"
        );
    }
    let engine = StorageEngine::open(&config).expect("reopen");
    assert_eq!(
        engine
            .footprints()
            .get("rec")
            .expect("capped")
            .file(&foreign)
            .bytes(),
        0,
        "the reopened store measured the files as they are"
    );
}

/// `HardLimitStrategy::CascadeEvict` MUST fire a cascade-eviction
/// when the scanner observes Emergency severity. The endpoint's
/// SSTs get demoted via major compaction (the per-LSM-level routing
/// mechanism).
#[test]
fn cascade_evict_strategy_fires_at_emergency_threshold() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let mut config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(16 * ROOMY)
        .with_hard_limit_strategy(HardLimitStrategy::CascadeEvict),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        )
        .with_hard_limit_bytes(0), // unlimited
    ]);
    // The cascade is what must move the data. A background compaction routes
    // the tables to cold by itself, before the limit is lowered on a fast
    // host, and leaves the hot endpoint holding nothing to judge.
    config.compaction_workers = 0;
    let engine = StorageEngine::open(&config).expect("open");

    // Rows of 1 KiB no codec shrinks, counted by what was written rather than
    // by the charge, which also holds what files still being written are
    // allowed.
    const ROW: u64 = 16 * 64;
    let written = 2 * MIN_HARD_LIMIT_BYTES + 2 * (1 << 20);
    let mut i = 0u32;
    while u64::from(i) * ROW < written {
        let key = format!("node:0:{i:010}");
        let value: Vec<u8> = (0..16u32)
            .flat_map(|k| incompressible(i * 16 + k))
            .collect();
        engine
            .put(Partition::Node, key.as_bytes(), &value)
            .expect("put under a roomy limit");
        i += 1;
    }
    engine.persist().expect("persist");

    // The limit lowered at runtime so that the client share is the supported
    // minimum, under half of the data the hot endpoint holds, puts it at
    // Full at once. The reserve left above the client share is the room the
    // cascade's own work uses.
    let cold_tables = cold.path().join(Partition::Node.name()).join("tables");
    let hot_before = engine.footprints().get("ep-hot").expect("capped").charged();
    let reserve = coordinode_storage::engine::footprint::OVER_LIMIT_ALLOWANCE;
    engine
        .set_hard_limit("ep-hot", MIN_HARD_LIMIT_BYTES + reserve, Some(reserve))
        .expect("lower the hot limit");

    // The refresh sees the endpoint at Emergency or Full and cascades. The
    // engine's own scanner runs the same refresh every few seconds, and on a
    // loaded host its pass can come first: then the cascade has already moved
    // the data and this refresh finds the endpoint back to Normal.
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep-hot").expect("tracked");
    assert!(
        matches!(
            usage.severity(),
            CapacitySeverity::Emergency | CapacitySeverity::Full
        ) || tables_in(&cold_tables) > 0,
        "a filled endpoint must be at Emergency or Full, or cascaded already; got {:?} \
         (used={}) and nothing on the cold endpoint",
        usage.severity(),
        usage.used()
    );

    // The cascade is a major compaction routing the bottom level to cold: the
    // tables there are merged, so their number says nothing; what moves is
    // the data, which stays readable, and the hot endpoint's footprint.
    assert!(
        tables_in(&cold_tables) > 0,
        "the cascade routes the bottom level to the cold endpoint"
    );
    // The tables the cascade replaced leave hot once no read still holds
    // them, which a loaded machine delays.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let hot_after = loop {
        let now = engine.footprints().get("ep-hot").expect("capped").charged();
        if now < hot_before || std::time::Instant::now() >= deadline {
            break now;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    assert!(
        hot_after < hot_before,
        "the cascade frees the hot endpoint (before {hot_before}, after {hot_after})"
    );
    assert!(
        engine
            .get(Partition::Node, b"node:0:0000000000")
            .expect("read")
            .is_some(),
        "the cascaded rows stay readable"
    );
}

/// `HardLimitStrategy::Reject` (default) MUST NOT auto-fire cascade
/// eviction even at Emergency severity. The endpoint just sits there
/// near-full until operator intervention. Verified by filling an endpoint
/// under the `Reject` strategy, then observing that the strategy field on
/// the tracker is preserved as `Reject` after the refresh pass (the
/// auto-cascade branch is gated on `CascadeEvict`).
#[test]
fn reject_strategy_preserved_no_auto_cascade() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(ROOMY),
        // strategy defaults to Reject
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Node tables move down to the cold tier as they compact; the schema
    // partition stays on the first endpoint, so it is what fills hot.
    fill(&engine, Partition::Schema, "schema:fill:");
    engine.refresh_capacity();

    let usage = engine.capacity().get("ep-hot").expect("tracked");
    // The Reject strategy is stored as-configured (would be
    // CascadeEvict if the auto-cascade path had upgraded the field).
    assert_eq!(
        usage.strategy,
        HardLimitStrategy::Reject,
        "strategy field must persist as Reject — not swapped to CascadeEvict",
    );
    assert_eq!(
        usage.severity(),
        CapacitySeverity::Full,
        "a store that refuses writes reports Full",
    );
}

/// A reopened store's footprint starts from the files: it covers the space
/// the file system allocated to every file under the endpoint, each once,
/// and the report carries it before any refresh.
#[test]
fn reopen_measures_the_footprint_from_the_files() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("ep", dir.path(), 10_000_000)]);
    {
        let engine = StorageEngine::open(&config).expect("first open");
        for i in 0..500u32 {
            let key = format!("node:0:{i:010}");
            engine
                .put(Partition::Node, key.as_bytes(), b"payload-bytes")
                .expect("put");
        }
        engine.persist().expect("persist");
    }
    let engine = StorageEngine::open(&config).expect("reopen");
    assert!(engine.capacity().get("ep").unwrap().used() > 0);
    // Open may itself remove and write files: compare with the files as they
    // are once it returned.
    let on_disk = allocated_under(dir.path());
    let charged = engine.footprints().get("ep").expect("capped").charged();
    assert!(
        charged >= on_disk,
        "the reopened footprint ({charged}) must cover what the files take ({on_disk})",
    );
}

/// Prometheus counter `endpoint_threshold_alerts_total` MUST
/// increment on every severity UP-crossing. Drives the endpoint into
/// `Full` severity and inspects the captured counter value.
///
/// The crossing is recorded by whichever refresh sees it first: this
/// thread's, or the engine's background capacity scanner on its own thread
/// once its interval passes (a loaded machine takes that long). The
/// recorder is therefore the process's, installed before the engine opens;
/// the test runs in a process of its own.
#[test]
fn threshold_alert_counter_increments_on_severity_crossing() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder.install().expect("the process's metrics recorder");

    let dir = TempDir::new().expect("tempdir");
    // Filled until refused: the severity climbs from Normal to Full.
    let config = StorageConfig::with_endpoints(vec![capped("alert-ep", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");
    fill(&engine, Partition::Node, "node:0:");
    engine.refresh_capacity();

    // Walk the snapshot and find the threshold-alert counter for our
    // endpoint. Any UP-crossing (Normal→Warning, Warning→Full, etc.)
    // contributes one increment.
    let snapshot = snapshotter.snapshot();
    let mut alert_count = 0u64;
    for (key, _unit, _desc, value) in snapshot.into_vec() {
        if key.key().name() == "endpoint_threshold_alerts_total" {
            let has_endpoint = key
                .key()
                .labels()
                .any(|l| l.key() == "endpoint_id" && l.value() == "alert-ep");
            if has_endpoint {
                if let DebugValue::Counter(n) = value {
                    alert_count += n;
                }
            }
        }
    }
    assert!(
        alert_count >= 1,
        "expected at least one threshold alert counter increment on severity crossing, got {alert_count}",
    );
}

/// Prometheus counter `endpoint_cascade_events_total` MUST increment
/// each time auto-cascade fires. Triggers Full severity on a
/// CascadeEvict-strategy endpoint and inspects the counter.
#[test]
fn cascade_events_counter_increments_on_auto_cascade() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");
    // Filled until it refuses a write: its refresh then sees Full, so
    // auto-cascade fires. Node tables move down to the cold tier as they
    // compact; the schema partition stays on the first endpoint, so it is
    // what fills hot.
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(ROOMY)
        .with_hard_limit_strategy(HardLimitStrategy::CascadeEvict),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    fill(&engine, Partition::Schema, "schema:fill:");
    metrics::with_local_recorder(&recorder, || {
        engine.refresh_capacity();
    });

    let snapshot = snapshotter.snapshot();
    let mut cascade_count = 0u64;
    for (key, _unit, _desc, value) in snapshot.into_vec() {
        if key.key().name() == "endpoint_cascade_events_total" {
            let has_endpoint = key
                .key()
                .labels()
                .any(|l| l.key() == "endpoint_id" && l.value() == "ep-hot");
            if has_endpoint {
                if let DebugValue::Counter(n) = value {
                    cascade_count += n;
                }
            }
        }
    }
    assert!(
        cascade_count >= 1,
        "expected at least one cascade event counter increment when auto-cascade fired, got {cascade_count}",
    );
}

/// Gauges `endpoint_used_bytes`, `endpoint_hard_limit_bytes` and the
/// footprint's own (`coordinode_footprint_*`) MUST be set on every refresh
/// of a capped endpoint, observable via the same debugging recorder.
/// Pins the gauge contract.
#[test]
fn capacity_gauges_emitted_on_refresh() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("gauge-ep", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        for i in 0..200u32 {
            let key = format!("node:0:{i:010}");
            engine
                .put(Partition::Node, key.as_bytes(), b"payload-bytes")
                .expect("put");
        }
        engine.persist().expect("persist");
        engine.refresh_capacity();
    });

    let snapshot = snapshotter.snapshot();
    let mut seen = std::collections::HashMap::new();
    for (key, _unit, _desc, value) in snapshot.into_vec() {
        let endpoint_ok = key
            .key()
            .labels()
            .any(|l| l.key() == "endpoint_id" && l.value() == "gauge-ep");
        if let (true, DebugValue::Gauge(g)) = (endpoint_ok, value) {
            seen.insert(key.key().name().to_string(), g.into_inner());
        }
    }
    assert!(seen["endpoint_used_bytes"] > 0.0, "used gauge after writes");
    assert_eq!(seen["endpoint_hard_limit_bytes"], ROOMY as f64);
    assert!(seen["coordinode_footprint_charged_bytes"] > 0.0);
    assert_eq!(seen["coordinode_footprint_limit_bytes"], ROOMY as f64);
    assert_eq!(
        seen["coordinode_footprint_reserve_bytes"],
        (ROOMY / 16) as f64,
        "the default reserve is a sixteenth of the limit"
    );
    assert!(seen.contains_key("coordinode_footprint_admitted_bytes"));
    assert!(seen.contains_key("coordinode_footprint_flush_debt_bytes"));
}

/// Concurrent put + refresh: writers and the capacity scanner must
/// not race in ways that produce torn reads or panics. Spawns N
/// writer threads doing bulk puts while a parallel thread loops on
/// `refresh_capacity()`. Final state must be consistent: every key
/// written before the test stopped is readable.
#[test]
fn concurrent_put_and_refresh_no_torn_reads() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let dir = TempDir::new().expect("tempdir");
    // Generous limit so writers are not refused mid-test.
    let config = StorageConfig::with_endpoints(vec![capped("ep", dir.path(), 1_000_000_000)]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));

    let stop = Arc::new(AtomicBool::new(false));

    // Refresher thread: bangs on refresh_capacity() while writers push
    // data through. A race in the AtomicU64 / Mutex<CapacitySeverity>
    // code path would surface as a panic (poison).
    let refresher_handle = {
        let engine = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                engine.refresh_capacity();
                let usage = engine.capacity().get("ep").expect("tracked");
                // u64 atomics never expose half-written values; the
                // footprint grows with flushes and shrinks with
                // compactions, so no ordering across reads is asserted.
                let _ = usage.used();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        })
    };

    // 4 writer threads each pushing 250 puts = 1000 writes total.
    let mut writer_handles = Vec::new();
    for w in 0..4 {
        let engine = Arc::clone(&engine);
        writer_handles.push(std::thread::spawn(move || {
            for i in 0..250u32 {
                let key = format!("node:0:w{w}:{i:010}");
                engine
                    .put(Partition::Node, key.as_bytes(), b"payload-bytes")
                    .expect("put");
            }
        }));
    }
    for h in writer_handles {
        h.join().expect("writer join");
    }
    engine.persist().expect("persist");
    stop.store(true, Ordering::Release);
    refresher_handle.join().expect("refresher join");

    for w in 0..4 {
        for i in 0..250u32 {
            let key = format!("node:0:w{w}:{i:010}");
            let v = engine
                .get(Partition::Node, key.as_bytes())
                .expect("get")
                .unwrap_or_else(|| panic!("missing key after concurrent run: {key}"));
            assert_eq!(&v[..], b"payload-bytes");
        }
    }
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");
    assert!(usage.used() > 0);
    assert_eq!(
        usage.severity(),
        CapacitySeverity::Normal,
        "well under 1 GB"
    );
}

/// A limit lowered below what the store holds deletes nothing: the store
/// reopened under it keeps every row readable and refuses client writes at
/// once, its report at Full before any scan.
#[test]
fn a_store_reopened_under_a_lower_limit_refuses_writes_and_keeps_its_data() {
    let dir = TempDir::new().expect("tempdir");
    let accepted = {
        let engine = StorageEngine::open(&StorageConfig::with_endpoints(vec![capped(
            "ep",
            dir.path(),
            2 * ROOMY,
        )]))
        .expect("first open");
        let accepted = fill(&engine, Partition::Node, "node:0:");
        engine.persist().expect("persist");
        accepted
    };

    let engine = StorageEngine::open(&StorageConfig::with_endpoints(vec![capped(
        "ep",
        dir.path(),
        ROOMY,
    )]))
    .expect("reopen under a lower limit");
    let usage = engine.capacity().get("ep").unwrap();
    assert_eq!(usage.severity(), CapacitySeverity::Full, "before any scan");
    let denied = engine.put(Partition::Node, b"node:0:after-reopen", b"x");
    assert!(
        matches!(denied, Err(StorageError::CapacityExhausted { .. })),
        "a store past its limit refuses writes on reopen, got: {denied:?}",
    );
    for i in 0..accepted {
        let key = format!("node:0:{i:010}");
        assert!(
            engine
                .get(Partition::Node, key.as_bytes())
                .unwrap()
                .is_some(),
            "row {i} was deleted to fit the lower limit"
        );
    }
}

/// Endpoints without a `hard_limit_bytes` configuration (default `0`)
/// MUST never enter alert severity or refuse writes. The tracker still
/// records `used_bytes` for diagnostics but the limit is "unlimited"
/// semantically.
#[test]
fn endpoint_without_hard_limit_never_alerts() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "untracked",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload-bytes")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();

    let usage = engine.capacity().get("untracked").unwrap();
    assert_eq!(usage.hard_limit_bytes, 0, "no limit configured");
    assert_eq!(
        usage.severity(),
        CapacitySeverity::Normal,
        "unlimited endpoint must stay Normal regardless of usage",
    );
    // Subsequent writes continue to succeed.
    engine
        .put(Partition::Node, b"node:0:no-limit", b"v")
        .expect("unlimited endpoint accepts writes indefinitely");
}

/// Multi-endpoint capacity tracking: each endpoint's `used_bytes` is its
/// own footprint, and severity on one endpoint does NOT affect any other
/// endpoint's report. Production CE+EE tier deployments rely on this: the
/// scanner must report per-endpoint state, never aggregate.
#[test]
fn multi_endpoint_tracking_is_independent_per_endpoint() {
    let hot = TempDir::new().expect("hot tempdir");
    let warm = TempDir::new().expect("warm tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    // Three endpoints, three different hard_limits. Default routing
    // places L0-L1 on Hot (the first endpoint), L2-L3 on Warm,
    // L4-L6 on Cold. We will force data through compaction so all
    // three tiers receive bytes, then assert each endpoint's tracker
    // is updated independently.
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(1_000_000_000),
        EndpointConfig::new(
            "ep-warm",
            warm.path(),
            Media::Ssd,
            Durability::Durable,
            Tier::Warm,
        )
        .with_hard_limit_bytes(2_000_000_000),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        )
        .with_hard_limit_bytes(10_000_000_000),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write + persist + major-compact to populate L0 and L4+ levels.
    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload-bytes")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine
        .major_compact(Partition::Node)
        .expect("major compact");
    engine.refresh_capacity();

    // Each endpoint must have an independent tracker entry.
    let hot_usage = engine.capacity().get("ep-hot").expect("hot tracked");
    let warm_usage = engine.capacity().get("ep-warm").expect("warm tracked");
    let cold_usage = engine.capacity().get("ep-cold").expect("cold tracked");

    // hard_limit_bytes preserved per-endpoint.
    assert_eq!(hot_usage.hard_limit_bytes, 1_000_000_000);
    assert_eq!(warm_usage.hard_limit_bytes, 2_000_000_000);
    assert_eq!(cold_usage.hard_limit_bytes, 10_000_000_000);

    // Cold gets the bottom-level SSTs after major compaction → its
    // tracker MUST report > 0 used_bytes. Hot keeps the partition
    // manifest, so it also reports > 0 (manifest bytes).
    assert!(
        cold_usage.used() > 0,
        "cold endpoint must hold bottom-level SSTs after major_compact, got 0",
    );
    assert!(
        hot_usage.used() > 0,
        "hot endpoint must hold the partition manifest, got 0",
    );

    // All three are well under their limits, so severity stays Normal
    // independently — no endpoint's severity is "inherited" from
    // another.
    assert_eq!(hot_usage.severity(), CapacitySeverity::Normal);
    assert_eq!(warm_usage.severity(), CapacitySeverity::Normal);
    assert_eq!(cold_usage.severity(), CapacitySeverity::Normal);
}

/// Multi-endpoint scenario: one endpoint past its limit MUST NOT report
/// another as full, and MUST NOT trigger auto-cascade for an endpoint that
/// is itself not in Emergency. Pins per-endpoint isolation of the report.
#[test]
fn multi_endpoint_full_state_does_not_leak_across_endpoints() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        // The smallest Hot limit, filled below.
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(ROOMY),
        // Huge Cold limit so it stays Normal.
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        )
        .with_hard_limit_bytes(10_000_000_000),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Node tables move down to the cold tier as they compact; the schema
    // partition stays on the first endpoint, so it is what fills hot.
    fill(&engine, Partition::Schema, "schema:fill:");
    engine.refresh_capacity();

    let hot_usage = engine.capacity().get("ep-hot").expect("tracked");
    let cold_usage = engine.capacity().get("ep-cold").expect("tracked");
    assert_eq!(hot_usage.severity(), CapacitySeverity::Full);
    assert_eq!(
        cold_usage.severity(),
        CapacitySeverity::Normal,
        "cold endpoint must NOT inherit Hot's Full state",
    );
    assert!(allocated_under(cold.path()) < 10_000_000_000);
}

/// DOWN-crossing severity (Full → Normal after the limit is raised at
/// runtime) MUST NOT increment `endpoint_threshold_alerts_total`. The
/// counter measures "alerts went off", not "severity changed in either
/// direction". Pins the asymmetric semantics of the alert counter, and a
/// raised limit admitting writes again without a restart.
#[test]
fn down_crossing_severity_does_not_increment_alert_counter() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("ep", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Step 1: write until refused (UP-crossing increments counter).
    fill(&engine, Partition::Node, "node:0:");
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");
    assert!(
        matches!(
            usage.severity(),
            CapacitySeverity::Emergency | CapacitySeverity::Full
        ),
        "a filled endpoint is at Emergency or Full, got {:?}",
        usage.severity()
    );

    // Step 2: raise the limit inside a fresh recorder: the DOWN-crossing
    // back to Normal must NOT increment the alert counter.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        engine
            .footprints()
            .get("ep")
            .expect("capped")
            .set_limit(64 * ROOMY, 0);
        engine.refresh_capacity();
    });

    let usage = engine.capacity().get("ep").expect("tracked");
    assert_eq!(
        usage.severity(),
        CapacitySeverity::Normal,
        "a raised limit drops severity back to Normal",
    );
    engine
        .put(Partition::Node, b"node:0:after-raise", b"v")
        .expect("a raised limit admits writes without a restart");

    let snapshot = snapshotter.snapshot();
    for (key, _unit, _desc, value) in snapshot.into_vec() {
        if key.key().name() == "endpoint_threshold_alerts_total" {
            if let DebugValue::Counter(n) = value {
                assert_eq!(
                    n, 0,
                    "DOWN-crossing must not increment the alert counter — \
                     the counter records severity escalations only",
                );
            }
        }
    }
}

// ── Every write path is admitted before it lands ──────────────────────
//
// A write path left out of admission would succeed on a full endpoint and
// pass the limit. Each test below fills the endpoint, then drives one path.

/// `engine.delete` on a full endpoint MUST reject with `CapacityExhausted`.
/// A delete tombstone still consumes memtable bytes that will eventually
/// flush to an SST, so a full endpoint can't accept it.
#[test]
fn delete_on_full_endpoint_rejects() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("small", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");
    saturate(&engine, Partition::Node, "node:0:");

    let result = engine.delete(Partition::Node, b"node:0:any-key");
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "delete on a full endpoint must reject with CapacityExhausted, got: {result:?}",
    );
}

/// `engine.merge` on a full endpoint MUST reject with `CapacityExhausted`.
/// Merge operands accumulate in the memtable until compaction folds them:
/// they are additive in storage cost.
#[test]
fn merge_on_full_endpoint_rejects() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("small", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");
    // Fill via the Adj partition (Adj is the canonical merge target —
    // posting-list deltas via `engine.merge`).
    saturate(&engine, Partition::Adj, "adj:T:out:");

    let result = engine.merge(Partition::Adj, b"adj:T:out:42", &[0u8; 8]);
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "merge on a full endpoint must reject with CapacityExhausted, got: {result:?}",
    );
}

/// `WriteBatch::commit` on a full endpoint MUST reject, and nothing of the
/// batch lands.
#[test]
fn write_batch_commit_on_full_endpoint_rejects() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("small", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");
    saturate(&engine, Partition::Node, "node:0:");

    let mut batch = WriteBatch::new(&engine);
    batch.put(Partition::Node, b"node:0:batched", &incompressible(7)[..]);
    let result = batch.commit();
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "WriteBatch::commit on a full endpoint must reject, got: {result:?}",
    );
    assert!(
        engine
            .get(Partition::Node, b"node:0:batched")
            .unwrap()
            .is_none()
    );
}

/// A `WriteBatch` mixing the store's metadata with client data is a client
/// write: the whole commit MUST reject on a full endpoint, its Schema part
/// included (atomicity: partial commits across partitions are not a thing
/// in our model).
#[test]
fn write_batch_commit_rejects_when_any_partition_endpoint_full() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("small", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");
    saturate(&engine, Partition::Node, "node:0:");

    let mut batch = WriteBatch::new(&engine);
    batch.put(Partition::Schema, b"schema:label:Test", b"{}");
    batch.put(Partition::Node, b"node:0:mixed", b"v");
    let result = batch.commit();
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "a mixed batch on a full endpoint must reject, got: {result:?}",
    );
    assert!(
        engine
            .get(Partition::Schema, b"schema:label:Test")
            .unwrap()
            .is_none(),
        "no part of a refused batch lands"
    );
}

/// Read paths MUST remain functional on a full endpoint:
/// `engine.get`, `prefix_scan` and friends MUST NOT consult the
/// footprint. Operators need to be able to inspect data on a full
/// endpoint to decide what to evict.
#[test]
fn reads_remain_functional_on_full_endpoint() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("small", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");
    let accepted = fill(&engine, Partition::Node, "node:0:");
    assert!(accepted > 42);

    let v = engine
        .get(Partition::Node, b"node:0:0000000042")
        .expect("get must work on a full endpoint");
    assert!(v.is_some(), "key must still be readable");

    let count = engine
        .prefix_scan(Partition::Node, b"node:0:")
        .expect("prefix_scan must work on a full endpoint")
        .count();
    assert_eq!(count as u32, accepted, "every accepted row scans back");
}

/// The first node a full database creates needs a NodeId lease, whose record
/// is itself a write: refused by the limit, it reaches the caller as the
/// typed `CapacityExhausted` naming the endpoint, like any write past the
/// limit, never as an internal failure.
#[test]
fn a_node_id_lease_refused_by_the_limit_is_capacity_exhausted() {
    use coordinode_embed::{Database, DatabaseError};

    let dir = TempDir::new().expect("tempdir");
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let storage_config = StorageConfig::with_endpoints(vec![capped("ep", dir.path(), ROOMY)]);
    let engine = std::sync::Arc::new(
        StorageEngine::open_with_oracle(&storage_config, std::sync::Arc::clone(&oracle))
            .expect("open engine"),
    );
    // Filled before any node exists, so no lease was granted yet.
    saturate(&engine, Partition::Node, "node:0:");
    let pipeline: std::sync::Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        std::sync::Arc::new(
            coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline::new(&engine),
        );
    let mut db =
        Database::from_engine(dir.path(), engine, oracle, pipeline).expect("a full store opens");
    match db.execute_cypher("CREATE (:First)") {
        Err(DatabaseError::CapacityExhausted {
            endpoint_id,
            hard_limit_bytes,
            ..
        }) => {
            assert_eq!(endpoint_id, "ep");
            assert_eq!(hard_limit_bytes, ROOMY);
        }
        other => panic!("expected CapacityExhausted, got {other:?}"),
    }
}

/// Triggers fire bodies through the same write path as the originating
/// mutation. When the endpoint is full, the whole transaction (the user
/// CREATE and the trigger's CREATE) MUST fail fast with
/// `CapacityExhausted`: no implicit retry, no infinite loop.
#[test]
fn trigger_fail_fast_on_capacity_exhausted() {
    use coordinode_embed::Database;
    use std::time::Instant;

    let dir = TempDir::new().expect("tempdir");
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let storage_config = StorageConfig::with_endpoints(vec![capped("ep", dir.path(), ROOMY)]);
    let engine = std::sync::Arc::new(
        coordinode_storage::engine::core::StorageEngine::open_with_oracle(
            &storage_config,
            std::sync::Arc::clone(&oracle),
        )
        .expect("open engine"),
    );
    let pipeline: std::sync::Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        std::sync::Arc::new(
            coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline::new(&engine),
        );
    let mut db =
        Database::from_engine(dir.path(), std::sync::Arc::clone(&engine), oracle, pipeline)
            .expect("from_engine");

    // Setup: trigger on User CREATE creates an Audit node, before the
    // endpoint fills.
    db.execute_cypher(
        "CREATE TRIGGER audit_user ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (a:Audit)",
    )
    .expect("create trigger");

    // Fill through direct engine writes, so the trigger is not fired.
    saturate(&engine, Partition::Node, "node:0:");

    let started = Instant::now();
    let result = db.execute_cypher("CREATE (u:User {name: 'alice'})");
    let elapsed = started.elapsed();

    let err = match result {
        Err(e) => e,
        Ok(rows) => panic!(
            "expected CapacityExhausted, got Ok with {} rows",
            rows.len(),
        ),
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("capacity exhausted") || msg.contains("CapacityExhausted"),
        "expected capacity-exhausted error from trigger path, got: {msg}",
    );
    assert!(
        db.execute_cypher("MATCH (a:Audit) RETURN a")
            .unwrap()
            .is_empty(),
        "the trigger's effect is refused with the statement"
    );

    // Wall-time guard: a retry loop with exponential backoff takes
    // seconds; a refusal plus one reclamation takes milliseconds. The
    // ceiling sits between the two, clear of a debug build on a fully
    // loaded CI runner.
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "trigger fail-fast must complete in < 5 s (got {elapsed:?}): \
         a longer wall time suggests a hidden retry loop",
    );
}

/// Compaction-driven recovery: fill with rewrites of the same keys until
/// refused, fire `major_compact` (the production path that reclaims space
/// from stale MVCC versions), and assert writes RESUME without a restart
/// because the files compaction removed gave their space back.
///
/// Deletes produce tombstones which still occupy space; only compaction
/// physically reclaims it, and the footprint follows the files.
#[test]
fn writes_resume_after_compaction_frees_space() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("ep", dir.path(), ROOMY)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Rewrite the SAME 100 keys round after round until refused: every
    // rewrite is a new MVCC version compaction can fold away.
    let mut refused = false;
    'fill: for round in 0..100_000u32 {
        for i in 0..100u32 {
            let key = format!("node:0:{i:010}");
            match engine.put(Partition::Node, key.as_bytes(), &incompressible(round)) {
                Ok(()) => {}
                Err(StorageError::CapacityExhausted { .. }) => {
                    refused = true;
                    break 'fill;
                }
                Err(other) => panic!("unexpected refusal: {other:?}"),
            }
        }
    }
    assert!(refused, "the rewrite barrage never filled the endpoint");
    assert!(allocated_under(dir.path()) <= ROOMY);

    engine
        .major_compact(Partition::Node)
        .expect("major compact");

    // The tables compaction replaced are removed once no read still holds
    // them, which a loaded machine delays: poll for the room they leave.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match engine.put(Partition::Node, b"node:0:after-compact", b"v") {
            Ok(()) => break,
            Err(StorageError::CapacityExhausted { .. }) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(other) => panic!("writes must resume after compaction reclaims space: {other:?}"),
        }
    }
    assert!(allocated_under(dir.path()) <= ROOMY);
}

/// Cascade-eviction-driven recovery on a multi-tier config: a capped Hot
/// endpoint under the `CascadeEvict` strategy takes three times its limit
/// of rows. Each write it refuses goes through once the cascade (fired by
/// the capacity refresh at Emergency or Full) has moved bottom-level tables
/// to the Cold endpoint; Hot never passes its limit and every row reads
/// back.
#[test]
fn writes_resume_after_cascade_eviction_frees_hot_endpoint() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(ROOMY)
        .with_hard_limit_strategy(HardLimitStrategy::CascadeEvict),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // A row of 64 incompressible bytes takes more than that on disk: rows
    // for three limits' worth of their values push well past Hot's limit.
    let rows = (3 * ROOMY / 64) as u32;
    for i in 0..rows {
        let key = format!("node:0:{i:010}");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match engine.put(Partition::Node, key.as_bytes(), &incompressible(i)) {
                Ok(()) => break,
                Err(StorageError::CapacityExhausted { .. })
                    if std::time::Instant::now() < deadline =>
                {
                    // The refresh fires the cascade; the tables it replaces
                    // leave Hot once no compaction still reads them.
                    engine.refresh_capacity();
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                Err(other) => panic!(
                    "row {i}: Hot must recover after cascade eviction: {other:?} \
                     (hot_used={}, cold_used={})",
                    engine.capacity().get("ep-hot").unwrap().used(),
                    engine.capacity().get("ep-cold").unwrap().used(),
                ),
            }
        }
        if i % 1024 == 0 {
            assert!(
                allocated_under(hot.path()) <= ROOMY,
                "row {i}: Hot past its limit"
            );
        }
    }
    assert!(allocated_under(hot.path()) <= ROOMY);
    assert!(
        allocated_under(cold.path()) > ROOMY,
        "more than Hot's limit moved to Cold"
    );
    for i in (0..rows).step_by(997) {
        let key = format!("node:0:{i:010}");
        assert!(
            engine
                .get(Partition::Node, key.as_bytes())
                .expect("read")
                .is_some(),
            "row {i} reads back"
        );
    }
}

/// A cache layer placed under a capped endpoint writes its file directly, so
/// open reserves the room it may reach (twice its size, for its compaction)
/// on the footprint; one that does not fit leaves the cache off and nothing
/// reserved, the store opening as before.
#[test]
fn a_cache_layer_under_a_capped_endpoint_is_reserved_or_left_off() {
    use coordinode_storage::cache::config::{CacheLayerConfig, TieredCacheConfig};
    let open = |dir: &std::path::Path, layer_bytes: u64| {
        let mut config = StorageConfig::with_endpoints(vec![capped("ep", dir, ROOMY)]);
        config.cache = TieredCacheConfig {
            layers: vec![CacheLayerConfig::new(dir.join("block.cache"), layer_bytes)],
            ..TieredCacheConfig::default()
        };
        StorageEngine::open(&config).expect("open")
    };

    let dir = TempDir::new().expect("tempdir");
    let engine = open(dir.path(), 64 * 1024);
    assert!(engine.tiered_cache().is_some(), "a layer that fits is on");
    let reserved = engine
        .footprints()
        .get("ep")
        .expect("capped")
        .file(&dir.path().join("block.cache"))
        .bytes();
    assert!(
        reserved >= 2 * 64 * 1024,
        "the layer is reserved twice over"
    );
    drop(engine);

    let dir = TempDir::new().expect("tempdir");
    let engine = open(dir.path(), ROOMY);
    assert!(
        engine.tiered_cache().is_none(),
        "a layer past the limit leaves the cache off"
    );
    assert!(
        engine
            .footprints()
            .get("ep")
            .expect("capped")
            .file(&dir.path().join("block.cache"))
            .bytes()
            < ROOMY,
        "nothing stays reserved for a layer left off"
    );
    engine
        .put(Partition::Node, b"node:0:with-cache-off", b"v")
        .expect("the store opens and writes as before");
}

/// The footprint open measures covers every file under the endpoint root,
/// not just `<part>/tables/`: journal segments, text-index files and
/// anything else a previous run left there take room on the endpoint.
#[test]
fn open_counts_all_endpoint_files_not_only_partition_tables() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("ep", dir.path(), 10_000_000)]);
    drop(StorageEngine::open(&config).expect("lay the store out"));
    let before = allocated_under(dir.path());
    // Files of every kind, planted before the store opens again.
    let oplog_dir = dir.path().join("oplog").join("0");
    std::fs::create_dir_all(&oplog_dir).expect("mkdir oplog");
    std::fs::write(oplog_dir.join("leftover.bin"), vec![0u8; 50_000]).expect("write oplog file");
    let text_dir = dir.path().join("text_indexes").join("text_idx_x_body");
    std::fs::create_dir_all(&text_dir).expect("mkdir text idx");
    std::fs::write(text_dir.join("meta.json"), vec![0u8; 30_000]).expect("write tantivy meta");
    std::fs::write(text_dir.join("segment.idx"), vec![0u8; 200_000])
        .expect("write tantivy segment");
    std::fs::write(dir.path().join("operator.log"), vec![0u8; 100_000]).expect("write misc");

    let engine = StorageEngine::open(&config).expect("open");
    let observed = engine.capacity().get("ep").expect("tracked").used();
    let planted = 100_000 + 50_000 + 30_000 + 200_000;
    assert!(
        observed >= before + planted,
        "the footprint must cover the store ({before}) and the planted {planted} bytes, \
         got {observed}",
    );
}

/// Text-index bytes left on an endpoint count against its limit like any
/// other file: a store whose files already pass the limit refuses client
/// writes, whatever subsystem wrote them.
#[test]
fn text_index_bytes_can_push_endpoint_to_full() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![capped("ep", dir.path(), ROOMY)]);
    drop(StorageEngine::open(&config).expect("lay the store out"));
    let text_dir = dir.path().join("text_indexes").join("text_idx_X_body");
    std::fs::create_dir_all(&text_dir).expect("mkdir");
    std::fs::write(text_dir.join("fat.idx"), vec![0u8; 3 * ROOMY as usize]).expect("write fat idx");

    let engine = StorageEngine::open(&config).expect("a store past its limit opens");
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");
    assert_eq!(usage.severity(), CapacitySeverity::Full);

    let result = engine.put(Partition::Node, b"node:0:after-fat-idx", b"x");
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "a store past its limit by text-index bytes refuses writes, got: {result:?}",
    );
}

/// An embedded database whose disk is below its free-space reserve refuses
/// every write with "no space" before anything reaches its journal, answers
/// reads as before, and takes writes again once space is freed. Queued
/// AFTER COMMIT trigger events wait instead of spending retry attempts.
#[test]
fn an_embedded_database_out_of_space_refuses_writes_and_keeps_reading() {
    use coordinode_embed::Database;
    let dir = TempDir::new().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE (:User {name: 'before'})")
        .expect("write with room");

    db.engine().space().set_reserve(u64::MAX, u64::MAX);
    let err = db
        .execute_cypher("CREATE (:User {name: 'full'})")
        .expect_err("no space");
    assert!(err.to_string().contains("no space"), "{err}");
    let rows = db
        .execute_cypher("MATCH (u:User) RETURN u.name AS name")
        .expect("reads go on");
    assert_eq!(rows.len(), 1, "the refused write left nothing behind");

    // Nothing an AFTER COMMIT dispatch does may land while paused, and it
    // reports when to look again.
    let report = db.dispatch_after_commit_triggers();
    assert_eq!(report.dead_lettered + report.retried + report.fired, 0);
    assert!(
        report.next_due_us.is_some(),
        "the paused queue is looked at again"
    );

    db.engine().space().set_reserve(0, 0);
    db.execute_cypher("CREATE (:User {name: 'after'})")
        .expect("writes resume once space is freed");
    assert_eq!(
        db.execute_cypher("MATCH (u:User) RETURN u.name AS name")
            .expect("read")
            .len(),
        2
    );
}
