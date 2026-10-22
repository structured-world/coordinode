use std::time::{Duration, Instant};

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};

/// One mebibyte a second: four mebibytes of tables take three seconds past
/// the one-second burst the budget starts with.
const RATE: u64 = 1 << 20;

fn config(dir: &Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// Bytes no codec shrinks, so the tables are as large as what was written.
fn incompressible(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        })
        .collect()
}

/// Four flushed tables of 1 MiB each in the Node partition.
fn fill(engine: &StorageEngine) {
    for table in 0..4u64 {
        for row in 0..16u64 {
            engine
                .put(
                    Partition::Node,
                    format!("node:1:{table:02}{row:04}").as_bytes(),
                    &incompressible(64 * 1024, table * 100 + row),
                )
                .expect("put");
        }
        engine.persist().expect("flush a table");
    }
}

/// A store opens with its maintenance budget unthrottled unless the
/// configuration names a rate, and with the rate and latency backoff it
/// names.
#[test]
fn the_maintenance_budget_opens_as_configured() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    assert_eq!(engine.maintenance_budget().rate(), 0, "unthrottled");
    assert!(engine.maintenance_budget().latency_backoff().is_none());
    drop(engine);

    let mut configured = config(dir.path());
    configured.maintenance_bytes_per_sec = RATE;
    configured.maintenance_backoff_ms = 20;
    let engine = StorageEngine::open(&configured).expect("reopen");
    assert_eq!(engine.maintenance_budget().rate(), RATE);
    assert_eq!(
        engine
            .maintenance_budget()
            .latency_backoff()
            .map(|backoff| backoff.ceiling()),
        Some(Duration::from_millis(20))
    );
}

/// A compaction draws on the maintenance budget retuned on the live engine:
/// four mebibytes at one a second take at least the three seconds past the
/// burst, where the unthrottled store compacts them at once.
#[test]
fn a_compaction_moves_no_faster_than_the_maintenance_budget() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    fill(&engine);
    engine.set_maintenance_rate(RATE);
    assert_eq!(engine.maintenance_budget().rate(), RATE);

    let started = Instant::now();
    engine.major_compact(Partition::Node).expect("compact");
    let took = started.elapsed();
    assert!(
        took >= Duration::from_secs(2),
        "four MiB compacted in {took:?} under a budget of one MiB a second"
    );
}

/// The scrub reads under the same budget: a store the budget holds to one
/// mebibyte a second takes at least the seconds its tables need past the
/// burst to verify, and every block still verifies.
#[test]
fn a_scrub_reads_no_faster_than_the_maintenance_budget() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    fill(&engine);
    engine.set_maintenance_rate(RATE);

    let started = Instant::now();
    let report =
        crate::scrub::scrub_all(&engine, &crate::scrub::ScrubConfig::default()).expect("scrub");
    let took = started.elapsed();
    assert!(!report.has_errors(), "{:?}", report.errors);
    assert!(
        took >= Duration::from_secs(2),
        "four MiB scrubbed in {took:?} under a budget of one MiB a second"
    );
}
