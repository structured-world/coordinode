//! Orphan blob collection on a real filesystem: does deleting blobs give the
//! space back, does storage stay bounded over repeated upload/delete cycles,
//! and what does the running collector cost the foreground.
//!
//! Two modes under the same durability, retention (none) and load: rounds of
//! uploading incompressible 1 MiB blobs and deleting them, once with the
//! store's collector running in the background and once without it. Reported
//! per mode: upload latency P50/P99, the Blob partition's size after the first
//! round (baseline), at its largest (peak) and at the end, what the collector
//! deleted and the partition shrank by, and how long the store takes to open
//! again and finish its first pass afterwards.
//!
//! Run: cargo bench -p coordinode-storage --bench blob_gc

#![allow(clippy::expect_used, clippy::panic, clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::graph::blob::{SharingScope, direct_owner_key};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::blob::gc::{BlobCollector, GcPolicy};
use coordinode_storage::blob::{BlobStore, OwnerWrite};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use lsm_tree::AbstractTree as _;

/// Upload/delete rounds per mode.
const ROUNDS: u64 = 20;
/// Blobs uploaded, then deleted, per round.
const BLOBS: u64 = 16;
/// Size of each blob: four full chunks.
const BLOB_BYTES: usize = 1 << 20;

fn config(dir: &tempfile::TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Ssd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// `len` bytes that do not compress, distinct per `seed`.
fn incompressible(seed: u64, len: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(len);
    out
}

fn owner(n: u64) -> Vec<u8> {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&n.to_be_bytes());
    direct_owner_key(&id)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn open(dir: &tempfile::TempDir) -> Arc<BlobCollector> {
    let oracle = Arc::new(TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_embedded(&config(dir), Arc::clone(&oracle)).expect("open"));
    engine.set_retention_window(Duration::ZERO);
    BlobCollector::new(
        BlobStore::direct(engine, oracle),
        // Collected as soon as released, compacted by the default 64 MiB of
        // deleted payload; the delay is short only so the last round's
        // remainder is compacted before the run ends.
        GcPolicy {
            grace: Duration::ZERO,
            compact_after: Duration::from_secs(5),
            ..GcPolicy::default()
        },
        Arc::new(|| true),
    )
}

fn blob_disk_bytes(gc: &BlobCollector) -> u64 {
    gc.store()
        .engine()
        .tree(Partition::Blob)
        .expect("tree")
        .disk_space()
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1 << 20)
}

fn run(collect: bool) {
    let dir = tempfile::tempdir().expect("tempdir");
    let gc = open(&dir);
    let worker = collect.then(|| gc.spawn());
    let mut uploads = Vec::new();
    let mut baseline = 0;
    let mut peak = 0;
    for round in 0..ROUNDS {
        for n in 0..BLOBS {
            let data = incompressible(round * BLOBS + n, BLOB_BYTES);
            let start = Instant::now();
            gc.store()
                .put(
                    &owner(round * BLOBS + n),
                    SharingScope::STORE,
                    &data,
                    OwnerWrite::Create,
                    None,
                )
                .expect("put");
            uploads.push(start.elapsed());
            peak = peak.max(blob_disk_bytes(&gc));
        }
        for n in 0..BLOBS {
            gc.store()
                .delete(&owner(round * BLOBS + n))
                .expect("delete");
        }
        if round == 0 {
            baseline = blob_disk_bytes(&gc);
        }
    }
    // Let the worker finish what the last round released.
    if collect {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let status = gc.status().expect("status");
            if status.candidates == 0 && status.retired_pending == 0 && !status.compaction_pending {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the collector fell behind: {status:?}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    drop(worker);
    let status = gc.status().expect("status");
    let end = blob_disk_bytes(&gc);
    drop(gc);

    let reopen_start = Instant::now();
    let gc = open(&dir);
    let reopened = reopen_start.elapsed();
    let pass_start = Instant::now();
    gc.run(false).expect("first pass after reopen");
    let first_pass = pass_start.elapsed();

    uploads.sort_unstable();
    let written = ROUNDS * BLOBS * BLOB_BYTES as u64;
    println!(
        "{:<12} upload p50 {:>8.2?} p99 {:>8.2?} | disk MiB: baseline {:>7.1} peak {:>7.1} end {:>7.1} \
         (written {:>7.1}) | reclaimed: payload {:>7.1} MiB, disk {:>7.1} MiB | reopen {:>8.2?}, \
         first pass {:>8.2?}",
        if collect { "collector" } else { "no collector" },
        percentile(&uploads, 0.50),
        percentile(&uploads, 0.99),
        mib(baseline),
        mib(peak),
        mib(end),
        mib(written),
        mib(status.reclaimed_bytes),
        mib(status.reclaimed_disk_bytes),
        reopened,
        first_pass,
    );
}

fn main() {
    // Alternate the modes so drift in the machine falls on both.
    for _ in 0..2 {
        run(false);
        run(true);
    }
}
