//! Integration tests: TieredCache wired into StorageEngine.
//!
//! Verifies the cache-aware read path: DRAM → tiered cache → persistent storage.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_storage::cache::config::{CacheLayerConfig, TieredCacheConfig};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn engine_with_cache(dir: &std::path::Path) -> StorageEngine {
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.cache = TieredCacheConfig {
        compaction_interval_secs: 0, // Disable background thread in tests
        layers: vec![CacheLayerConfig {
            path: dir.join("cache_layer0.cache"),
            max_bytes: 1024 * 1024, // 1MB
            max_entries: 10000,
            compaction_threshold: 0.5,
        }],
        ..Default::default()
    };
    StorageEngine::open(&config).expect("open engine with cache")
}

/// Reads go through tiered cache: first miss populates cache, second read is a cache hit.
#[test]
fn cache_populated_on_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine_with_cache(dir.path());

    // Write directly to storage (bypasses cache on write)
    engine
        .put(Partition::Node, b"key1", b"value1")
        .expect("put");

    // First read: cache miss → storage hit → populates cache
    let v1 = engine.get(Partition::Node, b"key1").expect("get");
    assert_eq!(v1.as_deref(), Some(b"value1".as_slice()));

    let cache = engine.tiered_cache().expect("cache should be enabled");
    let stats = cache.stats();
    assert_eq!(stats.total_hits, 0, "first read should be cache miss");

    // Second read: cache hit
    let v2 = engine.get(Partition::Node, b"key1").expect("get");
    assert_eq!(v2.as_deref(), Some(b"value1".as_slice()));

    let stats = cache.stats();
    assert_eq!(stats.total_hits, 1, "second read should be cache hit");
}

/// Delete invalidates the cache entry.
#[test]
fn cache_invalidated_on_delete() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine_with_cache(dir.path());

    engine
        .put(Partition::Node, b"key1", b"value1")
        .expect("put");

    // Populate cache
    let _ = engine.get(Partition::Node, b"key1").expect("get");

    // Delete — should invalidate cache
    engine.delete(Partition::Node, b"key1").expect("delete");

    // Cache should not serve stale data
    let result = engine.get(Partition::Node, b"key1").expect("get");
    assert!(result.is_none(), "deleted key should return None");
}

/// Access tracker records reads.
#[test]
fn access_tracker_records_reads() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine_with_cache(dir.path());

    engine
        .put(Partition::Node, b"hot_key", b"value")
        .expect("put");

    for _ in 0..5 {
        let _ = engine.get(Partition::Node, b"hot_key").expect("get");
    }

    let stats = engine
        .access_tracker()
        .get(Partition::Node, b"hot_key")
        .expect("should be tracked");
    assert_eq!(stats.count, 5, "should have 5 accesses");
}

/// Cache works across multiple partitions.
#[test]
fn cache_across_partitions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine_with_cache(dir.path());

    engine
        .put(Partition::Node, b"nk", b"node_val")
        .expect("put");
    engine.put(Partition::Adj, b"ak", b"adj_val").expect("put");

    // Populate cache
    let _ = engine.get(Partition::Node, b"nk").expect("get");
    let _ = engine.get(Partition::Adj, b"ak").expect("get");

    // Second reads should be cache hits
    let n = engine.get(Partition::Node, b"nk").expect("get");
    let a = engine.get(Partition::Adj, b"ak").expect("get");

    assert_eq!(n.as_deref(), Some(b"node_val".as_slice()));
    assert_eq!(a.as_deref(), Some(b"adj_val".as_slice()));

    let cache = engine.tiered_cache().expect("cache");
    assert_eq!(cache.stats().total_hits, 2);
}

/// Engine without cache works normally (no regression).
#[test]
fn engine_without_cache_works() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    // Default config has no cache layers
    let engine = StorageEngine::open(&config).expect("open");

    assert!(engine.tiered_cache().is_none());

    engine.put(Partition::Node, b"key", b"value").expect("put");
    let result = engine.get(Partition::Node, b"key").expect("get");
    assert_eq!(result.as_deref(), Some(b"value".as_slice()));
}

/// Two-layer cache: drain-through works end-to-end via StorageEngine.
#[test]
fn two_layer_drain_through_via_engine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.cache = TieredCacheConfig {
        compaction_interval_secs: 0,
        layers: vec![
            CacheLayerConfig {
                path: dir.path().join("fast.cache"),
                max_bytes: 300, // Tiny L0 — forces eviction quickly
                max_entries: 3,
                compaction_threshold: 0.5,
            },
            CacheLayerConfig {
                path: dir.path().join("slow.cache"),
                max_bytes: 10 * 1024 * 1024,
                max_entries: 10000,
                compaction_threshold: 0.5,
            },
        ],
        ..Default::default()
    };
    let engine = StorageEngine::open(&config).expect("open");

    // Write 10 entries — L0 holds ~3, rest drain to L1
    for i in 0..10u32 {
        let key = format!("key_{i:03}");
        let value = format!("value_{i:03}");
        engine
            .put(Partition::Node, key.as_bytes(), value.as_bytes())
            .expect("put");
    }

    // Populate cache (read all entries)
    for i in 0..10u32 {
        let key = format!("key_{i:03}");
        let _ = engine.get(Partition::Node, key.as_bytes()).expect("get");
    }

    let cache = engine.tiered_cache().expect("cache");
    let stats = cache.stats();

    // L0 should have evicted and L1 should have received entries
    assert!(
        stats.layers[0].evictions > 0 || stats.layers[0].drains > 0,
        "layer 0 should have evicted/drained: {:?}",
        stats.layers[0]
    );
    assert!(
        stats.layers[1].puts > 0,
        "layer 1 should have received drained entries: {:?}",
        stats.layers[1]
    );

    // All 10 entries should still be readable (from cache or storage)
    for i in 0..10u32 {
        let key = format!("key_{i:03}");
        let expected = format!("value_{i:03}");
        let result = engine.get(Partition::Node, key.as_bytes()).expect("get");
        assert_eq!(
            result.as_deref(),
            Some(expected.as_bytes()),
            "key {i} missing"
        );
    }
}

/// Overwrite via engine returns fresh value, not stale cache.
#[test]
fn cache_no_stale_after_overwrite() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine_with_cache(dir.path());

    engine.put(Partition::Node, b"k", b"v1").expect("put");
    // Populate cache with v1
    let _ = engine.get(Partition::Node, b"k").expect("get");

    // Overwrite — put() must invalidate cache entry
    engine.put(Partition::Node, b"k", b"v2").expect("put");

    // Read must return v2, not stale v1 from cache
    let result = engine.get(Partition::Node, b"k").expect("get");
    assert_eq!(
        result.as_deref(),
        Some(b"v2".as_slice()),
        "must return fresh v2, not stale v1 from cache"
    );
}

/// Engine persist + reopen: cache file survives, entries recoverable.
#[test]
fn cache_survives_engine_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    let cache_config = TieredCacheConfig {
        compaction_interval_secs: 0,
        layers: vec![CacheLayerConfig {
            path: dir.path().join("cache.dat"),
            max_bytes: 1024 * 1024,
            max_entries: 10000,
            compaction_threshold: 0.5,
        }],
        ..Default::default()
    };

    // Write + populate cache
    {
        let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path().join("data"),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        config.cache = cache_config.clone();
        let engine = StorageEngine::open(&config).expect("open");

        engine
            .put(Partition::Node, b"persist_k", b"persist_v")
            .expect("put");
        let _ = engine.get(Partition::Node, b"persist_k").expect("get");
        engine.persist().expect("persist");

        let cache = engine.tiered_cache().expect("cache");
        assert_eq!(cache.total_entries(), 1);
    }

    // Reopen — cache file should be recovered
    {
        let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path().join("data"),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        config.cache = cache_config;
        let engine = StorageEngine::open(&config).expect("reopen");

        let cache = engine.tiered_cache().expect("cache");
        assert_eq!(
            cache.total_entries(),
            1,
            "cache entry should survive reopen"
        );

        // Read from recovered cache — should be a hit
        let result = engine.get(Partition::Node, b"persist_k").expect("get");
        assert_eq!(result.as_deref(), Some(b"persist_v".as_slice()));

        assert_eq!(
            cache.stats().total_hits,
            1,
            "should be cache hit from recovered file"
        );
    }
}

/// Background compaction via engine doesn't lose data.
#[test]
fn background_compaction_via_engine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.cache = TieredCacheConfig {
        compaction_interval_secs: 1,
        layers: vec![CacheLayerConfig {
            path: dir.path().join("bg.cache"),
            max_bytes: 10 * 1024 * 1024,
            max_entries: 10000,
            compaction_threshold: 0.3,
        }],
        ..Default::default()
    };

    let engine = StorageEngine::open(&config).expect("open");

    // Write + read (populates cache)
    for i in 0..20u32 {
        let key = format!("k{i}");
        engine
            .put(Partition::Node, key.as_bytes(), &[0xFF; 100])
            .expect("put");
        let _ = engine.get(Partition::Node, key.as_bytes()).expect("get");
    }

    // Overwrite half to create dead space in cache
    for i in 0..10u32 {
        let key = format!("k{i}");
        engine
            .put(Partition::Node, key.as_bytes(), &[0xAA; 100])
            .expect("put");
        let _ = engine.get(Partition::Node, key.as_bytes()).expect("get");
    }

    // Wait for background compaction
    std::thread::sleep(std::time::Duration::from_secs(2));

    // All data should still be readable from CoordiNode storage (source of truth)
    for i in 0..20u32 {
        let key = format!("k{i}");
        let result = engine.get(Partition::Node, key.as_bytes()).expect("get");
        assert!(result.is_some(), "key k{i} should exist after compaction");
    }
}

/// Large dataset with cache — cache eviction doesn't lose data.
#[test]
fn cache_eviction_doesnt_lose_data() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.cache = TieredCacheConfig {
        compaction_interval_secs: 0, // Disable background thread in tests
        layers: vec![CacheLayerConfig {
            path: dir.path().join("small.cache"),
            max_bytes: 1024, // Tiny: forces eviction
            max_entries: 10,
            compaction_threshold: 0.5,
        }],
        ..Default::default()
    };
    let engine = StorageEngine::open(&config).expect("open");

    // Write 100 entries — cache can hold ~10
    for i in 0..100u32 {
        let key = format!("key_{i:06}");
        let value = format!("value_{i:06}");
        engine
            .put(Partition::Node, key.as_bytes(), value.as_bytes())
            .expect("put");
    }

    // Populate cache (only latest ~10 will fit)
    for i in 0..100u32 {
        let key = format!("key_{i:06}");
        let _ = engine.get(Partition::Node, key.as_bytes()).expect("get");
    }

    // ALL 100 entries should still be readable (cache miss → storage)
    for i in 0..100u32 {
        let key = format!("key_{i:06}");
        let expected = format!("value_{i:06}");
        let result = engine.get(Partition::Node, key.as_bytes()).expect("get");
        assert_eq!(
            result.as_deref(),
            Some(expected.as_bytes()),
            "data loss at key {i}"
        );
    }
}
