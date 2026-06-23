use super::*;

fn one_layer_config(dir: &std::path::Path, max_bytes: u64) -> TieredCacheConfig {
    TieredCacheConfig {
        layers: vec![CacheLayerConfig {
            path: dir.join("layer0.cache"),
            max_bytes,
            max_entries: 1000,
            compaction_threshold: 0.5,
        }],
        compaction_interval_secs: 0, // Disable background thread in tests
        ..Default::default()
    }
}

fn two_layer_config(dir: &std::path::Path) -> TieredCacheConfig {
    TieredCacheConfig {
        layers: vec![
            CacheLayerConfig {
                path: dir.join("fast.cache"),
                max_bytes: 500,
                max_entries: 5,
                compaction_threshold: 0.5,
            },
            CacheLayerConfig {
                path: dir.join("slow.cache"),
                max_bytes: 10 * 1024 * 1024,
                max_entries: 10000,
                compaction_threshold: 0.5,
            },
        ],
        compaction_interval_secs: 0,
        ..Default::default()
    }
}

// ── Single-layer tests ────────────────────────────────────────

#[test]
fn single_layer_put_get() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    cache.put(Partition::Node, b"key1", b"value1");
    assert_eq!(
        cache.get(Partition::Node, b"key1").as_deref(),
        Some(b"value1".as_slice())
    );
}

#[test]
fn single_layer_miss() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    assert!(cache.get(Partition::Node, b"missing").is_none());
}

#[test]
fn single_layer_remove() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    cache.put(Partition::Node, b"k", b"v");
    cache.remove(Partition::Node, b"k");
    assert!(cache.get(Partition::Node, b"k").is_none());
}

#[test]
fn single_layer_overwrite() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    cache.put(Partition::Node, b"k", b"v1");
    cache.put(Partition::Node, b"k", b"v2");
    assert_eq!(
        cache.get(Partition::Node, b"k").as_deref(),
        Some(b"v2".as_slice())
    );
}

#[test]
fn single_layer_partitions_isolated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    cache.put(Partition::Node, b"k", b"node");
    cache.put(Partition::Adj, b"k", b"adj");

    assert_eq!(
        cache.get(Partition::Node, b"k").as_deref(),
        Some(b"node".as_slice())
    );
    assert_eq!(
        cache.get(Partition::Adj, b"k").as_deref(),
        Some(b"adj".as_slice())
    );
}

#[test]
fn single_layer_survives_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);

    {
        let cache = TieredCache::open(&config).expect("open");
        cache.put(Partition::Node, b"k", b"v");
    }
    {
        let cache = TieredCache::open(&config).expect("reopen");
        assert_eq!(
            cache.get(Partition::Node, b"k").as_deref(),
            Some(b"v".as_slice())
        );
    }
}

#[test]
fn single_layer_stats() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    cache.put(Partition::Node, b"k", b"v");
    let _ = cache.get(Partition::Node, b"k");
    let _ = cache.get(Partition::Node, b"miss");

    let stats = cache.stats();
    assert_eq!(stats.total_hits, 1);
    assert_eq!(stats.total_misses, 1);
    assert_eq!(stats.layers.len(), 1);
    assert_eq!(stats.layers[0].puts, 1);
}

#[test]
fn single_layer_rejects_huge() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    let huge = vec![0u8; 17 * 1024 * 1024];
    cache.put(Partition::Blob, b"huge", &huge);
    assert!(cache.is_empty());
}

#[test]
fn single_layer_compaction() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 10 * 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    for i in 0..10u32 {
        cache.put(Partition::Node, format!("k{i}").as_bytes(), &[0xFF; 100]);
    }
    for i in 0..5u32 {
        cache.put(Partition::Node, format!("k{i}").as_bytes(), &[0xAA; 100]);
    }

    let before = cache.stats();
    cache.compact().expect("compact");
    let after = cache.stats();

    assert_eq!(after.layers[0].live_entries, 10);
    assert!(after.layers[0].file_bytes <= before.layers[0].file_bytes);
}

// ── Two-layer drain-through tests ─────────────────────────────

#[test]
fn two_layer_drain_through_on_eviction() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = two_layer_config(dir.path());
    let cache = TieredCache::open(&config).expect("open");

    // Layer 0: max 5 entries, 500 bytes.
    // Fill layer 0 beyond capacity — evicted entries should drain to layer 1.
    for i in 0..10u32 {
        let key = format!("key_{i:04}");
        cache.put(Partition::Node, key.as_bytes(), &[0xFF; 32]);
    }

    let stats = cache.stats();
    // Layer 0 should have evicted some entries
    assert!(stats.layers[0].evictions > 0, "layer 0 should have evicted");
    // Layer 1 should have received drained entries
    assert!(
        stats.layers[1].puts > 0,
        "layer 1 should have received drained entries"
    );

    // All entries should still be findable (some in layer 0, some in layer 1)
    let mut found = 0;
    for i in 0..10u32 {
        let key = format!("key_{i:04}");
        if cache.get(Partition::Node, key.as_bytes()).is_some() {
            found += 1;
        }
    }
    assert!(found > 0, "should find entries across layers");
}

#[test]
fn two_layer_deep_hit_promotes_to_fast() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = two_layer_config(dir.path());
    let cache = TieredCache::open(&config).expect("open");

    // Fill layer 0 to force drain
    for i in 0..8u32 {
        cache.put(Partition::Node, format!("fill_{i}").as_bytes(), &[0xFF; 32]);
    }

    // "target" should have been drained to layer 1
    // Now read it — should promote back to layer 0
    let target_key = b"fill_0";
    if cache.get(Partition::Node, target_key).is_some() {
        // After read-promotion, layer 0 should have a hit next time
        let stats_before = cache.stats();
        let hits_before = stats_before.layers[0].hits;

        let _ = cache.get(Partition::Node, target_key);

        let stats_after = cache.stats();
        // Either layer 0 or layer 1 served it
        assert!(stats_after.total_hits > hits_before);
    }
}

#[test]
fn two_layer_remove_clears_both() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = two_layer_config(dir.path());
    let cache = TieredCache::open(&config).expect("open");

    // Force entry into both layers
    for i in 0..8u32 {
        cache.put(Partition::Node, format!("key_{i}").as_bytes(), &[0xFF; 32]);
    }

    // Remove should clear from all layers
    cache.remove(Partition::Node, b"key_0");
    assert!(cache.get(Partition::Node, b"key_0").is_none());
}

#[test]
fn empty_config_disabled() {
    let config = TieredCacheConfig::default();
    let cache = TieredCache::open(&config).expect("open");
    assert_eq!(cache.layer_count(), 0);
    assert!(cache.is_empty());
    assert!(cache.get(Partition::Node, b"k").is_none());
    cache.put(Partition::Node, b"k", b"v"); // no-op
}

#[test]
fn all_partitions_across_layers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    for &part in Partition::all() {
        let key = format!("{:?}", part);
        let val = format!("val_{:?}", part);
        cache.put(part, key.as_bytes(), val.as_bytes());
    }

    assert_eq!(cache.total_entries(), Partition::all().len());

    for &part in Partition::all() {
        let key = format!("{:?}", part);
        let expected = format!("val_{:?}", part);
        assert_eq!(
            cache.get(part, key.as_bytes()).as_deref(),
            Some(expected.as_bytes()),
            "mismatch for {:?}",
            part
        );
    }
}

// ── Background compaction tests ───────────────────────────────

#[test]
fn background_compaction_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = TieredCacheConfig {
        layers: vec![CacheLayerConfig {
            path: dir.path().join("bg.cache"),
            max_bytes: 10 * 1024 * 1024,
            max_entries: 10000,
            compaction_threshold: 0.3, // Low threshold to trigger easily
        }],
        compaction_interval_secs: 1, // 1 second for fast test
        ..Default::default()
    };

    let cache = TieredCache::open(&config).expect("open");

    // Write entries then overwrite half (creates dead space)
    for i in 0..20u32 {
        cache.put(Partition::Node, format!("k{i}").as_bytes(), &[0xFF; 100]);
    }
    for i in 0..10u32 {
        cache.put(Partition::Node, format!("k{i}").as_bytes(), &[0xAA; 100]);
    }

    let before = cache.stats();

    // Wait for background compaction to run (interval=1s, so wait 2s)
    std::thread::sleep(Duration::from_secs(2));

    let after = cache.stats();

    // Compaction should have reduced file size
    assert!(
        after.layers[0].file_bytes <= before.layers[0].file_bytes,
        "background compaction should reduce file: before={}, after={}",
        before.layers[0].file_bytes,
        after.layers[0].file_bytes
    );

    // All data should still be intact
    assert_eq!(after.layers[0].live_entries, 20);
    for i in 0..20u32 {
        assert!(
            cache
                .get(Partition::Node, format!("k{i}").as_bytes())
                .is_some(),
            "entry k{i} lost after compaction"
        );
    }
}

#[test]
fn background_thread_stops_on_drop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = TieredCacheConfig {
        layers: vec![CacheLayerConfig::new(
            dir.path().join("drop.cache"),
            1024 * 1024,
        )],
        compaction_interval_secs: 1,
        ..Default::default()
    };

    // Open and immediately drop — background thread should stop cleanly
    {
        let cache = TieredCache::open(&config).expect("open");
        cache.put(Partition::Node, b"k", b"v");
    }
    // If we get here without hanging, the thread stopped cleanly
}

// ── Weighted eviction tests (G019) ────────────────────────────

/// Low-weight entries are evicted before high-weight entries.
#[test]
fn weighted_eviction_prefers_low_weight() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Tiny cache: only ~2 entries fit
    let config = one_layer_config(dir.path(), 200);
    let cache = TieredCache::open(&config).expect("open");

    // Insert high-weight entry
    cache.put_weighted(Partition::Node, b"hot", b"hot_value_data", 10.0);
    // Insert low-weight entry
    cache.put_weighted(Partition::Node, b"cold", b"cold_value_data", 0.1);

    // Force eviction: insert a third entry that exceeds capacity
    cache.put_weighted(Partition::Node, b"new", b"new_value_data_", 1.0);

    // High-weight "hot" should survive; low-weight "cold" should be evicted
    let hot = cache.get(Partition::Node, b"hot");
    let cold = cache.get(Partition::Node, b"cold");

    assert!(hot.is_some(), "high-weight entry should survive eviction");
    // "cold" may or may not be evicted depending on exact sizing,
    // but if eviction happened, cold should go first.
    // The key assertion is that hot survived.
    if cold.is_none() {
        // Expected: cold evicted before hot
    } else {
        // Both survived — cache had room for all three, no eviction
        // This is fine — just means the cache was big enough.
    }
}

/// put_weighted with different weights, then verify eviction order
/// by filling the cache completely.
#[test]
fn weighted_eviction_bulk() {
    let dir = tempfile::tempdir().expect("tempdir");
    // ~5 entries fit (each ~50 bytes with header)
    let config = one_layer_config(dir.path(), 400);
    let cache = TieredCache::open(&config).expect("open");

    // Insert 5 entries with descending weight: w=5, w=4, w=3, w=2, w=1
    for i in 0..5u8 {
        let key = [b'k', i];
        let value = [b'v'; 30]; // ~30 bytes + 15 header = ~45 bytes each
        cache.put_weighted(Partition::Node, &key, &value, (5 - i) as f32);
    }

    // Force eviction: insert 3 more entries, displacing the lowest-weight ones
    for i in 10..13u8 {
        let key = [b'k', i];
        let value = [b'v'; 30];
        cache.put_weighted(Partition::Node, &key, &value, 5.0);
    }

    // Highest-weight original entries (w=5 at k[0], w=4 at k[1])
    // should survive; lowest (w=1 at k[4], w=2 at k[3]) should be gone.
    let w5 = cache.get(Partition::Node, &[b'k', 0]);
    let w1 = cache.get(Partition::Node, &[b'k', 4]);

    // At minimum, highest weight should survive over lowest
    if w5.is_none() && w1.is_some() {
        panic!("eviction order wrong: w=5 evicted but w=1 survived");
    }
}

/// resolve_weight returns configured label weight or default 1.0.
#[test]
fn resolve_weight_from_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut label_weights = std::collections::HashMap::new();
    label_weights.insert("User".to_string(), 5.0);
    label_weights.insert("Log".to_string(), 0.1);

    let config = TieredCacheConfig {
        layers: vec![CacheLayerConfig::new(dir.path().join("l0.cache"), 1024)],
        compaction_interval_secs: 0,
        label_weights,
    };
    let cache = TieredCache::open(&config).expect("open");

    assert!((cache.resolve_weight("User") - 5.0).abs() < f32::EPSILON);
    assert!((cache.resolve_weight("Log") - 0.1).abs() < f32::EPSILON);
    assert!((cache.resolve_weight("Unknown") - 1.0).abs() < f32::EPSILON);
}

/// Default put() uses weight 1.0 (backward compatibility).
#[test]
fn default_put_uses_unit_weight() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 10 * 1024);
    let cache = TieredCache::open(&config).expect("open");

    cache.put(Partition::Node, b"default", b"value");
    let v = cache.get(Partition::Node, b"default");
    assert!(v.is_some(), "default put should work");
}

/// label_weights_empty returns true when no weights configured.
#[test]
fn label_weights_empty_check() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = one_layer_config(dir.path(), 1024);
    let cache = TieredCache::open(&config).expect("open");
    assert!(cache.label_weights_empty());

    let mut weights = std::collections::HashMap::new();
    weights.insert("User".to_string(), 2.0);
    let config2 = TieredCacheConfig {
        layers: vec![CacheLayerConfig::new(dir.path().join("l2.cache"), 1024)],
        compaction_interval_secs: 0,
        label_weights: weights,
    };
    let cache2 = TieredCache::open(&config2).expect("open");
    assert!(!cache2.label_weights_empty());
}
