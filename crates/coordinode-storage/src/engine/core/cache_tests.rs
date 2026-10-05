//! The tiered cache in front of the trees: a cached value is served only
//! while no write has replaced it since it was read from the tree.

use super::*;
use crate::cache::config::{CacheLayerConfig, TieredCacheConfig};
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use tempfile::TempDir;

/// A store at `dir` with one cache layer beside it.
fn cached_config(dir: &std::path::Path) -> StorageConfig {
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.cache = TieredCacheConfig {
        layers: vec![CacheLayerConfig::new(dir.join("cache/layer0"), 1 << 20)],
        compaction_interval_secs: 0,
        ..Default::default()
    };
    config
}

/// A value cached before a later write is not served after the store
/// reopens: the cache starts cold, and the write stands.
#[test]
fn a_reopened_store_never_serves_a_value_cached_before_a_later_write() {
    let dir = TempDir::new().expect("tempdir");
    let config = cached_config(dir.path());
    {
        let engine = StorageEngine::open(&config).expect("open");
        engine.put(Partition::Node, b"k", b"v1").expect("put v1");
        // Read once so the cache holds v1.
        assert_eq!(
            engine.get(Partition::Node, b"k").expect("get").as_deref(),
            Some(&b"v1"[..])
        );
        // Not read again before the store closes: the cache file still holds
        // v1 as its last entry for the key.
        engine.put(Partition::Node, b"k", b"v2").expect("put v2");
        engine.persist().expect("persist");
    }
    let engine = StorageEngine::open(&config).expect("reopen");
    assert_eq!(
        engine.get(Partition::Node, b"k").expect("get").as_deref(),
        Some(&b"v2"[..]),
        "the reopened store served the value cached before the write"
    );
}

/// Keys a range drop removed are not served from the cache.
#[test]
fn a_dropped_range_is_not_served_from_the_cache() {
    let dir = TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&cached_config(dir.path())).expect("open");
    engine.put(Partition::Idx, b"idx:a", b"v").expect("put");
    engine.persist().expect("persist");
    assert!(engine.get(Partition::Idx, b"idx:a").expect("get").is_some());
    engine
        .drop_range::<&[u8], _>(Partition::Idx, &b"idx:"[..]..&b"idx;"[..])
        .expect("drop range");
    assert_eq!(
        engine.get(Partition::Idx, b"idx:a").expect("get"),
        None,
        "the cache served a key the range drop removed"
    );
}
