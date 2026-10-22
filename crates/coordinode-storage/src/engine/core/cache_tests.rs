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

/// How a separated tree fills the block cache with the values it reads.
fn blob_cache_fill(engine: &StorageEngine, part: Partition) -> lsm_tree::config::BlobCacheFill {
    match engine.tree(part).expect("tree") {
        lsm_tree::AnyTree::Blob(tree) => tree.runtime_config().blob_cache_fill,
        lsm_tree::AnyTree::Standard(_) => panic!("{part:?} separates its values"),
    }
}

/// The vector truth tier keeps the values it reads in the block cache, since
/// the rerank candidates of different searches overlap on a hot set; the
/// large blobs, mostly read once, stay out of it so they do not evict the
/// index and filter blocks of every tree.
#[test]
fn only_the_vector_truth_tier_caches_its_values() {
    let dir = TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&cached_config(dir.path())).expect("open");
    assert_eq!(
        blob_cache_fill(&engine, Partition::VectorF32),
        lsm_tree::config::BlobCacheFill::Always
    );
    assert_eq!(
        blob_cache_fill(&engine, Partition::Blob),
        lsm_tree::config::BlobCacheFill::Never
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
