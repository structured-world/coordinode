use super::*;

#[test]
fn default_config_disabled() {
    let config = TieredCacheConfig::default();
    assert!(!config.is_enabled());
    assert!(config.layers.is_empty());
}

#[test]
fn single_layer_config() {
    let config = TieredCacheConfig {
        layers: vec![CacheLayerConfig::new(
            "/mnt/nvme/cache",
            100 * 1024 * 1024 * 1024,
        )],
        ..Default::default()
    };
    assert!(config.is_enabled());
    assert_eq!(config.layers.len(), 1);
    assert_eq!(config.layers[0].max_bytes, 100 * 1024 * 1024 * 1024);
}

#[test]
fn multi_layer_config() {
    let config = TieredCacheConfig {
        layers: vec![
            CacheLayerConfig::new("/mnt/nvme/cache", 100 * 1024 * 1024 * 1024),
            CacheLayerConfig::new("/mnt/ssd/cache", 500 * 1024 * 1024 * 1024),
        ],
        ..Default::default()
    };
    assert_eq!(config.layers.len(), 2);
}

#[test]
fn layer_defaults() {
    let layer = CacheLayerConfig::new("/tmp/test", 1024);
    assert_eq!(layer.max_entries, 1_000_000);
    assert!((layer.compaction_threshold - 0.5).abs() < f64::EPSILON);
}
