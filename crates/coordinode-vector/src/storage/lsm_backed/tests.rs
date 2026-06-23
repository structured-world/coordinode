use super::*;
use crate::storage::VectorTierHandle;

fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    (Arc::new(engine), dir)
}

#[test]
fn f32_round_trip_through_lsm() {
    let (engine, _dir) = open_engine();
    let tier = LsmVectorTier::new(engine);
    let v = vec![1.0_f32, -0.5, 3.25, -42.0, 0.0];
    tier.put_f32(7, 13, 99, &v).expect("put_f32");
    let got = tier.multi_get_f32(7, 13, &[99, 100]).expect("multi_get");
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].as_deref(), Some(v.as_slice()));
    assert!(got[1].is_none(), "missing id must report None, not garbage");
}

#[test]
fn handle_wires_through_to_lsm() {
    let (engine, _dir) = open_engine();
    let tier: Arc<dyn VectorTierStorage> = Arc::new(LsmVectorTier::new(engine));
    let h = VectorTierHandle::new(tier, 5, 9);
    h.put_f32(123, &[7.0, 8.0]).unwrap();
    assert_eq!(
        h.multi_get_f32(&[123]).unwrap()[0].as_deref(),
        Some(&[7.0_f32, 8.0][..])
    );
}

#[test]
fn corrupted_f32_value_decodes_to_none() {
    // Direct put of a non-multiple-of-4 byte string under the
    // f32 key — decoder must report None, not crash.
    let (engine, _dir) = open_engine();
    let key = encode_vec_f32_key(1, 1, 1);
    engine
        .put(Partition::VectorF32, &key, &[0xFFu8, 0xFF, 0xFF])
        .unwrap();
    let tier = LsmVectorTier::new(engine);
    let got = tier.multi_get_f32(1, 1, &[1]).unwrap();
    assert!(got[0].is_none(), "non-mod-4 byte slice must decode to None");
}
