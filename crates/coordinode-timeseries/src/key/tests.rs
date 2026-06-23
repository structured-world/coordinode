use super::*;

#[test]
fn equal_metas_produce_equal_keys() {
    let a = BucketKey::from_meta(7, &rmpv::Value::String("s42".into()));
    let b = BucketKey::from_meta(7, &rmpv::Value::String("s42".into()));
    assert_eq!(a, b);
}

#[test]
fn different_metas_produce_different_keys() {
    let a = BucketKey::from_meta(7, &rmpv::Value::String("s42".into()));
    let b = BucketKey::from_meta(7, &rmpv::Value::String("s43".into()));
    assert_ne!(a.meta_hash, b.meta_hash);
}

#[test]
fn different_labels_produce_different_keys_even_for_same_meta() {
    let a = BucketKey::from_meta(7, &rmpv::Value::String("s42".into()));
    let b = BucketKey::from_meta(8, &rmpv::Value::String("s42".into()));
    // label_id differs even though meta_hash matches — the full
    // BucketKey must be unequal.
    assert_eq!(a.meta_hash, b.meta_hash);
    assert_ne!(a, b);
}

#[test]
fn stripe_idx_bounded_by_stripe_count() {
    for label_id in 0u16..256 {
        let k = BucketKey::from_meta(label_id, &rmpv::Value::Integer(label_id.into()));
        assert!(k.stripe_idx() < crate::config::STRIPE_COUNT);
    }
}

#[test]
fn nested_map_metas_round_trip_to_stable_hash() {
    let m = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("k".into()),
            rmpv::Value::Integer(42.into()),
        ),
        (rmpv::Value::String("kk".into()), rmpv::Value::Boolean(true)),
    ]);
    let a = BucketKey::from_meta(1, &m);
    let b = BucketKey::from_meta(1, &m);
    assert_eq!(a, b);
}
