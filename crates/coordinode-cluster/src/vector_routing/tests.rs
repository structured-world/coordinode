use super::*;

#[test]
fn single_partition_assigns_to_zero() {
    let r = SinglePartitionRouter::new();
    // Any vector lands in the one partition.
    assert_eq!(&r.assign(&[1.0, 2.0, 3.0])[..], &[0]);
    assert_eq!(&r.assign(&[])[..], &[0]);
}

#[test]
fn single_partition_routes_to_zero_regardless_of_top_m() {
    let r = SinglePartitionRouter::new();
    // top_m never widens the single-partition fan-out.
    assert_eq!(&r.route(&[0.5, 0.5], 1)[..], &[0]);
    assert_eq!(&r.route(&[0.5, 0.5], 64)[..], &[0]);
}

#[test]
fn single_partition_count_is_one() {
    assert_eq!(SinglePartitionRouter::new().n_partitions(), 1);
}

#[test]
fn router_is_object_safe_and_shareable() {
    // The registry holds the router as a trait object behind an Arc; this
    // pins object-safety (the EE adapter plugs into the same slot).
    use std::sync::Arc;
    let r: Arc<dyn VectorShardRouter> = Arc::new(SinglePartitionRouter::new());
    assert_eq!(r.n_partitions(), 1);
    assert_eq!(&r.route(&[0.1], 8)[..], &[0]);
}
