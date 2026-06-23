use super::*;

#[test]
fn is_commutative_classifies_partitions_correctly() {
    // Commutative — merge operators compose concurrent writers.
    assert!(Partition::Adj.is_commutative());
    assert!(Partition::Counter.is_commutative());
    // Non-commutative — last-write-wins, conflict-detection required.
    assert!(!Partition::Node.is_commutative());
    assert!(!Partition::EdgeProp.is_commutative());
    assert!(!Partition::Blob.is_commutative());
    assert!(!Partition::BlobRef.is_commutative());
    assert!(!Partition::Schema.is_commutative());
    assert!(!Partition::Idx.is_commutative());
    assert!(!Partition::Raft.is_commutative());
}

#[test]
fn is_commutative_total_over_all_partitions() {
    // Every variant covered — guard against future additions
    // forgetting to declare commutativity.
    for &part in Partition::all() {
        // Just call it — exhaustive match in the function body
        // means an un-handled variant would not compile.
        let _ = part.is_commutative();
    }
}
