use super::*;
use crate::txn::proposal::PartitionId;

fn k(id: u64) -> Vec<u8> {
    id.to_be_bytes().to_vec()
}
fn del(p: PartitionId, id: u64) -> Mutation {
    Mutation::Delete {
        partition: p,
        key: k(id),
    }
}
fn put(p: PartitionId, id: u64) -> Mutation {
    Mutation::Put {
        partition: p,
        key: k(id),
        value: b"v".to_vec(),
    }
}
fn range(p: PartitionId, start: u64, end_incl: u64) -> Mutation {
    Mutation::RemoveRange {
        partition: p,
        start: k(start),
        end: k(end_incl + 1),
    }
}

#[test]
fn dense_run_becomes_one_range() {
    // The user's case: deleting all relationships -> a consecutive key run.
    let muts: Vec<Mutation> = (56783..=56984)
        .map(|i| del(PartitionId::EdgeProp, i))
        .collect();
    let out = coalesce_delete_mutations(muts, 4);
    assert_eq!(out, vec![range(PartitionId::EdgeProp, 56783, 56984)]);
}

#[test]
fn run_never_crosses_a_gap() {
    // 1,2,3 gap 5,6,7 -> two ranges; key 4 is never covered.
    let muts = vec![
        del(PartitionId::Adj, 1),
        del(PartitionId::Adj, 2),
        del(PartitionId::Adj, 3),
        del(PartitionId::Adj, 5),
        del(PartitionId::Adj, 6),
        del(PartitionId::Adj, 7),
    ];
    let out = coalesce_delete_mutations(muts, 3);
    assert_eq!(
        out,
        vec![range(PartitionId::Adj, 1, 3), range(PartitionId::Adj, 5, 7)]
    );
}

#[test]
fn non_delete_mutations_break_runs_and_pass_through() {
    // A Put between deletes is a run boundary and is preserved in order.
    let muts = vec![
        del(PartitionId::Node, 1),
        del(PartitionId::Node, 2),
        put(PartitionId::Node, 3),
        del(PartitionId::Node, 4),
        del(PartitionId::Node, 5),
    ];
    let out = coalesce_delete_mutations(muts, 2);
    assert_eq!(
        out,
        vec![
            range(PartitionId::Node, 1, 2),
            put(PartitionId::Node, 3),
            range(PartitionId::Node, 4, 5),
        ]
    );
}

#[test]
fn different_partitions_break_runs() {
    let muts = vec![
        del(PartitionId::Node, 1),
        del(PartitionId::Adj, 2),
        del(PartitionId::Adj, 3),
    ];
    let out = coalesce_delete_mutations(muts, 2);
    // Node singleton stays a point; the two Adj deletes coalesce.
    assert_eq!(
        out,
        vec![del(PartitionId::Node, 1), range(PartitionId::Adj, 2, 3)]
    );
}

#[test]
fn threshold_gates_short_runs() {
    let muts: Vec<Mutation> = (10..=12).map(|i| del(PartitionId::Idx, i)).collect();
    assert_eq!(
        coalesce_delete_mutations(muts.clone(), 4),
        vec![
            del(PartitionId::Idx, 10),
            del(PartitionId::Idx, 11),
            del(PartitionId::Idx, 12)
        ]
    );
    assert_eq!(
        coalesce_delete_mutations(muts, 3),
        vec![range(PartitionId::Idx, 10, 12)]
    );
}

#[test]
fn different_lengths_never_coalesce() {
    let short = Mutation::Delete {
        partition: PartitionId::Node,
        key: 1u32.to_be_bytes().to_vec(),
    };
    let long = del(PartitionId::Node, 2);
    let out = coalesce_delete_mutations(vec![short.clone(), long.clone()], 2);
    assert_eq!(out, vec![short, long]);
}

#[test]
fn all_ff_tail_stays_points() {
    let prev = Mutation::Delete {
        partition: PartitionId::Node,
        key: {
            let mut v = vec![0xFFu8; 8];
            v[7] = 0xFE;
            v
        },
    };
    let max = Mutation::Delete {
        partition: PartitionId::Node,
        key: vec![0xFFu8; 8],
    };
    let out = coalesce_delete_mutations(vec![prev.clone(), max.clone()], 2);
    assert_eq!(out, vec![prev, max]);
}

#[test]
fn empty_passes_through() {
    assert!(coalesce_delete_mutations(vec![], 4).is_empty());
}
