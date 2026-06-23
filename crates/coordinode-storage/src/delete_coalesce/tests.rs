use super::*;

/// Fixed-width (8-byte be) key in one family, the common CoordiNode case.
fn k(id: u64) -> Vec<u8> {
    id.to_be_bytes().to_vec()
}

fn point(p: u8, id: u64) -> CoalescedDelete {
    CoalescedDelete::Point {
        partition: p,
        key: k(id),
    }
}
fn range(p: u8, start: u64, end_inclusive: u64) -> CoalescedDelete {
    CoalescedDelete::Range {
        partition: p,
        start: k(start),
        end: k(end_inclusive + 1), // exclusive successor
    }
}

#[test]
fn dense_run_becomes_one_range() {
    // The user's example shape: a long consecutive run collapses to one range.
    let dels: Vec<(u8, Vec<u8>)> = (56783..=56984).map(|id| (0, k(id))).collect();
    let out = coalesce_deletes(dels, 4);
    assert_eq!(out, vec![range(0, 56783, 56984)]);
}

#[test]
fn mixed_runs_and_singletons() {
    // 1..=4 (run), 3943 (single), 34853 (single), 56783..=56785 (run).
    let mut dels: Vec<(u8, Vec<u8>)> = (1..=4).map(|id| (0u8, k(id))).collect();
    dels.push((0, k(3943)));
    dels.push((0, k(34853)));
    dels.extend((56783..=56785).map(|id| (0u8, k(id))));
    let out = coalesce_deletes(dels, 4);
    assert_eq!(
        out,
        vec![
            range(0, 1, 4),
            point(0, 3943),
            point(0, 34853),
            // run of 3 is below threshold 4 -> stays points
            point(0, 56783),
            point(0, 56784),
            point(0, 56785),
        ]
    );
}

#[test]
fn run_never_crosses_a_gap() {
    // 1,2,3 then a gap to 5,6,7 — must be TWO runs, never [1,7) (which would
    // wrongly delete the surviving key 4).
    let dels = vec![
        (0u8, k(1)),
        (0, k(2)),
        (0, k(3)),
        (0, k(5)),
        (0, k(6)),
        (0, k(7)),
    ];
    let out = coalesce_deletes(dels, 3);
    assert_eq!(out, vec![range(0, 1, 3), range(0, 5, 7)]);
    // Crucially, no range spans [1,7): key 4 is untouched.
    for d in &out {
        if let CoalescedDelete::Range { start, end, .. } = d {
            let covers_4 = start.as_slice() <= k(4).as_slice() && k(4).as_slice() < end.as_slice();
            assert!(!covers_4, "no range may cover the surviving key 4");
        }
    }
}

#[test]
fn different_partitions_never_coalesce() {
    // Adjacent ids but different partitions are independent.
    let dels = vec![(0u8, k(1)), (1u8, k(2)), (0u8, k(2)), (1u8, k(3))];
    let out = coalesce_deletes(dels, 2);
    // partition 0: {1,2} -> range; partition 1: {2,3} -> range.
    assert_eq!(out, vec![range(0, 1, 2), range(1, 2, 3)]);
}

#[test]
fn different_lengths_never_coalesce() {
    // A 4-byte key and an 8-byte key are never adjacent even if numerically
    // "next" — guards against a longer key falling inside a same-length range.
    let short = 1u32.to_be_bytes().to_vec();
    let long = 2u64.to_be_bytes().to_vec();
    let out = coalesce_deletes(vec![(0, short.clone()), (0, long.clone())], 2);
    // Both stay points (different lengths never coalesce). Output is sorted by
    // key: the 8-byte `[0,0,0,0,0,0,0,2]` sorts before the 4-byte `[0,0,0,1]`.
    assert_eq!(
        out,
        vec![
            CoalescedDelete::Point {
                partition: 0,
                key: long,
            },
            CoalescedDelete::Point {
                partition: 0,
                key: short,
            },
        ]
    );
}

#[test]
fn threshold_gates_short_runs() {
    let dels: Vec<(u8, Vec<u8>)> = (10..=12).map(|id| (0u8, k(id))).collect();
    // min_run 4: a 3-run stays points.
    assert_eq!(
        coalesce_deletes(dels.clone(), 4),
        vec![point(0, 10), point(0, 11), point(0, 12)]
    );
    // min_run 3: same 3-run becomes a range.
    assert_eq!(coalesce_deletes(dels, 3), vec![range(0, 10, 12)]);
}

#[test]
fn min_run_below_two_is_clamped() {
    // A single key is never a range regardless of min_run.
    assert_eq!(coalesce_deletes(vec![(0, k(7))], 0), vec![point(0, 7)]);
    assert_eq!(coalesce_deletes(vec![(0, k(7))], 1), vec![point(0, 7)]);
}

#[test]
fn duplicates_are_deduped() {
    let out = coalesce_deletes(vec![(0, k(1)), (0, k(1)), (0, k(2)), (0, k(2))], 2);
    assert_eq!(out, vec![range(0, 1, 2)]);
}

#[test]
fn empty_input_yields_nothing() {
    assert!(coalesce_deletes(vec![], 4).is_empty());
}

#[test]
fn all_ff_tail_falls_back_to_points() {
    // A run ending at the all-0xFF key has no same-length successor for `end`;
    // it must not panic and must degrade to point deletes rather than emit a
    // bad range.
    let max = vec![0xFFu8; 8];
    let prev = {
        let mut v = vec![0xFFu8; 8];
        v[7] = 0xFE;
        v
    };
    let out = coalesce_deletes(vec![(0, prev.clone()), (0, max.clone())], 2);
    assert_eq!(
        out,
        vec![
            CoalescedDelete::Point {
                partition: 0,
                key: prev,
            },
            CoalescedDelete::Point {
                partition: 0,
                key: max,
            },
        ]
    );
}
