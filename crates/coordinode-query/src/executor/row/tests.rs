use super::*;
use coordinode_core::budget::QueryBudget;

/// The index is charged before it allocates, so the charge must cover what
/// the hash table and the link list actually hold for every size: a bound
/// below the table's real buckets would let a statement keep memory its
/// limit does not count.
#[test]
fn the_index_charge_covers_what_it_allocates() {
    for positions in (0..2_000).chain([4_095, 4_096, 65_536, 100_000]) {
        let budget = QueryBudget::new(u64::MAX);
        let mut charge = budget.empty_charge();
        let mut index = DistinctIndex::default();
        index.reserve(positions, &mut charge).unwrap();
        // hashbrown holds at most 7/8 of its buckets, a power of two.
        let cap = index.heads.capacity();
        let buckets = if cap == 0 {
            0
        } else if cap < 8 {
            cap + 1
        } else {
            (cap * 8 / 7).next_power_of_two()
        };
        let held = buckets * (core::mem::size_of::<(u64, usize)>() + 1)
            + if buckets == 0 { 0 } else { 16 }
            + index.previous.capacity() * core::mem::size_of::<usize>();
        assert!(cap >= positions, "room for {positions}: capacity {cap}");
        assert!(
            charge.bytes() >= held as u64,
            "{positions} positions: charged {} for {held} held",
            charge.bytes()
        );
    }
}

/// Room made in steps is charged in steps, never twice and never short: the
/// running charge equals the bound for the total room.
#[test]
fn the_index_charges_growth_once() {
    let budget = QueryBudget::new(u64::MAX);
    let mut charge = budget.empty_charge();
    let mut index = DistinctIndex::default();
    index.reserve(10, &mut charge).unwrap();
    for hash in 0..10 {
        index.keep(hash);
    }
    // Room is counted from the kept positions: no new room, no new charge.
    index.reserve(0, &mut charge).unwrap();
    assert_eq!(charge.bytes(), DistinctIndex::bytes_for(10));
    index.reserve(990, &mut charge).unwrap();
    assert_eq!(charge.bytes(), DistinctIndex::bytes_for(1_000));
}

/// A limit the index would pass refuses it before anything is allocated.
#[test]
fn the_index_is_refused_past_the_limit() {
    let budget = QueryBudget::new(1_000);
    let mut charge = budget.empty_charge();
    let mut index = DistinctIndex::default();
    assert!(index.reserve(10_000, &mut charge).is_err());
    assert_eq!(index.heads.capacity(), 0);
    assert_eq!(index.previous.capacity(), 0);
}

/// Items with one hash are told apart by the comparison, and a later equal
/// item finds the kept one: correctness does not rest on the hash.
#[test]
fn colliding_hashes_are_compared() {
    let budget = QueryBudget::new(u64::MAX);
    let mut charge = budget.empty_charge();
    let mut index = DistinctIndex::default();
    index.reserve(3, &mut charge).unwrap();
    let kept = ["a", "b"];
    for (at, item) in kept.iter().enumerate() {
        assert_eq!(index.find(7, |k| kept[k] == *item).0, None);
        assert_eq!(index.keep(7), at);
    }
    let (found, asked) = index.find(7, |k| kept[k] == "a");
    assert_eq!(found, Some(0));
    assert_eq!(asked, 2);
    assert_eq!(index.find(7, |k| kept[k] == "c"), (None, 2));
}

/// A charged push charges the slots a full list allocates before it grows.
#[test]
fn a_charged_push_covers_the_list_capacity() {
    let budget = QueryBudget::new(u64::MAX);
    let mut charge = budget.empty_charge();
    let mut items: Vec<u64> = Vec::new();
    for i in 0..1_000 {
        push_charged(&mut items, i, &mut charge).unwrap();
        assert!(charge.bytes() >= (items.capacity() * 8) as u64);
    }
    let small = QueryBudget::new(16);
    let mut refused = small.empty_charge();
    let mut list: Vec<u64> = Vec::new();
    assert!(push_charged(&mut list, 1, &mut refused).is_err());
    assert!(list.is_empty());
}
