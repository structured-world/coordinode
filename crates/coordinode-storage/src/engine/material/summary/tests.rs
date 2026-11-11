use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use tempfile::TempDir;

fn engine(dir: &std::path::Path) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open")
}

/// Write `moves` to the Node tree's summary as one batch.
fn write(engine: &StorageEngine, moves: Moves) {
    let tree = engine.tree(Partition::Node).expect("tree");
    let mut batch = lsm_tree::WriteBatch::with_capacity(16);
    moves.write(tree, &mut batch).expect("write");
    tree.apply_batch(batch, engine.next_seqno()).expect("apply");
}

/// The bytes of every key whose newest change is at or after `s`, counted
/// key by key: what the summary must answer.
fn oracle(keys: &BTreeMap<u32, (SeqNo, u64)>, s: SeqNo) -> u64 {
    keys.values().filter(|(n, _)| *n >= s).map(|(_, b)| b).sum()
}

/// Seqnos on every side of bucket boundaries at several levels, and both
/// ends of the domain.
const EDGES: [SeqNo; 12] = [
    0,
    1,
    255,
    256,
    257,
    65_535,
    65_536,
    1 << 24,
    (1 << 40) + 3,
    1_795_195_217_139_970,
    u64::MAX - 1,
    u64::MAX,
];

/// Keys charged, moved to a later newest change, recharged and dropped, in
/// any order and across bucket edges: the sum from any continuation equals
/// the key-by-key count, after every step.
#[test]
fn the_sum_from_a_continuation_matches_the_charges_key_by_key() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let tree = engine.tree(Partition::Node).expect("tree");
    let mut keys: BTreeMap<u32, (SeqNo, u64)> = BTreeMap::new();
    // A deterministic sequence: an xorshift over the steps.
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for step in 0..400u32 {
        let mut moves = Moves::default();
        for _ in 0..(1 + next() % 4) {
            let key = (next() % 40) as u32;
            let pick = next();
            let n = if pick % 3 == 0 {
                EDGES[(pick as usize / 3) % EDGES.len()]
            } else {
                next()
            };
            let bytes = 1 + next() % 5_000;
            let from = keys.get(&key).copied();
            let to = if next() % 7 == 0 {
                None
            } else {
                Some((n, bytes))
            };
            moves.shift(from, to);
            match to {
                Some(t) => keys.insert(key, t),
                None => keys.remove(&key),
            };
        }
        write(&engine, moves);
        let mut probes: Vec<SeqNo> = EDGES.to_vec();
        probes.extend(keys.values().flat_map(|(n, _)| [*n, n.saturating_add(1)]));
        probes.push(next());
        for s in probes {
            assert_eq!(
                charge_from(tree, s, SeqNo::MAX).expect("read"),
                oracle(&keys, s),
                "step {step}, from {s}"
            );
        }
    }
}

/// A charge moved away leaves its bucket empty rather than holding zero, and
/// a summary with nothing charged answers zero from anywhere.
#[test]
fn an_empty_summary_answers_zero_and_keeps_no_buckets() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let tree = engine.tree(Partition::Node).expect("tree");
    let mut moves = Moves::default();
    moves.shift(None, Some((1_000, 40)));
    write(&engine, moves);
    assert_eq!(charge_from(tree, 0, SeqNo::MAX).expect("read"), 40);
    let mut moves = Moves::default();
    moves.shift(Some((1_000, 40)), None);
    write(&engine, moves);
    for s in EDGES {
        assert_eq!(charge_from(tree, s, SeqNo::MAX).expect("read"), 0);
    }
    let left = tree
        .range(
            PREFIX.as_slice()..[0x00, b'a', b't'].as_slice(),
            SeqNo::MAX,
            None,
        )
        .count();
    assert_eq!(left, 0, "an emptied bucket was kept");
}

/// Moving more out of a bucket than it holds is refused: the charges the
/// mover started from are not the ones recorded.
#[test]
fn a_move_past_what_a_bucket_holds_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let tree = engine.tree(Partition::Node).expect("tree");
    let mut moves = Moves::default();
    moves.shift(Some((7, 10)), None);
    let mut batch = lsm_tree::WriteBatch::with_capacity(16);
    assert!(moves.write(tree, &mut batch).is_err());
}

/// One read of the sum walks at most a bounded run of buckets per level,
/// however many keys are charged.
#[test]
fn a_sum_reads_a_bounded_number_of_buckets() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let tree = engine.tree(Partition::Node).expect("tree");
    let mut moves = Moves::default();
    for n in 0..20_000u64 {
        moves.shift(None, Some((n * 7_919, 1)));
    }
    write(&engine, moves);
    let s = 10_000 * 7_919 + 1;
    let mut read = 0usize;
    for level in 0..LEVELS {
        let index = index_at(level, s);
        read += run(tree, level, index & !(FANOUT - 1), index, SeqNo::MAX)
            .expect("run")
            .len();
    }
    read += run(tree, LEVELS - 1, 0, FANOUT, SeqNo::MAX)
        .expect("top")
        .len();
    assert!(
        read <= usize::from(LEVELS) * (FANOUT as usize - 1) + FANOUT as usize,
        "read {read} buckets"
    );
    assert_eq!(charge_from(tree, s, SeqNo::MAX).expect("read"), 9_999);
}
