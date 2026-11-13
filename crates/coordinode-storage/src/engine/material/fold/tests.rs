use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::material::summary::charge_from;
use crate::engine::partition::Partition;
use coordinode_core::budget::QueryBudget;
use tempfile::TempDir;

/// The Node tree's accounting scope: where its folds read and write.
fn scope(engine: &StorageEngine) -> Scope<'_> {
    let heads = *engine.material_heads();
    engine
        .material_scope(&heads, Partition::Node)
        .expect("scope")
}

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

/// A unit as the index holds it: commit, ordinal, kind, key, range end,
/// encoded length.
#[derive(Debug, Clone)]
struct U {
    seqno: SeqNo,
    ordinal: u32,
    kind: UnitKind,
    key: Vec<u8>,
    end: Vec<u8>,
    bytes: u64,
}

fn point(seqno: SeqNo, key: &[u8], kind: UnitKind, bytes: u64) -> U {
    U {
        seqno,
        ordinal: 0,
        kind,
        key: key.to_vec(),
        end: Vec::new(),
        bytes,
    }
}

/// The cursor key a test's folds write with their batches.
const TEST_CURSOR: &[u8] = b"\0ac";
/// The snapshot bytes these tests' folds start from: they do not track
/// them, and a fold may move them down.
const TEST_BASE: u64 = 1 << 62;

/// `fold`'s output batch, with a cursor record as long as a real one
/// without a key to resume from.
fn batch_of(fold: Fold<'_>) -> lsm_tree::WriteBatch {
    fold.write(TEST_BASE, TEST_CURSOR, |_| vec![0; 21])
        .expect("write")
        .batch
}

/// Fold `units` (in commit order): points in batches of `per_batch`, each
/// range deletion alone, in pages of `page` keys, each page its own batch.
fn fold(engine: &StorageEngine, units: &[U], per_batch: usize, page: usize) {
    let scope = scope(engine);
    let tree = scope.tree;
    let commit = |fold: Fold<'_>| {
        tree.apply_batch(batch_of(fold), engine.next_seqno())
            .expect("apply");
    };
    let mut current = Fold::new(scope);
    let mut in_batch = 0;
    for u in units {
        if u.kind == UnitKind::Range {
            commit(std::mem::replace(&mut current, Fold::new(scope)));
            in_batch = 0;
            let tomb = Tomb {
                seqno: u.seqno,
                ordinal: u.ordinal,
            };
            let mut from: Option<Vec<u8>> = None;
            loop {
                let mut paged = Fold::new(scope);
                let next = paged
                    .range_page(tomb, &u.key, &u.end, u.bytes, from.as_deref(), page)
                    .expect("range page");
                commit(paged);
                match next {
                    RangePage::Done(_) => break,
                    RangePage::From(k, _) => from = Some(k.to_vec()),
                    RangePage::NoRoom => panic!("an unmetered fold ran out of room"),
                }
            }
            continue;
        }
        current
            .unit(u.seqno, &u.key, u.kind, u.bytes)
            .expect("fold");
        in_batch += 1;
        if in_batch >= per_batch.max(1) {
            commit(std::mem::replace(&mut current, Fold::new(scope)));
            in_batch = 0;
        }
    }
    commit(current);
}

/// What a coalesced state delta from `s` needs, by definition, computed
/// independently of the descriptors: every key changed at or after `s` (a
/// range deletion changes every key it deletes), with each unit of its
/// closure since its last value, deletion or range deletion, a shared range
/// deletion once.
fn oracle(units: &[U], s: SeqNo) -> u64 {
    struct Key {
        newest: SeqNo,
        own: u64,
        rests_on: Option<(SeqNo, u32)>,
    }
    let mut keys: BTreeMap<Vec<u8>, Key> = BTreeMap::new();
    let mut tombs: BTreeMap<(SeqNo, u32), u64> = BTreeMap::new();
    for u in units {
        match u.kind {
            UnitKind::Value | UnitKind::Delete => {
                keys.insert(
                    u.key.clone(),
                    Key {
                        newest: u.seqno,
                        own: u.bytes,
                        rests_on: None,
                    },
                );
            }
            UnitKind::Operand => {
                let k = keys.entry(u.key.clone()).or_insert(Key {
                    newest: u.seqno,
                    own: 0,
                    rests_on: None,
                });
                k.newest = u.seqno;
                k.own += u.bytes;
            }
            UnitKind::Range => {
                tombs.insert((u.seqno, u.ordinal), u.bytes);
                for (key, k) in keys.range_mut(u.key.clone()..u.end.clone()) {
                    let _ = key;
                    k.newest = u.seqno;
                    k.own = 0;
                    k.rests_on = Some((u.seqno, u.ordinal));
                }
            }
        }
    }
    let changed = keys.values().filter(|k| k.newest >= s);
    let own: u64 = changed.clone().map(|k| k.own).sum();
    let shared: std::collections::BTreeSet<_> = changed.filter_map(|k| k.rests_on).collect();
    own + shared.iter().map(|t| tombs[t]).sum::<u64>()
}

fn key(n: u64) -> Vec<u8> {
    format!("node:{n:02}").into_bytes()
}

/// Random histories over a few keys, in commit order: several units at one
/// commit, same-commit operands repeated, deletions, recreations and range
/// deletions over random spans, nested and partly rewritten. The summary's
/// answer from every continuation equals the definition, however the units
/// are batched and the ranges paged.
#[test]
fn the_folded_charge_matches_the_closure_definition() {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for round in 0..16 {
        let dir = TempDir::new().expect("tempdir");
        let engine = engine(dir.path());
        let mut units: Vec<U> = Vec::new();
        let mut seqno: SeqNo = 1_000;
        let mut ordinal = 0u32;
        for _ in 0..300 {
            if next() % 3 != 0 {
                seqno += 1 + next() % 50;
                ordinal = 0;
            }
            let roll = next() % 20;
            let k = key(next() % 12);
            let u = if roll == 0 {
                let a = next() % 12;
                let b = a + 1 + next() % 6;
                let (start, end) = (key(a), key(b));
                let bytes = (start.len() + end.len()) as u64;
                U {
                    seqno,
                    ordinal,
                    kind: UnitKind::Range,
                    key: start,
                    end,
                    bytes,
                }
            } else {
                let kind = match roll % 6 {
                    0 => UnitKind::Value,
                    1 => UnitKind::Delete,
                    _ => UnitKind::Operand,
                };
                let bytes = k.len() as u64
                    + if kind == UnitKind::Delete {
                        0
                    } else {
                        next() % 300
                    };
                U {
                    seqno,
                    ordinal,
                    kind,
                    key: k,
                    end: Vec::new(),
                    bytes,
                }
            };
            ordinal += 1;
            if u.kind == UnitKind::Operand && next() % 5 == 0 {
                // The same operand twice at one commit: two units.
                let mut twin = u.clone();
                twin.ordinal = ordinal;
                ordinal += 1;
                units.push(u);
                units.push(twin);
            } else {
                units.push(u);
            }
        }
        fold(
            &engine,
            &units,
            1 + (next() % 40) as usize,
            1 + (next() % 4) as usize,
        );
        let scope = scope(&engine);
        let mut probes: Vec<SeqNo> = vec![0, 999, 1_000, seqno, seqno + 1, SeqNo::MAX];
        probes.extend(units.iter().flat_map(|u| [u.seqno, u.seqno + 1]));
        for s in probes {
            assert_eq!(
                charge_from(&scope, s, SeqNo::MAX).expect("read"),
                oracle(&units, s),
                "round {round}, from {s}"
            );
        }
    }
}

/// A key overwritten a thousand times since the continuation needs its last
/// value only, not the cumulative payload written.
#[test]
fn repeated_overwrites_charge_the_last_value_only() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let units: Vec<U> = (0..1_000u64)
        .map(|n| point(100 + n, b"node:hot", UnitKind::Value, 8 + 1_024))
        .collect();
    fold(&engine, &units, 64, 8);
    let scope = scope(&engine);
    assert_eq!(charge_from(&scope, 100, SeqNo::MAX).expect("read"), 1_032);
    assert_eq!(charge_from(&scope, 1_100, SeqNo::MAX).expect("read"), 0);
}

/// An operand after the continuation needs the base below it, and every
/// operand since that base.
#[test]
fn an_operand_after_the_continuation_needs_its_base_below_it() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let k = b"node:counter";
    let units = vec![
        point(10, k, UnitKind::Value, 100),
        point(20, k, UnitKind::Operand, 15),
        point(30, k, UnitKind::Operand, 15),
        point(40, k, UnitKind::Operand, 15),
    ];
    fold(&engine, &units, 4, 8);
    let scope = scope(&engine);
    assert_eq!(charge_from(&scope, 35, SeqNo::MAX).expect("read"), 145);
    assert_eq!(charge_from(&scope, 41, SeqNo::MAX).expect("read"), 0);
}

/// A deleted key still needs its deletion as evidence of the change; a key
/// recreated after it needs only the new value.
#[test]
fn a_deletion_is_evidence_and_a_recreation_replaces_it() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let k = b"node:gone";
    fold(
        &engine,
        &[
            point(10, k, UnitKind::Value, 500),
            point(20, k, UnitKind::Delete, 9),
        ],
        8,
        8,
    );
    let scope = scope(&engine);
    assert_eq!(charge_from(&scope, 15, SeqNo::MAX).expect("read"), 9);
    fold(&engine, &[point(30, k, UnitKind::Value, 40)], 8, 8);
    assert_eq!(charge_from(&scope, 15, SeqNo::MAX).expect("read"), 40);
}

/// A range deletion over many keys is charged once, while a key it deleted
/// is still resting on it; once every one of them was written again, it is
/// charged nothing.
#[test]
fn a_range_deletion_is_charged_once_while_a_key_rests_on_it() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let mut units: Vec<U> = (0..10u64)
        .map(|n| point(10 + n, &key(n), UnitKind::Value, 100))
        .collect();
    units.push(U {
        seqno: 50,
        ordinal: 0,
        kind: UnitKind::Range,
        key: key(0),
        end: key(10),
        bytes: 14,
    });
    fold(&engine, &units, 4, 3);
    let scope = scope(&engine);
    assert_eq!(charge_from(&scope, 50, SeqNo::MAX).expect("read"), 14);
    // Keys written again leave it; an operand onto one keeps it needed.
    let mut again: Vec<U> = (0..9u64)
        .map(|n| point(60 + n, &key(n), UnitKind::Value, 30))
        .collect();
    again.push(point(80, &key(9), UnitKind::Operand, 12));
    fold(&engine, &again, 3, 3);
    assert_eq!(charge_from(&scope, 75, SeqNo::MAX).expect("read"), 14 + 12);
    fold(&engine, &[point(90, &key(9), UnitKind::Value, 30)], 3, 3);
    assert_eq!(charge_from(&scope, 85, SeqNo::MAX).expect("read"), 30);
    assert_eq!(charge_from(&scope, 0, SeqNo::MAX).expect("read"), 10 * 30);
}

fn long_key(n: u8) -> Vec<u8> {
    let mut key = b"node:".to_vec();
    key.resize(u16::MAX as usize - 1, b'x');
    key.push(n);
    key
}

/// Keys stored as they are and of the longest length, alternately.
fn mixed_keys(count: u8) -> Vec<Vec<u8>> {
    (0..count)
        .map(|n| {
            if n % 2 == 0 {
                long_key(n)
            } else {
                format!("node:{n}").into_bytes()
            }
        })
        .collect()
}

/// Run `fold` from its making to its output built as the store writes it
/// (the fold reserving what writing takes, its batch, marks and cursor
/// record, the record carrying the key a page resumes from, copied out of
/// the tree's read), and check the most memory it held at once (counted by
/// the allocator, reads of the tree included) against what it had reserved
/// by then. `fold` returns the fold and the key its page resumes from.
fn within_reserve<'t>(
    what: &str,
    fold: impl FnOnce() -> (Fold<'t>, Option<lsm_tree::UserKey>),
) -> lsm_tree::WriteBatch {
    let ((batch, reserved), peak) = crate::test_alloc::peak_of(|| {
        let (mut fold, resume) = fold();
        if !fold.ready() {
            return (lsm_tree::WriteBatch::new(), 0);
        }
        // The cursor takes its own copy of the key, encoded after the rest.
        let resume = resume.map(|k| k.to_vec());
        let written = fold
            .write(TEST_BASE, TEST_CURSOR, |_| {
                let mut record = vec![0; 21];
                record.extend_from_slice(resume.as_deref().unwrap_or_default());
                record
            })
            .expect("write");
        let reserved = written.held.as_ref().map_or(0, |h| h.bytes());
        (written.batch, reserved)
    });
    let peak = peak as u64;
    assert!(
        peak <= reserved,
        "{what}: {peak} bytes held at once of {reserved} reserved"
    );
    batch
}

/// What a metered fold allocates, from its first unit to its batch built,
/// never exceeds what it reserved first: for keys stored as they are and
/// for the longest ones, through points, range pages and operands onto a
/// range. Counted by the allocator, not reckoned.
#[test]
fn a_metered_fold_holds_no_more_than_it_reserved() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let scope = scope(&engine);
    let tree = scope.tree;
    let budget = QueryBudget::new(1 << 30);
    let keys = mixed_keys(6);
    let apply = |batch| tree.apply_batch(batch, engine.next_seqno()).expect("apply");
    apply(within_reserve("points", || {
        let mut fold = Fold::metered(scope, &budget);
        for (i, key) in keys.iter().enumerate() {
            let kind = if i % 3 == 0 {
                UnitKind::Value
            } else {
                UnitKind::Operand
            };
            assert!(fold.unit(10 + i as u64, key, kind, 100).expect("unit"));
        }
        (fold, None)
    }));
    let tomb = Tomb {
        seqno: 50,
        ordinal: 0,
    };
    let mut from: Option<Vec<u8>> = None;
    loop {
        let mut next = None;
        apply(within_reserve("a range page", || {
            let mut page = Fold::metered(scope, &budget);
            let paged = page
                .range_page(tomb, b"node:", b"node;", 10, from.as_deref(), 2)
                .expect("page");
            let resume = match &paged {
                RangePage::From(k, _) => Some(k.clone()),
                _ => None,
            };
            next = Some(paged);
            (page, resume)
        }));
        match next.expect("paged") {
            RangePage::Done(_) => break,
            RangePage::From(k, _) => from = Some(k.to_vec()),
            RangePage::NoRoom => panic!("no room in a gigabyte"),
        }
    }
    apply(within_reserve("operands onto a range", || {
        let mut fold = Fold::metered(scope, &budget);
        for (i, key) in keys.iter().enumerate() {
            assert!(
                fold.unit(60 + i as u64, key, UnitKind::Operand, 7)
                    .expect("unit")
            );
        }
        (fold, None)
    }));
    // Many short keys, one commit each: what the maps' nodes and the marks
    // take beside the keys dominates.
    let short: Vec<Vec<u8>> = (0..3000u32).map(|n| format!("a{n}").into_bytes()).collect();
    apply(within_reserve("short keys", || {
        let mut fold = Fold::metered(scope, &budget);
        for (i, key) in short.iter().enumerate() {
            assert!(
                fold.unit(100 + i as u64, key, UnitKind::Value, 3)
                    .expect("unit")
            );
        }
        (fold, None)
    }));
    // A fold that only completes the commit folded before it: its mark,
    // batch and cursor record alone.
    apply(within_reserve("a commit completed", || {
        let mut fold = Fold::metered(scope, &budget).after(Some(3_099));
        assert!(fold.close(), "room for the mark");
        (fold, None)
    }));
}

/// The same within its reservation over a tree read cold from several
/// tables: each run the reads go through, its blocks and the range
/// tombstones they carry.
#[test]
fn a_fold_over_a_cold_tree_of_many_tables_holds_no_more_than_it_reserved() {
    const TABLES: u64 = 12;
    let dir = TempDir::new().expect("tempdir");
    let keys = mixed_keys(6);
    // Without compaction the flushed tables stay runs of their own, which
    // the reads each go through.
    let config = |cache: Option<u64>| {
        let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        config.compaction_workers = 0;
        if let Some(bytes) = cache {
            config.block_cache_bytes = bytes;
        }
        config
    };
    {
        let engine = StorageEngine::open(&config(None)).expect("open");
        let scope = scope(&engine);
        let tree = scope.tree;
        // Each table holds descriptors of the keys and a stretch of others,
        // with a range deletion over part of them.
        for t in 0..TABLES {
            let mut fold = Fold::new(scope);
            for (i, key) in keys.iter().enumerate() {
                fold.unit(t * 100 + i as u64, key, UnitKind::Operand, 5)
                    .expect("unit");
            }
            for n in 0..200u64 {
                fold.unit(
                    t * 100 + 50,
                    format!("node:filler:{t}:{n}").as_bytes(),
                    UnitKind::Value,
                    9,
                )
                .expect("unit");
            }
            tree.apply_batch(batch_of(fold), engine.next_seqno())
                .expect("apply");
            tree.remove_range(
                format!("node:filler:{t}:1").as_bytes(),
                format!("node:filler:{t}:2").as_bytes(),
                engine.next_seqno(),
            );
            engine.persist().expect("a table");
        }
    }
    // Without a block cache every block the reads load is theirs alone; with
    // one, what they load is the cache's, held to the cache's own capacity.
    let engine = StorageEngine::open(&config(Some(0))).expect("open cold");
    let scope = scope(&engine);
    let tree = scope.tree;
    assert!(
        super::runs(tree) >= 4,
        "the tree has {} runs",
        super::runs(tree)
    );
    let budget = QueryBudget::new(1 << 30);
    let apply = |batch| tree.apply_batch(batch, engine.next_seqno()).expect("apply");
    apply(within_reserve("points over a cold tree", || {
        let mut fold = Fold::metered(scope, &budget);
        for (i, key) in keys.iter().enumerate() {
            assert!(
                fold.unit(5_000 + i as u64, key, UnitKind::Operand, 7)
                    .expect("unit")
            );
        }
        (fold, None)
    }));
    let tomb = Tomb {
        seqno: 6_000,
        ordinal: 0,
    };
    let mut from: Option<Vec<u8>> = None;
    loop {
        let mut next = None;
        apply(within_reserve("a range page over a cold tree", || {
            let mut page = Fold::metered(scope, &budget);
            let paged = page
                .range_page(tomb, b"node:", b"node;", 10, from.as_deref(), 64)
                .expect("page");
            let resume = match &paged {
                RangePage::From(k, _) => Some(k.clone()),
                _ => None,
            };
            next = Some(paged);
            (page, resume)
        }));
        match next.expect("paged") {
            RangePage::Done(_) => break,
            RangePage::From(k, _) => from = Some(k.to_vec()),
            RangePage::NoRoom => panic!("no room in a gigabyte"),
        }
    }
}

/// A tree whose deeper level holds many tables in one run goes on folding
/// within a round's share of memory: a read holds a run's tables one at a
/// time past its first, so the reads are reckoned by runs, not tables, and
/// what the fold holds at once (counted by the allocator) stays within it.
/// Before, the reckoning counted every table, and a tree of 150 tables
/// asked more than the share for its reads: nothing was folded until a
/// compaction left fewer files.
#[test]
fn a_deep_run_of_many_tables_folds_within_a_rounds_share() {
    const TABLES: usize = 150;
    let dir = TempDir::new().expect("tempdir");
    let keys = mixed_keys(6);
    {
        let engine = engine(dir.path());
        let scope = scope(&engine);
        let tree = scope.tree;
        let mut fold = Fold::new(scope);
        for (i, key) in keys.iter().enumerate() {
            fold.unit(10 + i as u64, key, UnitKind::Operand, 5)
                .expect("unit");
        }
        for n in 0..(TABLES as u64 * 80) {
            fold.unit(
                100,
                format!("node:filler:{n:08}").as_bytes(),
                UnitKind::Value,
                9,
            )
            .expect("unit");
        }
        tree.apply_batch(batch_of(fold), engine.next_seqno())
            .expect("apply");
        engine.persist().expect("a table");
        // Tables of a few kilobytes, all in one run below the first level.
        tree.major_compact(4 << 10, 0).expect("compact");
        assert!(
            tree.table_count() >= TABLES,
            "the run holds {} tables",
            tree.table_count()
        );
    }
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.block_cache_bytes = 0;
    let engine = StorageEngine::open(&config).expect("open cold");
    let scope = scope(&engine);
    let tree = scope.tree;
    assert!(super::runs(tree) <= 2, "{} runs", super::runs(tree));
    // What a fold through two runs goes on with, and room for its units: a
    // fraction of what reads reckoned per table would ask.
    let share = least_memory(2) + (4 << 20);
    assert!(
        share < reads_cost(TABLES),
        "the share is below a per-table reckoning"
    );
    let budget = QueryBudget::new(share);
    let apply = |batch| tree.apply_batch(batch, engine.next_seqno()).expect("apply");
    apply(within_reserve("points through a deep run", || {
        let mut fold = Fold::metered(scope, &budget);
        for (i, key) in keys.iter().enumerate() {
            assert!(
                fold.unit(5_000 + i as u64, key, UnitKind::Operand, 7)
                    .expect("unit"),
                "room for a unit in a round's share"
            );
        }
        (fold, None)
    }));
    let tomb = Tomb {
        seqno: 6_000,
        ordinal: 0,
    };
    let paged = std::cell::Cell::new(0usize);
    let mut from: Option<Vec<u8>> = None;
    loop {
        let mut next = None;
        apply(within_reserve("a range page through a deep run", || {
            let mut page = Fold::metered(scope, &budget);
            let result = page
                .range_page(tomb, b"node:", b"node;", 10, from.as_deref(), 64)
                .expect("page");
            let resume = match &result {
                RangePage::From(k, _) => Some(k.clone()),
                _ => None,
            };
            next = Some(result);
            (page, resume)
        }));
        paged.set(paged.get() + 1);
        match next.expect("paged") {
            RangePage::Done(_) => break,
            RangePage::From(k, _) => from = Some(k.to_vec()),
            RangePage::NoRoom => panic!("no room for a page in a round's share"),
        }
        assert!(paged.get() < 10_000, "the range never finished");
    }
}

/// A run a flush adds while a fold goes on is reserved for at the fold's
/// next reservation, before the reads that go through it: what the fold
/// holds covers the runs the tree had at its last reservation and one more.
/// Before, the reads were reckoned once, as the tree stood when the fold
/// began.
#[test]
fn a_run_added_while_folding_is_reserved_before_it_is_read() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let scope = scope(&engine);
    let tree = scope.tree;
    let budget = QueryBudget::new(1 << 30);
    let mut fold = Fold::metered(scope, &budget);
    assert!(fold.unit(10, b"node:a", UnitKind::Value, 3).expect("unit"));
    let before = fold.runs_held;
    for n in 0..3u64 {
        tree.insert(format!("node:x{n}"), b"v", engine.next_seqno());
        engine.persist().expect("a table");
    }
    let now = super::runs(tree);
    assert!(
        now + 1 > before,
        "flushes added runs: {before} held, {now} now"
    );
    assert!(fold.unit(11, b"node:b", UnitKind::Value, 3).expect("unit"));
    assert_eq!(fold.runs_held, now + 1, "the added runs are reserved for");
}

/// A range deletion over the longest keys with no memory to fold even its
/// first key takes nothing in: it allocates no more than it could reserve,
/// leaves the range where it was, and the same range folds once there is
/// room.
#[test]
fn a_range_with_no_room_takes_nothing_in() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let scope = scope(&engine);
    let tree = scope.tree;
    let keys = mixed_keys(6);
    let mut fold = Fold::new(scope);
    for (i, key) in keys.iter().enumerate() {
        fold.unit(10 + i as u64, key, UnitKind::Value, 100)
            .expect("unit");
    }
    tree.apply_batch(batch_of(fold), engine.next_seqno())
        .expect("apply");
    let tomb = Tomb {
        seqno: 50,
        ordinal: 0,
    };
    for room in [0u64, 1, 4096] {
        let budget = QueryBudget::new(room);
        let mut outcome = None;
        let batch = within_reserve(&format!("a range page in {room} bytes"), || {
            let mut page = Fold::metered(scope, &budget);
            outcome = Some(
                page.range_page(tomb, b"node:", b"node;", 10, None, 4)
                    .expect("page"),
            );
            (page, None)
        });
        assert!(
            matches!(outcome, Some(RangePage::NoRoom)),
            "a page with {room} bytes went on: {outcome:?}"
        );
        assert!(batch.is_empty(), "a page with no room wrote records");
        // Completing a commit with no room for its mark or its write takes
        // nothing in and writes nothing either.
        let batch = within_reserve(&format!("a commit completed in {room} bytes"), || {
            let mut fold = Fold::metered(scope, &budget).after(Some(40));
            assert!(!fold.close(), "a mark in {room} bytes");
            (fold, None)
        });
        assert!(batch.is_empty(), "a fold with no room wrote records");
    }
    // With room the same range folds whole.
    let budget = QueryBudget::new(1 << 30);
    let mut from: Option<Vec<u8>> = None;
    loop {
        let mut page = Fold::metered(scope, &budget);
        let next = page
            .range_page(tomb, b"node:", b"node;", 10, from.as_deref(), 4)
            .expect("page");
        tree.apply_batch(batch_of(page), engine.next_seqno())
            .expect("apply");
        match next {
            RangePage::Done(_) => break,
            RangePage::From(k, _) => from = Some(k.to_vec()),
            RangePage::NoRoom => panic!("no room in a gigabyte"),
        }
    }
    assert_eq!(charge_from(&scope, 50, SeqNo::MAX).expect("read"), 10);
}

/// A unit older than one already folded is refused: the fold runs in commit
/// order only.
#[test]
fn a_unit_out_of_commit_order_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(dir.path());
    let scope = scope(&engine);
    let mut fold = Fold::new(scope);
    fold.unit(20, b"node:a", UnitKind::Value, 10)
        .expect("first");
    assert!(fold.unit(10, b"node:b", UnitKind::Value, 10).is_err());
}
