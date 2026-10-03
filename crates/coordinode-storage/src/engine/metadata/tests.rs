//! Metadata commands decided at ordered application.

use std::sync::Arc;

use coordinode_core::graph::intern::{DictionaryError, field_id_key, field_name_key};
use coordinode_core::txn::proposal::{MetadataCommand, Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn durable_cfg(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open(dir: &TempDir) -> (StorageEngine, Arc<TimestampOracle>) {
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&durable_cfg(dir), oracle.clone()).unwrap();
    (engine, oracle)
}

fn register(names: &[&str]) -> Mutation {
    Mutation::Command(MetadataCommand::RegisterFields {
        names: names.iter().map(|n| (*n).to_owned()).collect(),
    })
}

fn adopt(bindings: &[(&str, u32)]) -> Mutation {
    Mutation::Command(MetadataCommand::AdoptFields {
        bindings: bindings
            .iter()
            .map(|(n, i)| ((*n).to_owned(), *i))
            .collect(),
    })
}

fn apply(engine: &StorageEngine, commit_ts: u64, mutation: Mutation) {
    engine.apply_proposal_at(&[mutation], commit_ts).unwrap();
}

/// New names get the next ids in the order the batch lists them; a name
/// already bound, or listed twice, keeps its one id.
#[test]
fn registration_assigns_increasing_ids_in_batch_order() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    apply(&engine, oracle.next().as_raw(), register(&["b", "a", "b"]));
    apply(&engine, oracle.next().as_raw(), register(&["a", "c"]));
    let dictionary = load_field_dictionary(&engine).unwrap();
    assert_eq!(dictionary.lookup("b"), Some(1));
    assert_eq!(dictionary.lookup("a"), Some(2));
    assert_eq!(dictionary.lookup("c"), Some(3));
    assert_eq!(dictionary.len(), 3);
    assert_eq!(field_frontier(&engine).unwrap(), 3);
}

/// Proposals can apply in another order than their commit timestamps sort
/// in (a new leader's entries after an old leader's with later stamps). The
/// decision follows application order; stamps change nothing, because no
/// record is ever written twice.
#[test]
fn application_order_decides_even_against_reversed_timestamps() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let early = oracle.next().as_raw();
    let late = oracle.next().as_raw();
    apply(&engine, late, register(&["first"]));
    apply(&engine, early, register(&["second"]));
    let dictionary = load_field_dictionary(&engine).unwrap();
    assert_eq!(dictionary.lookup("first"), Some(1));
    assert_eq!(dictionary.lookup("second"), Some(2));
}

/// A batch the id space cannot hold publishes nothing, not the part of it
/// that fits.
#[test]
fn an_exhausted_id_space_refuses_the_whole_batch() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    apply(
        &engine,
        oracle.next().as_raw(),
        adopt(&[("last", u32::MAX)]),
    );
    apply(&engine, oracle.next().as_raw(), register(&["x", "y"]));
    let dictionary = load_field_dictionary(&engine).unwrap();
    assert_eq!(dictionary.lookup("x"), None);
    assert_eq!(dictionary.lookup("y"), None);
    assert_eq!(dictionary.len(), 1);
}

/// A batch over the size limit publishes nothing.
#[test]
fn an_oversized_batch_is_refused() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let names: Vec<String> = (0..=coordinode_core::graph::intern::MAX_REGISTRATION_BATCH)
        .map(|i| format!("n{i}"))
        .collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    apply(&engine, oracle.next().as_raw(), register(&refs));
    assert!(load_field_dictionary(&engine).unwrap().is_empty());
}

/// Adoption publishes exactly the ids given, accepts bindings already held,
/// and refuses, as a whole, a batch that contradicts one.
#[test]
fn adoption_keeps_ids_and_refuses_contradictions() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    apply(
        &engine,
        oracle.next().as_raw(),
        adopt(&[("a", 5), ("b", 9)]),
    );
    apply(
        &engine,
        oracle.next().as_raw(),
        adopt(&[("a", 5), ("c", 10)]),
    );
    // `b` would get a second id: nothing of this batch is published.
    apply(
        &engine,
        oracle.next().as_raw(),
        adopt(&[("d", 11), ("b", 12)]),
    );
    let dictionary = load_field_dictionary(&engine).unwrap();
    assert_eq!(dictionary.lookup("a"), Some(5));
    assert_eq!(dictionary.lookup("b"), Some(9));
    assert_eq!(dictionary.lookup("c"), Some(10));
    assert_eq!(dictionary.lookup("d"), None);
    // Registration continues above the adopted frontier.
    apply(&engine, oracle.next().as_raw(), register(&["e"]));
    assert_eq!(
        load_field_dictionary(&engine).unwrap().lookup("e"),
        Some(11)
    );
}

/// Records that disagree are reported, not repaired into a dictionary.
#[test]
fn inconsistent_records_fail_the_load() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    apply(&engine, oracle.next().as_raw(), register(&["a", "b"]));
    engine
        .put(Partition::Schema, &field_name_key("a"), &2u32.to_be_bytes())
        .unwrap();
    assert!(matches!(
        load_field_dictionary(&engine),
        Err(StorageError::FieldDictionary(_))
    ));

    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    apply(&engine, oracle.next().as_raw(), register(&["a"]));
    engine
        .put(Partition::Schema, &field_id_key(1), &[0xFF])
        .unwrap();
    assert!(matches!(
        load_field_dictionary(&engine),
        Err(StorageError::FieldDictionary(DictionaryError::Malformed(_)))
    ));

    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    apply(&engine, oracle.next().as_raw(), register(&["a"]));
    engine
        .delete(Partition::Schema, &field_name_key("a"))
        .unwrap();
    assert!(load_field_dictionary(&engine).is_err());
}

/// The bindings above a frontier are the ones registered after it.
#[test]
fn bindings_above_a_frontier() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    apply(&engine, oracle.next().as_raw(), register(&["a", "b"]));
    apply(&engine, oracle.next().as_raw(), register(&["c"]));
    assert_eq!(
        field_bindings_above(&engine, 2).unwrap(),
        vec![("c".to_owned(), 3)]
    );
    assert!(field_bindings_above(&engine, 3).unwrap().is_empty());
    assert!(field_bindings_above(&engine, u32::MAX).unwrap().is_empty());
}

/// Every applied binding moves the dictionary generation; a batch that
/// registers nothing new does not.
#[test]
fn applied_bindings_move_the_generation() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let start = engine.field_dictionary_generation();
    apply(&engine, oracle.next().as_raw(), register(&["a"]));
    let after = engine.field_dictionary_generation();
    assert!(after > start);
    apply(&engine, oracle.next().as_raw(), register(&["a"]));
    assert_eq!(engine.field_dictionary_generation(), after);
}

/// Only the first command of a proposal is decided: a second one would
/// decide against a dictionary lacking the first one's bindings.
#[test]
fn a_second_command_in_one_proposal_is_refused() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(
            &[register(&["a"]), register(&["b"])],
            oracle.next().as_raw(),
        )
        .unwrap();
    let dictionary = load_field_dictionary(&engine).unwrap();
    assert_eq!(dictionary.lookup("a"), Some(1));
    assert_eq!(dictionary.lookup("b"), None);
}

/// On the journalled path the journal records the decided bindings, so a
/// crash before any flush replays exactly the ids the live apply chose.
#[test]
fn journalled_bindings_replay_as_decided() {
    let dir = TempDir::new().unwrap();
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).unwrap();
        engine
            .commit_journaled(&[register(&["x", "y"])], oracle.next().as_raw())
            .unwrap();
        engine
            .commit_journaled(&[register(&["z"])], oracle.next().as_raw())
            .unwrap();
        // Dropped without a flush: only the journal holds the bindings.
    }
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).unwrap();
    let dictionary = load_field_dictionary(&engine).unwrap();
    assert_eq!(dictionary.lookup("x"), Some(1));
    assert_eq!(dictionary.lookup("y"), Some(2));
    assert_eq!(dictionary.lookup("z"), Some(3));
}

/// Concurrent registrations through the journal each land whole, with
/// distinct ids, and the same after a replay.
#[test]
fn concurrent_journalled_registrations_get_distinct_ids() {
    let dir = TempDir::new().unwrap();
    let live = {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).unwrap();
        std::thread::scope(|scope| {
            for t in 0..8 {
                let engine = &engine;
                let oracle = &oracle;
                scope.spawn(move || {
                    for i in 0..20 {
                        let name = format!("t{t}_{i}");
                        engine
                            .commit_journaled(&[register(&[&name])], oracle.next().as_raw())
                            .unwrap();
                    }
                });
            }
        });
        let dictionary = load_field_dictionary(&engine).unwrap();
        assert_eq!(dictionary.len(), 160);
        assert_eq!(dictionary.frontier(), 160);
        dictionary.to_bytes().unwrap()
    };
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).unwrap();
    assert_eq!(
        load_field_dictionary(&engine).unwrap().to_bytes().unwrap(),
        live,
        "a replay must reproduce the live bindings"
    );
}

/// A load that runs while bindings are being applied sees each binding whole
/// or not at all: its two records land together, so a reader never finds a
/// name record without the id record it points to.
#[test]
fn a_load_during_registrations_sees_whole_bindings() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for i in 0..2_000 {
                let name = format!("f{i}");
                apply(&engine, oracle.next().as_raw(), register(&[&name]));
            }
            done.store(true, std::sync::atomic::Ordering::Release);
        });
        while !done.load(std::sync::atomic::Ordering::Acquire) {
            let loaded = load_field_dictionary(&engine);
            assert!(
                loaded.is_ok(),
                "a load during registrations saw a torn binding: {:?}",
                loaded.err()
            );
            let above = field_bindings_above(&engine, 0);
            assert!(
                above.is_ok(),
                "a refresh during registrations saw a torn binding: {:?}",
                above.err()
            );
        }
    });
    assert_eq!(load_field_dictionary(&engine).unwrap().len(), 2_000);
}

fn grant(base: u64, ceiling: u64, token: u8) -> Mutation {
    Mutation::Command(MetadataCommand::GrantNodeLease {
        base,
        ceiling,
        token: [token; NODE_LEASE_TOKEN_LEN],
    })
}

/// A grant from the current ceiling is recorded; a grant from a ceiling that
/// has moved on is refused, whatever timestamps the two carry.
#[test]
fn a_lease_is_granted_only_from_the_current_ceiling() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    assert_eq!(node_lease_ceiling(&engine).unwrap(), 0);
    let early = oracle.next().as_raw();
    let late = oracle.next().as_raw();
    apply(&engine, late, grant(0, 100, 1));
    // Read the same base, applied after the first: refused.
    apply(&engine, early, grant(0, 100, 2));
    assert_eq!(node_lease_ceiling(&engine).unwrap(), 100);
    assert_eq!(
        node_lease_holder(&engine, 100).unwrap(),
        Some([1; NODE_LEASE_TOKEN_LEN])
    );
    apply(&engine, oracle.next().as_raw(), grant(100, 200, 3));
    assert_eq!(node_lease_ceiling(&engine).unwrap(), 200);
    assert_eq!(
        node_lease_holder(&engine, 200).unwrap(),
        Some([3; NODE_LEASE_TOKEN_LEN])
    );
}

/// A grant that is not a range, or leaves the NodeId space, is refused.
#[test]
fn a_lease_outside_the_id_space_is_refused() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    apply(&engine, oracle.next().as_raw(), grant(0, 0, 1));
    apply(
        &engine,
        oracle.next().as_raw(),
        grant(0, NODE_ID_MAX_SEQUENCE + 1, 1),
    );
    assert_eq!(node_lease_ceiling(&engine).unwrap(), 0);
}

#[test]
fn a_command_touches_only_the_schema_partition() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(
            &[
                register(&["a"]),
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:x".to_vec(),
                    value: b"v".to_vec(),
                },
            ],
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(bound_id(&engine, "a").unwrap(), Some(1));
    assert_eq!(bound_name(&engine, 1).unwrap().as_deref(), Some("a"));
    assert!(engine.get(Partition::Node, b"node:x").unwrap().is_some());
}

/// A journaled command that decides nothing (its names are bound already, or
/// it is refused) has nothing to record: it takes no journal entry, so it
/// pays no durable append. Proposers that lose a registration race hit this
/// on every name the winner bound.
#[test]
fn a_journaled_command_that_decides_nothing_appends_nothing() {
    let dir = TempDir::new().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).unwrap();
    let journaled = |engine: &StorageEngine| engine.oplog_read_since(0).unwrap().unwrap().len();

    engine
        .commit_journaled(&[register(&["a"])], oracle.next().as_raw())
        .unwrap();
    let after_binding = journaled(&engine);

    engine
        .commit_journaled(&[register(&["a"])], oracle.next().as_raw())
        .unwrap();
    let too_long = "x".repeat(usize::from(u16::MAX) + 1);
    engine
        .commit_journaled(&[register(&[too_long.as_str()])], oracle.next().as_raw())
        .unwrap();
    assert_eq!(
        journaled(&engine),
        after_binding,
        "a bound name and a refused batch append nothing"
    );

    engine
        .commit_journaled(&[register(&["a", "b"])], oracle.next().as_raw())
        .unwrap();
    assert_eq!(journaled(&engine), after_binding + 1, "a new binding does");
    assert_eq!(bound_id(&engine, "b").unwrap(), Some(2));
}

fn record_pair(engine: u32, host_epoch: u64) -> Mutation {
    Mutation::Command(MetadataCommand::RecordGroupPair {
        pair: VersionPair { engine, host_epoch },
    })
}

/// The group's pair records are numbered in application order; a leader
/// re-elected at the pair already recorded adds none, and a move to any
/// other pair, the host epoch alone included, adds the next one.
#[test]
fn group_pair_records_follow_application_order() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    assert_eq!(recorded_group_pair(&engine).unwrap(), None);
    apply(&engine, oracle.next().as_raw(), record_pair(1, 0));
    apply(&engine, oracle.next().as_raw(), record_pair(1, 0));
    let first = recorded_group_pair(&engine).unwrap().unwrap();
    assert_eq!(first.seq, 1);
    assert_eq!(
        first.pair,
        VersionPair {
            engine: 1,
            host_epoch: 0
        }
    );

    apply(&engine, oracle.next().as_raw(), record_pair(1, 7));
    let moved = recorded_group_pair(&engine).unwrap().unwrap();
    assert_eq!(moved.seq, 2);
    assert_eq!(moved.pair.host_epoch, 7);
    assert!(moved.is_later_than(&first));

    // A record of the earlier pair after the move is a new record too: the
    // log, not the pair's value, orders them.
    apply(&engine, oracle.next().as_raw(), record_pair(1, 0));
    assert_eq!(recorded_group_pair(&engine).unwrap().unwrap().seq, 3);
}
