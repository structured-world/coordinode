use std::sync::Arc;

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};

use super::*;

fn engine() -> (StorageEngine, Arc<TimestampOracle>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap();
    (engine, oracle, dir)
}

fn begin<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
    let snap = engine.snapshot();
    Transaction::new(engine, Some(oracle), Timestamp::from_raw(snap), Some(snap))
}

fn commit(txn: &mut Transaction<'_>) -> Result<(), CommitError> {
    let wc = coordinode_core::txn::write_concern::WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).map(|_| ())
}

/// A claim is read back by the claiming transaction and, once committed, by
/// the next one; a release frees the key.
#[test]
fn a_claimed_key_is_found_until_released() {
    let (engine, oracle, _d) = engine();
    let key = [Value::Int(1001)];

    let mut txn = begin(&engine, &oracle);
    assert_eq!(
        LocalTableKeyStore.lookup(&mut txn, "T", &key).unwrap(),
        None
    );
    LocalTableKeyStore
        .claim(&mut txn, "T", &key, NodeId::from_raw(7))
        .unwrap();
    assert_eq!(
        LocalTableKeyStore.lookup(&mut txn, "T", &key).unwrap(),
        Some(NodeId::from_raw(7)),
        "the claiming transaction reads its own claim"
    );
    commit(&mut txn).unwrap();

    let mut txn = begin(&engine, &oracle);
    assert_eq!(
        LocalTableKeyStore.lookup(&mut txn, "T", &key).unwrap(),
        Some(NodeId::from_raw(7))
    );
    LocalTableKeyStore.release(&mut txn, "T", &key).unwrap();
    commit(&mut txn).unwrap();

    let mut txn = begin(&engine, &oracle);
    assert_eq!(
        LocalTableKeyStore.lookup(&mut txn, "T", &key).unwrap(),
        None
    );
}

/// Two transactions that both saw a key free and both claim it cannot both
/// commit: the loser is refused, and the key holds the winner's row.
#[test]
fn concurrent_claims_of_one_key_conflict() {
    let (engine, oracle, _d) = engine();
    let key = [Value::String("alice".into())];
    let mut a = begin(&engine, &oracle);
    let mut b = begin(&engine, &oracle);
    for (txn, id) in [(&mut a, 1), (&mut b, 2)] {
        assert_eq!(LocalTableKeyStore.lookup(txn, "T", &key).unwrap(), None);
        LocalTableKeyStore
            .claim(txn, "T", &key, NodeId::from_raw(id))
            .unwrap();
    }
    commit(&mut a).unwrap();
    assert!(
        matches!(commit(&mut b), Err(CommitError::Conflict(_))),
        "the second claim of the key must be refused"
    );
    let mut txn = begin(&engine, &oracle);
    assert_eq!(
        LocalTableKeyStore.lookup(&mut txn, "T", &key).unwrap(),
        Some(NodeId::from_raw(1))
    );
}

/// Distinct keys never share an entry, however their encodings could be
/// confused: an embedded zero byte, a prefix of another string, a compound
/// key split at another point, the same bytes under another table, or the
/// same number as another type.
#[test]
fn distinct_keys_have_distinct_entries() {
    let keys: Vec<(&str, Vec<Value>)> = vec![
        ("T", vec![Value::String("a".into())]),
        ("T", vec![Value::String("a\0".into())]),
        ("T", vec![Value::String("a\0b".into())]),
        ("T", vec![Value::String("ab".into())]),
        (
            "T",
            vec![Value::String("a".into()), Value::String("b".into())],
        ),
        (
            "T",
            vec![Value::String("a\0".into()), Value::String("b".into())],
        ),
        ("T", vec![Value::Binary(b"a".to_vec())]),
        ("T", vec![Value::Int(1)]),
        ("T", vec![Value::Float(1.0)]),
        ("T", vec![Value::Timestamp(1)]),
        ("T", vec![Value::Bool(true)]),
        ("T2", vec![Value::String("a".into())]),
        ("T\0", vec![Value::String("a".into())]),
    ];
    let mut seen = std::collections::HashSet::new();
    for (table, key) in &keys {
        let entry = entry_key(table, key).unwrap();
        assert!(seen.insert(entry), "{table} {key:?} shares an entry");
    }
}

/// Within a table, entries sort as their keys do, so a key range is an entry
/// range.
#[test]
fn entries_sort_as_their_keys() {
    let ordered = [
        vec![Value::Int(i64::MIN)],
        vec![Value::Int(-1)],
        vec![Value::Int(0)],
        vec![Value::Int(1)],
        vec![Value::Int(i64::MAX)],
    ];
    let entries: Vec<Vec<u8>> = ordered.iter().map(|k| entry_key("T", k).unwrap()).collect();
    assert!(entries.windows(2).all(|w| w[0] < w[1]));

    let ordered = ["", "a", "a\0", "a\0b", "ab", "b"];
    let entries: Vec<Vec<u8>> = ordered
        .iter()
        .map(|s| entry_key("T", &[Value::String((*s).into())]).unwrap())
        .collect();
    assert!(entries.windows(2).all(|w| w[0] < w[1]), "{ordered:?}");

    let ordered = [f64::NEG_INFINITY, -2.5, 0.0, 1.5, f64::INFINITY];
    let entries: Vec<Vec<u8>> = ordered
        .iter()
        .map(|f| entry_key("T", &[Value::Float(*f)]).unwrap())
        .collect();
    assert!(entries.windows(2).all(|w| w[0] < w[1]));
}

/// Values that compare equal are one key.
#[test]
fn equal_values_are_one_key() {
    assert_eq!(
        entry_key("T", &[Value::Float(0.0)]).unwrap(),
        entry_key("T", &[Value::Float(-0.0)]).unwrap()
    );
}

/// A value with no key equality (NULL, NaN, collections, vectors) is refused
/// by name rather than encoded as some other key.
#[test]
fn values_without_key_equality_are_refused() {
    for value in [
        Value::Null,
        Value::Float(f64::NAN),
        Value::Array(vec![Value::Int(1)]),
        Value::Vector(vec![1.0]),
    ] {
        assert!(
            matches!(
                entry_key("T", std::slice::from_ref(&value)),
                Err(StoreError::UnsupportedKey(_))
            ),
            "{value:?}"
        );
    }
}

/// Dropping a table frees its keys and no other table's.
#[test]
fn release_all_frees_one_table_only() {
    let (engine, oracle, _d) = engine();
    let mut txn = begin(&engine, &oracle);
    for (table, id) in [("T", 1), ("T", 2), ("U", 3)] {
        LocalTableKeyStore
            .claim(
                &mut txn,
                table,
                &[Value::Int(id)],
                NodeId::from_raw(id as u64),
            )
            .unwrap();
    }
    commit(&mut txn).unwrap();

    let mut txn = begin(&engine, &oracle);
    LocalTableKeyStore.release_all(&mut txn, "T").unwrap();
    commit(&mut txn).unwrap();

    let mut txn = begin(&engine, &oracle);
    for id in [1, 2] {
        assert_eq!(
            LocalTableKeyStore
                .lookup(&mut txn, "T", &[Value::Int(id)])
                .unwrap(),
            None
        );
    }
    assert_eq!(
        LocalTableKeyStore
            .lookup(&mut txn, "U", &[Value::Int(3)])
            .unwrap(),
        Some(NodeId::from_raw(3))
    );
}
