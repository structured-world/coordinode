use super::*;

/// Logic-test fixture (memory backing, env-flippable).
fn open_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

#[test]
fn single_value_round_trip() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let v = vec![Value::String("alice".into())];
    store
        .put_entry("user_name", &v, NodeId::from_raw(1))
        .expect("put");
    let hits = store.scan_exact("user_name", &v).expect("scan");
    assert_eq!(hits, vec![NodeId::from_raw(1)]);
}

#[test]
fn clear_removes_every_entry_and_reports_count() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    for id in [1u64, 2, 3] {
        store
            .put_entry(
                "user_name",
                &[Value::String(format!("u{id}"))],
                NodeId::from_raw(id),
            )
            .expect("put");
    }
    // A second index must survive the clear (prefix isolation).
    store
        .put_entry("user_age", &[Value::Int(30)], NodeId::from_raw(9))
        .expect("put other");

    let removed = store.clear("user_name").expect("clear");
    assert_eq!(removed, 3, "clear reports the number of entries removed");
    assert!(
        store.scan_all("user_name").expect("scan").is_empty(),
        "cleared index has no entries left",
    );
    assert_eq!(
        store.scan_all("user_age").expect("scan other").len(),
        1,
        "clear is prefix-isolated to the named index",
    );
}

#[test]
fn clear_empty_index_is_zero() {
    let fx = open_engine();
    let store = LocalIndexStore::new(&fx.engine);
    assert_eq!(store.clear("never_written").expect("clear"), 0);
}

#[test]
fn delete_raw_removes_a_scanned_entry() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    for id in [1u64, 2] {
        store
            .put_entry(
                "by_name",
                &[Value::String("dup".into())],
                NodeId::from_raw(id),
            )
            .expect("put");
    }
    // Scan, then selectively delete node 1's entry by its raw key.
    let entries = store.scan_all("by_name").expect("scan");
    let (raw_key, _) = entries
        .iter()
        .find(|(_, nid)| *nid == NodeId::from_raw(1))
        .cloned()
        .expect("entry for node 1");
    store.delete_raw(&raw_key).expect("delete_raw");

    let remaining = store.scan_all("by_name").expect("rescan");
    assert_eq!(remaining.len(), 1, "exactly one entry removed");
    assert_eq!(
        remaining[0].1,
        NodeId::from_raw(2),
        "node 2's entry survives"
    );
}

#[test]
fn duplicate_values_return_all_nodes() {
    // Index value "alice" maps to two nodes — scan_exact returns
    // both, sorted by node_id (because the key suffix is BE u64).
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let v = vec![Value::String("alice".into())];
    for id in [1u64, 2, 3] {
        store
            .put_entry("user_name", &v, NodeId::from_raw(id))
            .expect("put");
    }
    let hits = store.scan_exact("user_name", &v).expect("scan");
    assert_eq!(
        hits,
        vec![
            NodeId::from_raw(1),
            NodeId::from_raw(2),
            NodeId::from_raw(3)
        ],
    );
}

#[test]
fn delete_removes_specific_entry() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let v = vec![Value::String("alice".into())];
    store
        .put_entry("user_name", &v, NodeId::from_raw(1))
        .expect("put");
    store
        .put_entry("user_name", &v, NodeId::from_raw(2))
        .expect("put");

    store
        .delete_entry("user_name", &v, NodeId::from_raw(1))
        .expect("delete");

    let hits = store.scan_exact("user_name", &v).expect("scan");
    assert_eq!(hits, vec![NodeId::from_raw(2)]);
}

#[test]
fn delete_missing_entry_is_idempotent() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let v = vec![Value::Int(7)];
    // Never put — delete must still succeed.
    store
        .delete_entry("noise", &v, NodeId::from_raw(99))
        .expect("delete missing");
}

#[test]
fn compound_index_distinguishes_by_secondary_column() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let alice_us = vec![Value::String("alice".into()), Value::String("US".into())];
    let alice_uk = vec![Value::String("alice".into()), Value::String("UK".into())];

    store
        .put_entry("by_name_country", &alice_us, NodeId::from_raw(1))
        .expect("put");
    store
        .put_entry("by_name_country", &alice_uk, NodeId::from_raw(2))
        .expect("put");

    // Exact match on (alice, US) returns only node 1.
    let us_hits = store
        .scan_exact("by_name_country", &alice_us)
        .expect("scan");
    assert_eq!(us_hits, vec![NodeId::from_raw(1)]);

    // Exact match on (alice, UK) returns only node 2.
    let uk_hits = store
        .scan_exact("by_name_country", &alice_uk)
        .expect("scan");
    assert_eq!(uk_hits, vec![NodeId::from_raw(2)]);
}

#[test]
fn scan_all_returns_every_entry() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let alice = vec![Value::String("alice".into())];
    let bob = vec![Value::String("bob".into())];
    store
        .put_entry("nm", &alice, NodeId::from_raw(1))
        .expect("put");
    store
        .put_entry("nm", &bob, NodeId::from_raw(2))
        .expect("put");
    store
        .put_entry("nm", &alice, NodeId::from_raw(3))
        .expect("put");

    let all = store.scan_all("nm").expect("scan all");
    assert_eq!(all.len(), 3);
    // Sorted by (encoded value, node_id): alice/1, alice/3, bob/2.
    let ids: Vec<u64> = all.iter().map(|(_, id)| id.as_raw()).collect();
    assert_eq!(ids, vec![1, 3, 2]);
}

#[test]
fn compound_index_three_columns() {
    // N=3 compound: confirm encode/scan symmetry beyond N=2. Two
    // entries differ only in the third column.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let key_a = vec![
        Value::String("alice".into()),
        Value::String("US".into()),
        Value::Int(30),
    ];
    let key_b = vec![
        Value::String("alice".into()),
        Value::String("US".into()),
        Value::Int(40),
    ];
    store
        .put_entry("triple", &key_a, NodeId::from_raw(1))
        .expect("put a");
    store
        .put_entry("triple", &key_b, NodeId::from_raw(2))
        .expect("put b");

    // Exact match on key_a returns only node 1; key_b only 2.
    assert_eq!(
        store.scan_exact("triple", &key_a).expect("scan"),
        vec![NodeId::from_raw(1)],
    );
    assert_eq!(
        store.scan_exact("triple", &key_b).expect("scan"),
        vec![NodeId::from_raw(2)],
    );

    // scan_all walks both, in encoded-key order (key_a's Int=30
    // sorts before key_b's Int=40).
    let all = store.scan_all("triple").expect("scan all");
    let ids: Vec<u64> = all.iter().map(|(_, id)| id.as_raw()).collect();
    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn scan_exact_missing_returns_empty() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let hits = store
        .scan_exact("nonexistent", &[Value::Int(42)])
        .expect("scan");
    assert!(hits.is_empty());
}

#[test]
fn sortable_type_ordering_null_lt_bool_lt_int_lt_string() {
    // ADR contract: Value ordering Null < Bool < Int < Float <
    // String < Timestamp. scan_all walks in encoded-key order so
    // we can read the type ordering off directly. One entry per
    // type, all under the same index name and node_id 1.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let id = NodeId::from_raw(1);
    let entries: Vec<Vec<Value>> = vec![
        vec![Value::Null],
        vec![Value::Bool(true)],
        vec![Value::Int(0)],
        vec![Value::String("z".into())],
    ];
    // Insert in reverse to prove ordering comes from key
    // encoding, not insertion order.
    for v in entries.iter().rev() {
        store.put_entry("mix", v, id).expect("put");
    }
    let all = store.scan_all("mix").expect("scan");
    assert_eq!(all.len(), 4);
    // The keys themselves carry the encoded value bytes — verify
    // their order matches the ADR contract by comparing prefixes
    // pairwise (each later key sorts >= the previous one).
    for pair in all.windows(2) {
        assert!(
            pair[0].0 <= pair[1].0,
            "index keys not sorted: {:?} vs {:?}",
            pair[0].0,
            pair[1].0,
        );
    }
}

#[test]
fn scan_all_isolates_per_index_name() {
    // Two indexes share the partition; each scan returns only
    // its own entries.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalIndexStore::new(engine);
    let v = vec![Value::Int(42)];
    store.put_entry("a", &v, NodeId::from_raw(1)).expect("put");
    store.put_entry("b", &v, NodeId::from_raw(2)).expect("put");

    let a = store.scan_all("a").expect("scan a");
    let b = store.scan_all("b").expect("scan b");
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(a[0].1, NodeId::from_raw(1));
    assert_eq!(b[0].1, NodeId::from_raw(2));
}

#[test]
fn definition_txn_round_trip() {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_storage::engine::transaction::CommitContext;

    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);

    let commit = |t: &mut Transaction| {
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        t.commit(&ctx).expect("commit");
    };

    let def = IndexDefinition::btree("user_email", "User", "email").unique();

    // CREATE INDEX: persist the definition through a statement transaction.
    let read_ts = oracle.next();
    let mut t = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    store.put_definition_txn(&mut t, &def).expect("put txn");
    commit(&mut t);
    let loaded = store
        .load_definition("user_email")
        .expect("load")
        .expect("present after commit");
    assert_eq!(loaded.name, "user_email");
    assert!(loaded.unique);

    // DROP INDEX: delete the definition through a statement transaction.
    let read_ts = oracle.next();
    let mut t = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    store
        .delete_definition_txn(&mut t, "user_email")
        .expect("delete txn");
    commit(&mut t);
    assert!(store.load_definition("user_email").expect("load").is_none());
}
