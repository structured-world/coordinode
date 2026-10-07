use super::*;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_modality::LocalIndexStore;
use coordinode_storage::engine::transaction::CommitContext;

struct Fixture {
    _dir: tempfile::TempDir,
    engine: StorageEngine,
    oracle: TimestampOracle,
    interner: FieldInterner,
}

fn fixture() -> Fixture {
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    Fixture {
        engine: StorageEngine::open(&config).expect("open engine"),
        _dir: dir,
        oracle: TimestampOracle::resume_from(Timestamp::from_raw(1)),
        interner: FieldInterner::new(),
    }
}

fn commit(txn: &mut Transaction<'_>) -> Result<(), CommitError> {
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).map(|_| ())
}

fn put_node(fx: &mut Fixture, node_id: u64, label: &str, props: &[(&str, Value)]) {
    let mut record = NodeRecord::new(label);
    for (name, value) in props {
        let field = fx.interner.intern(name);
        record.set(field, value.clone());
    }
    let mut txn = Transaction::begin(&fx.engine, Some(&fx.oracle), fx.oracle.next());
    LocalNodeStore
        .put(&mut txn, 1, NodeId::from_raw(node_id), &record)
        .expect("put");
    commit(&mut txn).expect("commit");
}

fn backfill(fx: &Fixture) -> Backfill<'_> {
    Backfill {
        engine: &fx.engine,
        oracle: Some(&fx.oracle),
        interner: &fx.interner,
        shard_id: 1,
        definition_version: None,
        older_transactions_wait: super::DEFAULT_OLDER_TRANSACTIONS_WAIT,
        progress: None,
        covered: None,
        repair: None,
    }
}

fn lookup(fx: &Fixture, index: &IndexDefinition, value: &str) -> Vec<u64> {
    let mut txn = Transaction::begin(&fx.engine, Some(&fx.oracle), fx.oracle.next());
    let mut ids: Vec<u64> = LocalIndexStore::new(&fx.engine)
        .scan_exact(&mut txn, index, &[Value::String(value.into())])
        .expect("scan")
        .expect("indexable")
        .into_iter()
        .map(|n| n.as_raw())
        .collect();
    ids.sort_unstable();
    ids
}

fn email(v: &str) -> [(&'static str, Value); 1] {
    [("email", Value::String(v.into()))]
}

/// `descriptor` as the index numbered 1, serving from generation 1: the
/// backfill is driven without a catalog here.
fn bound(descriptor: crate::index::IndexDescriptor) -> IndexDefinition {
    descriptor.bind(
        crate::index::IndexId::from_raw(1),
        crate::index::GenerationId::from_raw(1),
    )
}

/// The nodes of the index's label are indexed, others are not.
#[test]
fn a_backfill_indexes_the_nodes_of_its_label() {
    let mut fx = fixture();
    put_node(&mut fx, 1, "User", &email("alice@x"));
    put_node(&mut fx, 2, "User", &email("bob@x"));
    put_node(&mut fx, 3, "Movie", &email("alice@x"));

    let index = bound(crate::index::IndexDescriptor::btree(
        "user_email",
        "User",
        "email",
    ));
    let indexed = backfill(&fx).run(&index, &mut commit).expect("backfill");

    assert_eq!(indexed, 2);
    assert_eq!(lookup(&fx, &index, "alice@x"), vec![1]);
    assert_eq!(lookup(&fx, &index, "bob@x"), vec![2]);
}

/// Stored data that already breaks a unique index stops its build and names
/// the node holding the value.
#[test]
fn a_unique_backfill_stops_at_a_duplicate() {
    let mut fx = fixture();
    put_node(&mut fx, 1, "User", &email("same@x"));
    put_node(&mut fx, 2, "User", &email("same@x"));

    let index = bound(crate::index::IndexDescriptor::btree("user_email", "User", "email").unique());
    let err = backfill(&fx)
        .run(&index, &mut commit)
        .expect_err("duplicate data");
    assert!(
        matches!(&err, BackfillError::Duplicate(v) if v.holder == NodeId::from_raw(1)),
        "expected a duplicate held by node 1, got {err:?}"
    );
}

/// A sparse index skips a node missing the property.
#[test]
fn a_sparse_backfill_skips_missing_values() {
    let mut fx = fixture();
    put_node(&mut fx, 1, "User", &[("bio", Value::String("dev".into()))]);
    put_node(&mut fx, 2, "User", &[("name", Value::String("bob".into()))]);

    let index = bound(crate::index::IndexDescriptor::btree("user_bio", "User", "bio").sparse());
    assert_eq!(backfill(&fx).run(&index, &mut commit).expect("backfill"), 1);
}

/// A label larger than a page is indexed whole, one committed transaction
/// per page.
#[test]
fn a_backfill_spans_pages() {
    let mut fx = fixture();
    let total = PAGE as u64 * 2 + 7;
    for id in 1..=total {
        put_node(&mut fx, id, "User", &email(&format!("u{id}@x")));
    }
    let index = bound(crate::index::IndexDescriptor::btree("user_email", "User", "email").unique());
    let mut commits = 0;
    let indexed = backfill(&fx)
        .run(&index, &mut |txn| {
            commits += 1;
            commit(txn)
        })
        .expect("backfill");
    assert_eq!(indexed, total);
    assert_eq!(commits, 3);
    assert_eq!(lookup(&fx, &index, &format!("u{total}@x")), vec![total]);
}

/// A backfill reports the wait for older transactions, then the entries
/// committed after every page, ending at its total.
#[test]
fn a_backfill_reports_its_progress_per_page() {
    let mut fx = fixture();
    let total = PAGE as u64 + 3;
    for id in 1..=total {
        put_node(&mut fx, id, "User", &email(&format!("u{id}@x")));
    }
    let index = bound(crate::index::IndexDescriptor::btree(
        "user_email",
        "User",
        "email",
    ));
    let reports = std::cell::RefCell::new(Vec::new());
    let record = |p: BackfillProgress| reports.borrow_mut().push(p);
    let mut run = backfill(&fx);
    run.progress = Some(&record);
    run.run(&index, &mut |txn| commit(txn)).expect("backfill");

    assert_eq!(
        reports.into_inner(),
        [
            BackfillProgress::AwaitingOlderTransactions,
            BackfillProgress::Indexed(0),
            BackfillProgress::Indexed(PAGE as u64),
            BackfillProgress::Indexed(total),
        ]
    );
}

/// After every committed page the backfill tells the last node key it read,
/// so a writer knows which stored nodes already have their entries.
#[test]
fn a_backfill_tells_the_key_it_covered_through_per_page() {
    let mut fx = fixture();
    let total = PAGE as u64 + 3;
    for id in 1..=total {
        put_node(&mut fx, id, "User", &email(&format!("u{id}@x")));
    }
    let index = bound(crate::index::IndexDescriptor::btree(
        "user_email",
        "User",
        "email",
    ));
    let covered = std::cell::RefCell::new(Vec::new());
    let record = |key: &[u8]| covered.borrow_mut().push(key.to_vec());
    let mut run = backfill(&fx);
    run.covered = Some(&record);
    run.run(&index, &mut |txn| commit(txn)).expect("backfill");

    let node_key =
        |id: u64| coordinode_core::graph::node::encode_node_key(1, NodeId::from_raw(id)).to_vec();
    assert_eq!(
        covered.into_inner(),
        [node_key(PAGE as u64), node_key(total)]
    );
}

/// A node a writer changes after a page read it makes that page conflict at
/// commit; the page is read again, so the index holds the node's new value
/// and no entry for the old one.
#[test]
fn a_page_that_read_a_changed_node_is_read_again() {
    let mut fx = fixture();
    put_node(&mut fx, 1, "User", &email("alice@x"));
    let field = fx.interner.intern("email");

    let index = bound(crate::index::IndexDescriptor::btree(
        "user_email",
        "User",
        "email",
    ));
    let engine = &fx.engine;
    let oracle = &fx.oracle;
    let mut first = true;
    let indexed = backfill(&fx)
        .run(&index, &mut |txn| {
            if std::mem::take(&mut first) {
                // A writer changes the node between the page's read and its
                // commit.
                let mut record = NodeRecord::new("User");
                record.set(field, Value::String("changed@x".into()));
                let mut writer = Transaction::begin(engine, Some(oracle), oracle.next());
                LocalNodeStore
                    .put(&mut writer, 1, NodeId::from_raw(1), &record)
                    .expect("put");
                commit(&mut writer).expect("writer commit");
            }
            commit(txn)
        })
        .expect("backfill");

    assert_eq!(indexed, 1);
    assert_eq!(lookup(&fx, &index, "changed@x"), vec![1]);
    assert!(
        lookup(&fx, &index, "alice@x").is_empty(),
        "the value the node no longer holds must not be indexed"
    );
}

/// A build whose definition record moves (a drop, or a rebuild into another
/// generation) before one of its pages commits stops, and that page writes
/// no entry into the generation it was filling.
#[test]
fn a_page_of_a_replaced_definition_writes_nothing() {
    let mut fx = fixture();
    put_node(&mut fx, 1, "User", &email("alice@x"));

    let store = LocalIndexStore::new(&fx.engine);
    let (engine, oracle) = (&fx.engine, &fx.oracle);
    // A catalog write of its own, as a statement commits one.
    let put = |def: &IndexDefinition| {
        let mut txn = Transaction::begin(engine, Some(oracle), oracle.next());
        store.put_definition_txn(&mut txn, def).expect("stage");
        commit(&mut txn).expect("commit definition");
    };
    let index = bound(crate::index::IndexDescriptor::btree(
        "user_email",
        "User",
        "email",
    ));
    put(&index);
    let published = store.definition_version(index.id).expect("version");
    assert!(published.is_some());

    let mut build = backfill(&fx);
    build.definition_version = published;
    let mut first = true;
    let err = build
        .run(&index, &mut |txn| {
            if std::mem::take(&mut first) {
                // The record moves between the page's read and its commit.
                let mut moved = index.clone();
                moved.generation = crate::index::GenerationId::from_raw(2);
                put(&moved);
            }
            commit(txn)
        })
        .expect_err("the definition moved");
    assert!(matches!(err, BackfillError::Superseded), "{err:?}");
    assert!(
        lookup(&fx, &index, "alice@x").is_empty(),
        "a page of the superseded build must not land"
    );
}
