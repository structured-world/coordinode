use super::*;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::transaction::CommitContext;

fn open_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

#[test]
fn schema_key_format_is_stable() {
    // The DDL key contract is `encrypted_index:<name>` — integration
    // tests and any future scanner depend on this exact prefix.
    let def = EncryptedIndexDefinition::new("idx_ssn", "Patient", "ssn");
    assert_eq!(def.schema_key(), b"encrypted_index:idx_ssn");
}

#[test]
fn definition_txn_round_trip() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalEncryptedIndexStore::new(engine);

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

    let def = EncryptedIndexDefinition::new("idx_email", "User", "email");

    // CREATE: persist through a statement transaction.
    let read_ts = oracle.next();
    let mut t = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    store.put_definition_txn(&mut t, &def).expect("put txn");
    commit(&mut t);
    let loaded = store
        .load_definition("idx_email")
        .expect("load")
        .expect("present after commit");
    assert_eq!(loaded, def);

    // DROP: delete through a statement transaction.
    let read_ts = oracle.next();
    let mut t = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    store
        .delete_definition_txn(&mut t, "idx_email")
        .expect("delete txn");
    commit(&mut t);
    assert!(store.load_definition("idx_email").expect("load").is_none());
}
