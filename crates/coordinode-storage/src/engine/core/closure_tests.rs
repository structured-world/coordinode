//! The closure ledger is engine protocol metadata kept in the Schema tree's
//! reserved namespace: application scans, change enumeration, range
//! deletion, consumers and material accounting never see it, a copy of the
//! partition carries it explicitly, and a store holding it where an earlier
//! layout kept it is refused.

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use coordinode_core::txn::proposal::{Mutation, PartitionId};

fn config(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// Apply Raft entry `index` as the apply path does: its commits, the
/// closure records it plans with its coverage, then the plan installed.
fn closure_entry(engine: &StorageEngine, index: u64, commits: &[u64], below: u64) {
    for &ts in commits {
        engine.note_raft_applied(index, ts);
    }
    if let Some((records, at)) = engine
        .closure_record(below, index, false)
        .expect("plan the closure records")
    {
        engine
            .apply_raft_proposal(&records, at, index, 0, |_| false)
            .expect("write the closure records");
    }
    engine
        .raise_closure_frontier(below, index)
        .expect("install the plan");
}

/// Entries 1..=n, entry i committing at `i * 10`, then a bound closing them.
fn ledger(engine: &StorageEngine, n: u64) {
    for index in 1..=n {
        closure_entry(engine, index, &[index * 10], 0);
    }
    closure_entry(engine, n + 1, &[], n * 10 + 1);
}

/// No application read of the Schema partition meets a ledger record: a
/// scan of every row, a scan from the empty key, the keys changed since the
/// start, and an application range deletion of the whole partition leaves
/// the ledger answering. Kept among application rows, a scan returned the
/// head and the breakpoints and a range deletion erased them.
#[test]
fn ledger_records_are_not_application_rows() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    let rows = || -> Vec<Vec<u8>> {
        engine
            .prefix_scan(Partition::Schema, &[])
            .expect("scan")
            .map(|guard| guard.into_inner().expect("row").0.to_vec())
            .collect()
    };
    let changed = || {
        engine
            .changed_keys_since(Partition::Schema, 0)
            .expect("changed keys")
    };
    // The rows the store keeps of its own (its routing) before the ledger.
    let (rows_before, changed_before) = (rows(), changed());
    ledger(&engine, 5);

    assert_eq!(rows(), rows_before, "the scan met ledger records");
    assert_eq!(changed(), changed_before, "ledger records changed keys");

    engine
        .apply_proposal_at(
            &[Mutation::RemoveRange {
                partition: PartitionId::Schema,
                start: Vec::new(),
                end: vec![0xff; 16],
            }],
            engine.next_seqno(),
        )
        .expect("remove every application row");
    engine.persist().expect("persist");
    engine.reload_closure_frontier().expect("reload");
    for index in 1..=5 {
        assert_eq!(
            engine.commit_ceiling_through(index).expect("ceiling"),
            index * 10,
            "position {index}"
        );
    }
    assert_eq!(engine.closure_frontier(), 51);
}

/// The ledger's writes owe consumers nothing: a follower of the Schema
/// partition is handed no keys for them, and they add no logical material.
/// Written as application rows, each entry handed its keys to the follower
/// and counted their bytes as material a consumer must read.
#[test]
fn ledger_writes_owe_consumers_nothing() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    let schema = engine.subscribe_applied(Partition::Schema, 64);
    // The store's own rows (its routing) written when it opened.
    let material_before = engine.material_between(0, None).expect("material");
    ledger(&engine, 5);

    while let Some(event) = schema.try_next() {
        if let crate::engine::applied::AppliedEvent::Keys { keys, .. } = &event {
            assert!(keys.is_empty(), "keys handed to a consumer: {keys:?}");
        } else {
            panic!("the ledger replaced the partition for its consumers: {event:?}");
        }
    }
    assert_eq!(
        engine.material_between(0, None).expect("material"),
        material_before,
        "ledger records counted as material"
    );
}

/// A copy of the Schema partition carries the ledger from the same cut: the
/// store it is installed in answers every position's ceiling and the closed
/// bound as the source does, and holds exactly the source's records.
#[test]
fn a_partition_copy_carries_the_ledger() {
    let source_dir = tempfile::TempDir::new().expect("tempdir");
    let source = StorageEngine::open(&config(source_dir.path())).expect("open source");
    let positions =
        u64::try_from(crate::engine::closure_ledger::CACHED_BREAKPOINTS).expect("small") + 9;
    ledger(&source, positions);
    let copy = source.copy_partition(Partition::Schema).expect("copy");

    let target_dir = tempfile::TempDir::new().expect("tempdir");
    let target = StorageEngine::open(&config(target_dir.path())).expect("open target");
    // Answered from the installed ledger once the install returns: no
    // separate reload.
    target
        .install_partition(Partition::Schema, &copy, true)
        .expect("install");
    for index in 1..=positions {
        assert_eq!(
            target.commit_ceiling_through(index).expect("ceiling"),
            source.commit_ceiling_through(index).expect("ceiling"),
            "position {index}"
        );
    }
    assert_eq!(target.closure_frontier(), source.closure_frontier());
}

/// A Schema copy whose ledger cannot be read leaves the partition owed a
/// rebuild, through a consensus member's finish and a whole-copy install
/// alike: the intent stays and the install fails, rather than the repair
/// ending with readers answered from the ledger the store held before.
#[test]
fn a_ledger_that_cannot_be_read_leaves_the_rebuild_owed() {
    let source_dir = tempfile::TempDir::new().expect("tempdir");
    let source = StorageEngine::open(&config(source_dir.path())).expect("open source");
    ledger(&source, 5);
    let mut copy = source.copy_partition(Partition::Schema).expect("copy");
    let head = copy
        .rows
        .iter_mut()
        .find(|(key, _)| key.as_slice() == crate::engine::closure_ledger::HEAD_KEY)
        .expect("the copy carries the head");
    head.1.truncate(3);

    let target_dir = tempfile::TempDir::new().expect("tempdir");
    let target = StorageEngine::open(&config(target_dir.path())).expect("open target");
    let repair = target
        .begin_partition_rebuild(Partition::Schema, &copy.rows)
        .expect("begin");
    assert!(
        target.finish_raft_rebuild(repair, 7, &[]).is_err(),
        "the rebuild finished over a ledger it cannot read"
    );
    assert_eq!(
        target.pending_rebuilds().expect("intents"),
        [Partition::Schema],
        "the rebuild is no longer owed"
    );

    let other_dir = tempfile::TempDir::new().expect("tempdir");
    let other = StorageEngine::open(&config(other_dir.path())).expect("open other");
    assert!(
        other
            .install_partition(Partition::Schema, &copy, true)
            .is_err(),
        "the install finished over a ledger it cannot read"
    );
    assert_eq!(
        other.pending_rebuilds().expect("intents"),
        [Partition::Schema],
        "the install is no longer owed"
    );
}

/// A position older than the cached breakpoints is read from its stored
/// record, and a read of it that the disk holds up holds up no apply: the
/// next entries record their commits and close the log meanwhile, and the
/// read, once the record answers, gets the position's exact ceiling.
#[test]
fn a_stalled_stored_ceiling_read_holds_up_no_apply() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let engine = std::sync::Arc::new(StorageEngine::open(&config(dir.path())).expect("open"));
    let positions =
        u64::try_from(crate::engine::closure_ledger::CACHED_BREAKPOINTS).expect("small") + 9;
    for index in 1..=positions {
        closure_entry(&engine, index, &[index * 10], 0);
    }
    super::breakpoint_read_gate::hold();
    let reader = std::thread::spawn({
        let engine = std::sync::Arc::clone(&engine);
        move || engine.commit_ceiling_through(1)
    });
    while super::breakpoint_read_gate::waiting() == 0 {
        std::thread::yield_now();
    }
    let applied = std::thread::spawn({
        let engine = std::sync::Arc::clone(&engine);
        move || {
            closure_entry(&engine, positions + 1, &[(positions + 1) * 10], 0);
            closure_entry(&engine, positions + 2, &[], 15);
        }
    });
    let started = std::time::Instant::now();
    while !applied.is_finished() && started.elapsed() < std::time::Duration::from_secs(3) {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let done = applied.is_finished();
    super::breakpoint_read_gate::release();
    applied.join().expect("apply");
    assert!(done, "the entries waited on the stalled read");
    assert_eq!(engine.closure_frontier(), 15, "the entries closed the log");
    assert_eq!(reader.join().expect("read").expect("ceiling"), 10);
}

/// A store holding the head where an earlier layout kept it, among the
/// application's Schema rows, is refused when it opens and when its ledger
/// is read: read as having no ledger, every prefix would look closed.
#[test]
fn a_head_kept_among_application_rows_is_refused() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    {
        let engine = StorageEngine::open(&config(dir.path())).expect("open");
        engine
            .put(Partition::Schema, b"closure:below", &[0u8; 8])
            .expect("write the earlier layout's head");
        assert!(
            engine.reload_closure_frontier().is_err(),
            "the earlier layout's head was read"
        );
        engine.persist().expect("persist");
    }
    let reopened = StorageEngine::open(&config(dir.path()));
    let Err(err) = reopened else {
        panic!("a store of the earlier layout opened");
    };
    assert!(err.to_string().contains("layout"), "{err}");
}
