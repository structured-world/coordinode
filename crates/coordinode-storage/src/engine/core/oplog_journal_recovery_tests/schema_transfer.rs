//! A member's Schema tree replaced by a copy from a peer, cut short by a
//! power loss after its clear and before the copy's closure ledger is
//! installed: the next open keeps the transfer owed, with the member's own
//! records, and answers no closure ledger, neither the one the member held
//! before (another apply cut) nor the copy's (not installed). A transfer run
//! again from the start installs the copy's.

use super::*;

/// Apply Raft entry `index` committing at `ts` as the state machine does:
/// its commit, the closure records it plans, then the plan installed.
fn closure_entry(engine: &StorageEngine, index: u64, ts: u64, below: u64) {
    engine.note_raft_applied(index, ts);
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

/// Entries `1..=n` committing at `base + i * 10`, then a bound closing them.
fn ledger(engine: &StorageEngine, base: u64, n: u64) {
    for index in 1..=n {
        closure_entry(engine, index, base + index * 10, 0);
    }
    closure_entry(engine, n + 1, base, base + n * 10 + 1);
}

#[test]
fn a_schema_transfer_cut_short_after_its_clear_stays_owed() {
    let source_dir = TempDir::new().expect("tempdir");
    let source = StorageEngine::open_with_oracle(
        &durable_cfg(&source_dir),
        Arc::new(TimestampOracle::new()),
    )
    .expect("open the source");
    ledger(&source, 1_000, 5);
    let copy = source.copy_partition(Partition::Schema).expect("copy");

    let rig = PowerRig::new();
    let config = rig.config();
    let routes = {
        let target = rig.open_group(&config).expect("open the member");
        // The member's own ledger, at an apply cut the copy does not share.
        ledger(&target, 100, 2);
        target.persist().expect("persist");
        assert_eq!(target.closure_frontier(), 100 + 2 * 10 + 1);
        let routes = target
            .get(Partition::Schema, b"meta:routing:node")
            .expect("read the routes")
            .expect("the member's routes");
        let repair = target
            .begin_partition_rebuild(Partition::Schema, &copy.rows)
            .expect("begin the transfer");
        // Lost with the power: nothing after the clear reached the disk.
        drop(repair);
        rig.cut(target);
        routes
    };

    let target = rig.open_group(&config).expect("reopen the member");
    assert_eq!(
        target.pending_rebuilds().expect("intents"),
        [Partition::Schema],
        "the transfer is no longer owed"
    );
    assert_eq!(
        target
            .get(Partition::Schema, b"meta:routing:node")
            .expect("read the routes")
            .as_deref(),
        Some(routes.as_ref()),
        "the member's own records"
    );
    target.reload_closure_frontier().expect("read the ledger");
    assert_eq!(
        target.closure_frontier(),
        0,
        "a bound answered before the transfer"
    );
    assert!(
        target.commit_ceiling_through(1).is_err(),
        "a ceiling answered before the transfer installed a ledger: {:?}",
        target.commit_ceiling_through(1)
    );

    let repair = target
        .begin_partition_rebuild(Partition::Schema, &copy.rows)
        .expect("transfer again");
    target
        .finish_raft_rebuild(repair, 7, &[])
        .expect("finish the transfer");
    assert!(target.pending_rebuilds().expect("intents").is_empty());
    assert_eq!(target.closure_frontier(), source.closure_frontier());
    for index in 1..=5 {
        assert_eq!(
            target.commit_ceiling_through(index).expect("ceiling"),
            source.commit_ceiling_through(index).expect("ceiling"),
            "position {index}"
        );
    }
}
