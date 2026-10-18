use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_modality::{DuplicateRepairRecord, LocalNodeStore, NodeStore as _};

use super::test_env::{Fault, TestEnv, commit};
use super::*;
use crate::index::IndexDescriptor;

fn env() -> Arc<TestEnv> {
    let env = TestEnv::open();
    let mut fields = FieldInterner::new();
    fields.intern("email");
    env.set_fields(&fields);
    env
}

/// What these tests do with a member: write users, admit index builds as
/// CREATE INDEX does, and read back records, definitions and entries.
trait Users {
    fn put_user(&self, id: u64, email: &str);
    fn admit(&self, descriptor: IndexDescriptor, on_failure: BuildFailure) -> IndexDefinition;
    fn record(&self, def: &IndexDefinition) -> Option<IndexBuildRecord>;
    fn stored(&self, def: &IndexDefinition) -> Option<IndexDefinition>;
    fn holders(&self, def: &IndexDefinition, email: &str) -> Vec<u64>;
}

impl Users for TestEnv {
    fn put_user(&self, id: u64, email: &str) {
        let mut record = NodeRecord::new("User");
        let field = self.fields_now().lookup("email").expect("email field");
        record.set(field, Value::String(email.into()));
        let mut txn = self.begin();
        LocalNodeStore
            .put(&mut txn, 1, NodeId::from_raw(id), &record)
            .expect("put");
        commit(&mut txn).expect("commit node");
    }

    /// Publish `descriptor` as building with its build admitted, in one
    /// catalog commit, as CREATE INDEX does.
    fn admit(&self, descriptor: IndexDescriptor, on_failure: BuildFailure) -> IndexDefinition {
        let store = LocalIndexStore::new(&self.engine);
        let mut descriptor = descriptor;
        descriptor.state = IndexState::Building {
            written: 0,
            estimated_total: 0,
        };
        let mut txn = self.begin();
        let def = store
            .publish_definition_txn(&mut txn, descriptor)
            .expect("publish");
        store
            .put_build_txn(
                &mut txn,
                &IndexBuildRecord::accepted(def.id, def.generation, on_failure),
                None,
            )
            .expect("admit build");
        commit(&mut txn).expect("commit publication");
        self.registry
            .register_published(&self.engine, def.clone())
            .expect("register");
        def
    }

    fn record(&self, def: &IndexDefinition) -> Option<IndexBuildRecord> {
        LocalIndexStore::new(&self.engine)
            .load_build(def.generation)
            .expect("load build")
            .map(|(record, _)| record)
    }

    fn stored(&self, def: &IndexDefinition) -> Option<IndexDefinition> {
        LocalIndexStore::new(&self.engine)
            .load_definition(def.id)
            .expect("load definition")
    }

    fn holders(&self, def: &IndexDefinition, email: &str) -> Vec<u64> {
        let mut txn = self.begin();
        let budget = coordinode_core::budget::QueryBudget::new(
            coordinode_core::budget::DEFAULT_QUERY_MEMORY_LIMIT,
        );
        let mut ids: Vec<u64> = LocalIndexStore::new(&self.engine)
            .scan_exact(&mut txn, def, &[Value::String(email.into())], &budget)
            .expect("scan")
            .expect("indexable")
            .into_iter()
            .map(NodeId::as_raw)
            .collect();
        ids.sort_unstable();
        ids
    }
}

/// A transaction older than every build submitted after it: a backfill
/// waits for it to end before it reads, so the build stays unfinished while
/// it is held.
fn older_transaction(env: &TestEnv) -> Transaction<'_> {
    env.begin()
}

fn service(env: &Arc<TestEnv>) -> IndexBuildService {
    env.service(2)
}

/// Wait until an executor other than `besides` holds the build of `def`,
/// and return its token.
fn await_running(env: &TestEnv, def: &IndexDefinition, besides: Option<u64>) -> u64 {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        if let BuildState::Running { executor } = env.record(def).expect("record").state {
            if Some(executor) != besides {
                return executor;
            }
        }
        assert!(std::time::Instant::now() < deadline, "never taken");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The build of an admitted index runs on the service, not on the caller:
/// the caller only waits, and the outcome publishes the index ready with
/// its record and its entries.
#[test]
fn an_admitted_build_runs_on_the_engine_and_publishes_the_index() {
    let env = env();
    env.put_user(1, "a@x");
    env.put_user(2, "b@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(
            outcome,
            Some(IndexBuildOutcome::Published { indexed: Some(2) })
        ),
        "{outcome:?}"
    );
    assert_eq!(env.stored(&def).expect("defined").state, IndexState::Ready);
    assert_eq!(
        env.record(&def).expect("record").state,
        BuildState::Published
    );
    assert_eq!(env.holders(&def, "a@x"), [1]);
    assert_eq!(
        env.registry.get_by_id(def.id).expect("registered").state,
        IndexState::Ready
    );
}

/// A waiter that gives up does not cancel the build: it goes on, and a
/// later wait finds it published.
#[test]
fn a_waiter_that_gives_up_cancels_nothing() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    let older = older_transaction(&env);

    builds.submit(def.generation).expect("submit");
    let taken_by = std::time::Instant::now() + Duration::from_secs(20);
    while !matches!(
        env.record(&def).expect("record").state,
        BuildState::Running { .. }
    ) {
        assert!(std::time::Instant::now() < taken_by, "never taken");
        std::thread::sleep(Duration::from_millis(5));
    }
    let early = builds
        .wait(def.generation, Some(Duration::from_millis(100)))
        .expect("wait");
    assert!(early.is_none(), "the build waits for the older transaction");

    drop(older);
    let outcome = builds.wait(def.generation, None).expect("wait");
    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Published { .. })),
        "{outcome:?}"
    );
    assert_eq!(env.holders(&def, "a@x"), [1]);
}

/// Cancelling a build of a new index withdraws the index in the same
/// commit as the cancellation, and the executor, fenced, writes nothing
/// more and reports the cancellation.
#[test]
fn a_cancelled_build_withdraws_the_index_it_was_creating() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    let older = older_transaction(&env);
    builds.submit(def.generation).expect("submit");

    assert!(builds.cancel(def.generation).expect("cancel"));
    drop(older);
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Cancelled)),
        "{outcome:?}"
    );
    assert_eq!(
        env.record(&def).expect("record").state,
        BuildState::Cancelled
    );
    assert!(env.stored(&def).is_none(), "the index is withdrawn");
    assert!(env.registry.get_by_id(def.id).is_none());
    assert!(env.holders(&def, "a@x").is_empty(), "no entry was written");
    assert!(
        !builds.cancel(def.generation).expect("cancel again"),
        "a build with an outcome is not cancelled again"
    );
}

/// Stored data that breaks a new unique index fails its build: the index
/// is withdrawn and the waiter is told which node holds the value.
#[test]
fn a_duplicate_withdraws_a_new_unique_index() {
    let env = env();
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email").unique(),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(
            &outcome,
            Some(IndexBuildOutcome::Failed(BuildError::Duplicate(v)))
                if v.holder == NodeId::from_raw(1)
        ),
        "{outcome:?}"
    );
    // The duplicate is named by the stored outcome too, for whoever asks
    // after the waiter is gone.
    let state = env.record(&def).expect("record").state;
    assert!(
        matches!(&state, BuildState::Failed { reason } if reason.contains("same@x")),
        "{state:?}"
    );
    assert!(env.stored(&def).is_none());
    assert!(env.holders(&def, "same@x").is_empty());
}

/// A rebuild the stored data refuses keeps its index, failed: the index
/// stays defined so its constraint still holds for new writes.
#[test]
fn a_refused_rebuild_keeps_its_index_failed() {
    let env = env();
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email").unique(),
        BuildFailure::Keep,
    );
    let builds = service(&env);

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Failed(_))),
        "{outcome:?}"
    );
    assert!(matches!(
        env.stored(&def).expect("kept").state,
        IndexState::Failed { .. }
    ));
    assert!(matches!(
        env.registry
            .get_by_id(def.id)
            .expect("still registered")
            .state,
        IndexState::Failed { .. }
    ));
}

/// A build admitted and never finished, as after a crash, is taken up by
/// the service of the next process and runs to its outcome.
#[test]
fn an_unfinished_build_is_resumed() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );

    let builds = service(&env);
    let resumed = builds.resume().expect("resume");
    assert_eq!(resumed, [def.generation]);
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Published { .. })),
        "{outcome:?}"
    );
    assert!(builds.resume().expect("resume again").is_empty());
}

/// A member that cannot lead when it takes a build changes nothing: the
/// build is neither taken nor failed, its executor ends without an outcome,
/// and the record still waits for one. A later take, once the member leads,
/// finishes it.
#[test]
fn a_build_a_member_cannot_take_is_left_for_the_next_take() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    env.inject(Some(Fault::NotLeader));

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait_here(def.generation);
    assert!(outcome.is_none(), "not taken, so no outcome: {outcome:?}");
    assert_eq!(
        env.record(&def).expect("record").state,
        BuildState::Accepted
    );

    env.inject(None);
    assert_eq!(builds.resume().expect("resume"), [def.generation]);
    let outcome = builds.wait_here(def.generation);
    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Published { .. })),
        "{outcome:?}"
    );
}

/// Mark this member's copy of `generation` unfit, as a guard an earlier run
/// left open does at the next open, and admit what the mark asks for.
fn mark_and_admit(env: &TestEnv, builds: &IndexBuildService, generation: GenerationId) {
    let store = LocalIndexStore::new(&env.engine);
    let copy = store.local_copy(generation).expect("a copy here");
    store.mark_unfit_here(copy).expect("mark");
    env.registry.load_all(&env.engine).expect("load");
    assert!(env.registry.is_suspect(generation));
    builds.resume_checks().expect("admit");
}

/// Wait for the local check of `generation` to end.
fn await_local_check(builds: &IndexBuildService, generation: GenerationId) {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while builds.local_check_running(generation) {
        assert!(
            std::time::Instant::now() < deadline,
            "the local check never ended"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A member that takes no writes cannot run the shared check, which proves
/// only the copy of the member running it; it checks its own copy against
/// its own records, writing nothing shared. A copy that agrees serves
/// again; one missing an entry stays unfit here and answers from the
/// records.
#[test]
fn a_member_that_takes_no_writes_checks_its_own_copy() {
    let env = env();
    env.put_user(1, "a@x");
    env.put_user(2, "b@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    builds.submit(def.generation).expect("submit");
    assert!(matches!(
        builds.wait_here(def.generation),
        Some(IndexBuildOutcome::Published { .. })
    ));
    env.inject(Some(Fault::NotLeader));

    // The check may end before the admission returns, so it is proved by
    // what it leaves: the mark lifted with no shared integrity record, which
    // only a check of this member's own copy does.
    mark_and_admit(&env, &builds, def.generation);
    await_local_check(&builds, def.generation);
    assert!(
        !env.registry.is_suspect(def.generation),
        "agrees: serves again"
    );
    assert!(
        LocalIndexStore::new(&env.engine)
            .list_unfit_here()
            .expect("marks")
            .is_empty()
    );
    assert!(
        LocalIndexStore::new(&env.engine)
            .load_integrity(def.generation)
            .expect("read")
            .is_none(),
        "checked here: no shared integrity record"
    );

    // b's entry lost on this member alone: its copy disagrees.
    let tuple = coordinode_core::index::encoding::encode_tuple(&[Value::String("b@x".into())])
        .expect("tuple");
    env.engine
        .delete(
            coordinode_storage::engine::partition::Partition::Idx,
            &coordinode_core::index::encoding::encode_entry_key(def.generation, &tuple, 2),
        )
        .expect("lose b's entry");
    mark_and_admit(&env, &builds, def.generation);
    await_local_check(&builds, def.generation);
    assert!(
        env.registry.is_suspect(def.generation),
        "a copy missing an entry stays unfit here"
    );
    assert!(
        LocalIndexStore::new(&env.engine)
            .load_integrity(def.generation)
            .expect("read")
            .is_none(),
        "nothing shared was written: no integrity record"
    );
}

/// Lose the entry of user `id` under `email` in `def`.
fn lose_entry(env: &TestEnv, def: &IndexDefinition, email: &str, id: u64) {
    let tuple = coordinode_core::index::encoding::encode_tuple(&[Value::String(email.into())])
        .expect("tuple");
    env.engine
        .delete(
            coordinode_storage::engine::partition::Partition::Idx,
            &coordinode_core::index::encoding::encode_entry_key(def.generation, &tuple, id),
        )
        .expect("lose the entry");
}

/// A repair commits only while the record it read is as it read it: a write
/// moving the record between the check's read and the repair's commit
/// refuses the repair, and the next pass repairs the record as it stands.
/// The stale entry is never written.
#[test]
fn a_repair_raced_by_a_write_repairs_the_record_as_it_stands() {
    let (env, builds, def) = member_with_users();
    lose_entry(&env, &def, "b@x", 2);
    env.hold_pages_after(0);
    let operation = builds.request_check(def.id).expect("check");
    env.await_held_page();
    // The repair of b's entry waits to commit; b moves meanwhile, its
    // record written without its entry, as a second loss would leave it.
    env.put_user(2, "c@x");
    env.release_pages();
    let outcome = builds
        .wait_check(operation, Some(Duration::from_secs(30)))
        .expect("wait");
    assert!(
        matches!(outcome, Some(CheckOutcome::Verified { .. })),
        "{outcome:?}"
    );
    assert!(
        env.holders(&def, "b@x").is_empty(),
        "no stale entry for b@x"
    );
    assert_eq!(env.holders(&def, "c@x"), vec![2], "the record as it stands");
}

/// A repair commits only while the index still serves from the checked
/// generation: a rebuild between the check's read and the repair's commit
/// refuses the repair, and the check ends cancelled, leaving the rebuild's
/// generation to its own build.
#[test]
fn a_repair_raced_by_a_rebuild_is_refused() {
    let (env, builds, def) = member_with_users();
    lose_entry(&env, &def, "b@x", 2);
    env.hold_pages_after(0);
    let operation = builds.request_check(def.id).expect("check");
    env.await_held_page();
    let stored = env.stored(&def).expect("definition");
    let rebuilt = std::thread::spawn({
        let builds = builds.clone();
        move || builds.rebuild(stored)
    });
    // The rebuild's catalog commit lands while the repair is held; its
    // backfill pages wait behind the hold too.
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while env
        .stored(&def)
        .is_none_or(|d| d.generation == def.generation)
    {
        assert!(std::time::Instant::now() < deadline, "never rebuilt");
        std::thread::sleep(Duration::from_millis(5));
    }
    env.release_pages();
    let rebuilt = rebuilt.join().expect("rebuild thread").expect("rebuild");
    let outcome = builds
        .wait_check(operation, Some(Duration::from_secs(30)))
        .expect("wait");
    assert!(
        matches!(
            outcome,
            Some(CheckOutcome::Cancelled | CheckOutcome::Failed(_))
        ),
        "the check of a generation no longer served concludes nothing: {outcome:?}"
    );
    assert!(matches!(
        builds.wait_here(rebuilt.generation),
        Some(IndexBuildOutcome::Published { .. })
    ));
    assert_eq!(env.holders(&rebuilt, "b@x"), vec![2], "the rebuild has b");
}

/// A repair is bound to the check its executor holds: one held at its
/// commit while the check is cancelled (or taken over by another executor,
/// which moves the same record) commits nothing, and the check ends
/// cancelled with the damage left for the next one.
#[test]
fn a_repair_of_a_cancelled_check_commits_nothing() {
    let (env, builds, def) = member_with_users();
    lose_entry(&env, &def, "b@x", 2);
    env.hold_pages_after(0);
    let operation = builds.request_check(def.id).expect("check");
    env.await_held_page();
    assert!(builds.cancel_check(operation).expect("cancel"), "cancelled");
    env.release_pages();
    let outcome = builds
        .wait_check(operation, Some(Duration::from_secs(30)))
        .expect("wait");
    assert_eq!(outcome, Some(CheckOutcome::Cancelled));
    assert!(
        env.holders(&def, "b@x").is_empty(),
        "the cancelled check's repair did not commit"
    );
}

/// A record deleted while the repair of its missing entry waits to commit
/// takes no entry: the repair is bound to the record as it read it, and the
/// next pass finds nothing left to repair.
#[test]
fn a_repair_raced_by_a_delete_writes_no_entry() {
    let (env, builds, def) = member_with_users();
    lose_entry(&env, &def, "b@x", 2);
    env.hold_pages_after(0);
    let operation = builds.request_check(def.id).expect("check");
    env.await_held_page();
    let mut txn = env.begin();
    LocalNodeStore
        .delete(&mut txn, 1, NodeId::from_raw(2))
        .expect("stage the delete");
    commit(&mut txn).expect("delete b");
    env.release_pages();
    let outcome = builds
        .wait_check(operation, Some(Duration::from_secs(30)))
        .expect("wait");
    assert!(
        matches!(outcome, Some(CheckOutcome::Verified { .. })),
        "{outcome:?}"
    );
    assert!(
        env.holders(&def, "b@x").is_empty(),
        "no entry for a deleted record"
    );
    assert_eq!(env.holders(&def, "a@x"), vec![1], "the rest untouched");
}

/// A check taken over by another executor (a second service resuming it, as
/// a new leader does) is that executor's alone: the first executor's repair,
/// held at its commit across the takeover, commits nothing, and the check
/// ends once, repaired by the executor that holds it.
#[test]
fn a_check_taken_over_is_repaired_by_its_new_executor_only() {
    let (env, builds, def) = member_with_users();
    lose_entry(&env, &def, "b@x", 2);
    env.hold_pages_after(0);
    let operation = builds.request_check(def.id).expect("check");
    env.await_held_page();
    let store = LocalIndexStore::new(&env.engine);
    let first = store
        .load_integrity(def.generation)
        .expect("read")
        .and_then(|(record, _)| record.check)
        .map(|check| check.state)
        .expect("the first executor's check");
    let successor = service(&env);
    let resumed = successor.resume_checks().expect("take it over");
    assert_eq!(resumed, vec![def.generation]);
    // The successor's take lands before its own first page, which the hold
    // keeps from committing too.
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let state = store
            .load_integrity(def.generation)
            .expect("read")
            .and_then(|(record, _)| record.check)
            .map(|check| check.state);
        if state.as_ref().is_some_and(|s| *s != first) {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "never taken over");
        std::thread::sleep(Duration::from_millis(5));
    }
    env.release_pages();
    let outcome = successor
        .wait_check(operation, Some(Duration::from_secs(30)))
        .expect("wait");
    assert!(
        matches!(outcome, Some(CheckOutcome::Verified { .. })),
        "{outcome:?}"
    );
    let record = store
        .load_integrity(def.generation)
        .expect("read")
        .expect("record")
        .0;
    assert_eq!(
        record.check.as_ref().map(|c| c.repaired),
        Some(1),
        "repaired once, by the executor that held the check"
    );
    assert_eq!(env.holders(&def, "b@x"), vec![2]);
    assert_eq!(
        builds
            .wait_check(operation, Some(Duration::from_secs(30)))
            .expect("wait"),
        outcome,
        "both services report the one outcome"
    );
}

/// A write to the integrity record that leaves the check with its executor
/// (a report recorded against the generation, say) refuses the repair held
/// over it, but not the check: the next pass, still this executor's,
/// repairs the damage and the check verifies.
#[test]
fn a_record_write_that_keeps_the_check_does_not_cost_the_check() {
    let (env, builds, def) = member_with_users();
    lose_entry(&env, &def, "b@x", 2);
    env.hold_pages_after(0);
    let operation = builds.request_check(def.id).expect("check");
    env.await_held_page();
    let store = LocalIndexStore::new(&env.engine);
    let (record, version) = store
        .load_integrity(def.generation)
        .expect("read")
        .expect("the check's record");
    let mut txn = env.begin();
    store
        .put_integrity_txn(&mut txn, &record, Some(version))
        .expect("stage");
    commit(&mut txn).expect("rewrite the record, the check still held");
    env.release_pages();
    let outcome = builds
        .wait_check(operation, Some(Duration::from_secs(30)))
        .expect("wait");
    assert!(
        matches!(outcome, Some(CheckOutcome::Verified { .. })),
        "{outcome:?}"
    );
    assert_eq!(env.holders(&def, "b@x"), vec![2], "repaired by this check");
}

/// A new check admitted right after a check concluded (a report of the
/// damage it just repaired, recorded late, admits one) leaves the outcome the
/// first check recorded: its waiter is told that outcome, not a cancellation
/// read off the new check's state.
#[test]
fn a_check_followed_at_once_by_another_reports_its_own_outcome() {
    let (env, builds, def) = member_with_users();
    lose_entry(&env, &def, "b@x", 2);
    let generation = def.generation;
    // Weak: the hook lives in the environment it writes through.
    let member = Arc::downgrade(&env);
    env.after_catalog_commit(move |engine| {
        let Some(member) = member.upgrade() else {
            return true;
        };
        let store = LocalIndexStore::new(engine);
        let Some((mut record, version)) = store.load_integrity(generation).expect("read") else {
            return false;
        };
        if record
            .check
            .as_ref()
            .is_none_or(|c| c.state != coordinode_modality::CheckState::Done)
        {
            return false;
        }
        record.check = Some(coordinode_modality::IndexCheck::accepted(
            record.evidence_revision,
        ));
        let mut txn = member.begin();
        store
            .put_integrity_txn(&mut txn, &record, Some(version))
            .expect("stage");
        commit(&mut txn).expect("admit the next check");
        true
    });
    let operation = builds.request_check(def.id).expect("check");
    let outcome = builds
        .wait_check(operation, Some(Duration::from_secs(30)))
        .expect("wait");
    assert!(
        matches!(outcome, Some(CheckOutcome::Verified { repaired: 1, .. })),
        "{outcome:?}"
    );
    assert_eq!(env.holders(&def, "b@x"), vec![2], "repaired");
}

/// Settle the catalog's versions as a bottommost compaction does to records
/// it keeps no older version of: renumbered to zero, unwritten. Returns the
/// version `def`'s record is left at.
fn settle_catalog(env: &TestEnv, def: &IndexDefinition) -> Option<u64> {
    use coordinode_storage::engine::partition::Partition::Schema;
    env.engine.persist().expect("persist");
    // No time-travel window: only live snapshots hold versions back.
    env.engine.set_retention_window(Duration::ZERO);
    env.engine
        .major_compact(Schema)
        .expect("compact the catalog");
    LocalIndexStore::new(&env.engine)
        .definition_version(def.id)
        .expect("version")
}

/// A check binds each repair to its index's definition as stored when the
/// repair commits, not to a version read when the check was taken: storage
/// renumbering the unchanged definition meanwhile costs neither the repair
/// nor the check.
#[test]
fn a_check_repairs_across_a_settled_definition_version() {
    let (env, builds, def) = member_with_users();
    lose_entry(&env, &def, "b@x", 2);
    env.hold_pages_after(0);
    let operation = builds.request_check(def.id).expect("check");
    env.await_held_page();
    assert_eq!(
        settle_catalog(&env, &def),
        Some(0),
        "precondition: the definition's version settled without a write"
    );
    env.release_pages();
    let outcome = builds
        .wait_check(operation, Some(Duration::from_secs(30)))
        .expect("wait");
    assert!(
        matches!(outcome, Some(CheckOutcome::Verified { repaired: 1, .. })),
        "{outcome:?}"
    );
    assert_eq!(env.holders(&def, "b@x"), vec![2], "repaired");
}

/// A build binds each page and its publication to its definition as stored
/// at that commit: storage renumbering the unchanged definition during the
/// build neither supersedes the build nor refuses its publication.
#[test]
fn a_build_publishes_across_a_settled_definition_version() {
    let env = env();
    env.put_user(1, "a@x");
    env.put_user(2, "b@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    env.hold_pages_after(0);
    let builds = service(&env);
    builds.submit(def.generation).expect("submit");
    env.await_held_page();
    assert_eq!(
        settle_catalog(&env, &def),
        Some(0),
        "precondition: the definition's version settled without a write"
    );
    env.release_pages();
    assert!(
        matches!(
            builds.wait_here(def.generation),
            Some(IndexBuildOutcome::Published { .. })
        ),
        "published"
    );
    assert_eq!(env.holders(&def, "b@x"), vec![2]);
}

/// Fills a replacement from a peer's copy of the generation, as segment
/// transfer does between members: the peer's history imported at its own
/// seqnos.
struct PeerCopy {
    local: Arc<TestEnv>,
    peer: Arc<TestEnv>,
    prepared: core::sync::atomic::AtomicUsize,
    /// The peer's proof of its copy, as its history export carries it.
    proof: Option<Vec<u8>>,
}

impl IndexCopySource for PeerCopy {
    fn prepare(&self, generation: GenerationId) -> Result<Option<Vec<u8>>, String> {
        self.prepared
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let local = &self.local.engine;
        local
            .stage_generation(generation)
            .map_err(|e| e.to_string())?;
        let history = self
            .peer
            .engine
            .export_generation_history(generation)
            .map_err(|e| e.to_string())?;
        local
            .import_generation_history(generation, &history.entries)
            .map_err(|e| e.to_string())?;
        local
            .finish_generation_import(generation, history.covers_through, history.history_from)
            .map_err(|e| e.to_string())?;
        Ok(self.proof.clone())
    }
}

/// A member with users and a ready `email` index over them.
fn member_with_users() -> (Arc<TestEnv>, IndexBuildService, IndexDefinition) {
    let env = env();
    env.put_user(1, "a@x");
    env.put_user(2, "b@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    builds.submit(def.generation).expect("submit");
    assert!(matches!(
        builds.wait_here(def.generation),
        Some(IndexBuildOutcome::Published { .. })
    ));
    (env, builds, def)
}

/// A copy damaged on this member alone is replaced from a peer's copy of
/// the same generation: the replacement is guarded before it is published,
/// checked against this member's records, and only then serves; the
/// generation, its identity and every other copy stay as they were.
#[test]
fn a_copy_damaged_here_is_replaced_from_a_peer() {
    let (env, builds, def) = member_with_users();
    let (peer, _peer_builds, peer_def) = member_with_users();
    assert_eq!(peer_def.generation, def.generation, "one generation");
    let source = Arc::new(PeerCopy {
        local: Arc::clone(&env),
        peer: Arc::clone(&peer),
        prepared: core::sync::atomic::AtomicUsize::new(0),
        proof: None,
    });
    builds.set_copy_source(Arc::clone(&source) as Arc<dyn IndexCopySource>);
    env.inject(Some(Fault::NotLeader));
    let store = LocalIndexStore::new(&env.engine);
    let damaged = store.local_copy(def.generation).expect("a copy here");

    let tuple = coordinode_core::index::encoding::encode_tuple(&[Value::String("b@x".into())])
        .expect("tuple");
    env.engine
        .delete(
            coordinode_storage::engine::partition::Partition::Idx,
            &coordinode_core::index::encoding::encode_entry_key(def.generation, &tuple, 2),
        )
        .expect("lose b's entry here");
    mark_and_admit(&env, &builds, def.generation);
    await_local_check(&builds, def.generation);

    assert_eq!(
        source.prepared.load(core::sync::atomic::Ordering::Relaxed),
        1,
        "filled once from the peer"
    );
    let replaced = store.local_copy(def.generation).expect("a copy here");
    assert_eq!(replaced.generation, damaged.generation);
    assert_ne!(replaced.installation, damaged.installation, "a new copy");
    assert!(
        store.list_armed_here().expect("guards").contains(&replaced),
        "guarded before it serves"
    );
    assert!(
        !env.registry.is_suspect(def.generation),
        "checked: it serves"
    );
    assert!(store.list_unfit_here().expect("marks").is_empty());
    assert_eq!(env.holders(&def, "b@x"), vec![2], "b's entry is back");
    assert_eq!(env.holders(&def, "a@x"), vec![1]);
}

/// A replacement filled from a peer whose history carried the peer's proof
/// of its copy serves without a pass of its own over this member's records:
/// the proof qualifies it. The peer here holds a copy missing an entry that
/// nothing found, so the replacement is missing it too, which only a pass
/// would have shown; without the proof the pass shows it and the copy stays
/// unfit.
#[test]
fn a_replacement_proved_by_its_peer_needs_no_pass() {
    use coordinode_storage::engine::index_copies::IndexCopyQualifier as _;
    for proved in [true, false] {
        let (env, builds, def) = member_with_users();
        let (peer, _peer_builds, _) = member_with_users();
        lose_entry(&peer, &def, "b@x", 2);
        let peer_registry = Arc::new(crate::index::IndexRegistry::new());
        peer_registry.load_all(&peer.engine).expect("load");
        let proof = crate::index::transfer::CopyQualifier::new(peer_registry)
            .copy_proof(&peer.engine, def.generation.as_raw())
            .expect("proof")
            .map(|(bytes, _)| bytes);
        assert!(proof.is_some(), "the peer's build proved its copy");
        builds.set_copy_source(Arc::new(PeerCopy {
            local: Arc::clone(&env),
            peer: Arc::clone(&peer),
            prepared: core::sync::atomic::AtomicUsize::new(0),
            proof: proved.then_some(proof).flatten(),
        }) as Arc<dyn IndexCopySource>);
        env.inject(Some(Fault::NotLeader));
        lose_entry(&env, &def, "a@x", 1);
        mark_and_admit(&env, &builds, def.generation);
        await_local_check(&builds, def.generation);
        if proved {
            assert!(
                !env.registry.is_suspect(def.generation),
                "the peer's proof qualifies the replacement"
            );
            assert!(
                env.holders(&def, "b@x").is_empty(),
                "no pass read it: the peer's undetected loss came along"
            );
        } else {
            assert!(
                env.registry.is_suspect(def.generation),
                "without a proof the pass finds the loss and the copy stays unfit"
            );
        }
    }
}

/// A shutdown stops a build waiting for older transactions without waiting
/// them out and returns once its executor has ended: the build keeps its
/// running record, nothing new starts on the stopped service, and a service
/// started afterwards, as the next process opening the storage, finishes it.
#[test]
fn a_shutdown_stops_a_held_build_and_the_next_service_finishes_it() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    let older = older_transaction(&env);
    builds.submit(def.generation).expect("submit");
    await_running(&env, &def, None);

    let started = std::time::Instant::now();
    builds.shutdown();
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "waited out the older transaction: {:?}",
        started.elapsed()
    );
    assert!(matches!(
        env.record(&def).expect("record").state,
        BuildState::Running { .. }
    ));
    assert!(builds.submit(def.generation).is_err(), "stopped for good");
    drop(older);

    let next = service(&env);
    assert_eq!(next.resume().expect("resume"), [def.generation]);
    let outcome = next.wait(def.generation, None).expect("wait");
    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Published { .. })),
        "{outcome:?}"
    );
    assert_eq!(env.holders(&def, "a@x"), [1]);
}

/// Two executors of one build (a process that took it and the one that
/// took it over, as after a leader change) cannot both finish it: the
/// index is published once and holds each entry once.
#[test]
fn a_build_taken_over_is_finished_once() {
    let env = env();
    env.put_user(1, "a@x");
    env.put_user(2, "b@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let first = service(&env);
    let second = service(&env);
    let older = older_transaction(&env);

    first.submit(def.generation).expect("first takes it");
    let taken = await_running(&env, &def, None);
    second.submit(def.generation).expect("second takes it over");
    await_running(&env, &def, Some(taken));
    drop(older);
    let a = first.wait(def.generation, None).expect("first wait");
    let b = second.wait(def.generation, None).expect("second wait");

    let published = [&a, &b]
        .iter()
        .filter(|o| matches!(o, Some(IndexBuildOutcome::Published { indexed: Some(_) })))
        .count();
    assert_eq!(published, 1, "exactly one executor published: {a:?} {b:?}");
    assert_eq!(
        env.record(&def).expect("record").state,
        BuildState::Published
    );
    assert_eq!(env.stored(&def).expect("defined").state, IndexState::Ready);
    assert_eq!(env.holders(&def, "a@x"), [1]);
    assert_eq!(env.holders(&def, "b@x"), [2]);
}

/// A build a member runs for itself reports its outcome to waiters like any
/// other: the records it indexed, or why it failed.
#[test]
fn a_local_build_reports_its_outcome() {
    let env = env();
    let builds = service(&env);
    let (ok, failed) = (GenerationId::from_raw(10), GenerationId::from_raw(11));

    builds
        .run_local(ok, Box::new(|_| Ok(5)))
        .expect("run local");
    builds
        .run_local(
            failed,
            Box::new(|_| Err(BuildError::Other("no texts".into()))),
        )
        .expect("run local");

    assert!(matches!(
        builds.wait(ok, None).expect("wait"),
        Some(IndexBuildOutcome::Published { indexed: Some(5) })
    ));
    assert!(matches!(
        builds.wait(failed, None).expect("wait"),
        Some(IndexBuildOutcome::Failed(BuildError::Other(reason))) if reason == "no texts"
    ));
}

/// Builds a member runs for itself take seats like the others: with one
/// seat, a second build waits until the first has finished.
#[test]
fn local_builds_take_seats() {
    let env = env();
    let builds = env.service(1);
    let (release, held) = std::sync::mpsc::channel::<()>();
    let (first, second) = (GenerationId::from_raw(10), GenerationId::from_raw(11));

    builds
        .run_local(
            first,
            Box::new(move |_| {
                held.recv().expect("released");
                Ok(1)
            }),
        )
        .expect("run first");
    // The executors are threads: the second could take the seat first.
    await_phase(&builds, first, BuildPhase::Indexing { indexed: None });
    builds
        .run_local(second, Box::new(|_| Ok(2)))
        .expect("run second");

    assert!(
        builds
            .wait(second, Some(Duration::from_millis(200)))
            .expect("wait")
            .is_none(),
        "the second build waits for the seat the first holds"
    );
    release.send(()).expect("release");
    assert!(matches!(
        builds.wait(second, None).expect("wait"),
        Some(IndexBuildOutcome::Published { indexed: Some(2) })
    ));
    assert!(matches!(
        builds.wait(first, None).expect("wait"),
        Some(IndexBuildOutcome::Published { indexed: Some(1) })
    ));
}

/// Wait until inspection shows the build of `generation` in `phase`.
fn await_phase(builds: &IndexBuildService, generation: GenerationId, phase: BuildPhase) {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let status = builds.builds().expect("inspect");
        if status
            .iter()
            .any(|s| s.generation == generation && s.phase == Some(phase))
        {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "never {phase:?}: {status:?}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Inspection shows a build that waits for a seat as waiting, beside the
/// build holding the seat, and a finished build with no phase.
#[test]
fn a_build_without_a_seat_shows_it_waits_for_one() {
    let env = env();
    let builds = env.service(1);
    let (release, held) = std::sync::mpsc::channel::<()>();
    let (first, second) = (GenerationId::from_raw(10), GenerationId::from_raw(11));
    builds
        .run_local(
            first,
            Box::new(move |_| {
                held.recv().expect("released");
                Ok(1)
            }),
        )
        .expect("run first");
    await_phase(&builds, first, BuildPhase::Indexing { indexed: None });
    builds
        .run_local(second, Box::new(|_| Ok(2)))
        .expect("run second");

    await_phase(&builds, second, BuildPhase::AwaitingSeat);
    release.send(()).expect("release");
    builds.wait(second, None).expect("wait");
    builds.wait(first, None).expect("wait");

    assert!(
        builds.builds().expect("inspect").is_empty(),
        "member builds keep no record, and finished ones no phase"
    );
}

/// A key-shaped build waiting for a transaction older than its index shows
/// that wait; once published, its record remains and its phase is gone.
#[test]
fn a_build_behind_an_older_transaction_shows_the_wait() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    let older = older_transaction(&env);

    builds.submit(def.generation).expect("submit");
    await_phase(
        &builds,
        def.generation,
        BuildPhase::AwaitingOlderTransactions,
    );
    let status = builds.builds().expect("inspect");
    let running = status
        .iter()
        .find(|s| s.generation == def.generation)
        .expect("listed");
    assert!(
        matches!(
            running.record.as_ref().map(|r| &r.state),
            Some(BuildState::Running { .. })
        ),
        "{running:?}"
    );

    drop(older);
    builds.wait(def.generation, None).expect("wait");
    let status = builds.builds().expect("inspect");
    assert_eq!(status.len(), 1, "{status:?}");
    assert_eq!(status[0].phase, None);
    assert_eq!(
        status[0].record.as_ref().map(|r| &r.state),
        Some(&BuildState::Published)
    );
}

/// A backfill that outwaits the configured wait for older transactions
/// fails its build, and the new index is withdrawn.
#[test]
fn a_build_outwaited_by_an_older_transaction_fails() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = env.service_with(IndexBuildConfig {
        max_running: 1,
        older_transactions_wait: Duration::from_millis(50),
        ..IndexBuildConfig::default()
    });
    let older = older_transaction(&env);

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");
    drop(older);

    assert!(
        matches!(
            &outcome,
            Some(IndexBuildOutcome::Failed(BuildError::Other(reason)))
                if reason.contains("still open")
        ),
        "{outcome:?}"
    );
    assert!(env.stored(&def).is_none(), "the new index is withdrawn");
}

/// Raising the number of builds that run at once seats a waiting build
/// without waiting for a running one to finish.
#[test]
fn raising_the_seat_count_seats_a_waiting_build() {
    let env = env();
    let builds = env.service(1);
    let (release, held) = std::sync::mpsc::channel::<()>();
    let (first, second) = (GenerationId::from_raw(10), GenerationId::from_raw(11));
    builds
        .run_local(
            first,
            Box::new(move |_| {
                held.recv().expect("released");
                Ok(1)
            }),
        )
        .expect("run first");
    await_phase(&builds, first, BuildPhase::Indexing { indexed: None });
    builds
        .run_local(second, Box::new(|_| Ok(2)))
        .expect("run second");
    await_phase(&builds, second, BuildPhase::AwaitingSeat);

    builds.set_config(IndexBuildConfig {
        max_running: 2,
        ..builds.config()
    });

    assert!(matches!(
        builds.wait(second, None).expect("wait"),
        Some(IndexBuildOutcome::Published { indexed: Some(2) })
    ));
    release.send(()).expect("release");
    builds.wait(first, None).expect("wait");
}

/// A seat count of zero is taken as one: builds still run.
#[test]
fn a_zero_seat_count_still_runs_builds() {
    let env = env();
    let builds = env.service(0);
    assert_eq!(builds.config().max_running, 1);
    builds
        .run_local(GenerationId::from_raw(10), Box::new(|_| Ok(1)))
        .expect("run");
    assert!(matches!(
        builds.wait(GenerationId::from_raw(10), None).expect("wait"),
        Some(IndexBuildOutcome::Published { indexed: Some(1) })
    ));
}

/// A member build that panics fails with the panic's message instead of
/// leaving its waiters hanging, and frees its seat.
#[test]
fn a_panicking_local_build_fails_and_frees_its_seat() {
    let env = env();
    let builds = env.service(1);
    let (bad, good) = (GenerationId::from_raw(10), GenerationId::from_raw(11));
    builds
        .run_local(bad, Box::new(|_| panic!("text tokenizer broke")))
        .expect("run");

    assert!(matches!(
        builds.wait(bad, None).expect("wait"),
        Some(IndexBuildOutcome::Failed(BuildError::Other(reason))) if reason == "text tokenizer broke"
    ));
    builds.run_local(good, Box::new(|_| Ok(1))).expect("run");
    assert!(matches!(
        builds.wait(good, None).expect("wait"),
        Some(IndexBuildOutcome::Published { indexed: Some(1) })
    ));
}

/// A taken build that cannot go on settles its record failed, so the
/// catalog and the waiter agree and no later process resumes a build its
/// waiter was told had failed.
fn a_taken_build_that_breaks_settles_failed(fault: Fault, cause: &str) {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    env.inject(Some(fault));

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(
            &outcome,
            Some(IndexBuildOutcome::Failed(BuildError::Other(reason))) if reason.contains(cause)
        ),
        "{outcome:?}"
    );
    assert!(
        matches!(
            env.record(&def).expect("record").state,
            BuildState::Failed { .. }
        ),
        "{:?}",
        env.record(&def)
    );
    assert!(env.stored(&def).is_none(), "the new index is withdrawn");
    env.inject(None);
    assert!(
        builds.resume().expect("resume").is_empty(),
        "nothing is left to resume"
    );
}

#[test]
fn a_build_whose_environment_fails_settles_failed() {
    a_taken_build_that_breaks_settles_failed(Fault::FieldsUnavailable, "field dictionary");
}

#[test]
fn a_build_that_panics_settles_failed() {
    a_taken_build_that_breaks_settles_failed(Fault::PanicOnPage, "page commit broke");
}

/// Cancelling a build the catalog has no record of cancels nothing, and a
/// waiter on it is told at once that no such build will finish rather than
/// waiting.
#[test]
fn cancelling_an_unknown_build_cancels_nothing() {
    let env = env();
    let builds = service(&env);
    assert!(!builds.cancel(GenerationId::from_raw(999)).expect("cancel"));
    assert!(matches!(
        builds
            .wait(GenerationId::from_raw(999), None)
            .expect("wait"),
        Some(IndexBuildOutcome::Cancelled)
    ));
}

/// An index dropped while its build runs takes the build with it: the drop
/// leaves the running build's record alone, the executor, its pages fenced
/// by the deleted definition, writes no entry, removes its record and ends
/// cancelled.
#[test]
fn an_index_dropped_during_its_build_ends_the_build() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    let older = older_transaction(&env);
    builds.submit(def.generation).expect("submit");
    await_running(&env, &def, None);

    let store = LocalIndexStore::new(&env.engine);
    let mut txn = env.begin();
    store.delete_definition_txn(&mut txn, &def).expect("drop");
    store
        .delete_finished_builds_txn(&mut txn, def.id)
        .expect("drop finished builds");
    store.clear_txn(&mut txn, def.generation).expect("clear");
    commit(&mut txn).expect("commit drop");
    drop(txn);
    drop(older);
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Cancelled)),
        "{outcome:?}"
    );
    assert!(env.record(&def).is_none());
    assert!(env.stored(&def).is_none());
    assert!(env.holders(&def, "a@x").is_empty(), "no page landed");
}

/// A build whose index was dropped before an executor took it ends
/// cancelled without writing anything, and its record is removed.
/// Create user `id` holding `email` while `def` is being built, as a write
/// does: its entries staged through the registry, and each unique value it
/// takes stated as a claim decided over the stored nodes past `covered`.
fn take_while_building(
    env: &TestEnv,
    def: &IndexDefinition,
    id: u64,
    email: &str,
    covered: Option<Vec<u8>>,
) -> Result<(), String> {
    use coordinode_core::index::derive::tuples;
    use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope, UncoveredSource};

    let fields = env.fields_now();
    let mut record = NodeRecord::new("User");
    record.set(
        fields.lookup("email").expect("email field"),
        Value::String(email.into()),
    );
    let mut txn = env.begin();
    LocalNodeStore
        .put(&mut txn, 1, NodeId::from_raw(id), &record)
        .expect("put");
    let lookup = crate::index::registry::record_lookup(&record, &fields);
    let field_of = |name: &str| fields.lookup(name);
    let mut claims = Vec::new();
    env.registry
        .on_node_created(
            &env.engine,
            &mut txn,
            1,
            &crate::index::registry::NodeState {
                node_id: NodeId::from_raw(id),
                valid_from: None,
                label: "User",
                value_of: &lookup,
            },
            &field_of,
            &mut claims,
        )
        .map_err(|e| e.to_string())?;
    let revision = txn.schema_generation();
    for claim in claims {
        for tuple in tuples(&claim.values) {
            txn.claim(Claim::new(
                ClaimScope::UniqueValue {
                    generation: def.generation,
                    tuple,
                },
                ClaimPredicate::UniqueHolder {
                    node: NodeId::from_raw(id),
                    uncovered: Some(Box::new(UncoveredSource {
                        shard_id: 1,
                        label: "User".into(),
                        interpretation: def.interpretation(&field_of),
                        covered_through: covered.clone(),
                        read_limit: 100_000,
                    })),
                },
                revision,
            ));
        }
    }
    commit(&mut txn).map_err(|e| e.to_string())
}

/// With the build stopped part way, the service tells the key its backfill
/// covered through. A value held by a node before that key is refused on
/// the entry the build committed; one held by a node after it, which has no
/// entry yet, is refused by the commit reading the uncovered nodes. A free
/// value is taken, and once the build ends the service tells no key.
#[test]
fn a_build_stopped_part_way_covers_some_owners_and_reads_the_rest() {
    let env = env();
    let total = 600;
    for id in 1..=total {
        env.put_user(id, &format!("u{id}@x"));
    }
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email").unique(),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    env.hold_pages_after(1);
    builds.submit(def.generation).expect("submit");
    env.await_held_page();

    let covered = builds
        .covered_through(def.generation)
        .expect("the first page is covered");
    let key =
        |id: u64| coordinode_core::graph::node::encode_node_key(1, NodeId::from_raw(id)).to_vec();
    assert!(key(1) <= covered && covered < key(total), "{covered:?}");

    let early = take_while_building(&env, &def, 10_001, "u1@x", Some(covered.clone()))
        .expect_err("held by a covered node");
    assert!(early.contains("unique constraint violated"), "{early}");
    let late = take_while_building(
        &env,
        &def,
        10_002,
        &format!("u{total}@x"),
        Some(covered.clone()),
    )
    .expect_err("held by an uncovered node");
    assert!(late.contains(&format!("node {total}")), "{late}");
    take_while_building(&env, &def, 10_003, "free@x", Some(covered)).expect("free");

    env.release_pages();
    let outcome = builds.wait(def.generation, None).expect("wait");
    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Published { .. })),
        "{outcome:?}"
    );
    assert_eq!(builds.covered_through(def.generation), None);
    assert_eq!(env.holders(&def, "free@x"), vec![10_003]);
    assert_eq!(env.holders(&def, &format!("u{total}@x")), vec![total]);
}

/// The configuration is retuned and read while a build runs, here while it
/// waits for an older transaction: neither waits for the build.
#[test]
fn the_config_is_retuned_and_read_while_a_build_runs() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let builds = service(&env);
    let older = older_transaction(&env);
    builds.submit(def.generation).expect("submit");
    await_phase(
        &builds,
        def.generation,
        BuildPhase::AwaitingOlderTransactions,
    );

    let (done, retuned) = std::sync::mpsc::channel();
    let tuner = builds.clone();
    std::thread::spawn(move || {
        tuner.set_config(IndexBuildConfig {
            unique_admission_read_limit: 7,
            ..tuner.config()
        });
        done.send(tuner.config().unique_admission_read_limit)
            .expect("report");
    });
    let read = retuned
        .recv_timeout(Duration::from_secs(5))
        .expect("the retune waited for the build");
    assert_eq!(read, 7);

    drop(older);
    builds.wait(def.generation, None).expect("wait");
}

#[test]
fn a_build_of_a_dropped_index_is_cancelled() {
    let env = env();
    env.put_user(1, "a@x");
    let def = env.admit(
        IndexDescriptor::btree("user_email", "User", "email"),
        BuildFailure::Withdraw,
    );
    let store = LocalIndexStore::new(&env.engine);
    let mut txn = env.begin();
    store.delete_definition_txn(&mut txn, &def).expect("drop");
    commit(&mut txn).expect("commit drop");
    drop(txn);

    let builds = service(&env);
    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Cancelled)),
        "{outcome:?}"
    );
    assert!(env.record(&def).is_none());
    assert!(env.holders(&def, "a@x").is_empty());
}

// ── ON DUPLICATE RENAME ──────────────────────────────────────────────

/// Admit a build of a new unique index on `:User(email)` that repairs the
/// duplicates it meets by renaming `email`.
fn admit_repairing(env: &TestEnv) -> IndexDefinition {
    let store = LocalIndexStore::new(&env.engine);
    let mut descriptor = IndexDescriptor::btree("user_email", "User", "email").unique();
    descriptor.state = IndexState::Building {
        written: 0,
        estimated_total: 0,
    };
    let mut txn = env.begin();
    let def = store
        .publish_definition_txn(&mut txn, descriptor)
        .expect("publish");
    store
        .put_build_txn(
            &mut txn,
            &IndexBuildRecord::accepted(def.id, def.generation, BuildFailure::Withdraw).repairing(
                Some(DuplicateRepair {
                    property: "email".into(),
                }),
            ),
            None,
        )
        .expect("admit build");
    commit(&mut txn).expect("commit publication");
    env.registry
        .register_published(&env.engine, def.clone())
        .expect("register");
    def
}

/// The email node `id` holds now.
fn email_of(env: &TestEnv, id: u64) -> Option<Value> {
    let field = env.fields_now().lookup("email").expect("email field");
    let txn = env.begin();
    LocalNodeStore
        .get(&txn, 1, NodeId::from_raw(id))
        .expect("read node")
        .and_then(|record| record.get(field).cloned())
}

/// A build allowed to rename meets a value two stored nodes hold: the node
/// it reaches second gets the value plus a suffix, the build publishes the
/// index with every node, and the repair is recorded with the old and the
/// new value. The first holder and the other nodes are untouched.
#[test]
fn a_repairing_build_renames_the_second_holder_and_publishes() {
    let env = env();
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    env.put_user(3, "other@x");
    let def = admit_repairing(&env);
    let builds = service(&env);

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(
            outcome,
            Some(IndexBuildOutcome::Published { indexed: Some(3) })
        ),
        "{outcome:?}"
    );
    assert_eq!(env.holders(&def, "same@x"), [1]);
    assert_eq!(env.holders(&def, "other@x"), [3]);
    let Some(Value::String(renamed)) = email_of(&env, 2) else {
        panic!("node 2 keeps a string");
    };
    assert!(
        renamed.starts_with("same@x_") && renamed.len() == "same@x_".len() + 8,
        "{renamed}"
    );
    assert_eq!(env.holders(&def, &renamed), [2]);
    let record = env.record(&def).expect("record");
    assert_eq!(record.repaired, 1);
    let repairs = LocalIndexStore::new(&env.engine)
        .list_repairs(def.generation)
        .expect("repairs");
    assert_eq!(
        repairs,
        [DuplicateRepairRecord {
            generation: def.generation,
            node: 2,
            property: "email".into(),
            old: "same@x".into(),
            new: renamed,
        }]
    );
}

/// A repair commits on its own, before the build ends: it is visible while
/// the build still runs, and a cancellation of the build afterwards keeps
/// it, with its record, while the index itself is withdrawn.
#[test]
fn a_repair_is_visible_before_the_build_ends_and_survives_its_cancellation() {
    let env = env();
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    let def = admit_repairing(&env);
    let builds = service(&env);
    env.hold_pages_after(0);

    builds.submit(def.generation).expect("submit");
    env.await_held_page();
    let Some(Value::String(renamed)) = email_of(&env, 2) else {
        panic!("node 2 keeps a string");
    };
    assert_ne!(renamed, "same@x", "repaired while the build runs");
    assert_eq!(env.record(&def).expect("record").repaired, 1);

    assert!(builds.cancel(def.generation).expect("cancel"));
    env.release_pages();
    let outcome = builds.wait(def.generation, None).expect("wait");
    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Cancelled)),
        "{outcome:?}"
    );
    assert!(env.stored(&def).is_none(), "the index is withdrawn");
    assert_eq!(email_of(&env, 2), Some(Value::String(renamed.clone())));
    assert_eq!(email_of(&env, 1), Some(Value::String("same@x".into())));
    let repairs = LocalIndexStore::new(&env.engine)
        .list_repairs(def.generation)
        .expect("repairs");
    assert_eq!(repairs.len(), 1);
    assert_eq!(repairs[0].new, renamed);
}

/// A duplicate value a suffix cannot be appended to fails the build: the
/// index is withdrawn, no node is changed, and nothing is recorded.
#[test]
fn a_duplicate_that_is_not_a_string_fails_a_repairing_build() {
    let env = env();
    let field = env.fields_now().lookup("email").expect("email field");
    for id in [1, 2] {
        let mut record = NodeRecord::new("User");
        record.set(field, Value::Int(7));
        let mut txn = env.begin();
        LocalNodeStore
            .put(&mut txn, 1, NodeId::from_raw(id), &record)
            .expect("put");
        commit(&mut txn).expect("commit node");
    }
    let def = admit_repairing(&env);
    let builds = service(&env);

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(&outcome, Some(IndexBuildOutcome::Failed(BuildError::Other(why)))
            if why.contains("only a string")),
        "{outcome:?}"
    );
    assert!(env.stored(&def).is_none());
    assert_eq!(email_of(&env, 1), Some(Value::Int(7)));
    assert_eq!(email_of(&env, 2), Some(Value::Int(7)));
    assert!(
        LocalIndexStore::new(&env.engine)
            .list_repairs(def.generation)
            .expect("repairs")
            .is_empty()
    );
}

/// A repair decided from a value the node no longer holds (a writer
/// changed it after the scan) writes nothing: the writer's value stays, and
/// the backfill reads the node again.
#[test]
fn a_repair_of_a_value_changed_since_overwrites_nothing() {
    let env = env();
    env.put_user(2, "edited@x");
    let def = admit_repairing(&env);
    let builds = service(&env);

    let repaired = super::super::repair::rename_duplicate(
        &builds,
        env.as_ref(),
        &super::super::repair::RepairBuild {
            generation: def.generation,
            token: 0,
            index: &def,
            property: "email",
        },
        NodeId::from_raw(2),
        Some(&Value::String("same@x".into())),
        &super::super::repair::suffix,
    )
    .expect("repair");

    assert_eq!(repaired, super::super::repair::Repaired::Changed);
    assert_eq!(email_of(&env, 2), Some(Value::String("edited@x".into())));
    assert!(
        LocalIndexStore::new(&env.engine)
            .list_repairs(def.generation)
            .expect("repairs")
            .is_empty()
    );
}

/// Run one repair of node 2, whose `same@x` node 1 also holds, while the
/// build of `def` is running but has covered nothing yet, drawing suffixes
/// from `suffixes` in turn (the last one repeats), as the executor holding
/// the build or, with `stale`, as one the build no longer belongs to.
fn repair_node_two(
    env: &Arc<TestEnv>,
    def: &IndexDefinition,
    suffixes: &[&str],
    stale: bool,
) -> Result<super::super::repair::Repaired, BackfillError> {
    let builds = service(env);
    let older = older_transaction(env);
    builds.submit(def.generation).expect("submit");
    let running = await_running(env, def, None);
    let token = if stale { running + 1 } else { running };
    let drawn = std::cell::Cell::new(0usize);
    let next = || {
        let i = drawn.get();
        drawn.set(i + 1);
        suffixes[i.min(suffixes.len() - 1)].to_string()
    };

    let repaired = super::super::repair::rename_duplicate(
        &builds,
        env.as_ref(),
        &super::super::repair::RepairBuild {
            generation: def.generation,
            token,
            index: def,
            property: "email",
        },
        NodeId::from_raw(2),
        Some(&Value::String("same@x".into())),
        &next,
    );
    drop(older);
    builds.wait(def.generation, None).expect("wait");
    repaired
}

/// A suffix whose result another node already holds is not taken: the
/// repair draws another one and gives the node that value.
#[test]
fn a_repair_redraws_a_suffix_another_node_holds() {
    let env = env();
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    env.put_user(3, "same@x_aaaaaaaa");
    let def = admit_repairing(&env);

    let repaired = repair_node_two(&env, &def, &["aaaaaaaa", "bbbbbbbb"], false).expect("repair");

    assert_eq!(repaired, super::super::repair::Repaired::Done);
    assert_eq!(
        email_of(&env, 2),
        Some(Value::String("same@x_bbbbbbbb".into()))
    );
    assert_eq!(
        email_of(&env, 3),
        Some(Value::String("same@x_aaaaaaaa".into()))
    );
    let repairs = LocalIndexStore::new(&env.engine)
        .list_repairs(def.generation)
        .expect("repairs");
    assert_eq!(repairs.len(), 1);
    assert_eq!(repairs[0].new, "same@x_bbbbbbbb");
}

/// When every suffix drawn gives a value another node holds, the repair
/// gives up with an error naming the property and the value, and changes
/// nothing.
#[test]
fn a_repair_that_finds_no_free_suffix_fails_without_writing() {
    let env = env();
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    env.put_user(3, "same@x_aaaaaaaa");
    let def = admit_repairing(&env);

    let refused = repair_node_two(&env, &def, &["aaaaaaaa"], false);

    assert!(
        matches!(&refused, Err(BackfillError::Repair(why))
            if why.contains("no free value of `email`") && why.contains("same@x")),
        "{refused:?}"
    );
    assert_ne!(
        email_of(&env, 2),
        Some(Value::String("same@x_aaaaaaaa".into()))
    );
    assert!(
        LocalIndexStore::new(&env.engine)
            .list_repairs(def.generation)
            .expect("repairs")
            .iter()
            .all(|r| r.new != "same@x_aaaaaaaa")
    );
}

/// A repair prepared by an executor the build no longer belongs to (it was
/// cancelled or taken over while the repair was being prepared) commits
/// nothing: the node keeps its value and no repair is recorded for it.
#[test]
fn a_repair_by_an_executor_the_build_moved_from_writes_nothing() {
    let env = env();
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    let def = admit_repairing(&env);

    let refused = repair_node_two(&env, &def, &["cccccccc"], true);

    assert!(
        matches!(refused, Err(BackfillError::Superseded)),
        "{refused:?}"
    );
    assert!(
        LocalIndexStore::new(&env.engine)
            .list_repairs(def.generation)
            .expect("repairs")
            .iter()
            .all(|r| r.new != "same@x_cccccccc"),
        "the stale repair is not recorded"
    );
    assert_ne!(
        email_of(&env, 2),
        Some(Value::String("same@x_cccccccc".into())),
        "the stale repair changed no node"
    );
}

/// A build that fails after it repaired a duplicate leaves the repair in
/// place, value and record: a repair is its own committed change, and the
/// failure of the build later is not a reason to undo it.
#[test]
fn a_build_failing_after_a_repair_keeps_the_repair() {
    let env = env();
    let field = env.fields_now().lookup("email").expect("email field");
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    for id in [3, 4] {
        let mut record = NodeRecord::new("User");
        record.set(field, Value::Int(7));
        let mut txn = env.begin();
        LocalNodeStore
            .put(&mut txn, 1, NodeId::from_raw(id), &record)
            .expect("put");
        commit(&mut txn).expect("commit node");
    }
    let def = admit_repairing(&env);
    let builds = service(&env);

    builds.submit(def.generation).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(&outcome, Some(IndexBuildOutcome::Failed(BuildError::Other(why)))
            if why.contains("only a string")),
        "{outcome:?}"
    );
    let Some(Value::String(renamed)) = email_of(&env, 2) else {
        panic!("node 2 keeps a string");
    };
    assert!(renamed.starts_with("same@x_"), "{renamed}");
    let repairs = LocalIndexStore::new(&env.engine)
        .list_repairs(def.generation)
        .expect("repairs");
    assert_eq!(repairs.len(), 1, "{repairs:?}");
    assert_eq!(repairs[0].new, renamed);
    assert_eq!(env.record(&def).expect("the outcome stays").repaired, 1);
}

/// A build taken over after it repaired a duplicate (the executor lost, as
/// after a leader change or a restart) is finished without renaming that
/// node again: the new executor reads the repaired value and finds no
/// duplicate left.
#[test]
fn a_build_taken_over_after_a_repair_renames_nothing_again() {
    let env = env();
    env.put_user(1, "same@x");
    env.put_user(2, "same@x");
    let def = admit_repairing(&env);
    let first = service(&env);
    let second = service(&env);
    env.hold_pages_after(0);

    first.submit(def.generation).expect("first takes it");
    env.await_held_page();
    let Some(Value::String(renamed)) = email_of(&env, 2) else {
        panic!("node 2 keeps a string");
    };
    assert_ne!(renamed, "same@x", "repaired by the first executor");
    let taken = await_running(&env, &def, None);
    second.submit(def.generation).expect("second takes it over");
    await_running(&env, &def, Some(taken));
    env.release_pages();
    first.wait(def.generation, None).expect("first wait");
    let outcome = second.wait(def.generation, None).expect("second wait");

    assert!(
        matches!(
            outcome,
            Some(IndexBuildOutcome::Published { indexed: Some(2) })
        ),
        "{outcome:?}"
    );
    assert_eq!(email_of(&env, 2), Some(Value::String(renamed.clone())));
    assert_eq!(env.holders(&def, &renamed), [2]);
    let repairs = LocalIndexStore::new(&env.engine)
        .list_repairs(def.generation)
        .expect("repairs");
    assert_eq!(repairs.len(), 1, "{repairs:?}");
    assert_eq!(env.record(&def).expect("record").repaired, 1);
}
