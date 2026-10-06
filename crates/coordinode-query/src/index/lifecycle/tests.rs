use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_modality::{LocalNodeStore, NodeStore as _};

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
        let mut ids: Vec<u64> = LocalIndexStore::new(&self.engine)
            .scan_exact(&mut txn, def, &[Value::String(email.into())])
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

    builds.submit(def.generation, 0).expect("submit");
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

    builds.submit(def.generation, 0).expect("submit");
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
    builds.submit(def.generation, 0).expect("submit");

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

    builds.submit(def.generation, 0).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(
            &outcome,
            Some(IndexBuildOutcome::Failed(BuildError::Duplicate(v)))
                if v.holder == NodeId::from_raw(1)
        ),
        "{outcome:?}"
    );
    assert!(matches!(
        env.record(&def).expect("record").state,
        BuildState::Failed { .. }
    ));
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

    builds.submit(def.generation, 0).expect("submit");
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

    first.submit(def.generation, 0).expect("first takes it");
    let taken = await_running(&env, &def, None);
    second
        .submit(def.generation, 0)
        .expect("second takes it over");
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

    builds.submit(def.generation, 0).expect("submit");
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

    builds.submit(def.generation, 0).expect("submit");
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

    builds.submit(def.generation, 0).expect("submit");
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
    builds.submit(def.generation, 0).expect("submit");
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
    builds.submit(def.generation, 0).expect("submit");
    let outcome = builds.wait(def.generation, None).expect("wait");

    assert!(
        matches!(outcome, Some(IndexBuildOutcome::Cancelled)),
        "{outcome:?}"
    );
    assert!(env.record(&def).is_none());
    assert!(env.holders(&def, "a@x").is_empty());
}
