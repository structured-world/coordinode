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
    builds.submit(def.generation, 0).expect("submit");
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
    builds.submit(def.generation, 0).expect("submit");
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
    builds.submit(def.generation, 0).expect("submit");
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

    builds.submit(def.generation, 0).expect("submit");
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

    builds.submit(def.generation, 0).expect("submit");
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

    builds.submit(def.generation, 0).expect("submit");
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
/// from `suffixes` in turn (the last one repeats).
fn repair_node_two(
    env: &Arc<TestEnv>,
    def: &IndexDefinition,
    suffixes: &[&str],
) -> Result<super::super::repair::Repaired, BackfillError> {
    let builds = service(env);
    let older = older_transaction(env);
    builds.submit(def.generation, 0).expect("submit");
    let token = await_running(env, def, None);
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

    let repaired = repair_node_two(&env, &def, &["aaaaaaaa", "bbbbbbbb"]).expect("repair");

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

    let refused = repair_node_two(&env, &def, &["aaaaaaaa"]);

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
