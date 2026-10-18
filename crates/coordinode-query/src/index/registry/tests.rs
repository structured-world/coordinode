use super::*;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

/// A direct-mode transaction: its writes land as they are staged, so each
/// step below sees the ones before it.
fn txn(engine: &StorageEngine) -> Transaction<'_> {
    Transaction::new(engine, None, Timestamp::ZERO, None)
}

fn props(pairs: &[(&str, Value)]) -> Vec<(String, Value)> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

/// No property is bound to a field id: the tests keep values by name.
fn no_fields(_: &str) -> Option<u32> {
    None
}

/// The shard the tests' nodes live in.
const SHARD: u16 = 1;

/// Store the `User` node `id` with `pairs` as its properties, kept by name,
/// as a write stores the record its entries are derived from: a unique
/// entry's holder is checked against it.
fn put_record(t: &mut Transaction, id: u64, pairs: &[(&str, Value)]) {
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let mut record = coordinode_core::graph::node::NodeRecord::new("User");
    for (name, value) in pairs {
        record.set_extra(*name, value.clone());
    }
    LocalNodeStore
        .put(t, SHARD, NodeId::from_raw(id), &record)
        .expect("put node record");
}

fn create(
    reg: &IndexRegistry,
    engine: &StorageEngine,
    id: u64,
    pairs: &[(&str, Value)],
) -> Result<(), IndexWriteError> {
    let props = props(pairs);
    let lookup = props_lookup(&props);
    let mut t = txn(engine);
    put_record(&mut t, id, pairs);
    reg.on_node_created(
        engine,
        &mut t,
        SHARD,
        &NodeState {
            node_id: NodeId::from_raw(id),
            valid_from: None,
            label: "User",
            value_of: &lookup,
        },
        &no_fields,
        &mut Vec::new(),
    )
}

fn change(
    reg: &IndexRegistry,
    engine: &StorageEngine,
    id: u64,
    property: &str,
    before: &[(&str, Value)],
    after: &[(&str, Value)],
) -> Result<(), IndexWriteError> {
    let mut t = txn(engine);
    put_record(&mut t, id, after);
    let (before, after) = (props(before), props(after));
    let (before, after) = (props_lookup(&before), props_lookup(&after));
    reg.on_property_changed(
        engine,
        &mut t,
        SHARD,
        &PropertyChange {
            node_id: NodeId::from_raw(id),
            valid_from: None,
            label: "User",
            properties: &[property],
            before: &before,
            after: &after,
        },
        &no_fields,
        &mut Vec::new(),
    )
}

fn lookup(engine: &StorageEngine, index: &IndexDefinition, values: &[Value]) -> Vec<u64> {
    let budget = coordinode_core::budget::QueryBudget::new(
        coordinode_core::budget::DEFAULT_QUERY_MEMORY_LIMIT,
    );
    let mut ids: Vec<u64> = LocalIndexStore::new(engine)
        .scan_exact(&mut txn(engine), index, values, &budget)
        .expect("scan")
        .expect("indexable")
        .into_iter()
        .map(|n| n.as_raw())
        .collect();
    ids.sort_unstable();
    ids
}

fn all_ids(engine: &StorageEngine, index: &IndexDefinition) -> Vec<u64> {
    LocalIndexStore::new(engine)
        .scan_entry_ids(&mut txn(engine), index)
        .expect("scan")
        .into_iter()
        .map(|n| n.as_raw())
        .collect()
}

/// `descriptor` as the index numbered `raw`, serving from generation `raw`.
fn bound(descriptor: crate::index::IndexDescriptor, raw: u64) -> IndexDefinition {
    descriptor.bind(
        crate::index::IndexId::from_raw(raw),
        crate::index::GenerationId::from_raw(raw),
    )
}

fn btree(name: &str, property: &str, raw: u64) -> IndexDefinition {
    bound(
        crate::index::IndexDescriptor::btree(name, "User", property),
        raw,
    )
}

fn integrity(index: &IndexDefinition, state: Integrity) -> IndexIntegrityRecord {
    let mut record = IndexIntegrityRecord::new(index.id, index.generation);
    record.integrity = state;
    record
}

fn stray(node: u64) -> Mismatch {
    Mismatch::Extra {
        node,
        valid_from: None,
        tuple: vec![1],
    }
}

/// Report that node `node` holds a stray entry of `index`.
fn report(reg: &IndexRegistry, engine: &StorageEngine, index: &IndexDefinition, node: u64) {
    reg.report_mismatch(engine, index, stray(node));
}

/// The generations of this member's stored marks, ascending.
fn marked(engine: &StorageEngine) -> Vec<GenerationId> {
    let mut out: Vec<GenerationId> = LocalIndexStore::new(engine)
        .list_unfit_here()
        .expect("marks")
        .into_iter()
        .map(|copy| copy.generation)
        .collect();
    out.sort_unstable();
    out
}

/// The generations of this member's stored use guards, ascending.
fn guarded(engine: &StorageEngine) -> Vec<GenerationId> {
    let mut out: Vec<GenerationId> = LocalIndexStore::new(engine)
        .list_armed_here()
        .expect("guards")
        .into_iter()
        .map(|copy| copy.generation)
        .collect();
    out.sort_unstable();
    out
}

/// Give `generation` a copy on disk, as an earlier write leaves it: its
/// installation bound and flushed.
fn existing_copy(engine: &StorageEngine, generation: GenerationId) {
    LocalIndexStore::new(engine)
        .bind_local_copy(generation)
        .expect("bind");
    engine
        .persist_partition(coordinode_storage::engine::partition::Partition::Idx)
        .expect("persist the copy's installation");
}

/// A process opening the member: guards recovered, then the catalog loaded.
fn reopened(engine: &StorageEngine) -> IndexRegistry {
    let reg = IndexRegistry::new();
    reg.recover_local_guards(engine).expect("recover guards");
    reg.load_all(engine).expect("load");
    reg
}

/// A disagreement found here makes the generation suspect at once and waits
/// to be recorded; nothing else is suspect.
#[test]
fn a_reported_generation_is_suspect_and_its_report_waits() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    let other = btree("user_name", "name", 2);
    assert!(!reg.is_suspect(index.generation));
    report(&reg, &engine, &index, 7);
    report(&reg, &engine, &index, 7);
    assert!(reg.is_suspect(index.generation));
    assert!(!reg.is_suspect(other.generation));
    let reports = reg.take_reports(core::time::Duration::ZERO);
    assert_eq!(reports.len(), 1, "one disagreement, reported twice");
    assert_eq!(reports[0].generation, index.generation);
    assert!(reg.take_reports(core::time::Duration::ZERO).is_empty());
    assert!(
        reg.is_suspect(index.generation),
        "taking the report clears nothing"
    );
}

/// A suspicion found here is about this member's copy: a verified record in
/// the catalog proves another member's copy, so neither an older one nor one
/// written after the catalog recorded the suspicion clears it. Only a check
/// verified on this member does.
#[test]
fn a_catalog_verification_does_not_clear_a_suspicion_found_here() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    report(&reg, &engine, &index, 7);
    let revision = reg.local_revision(index.generation);
    reg.apply_integrity(&[integrity(&index, Integrity::Verified)]);
    assert!(
        reg.is_suspect(index.generation),
        "the older verified record"
    );
    reg.apply_integrity(&[integrity(&index, Integrity::Suspect)]);
    assert!(reg.is_suspect(index.generation));
    reg.apply_integrity(&[integrity(&index, Integrity::Verified)]);
    assert!(
        reg.is_suspect(index.generation),
        "verified elsewhere after the catalog recorded it"
    );
    assert!(!reg.answers_at(index.generation, u64::MAX));
    assert!(
        reg.verified_here(&engine, &index, revision, 0)
            .expect("lift")
    );
    assert!(!reg.is_suspect(index.generation), "verified on this member");
    assert!(
        LocalIndexStore::new(&engine)
            .list_unfit_here()
            .expect("marks")
            .is_empty(),
        "and its stored mark is gone"
    );
}

/// A check proves this member's copy only as it stood when its pass
/// started: a disagreement found here after that keeps the copy unfit,
/// in memory and in storage, whatever the pass then concluded.
#[test]
fn a_finding_after_the_pass_started_survives_its_verification() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    report(&reg, &engine, &index, 7);
    let started = reg.local_revision(index.generation);
    report(&reg, &engine, &index, 8);
    assert!(
        !reg.verified_here(&engine, &index, started, 0)
            .expect("lift")
    );
    assert!(reg.is_suspect(index.generation));
    assert_eq!(marked(&engine), vec![index.generation]);
    // A pass that started with no finding here at all does not lift a
    // finding that came after it either.
    let fresh = btree("user_name", "name", 2);
    let before = reg.local_revision(fresh.generation);
    report(&reg, &engine, &fresh, 9);
    assert!(!reg.verified_here(&engine, &fresh, before, 0).expect("lift"));
    assert!(reg.is_suspect(fresh.generation));
}

/// A finding is stored before the statement that made it goes on: a power
/// cut right after it, before any maintenance ran, still finds the copy
/// unfit when the member opens again.
#[test]
fn a_finding_survives_power_loss_before_any_maintenance() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let index = btree("user_email", "email", 1);
    {
        let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
        let reg = IndexRegistry::new();
        report(&reg, &engine, &index, 7);
        rig.cut(engine);
    }
    let engine = StorageEngine::open(&rig.config()).expect("reopen");
    let reg = IndexRegistry::new();
    reg.load_all(&engine).expect("load");
    assert!(reg.is_suspect(index.generation));
}

/// A mark whose write or sync failed has unknown durability: the statement
/// that found it goes on (it answers from the records), the copy stays unfit
/// here, nothing writes the mark again (a retry after a failed sync proves
/// nothing), and the copy's use guard is not closed, so the next open still
/// finds the copy unfit.
#[test]
fn a_mark_whose_flush_failed_is_not_retried_and_keeps_the_guard() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let index = btree("user_email", "email", 1);
    {
        let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
        let reg = IndexRegistry::new();
        existing_copy(&engine, index.generation);
        assert!(reg.arm_copy(&engine, index.generation), "armed before use");
        // The mark is made durable by the journal's sync, which fails from
        // here on, as on a disk that stopped taking writes.
        engine.fail_journal_syncs();
        report(&reg, &engine, &index, 7);
        assert!(reg.is_suspect(index.generation), "unfit here regardless");
        // Whatever follows must not ask for the mark to be written, removed
        // or the guard closed.
        let flushes = reg.recovery_flushes();
        report(&reg, &engine, &index, 8);
        let now = reg.local_revision(index.generation);
        assert!(
            !reg.verified_here(&engine, &index, now, 0)
                .expect("refused without I/O"),
            "a mark of unknown state is not lifted"
        );
        reg.close_local_guards(&engine);
        assert_eq!(reg.recovery_flushes(), flushes, "no I/O for a failed mark");
        assert!(reg.is_suspect(index.generation));
        rig.cut(engine);
    }
    let engine = StorageEngine::open(&rig.config()).expect("reopen");
    let reg = reopened(&engine);
    assert!(
        reg.is_suspect(index.generation),
        "the guard left open makes the copy unfit after the restart"
    );
}

/// A finding whose mark could not be written is still among the findings a
/// capture of the store reads from the registry: a transfer taken now
/// excludes that copy although no stored mark names it.
#[test]
fn a_finding_whose_mark_failed_is_among_the_findings() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    existing_copy(&engine, index.generation);
    assert!(reg.findings().is_empty());
    // The mark's journal sync fails.
    engine.fail_journal_syncs();
    report(&reg, &engine, &index, 7);
    let findings = reg.findings();
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].0, index.generation);
}

/// A removal whose sync failed may or may not have reached the disk: the
/// mark is no longer taken as stored, nothing writes or removes it again,
/// and the copy stays unfit here under its open guard.
#[test]
fn a_removal_whose_flush_failed_leaves_the_mark_of_unknown_durability() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let index = btree("user_email", "email", 1);
    let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
    let reg = IndexRegistry::new();
    assert!(reg.arm_copy(&engine, index.generation));
    report(&reg, &engine, &index, 7);
    let started = reg.local_revision(index.generation);
    // The removal is made durable by the journal's sync, which fails from
    // here on.
    engine.fail_journal_syncs();
    assert!(
        reg.verified_here(&engine, &index, started, 0).is_err(),
        "the removal's sync failed"
    );
    assert!(reg.is_suspect(index.generation), "unfit here regardless");
    // Whatever follows must not ask for the mark to be written or removed.
    let flushes = reg.recovery_flushes();
    report(&reg, &engine, &index, 8);
    // A later check verifying the copy does not touch the disk either: the
    // mark's stored state stays unknown, so the copy stays unfit here.
    let now = reg.local_revision(index.generation);
    assert!(
        !reg.verified_here(&engine, &index, now, 0).expect("refused"),
        "a mark of unknown state is not lifted"
    );
    reg.close_local_guards(&engine);
    assert_eq!(reg.recovery_flushes(), flushes, "no I/O for a failed mark");
    assert!(reg.is_suspect(index.generation));
}

/// A mark whose sync failed, or a removal whose sync failed, leaves the mark
/// of unknown state, with no further I/O for it and the guard left open, and
/// the copy unfit after a power cut and a restart. The journal's sync is the
/// one that makes either durable; it fails from that point on, so nothing
/// after it may touch the disk anyway.
#[test]
fn a_failed_sync_leaves_the_mark_of_unknown_durability() {
    let index = btree("user_email", "email", 1);
    for removal in [false, true] {
        let rig = coordinode_test_fixtures::PowerRig::new();
        {
            let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
            let reg = IndexRegistry::new();
            existing_copy(&engine, index.generation);
            assert!(reg.arm_copy(&engine, index.generation));
            if removal {
                report(&reg, &engine, &index, 7);
                let started = reg.local_revision(index.generation);
                engine.fail_journal_syncs();
                assert!(
                    reg.verified_here(&engine, &index, started, 0).is_err(),
                    "the removal's sync failed"
                );
            } else {
                engine.fail_journal_syncs();
                report(&reg, &engine, &index, 7);
            }
            assert!(reg.is_suspect(index.generation));
            // Whatever follows must not ask for the mark to be written or
            // removed, or the guard closed.
            let before = reg.recovery_flushes();
            report(&reg, &engine, &index, 9);
            let now = reg.local_revision(index.generation);
            assert!(!reg.verified_here(&engine, &index, now, 0).expect("refused"));
            reg.close_local_guards(&engine);
            assert_eq!(
                reg.recovery_flushes(),
                before,
                "removal {removal}: no I/O for a mark of unknown state"
            );
            assert!(reg.is_suspect(index.generation));
            rig.cut(engine);
        }
        let engine = StorageEngine::open(&rig.config()).expect("reopen");
        assert!(
            reopened(&engine).is_suspect(index.generation),
            "removal {removal}: unfit after the restart"
        );
    }
}

/// Marks are recovery state the free-space reserve exists for: a disk below
/// the reserve, where client writes are refused, still takes the mark (out
/// of the reserve's upper half), so a restart finds the copy unfit whatever
/// space is free then.
#[test]
fn a_mark_is_written_below_the_free_space_reserve() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = btree("user_email", "email", 1);
    {
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();
        let free = engine.space().available_bytes();
        engine.space().set_reserve(free + (64 << 20), u64::MAX);
        assert!(engine.space().is_paused(), "client writes are refused");
        assert!(reg.arm_copy(&engine, index.generation), "the guard too");
        report(&reg, &engine, &index, 7);
        assert_eq!(marked(&engine), vec![index.generation]);
        engine.space().set_reserve(0, 0);
        reg.close_local_guards(&engine);
    }
    let engine = test_engine(dir.path());
    assert!(
        reopened(&engine).is_suspect(index.generation),
        "the stored mark, not the space freed since, decides"
    );
}

/// A copy is guarded once per run: the first use stores the guard, every
/// later use reads one flag and makes no I/O.
#[test]
fn a_copy_is_armed_once_per_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    reg.register_in_memory(index.clone());
    let armed = reg.get_guarded(index.id).expect("registered").armed;
    assert!(!armed, "not armed before its first use");
    assert!(reg.arm_copy(&engine, index.generation));
    assert_eq!(reg.recovery_flushes(), 1);
    for _ in 0..1000 {
        assert!(reg.get_guarded(index.id).expect("registered").armed);
    }
    assert!(reg.arm_copy(&engine, index.generation));
    assert_eq!(
        reg.recovery_flushes(),
        1,
        "no guard I/O after the first use"
    );
    assert_eq!(guarded(&engine), vec![index.generation]);
    // A catalog refresh keeps the flag: the guard is this run's, not the
    // definition's.
    reg.load_all(&engine).expect("load");
    reg.register_in_memory(index.clone());
    assert!(reg.get_guarded(index.id).expect("registered").armed);
}

/// Copies first used at the same time share one flush: each waits for its
/// guard behind the one being stored, and the next flush takes them all.
#[test]
fn copies_waiting_together_are_armed_in_one_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = Arc::new(test_engine(dir.path()));
    let reg = Arc::new(IndexRegistry::new());
    let generations: Vec<GenerationId> = (1..=3).map(GenerationId::from_raw).collect();
    let held = reg.arming.lock();
    let threads: Vec<_> = generations
        .iter()
        .map(|g| {
            let (reg, engine, g) = (Arc::clone(&reg), Arc::clone(&engine), *g);
            std::thread::spawn(move || reg.arm_copy(&engine, g))
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while reg.integrity.lock().to_arm.len() < generations.len() {
        assert!(std::time::Instant::now() < deadline, "the uses never came");
        std::thread::yield_now();
    }
    drop(held);
    for thread in threads {
        assert!(thread.join().expect("arming thread"));
    }
    assert_eq!(reg.recovery_flushes(), 1, "one flush for the three guards");
    assert_eq!(guarded(&engine), generations);
}

/// A run that stops without closing its guards, at any point after a guard
/// is stored (before the copy's first use, or after uses that found
/// nothing), leaves its copies unfit at the next open, until a check
/// verifies them; a copy the run never used stays fit, so the open checks
/// only what was in use.
#[test]
fn an_unclosed_guard_makes_its_copy_unfit_at_the_next_open() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let used = btree("user_email", "email", 1);
    let unused = btree("user_name", "name", 2);
    {
        let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
        let reg = IndexRegistry::new();
        // Both copies exist on disk from an earlier write; only one is used.
        existing_copy(&engine, used.generation);
        existing_copy(&engine, unused.generation);
        assert!(reg.arm_copy(&engine, used.generation));
        rig.cut(engine);
    }
    let engine = StorageEngine::open(&rig.config()).expect("reopen");
    let reg = reopened(&engine);
    assert!(reg.is_suspect(used.generation), "in use when the run ended");
    assert!(!reg.answers_at(used.generation, u64::MAX));
    assert!(!reg.is_suspect(unused.generation), "never used: still fit");
    assert_eq!(marked(&engine), vec![used.generation], "stored as a mark");
    assert!(guarded(&engine).is_empty(), "the guard became the mark");
    // The mark, unlike a guard, outlives a clean run of its own.
    reg.close_local_guards(&engine);
    drop(reg);
    assert!(reopened(&engine).is_suspect(used.generation));
}

/// A clean stop closes the guards of copies with nothing outstanding, so the
/// next open uses them without a check; a copy with a stored finding is
/// closed too, its mark keeping it unfit.
#[test]
fn a_clean_stop_closes_the_guards() {
    let dir = tempfile::tempdir().expect("tempdir");
    let fit = btree("user_email", "email", 1);
    let found = btree("user_name", "name", 2);
    {
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();
        assert!(reg.arm_copy(&engine, fit.generation));
        assert!(reg.arm_copy(&engine, found.generation));
        report(&reg, &engine, &found, 7);
        reg.close_local_guards(&engine);
        assert!(guarded(&engine).is_empty(), "every guard closed");
        assert!(
            !reg.arm_copy(&engine, fit.generation),
            "nothing is armed once the guards close"
        );
    }
    let engine = test_engine(dir.path());
    let reg = reopened(&engine);
    assert!(!reg.is_suspect(fit.generation), "closed cleanly: fit");
    assert!(reg.is_suspect(found.generation), "its stored mark stands");
}

/// A guard that cannot be stored keeps its copy out of use for the rest of
/// the run, without trying the write again on every use.
#[test]
fn a_copy_whose_guard_cannot_be_stored_never_serves() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    reg.register_in_memory(index.clone());
    // The guard is stored by the journal's sync, which fails.
    engine.fail_journal_syncs();
    assert!(!reg.arm_copy(&engine, index.generation), "not armed");
    let flushes = reg.recovery_flushes();
    for _ in 0..10 {
        assert!(!reg.arm_copy(&engine, index.generation));
        assert!(!reg.get_guarded(index.id).expect("registered").armed);
    }
    assert_eq!(reg.recovery_flushes(), flushes, "the write is not retried");
}

/// A finding made while a check removes the mark does not return until the
/// mark is stored again: a power cut the moment the finding returns still
/// finds the copy unfit.
#[test]
fn a_finding_during_the_removal_returns_only_once_stored_again() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let index = btree("user_email", "email", 1);
    {
        let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
        let reg = Arc::new(IndexRegistry::new());
        report(&reg, &engine, &index, 7);
        let started = reg.local_revision(index.generation);
        let reporter = Arc::new(parking_lot::Mutex::new(None));
        {
            let (reg_in, engine_in, index_in, reporter_in) = (
                Arc::clone(&reg),
                Arc::clone(&engine),
                index.clone(),
                Arc::clone(&reporter),
            );
            *reg.after_mark_removed.lock() = Some(Box::new(move || {
                let (reg_t, engine_t, index_t) = (
                    Arc::clone(&reg_in),
                    Arc::clone(&engine_in),
                    index_in.clone(),
                );
                let handle = std::thread::spawn(move || {
                    reg_t.report_mismatch(&engine_t, &index_t, stray(8))
                });
                // The finding has taken its revision; it must now wait for
                // the removal to finish rather than return.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while reg_in.local_revision(index_in.generation) == started {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "the finding never came"
                    );
                    std::thread::yield_now();
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
                assert!(
                    !handle.is_finished(),
                    "the finding returned while its mark was being removed"
                );
                *reporter_in.lock() = Some(handle);
            }));
        }
        assert!(
            !reg.verified_here(&engine, &index, started, 0)
                .expect("lift")
        );
        *reg.after_mark_removed.lock() = None;
        let handle = reporter.lock().take().expect("the finding ran");
        handle.join().expect("reporter thread");
        drop(reg);
        rig.cut(engine);
    }
    let engine = StorageEngine::open(&rig.config()).expect("reopen");
    let reg = IndexRegistry::new();
    reg.load_all(&engine).expect("load");
    assert!(reg.is_suspect(index.generation));
}

/// The report queue stays within its bound: a report past it is dropped and
/// the drop is flagged, so the maintenance admits the check from the mark,
/// which every finding still stores.
#[test]
fn a_full_report_queue_drops_reports_but_keeps_marks() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    for node in 0..MAX_PENDING_REPORTS as u64 {
        report(&reg, &engine, &index, node);
    }
    assert!(!reg.take_overflow());
    let other = btree("user_name", "name", 2);
    report(&reg, &engine, &other, 1);
    assert!(reg.take_overflow(), "the drop is flagged");
    assert!(!reg.take_overflow(), "once");
    assert!(reg.unfit_here().contains(&other.generation));
    assert_eq!(
        reg.take_reports(core::time::Duration::ZERO).len(),
        MAX_PENDING_REPORTS
    );
    assert_eq!(marked(&engine), vec![index.generation, other.generation]);
}

/// A generation that left the catalog keeps a mark found here: a reader
/// pinned to an older snapshot, plan or cursor may still reach its
/// entries, and nothing says when the last one is gone.
#[test]
fn a_replaced_generation_keeps_its_mark() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    report(&reg, &engine, &index, 7);
    reg.load_all(&engine).expect("load without the generation");
    assert!(reg.is_suspect(index.generation));
    assert!(!reg.answers_at(index.generation, u64::MAX));
    let restarted = IndexRegistry::new();
    restarted.load_all(&engine).expect("load after a restart");
    assert!(restarted.is_suspect(index.generation));
}

/// A generation the catalog records suspect is suspect on a member that
/// never saw the disagreement itself (a follower, a restarted process).
#[test]
fn a_recorded_suspicion_reaches_a_member_that_found_nothing() {
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    reg.apply_integrity(&[integrity(&index, Integrity::Suspect)]);
    assert!(reg.is_suspect(index.generation));
    reg.apply_integrity(&[]);
    assert!(!reg.is_suspect(index.generation));
}

#[test]
fn register_and_lookup() {
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    );
    reg.register_in_memory(index.clone());

    assert_eq!(reg.len(), 1);
    assert_eq!(reg.get("user_email"), Some(index.clone()));
    assert_eq!(reg.get_by_id(index.id), Some(index));
    assert_eq!(reg.indexes_for_label("User").len(), 1);
    assert_eq!(reg.indexes_for_label("Movie").len(), 0);
}

/// A name names the index that holds it now: after a drop and a create of
/// the same name the name finds the new index, and the dropped identity
/// finds nothing, so work bound to it cannot reach its successor.
#[test]
fn a_name_follows_its_index_and_a_dropped_identity_finds_nothing() {
    let reg = IndexRegistry::new();
    let first = btree("user_email", "email", 1);
    reg.register_in_memory(first.clone());
    reg.unregister(first.id);
    assert_eq!(reg.get("user_email"), None);

    let second = btree("user_email", "email", 2);
    reg.register_in_memory(second.clone());
    assert_eq!(reg.get("user_email"), Some(second.clone()));
    assert_eq!(reg.get_by_id(first.id), None);

    // Re-registering an index under another name moves its binding.
    let mut renamed = second.clone();
    renamed.name = Some("by_email".into());
    reg.register_in_memory(renamed.clone());
    assert_eq!(reg.get("user_email"), None);
    assert_eq!(reg.get("by_email"), Some(renamed));
    assert_eq!(reg.len(), 1);
}

#[test]
fn indexes_for_property() {
    let reg = IndexRegistry::new();
    reg.register_in_memory(btree("user_email", "email", 1));
    reg.register_in_memory(btree("user_name", "name", 2));

    let email_idxs = reg.indexes_for_property("User", "email");
    assert_eq!(email_idxs.len(), 1);
    assert_eq!(email_idxs[0].name.as_deref(), Some("user_email"));
}

#[test]
fn distinct_unique_values_are_accepted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    reg.register_in_memory(bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    ));

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("first create");
    create(&reg, &engine, 2, &[("email", s("bob@test.com"))]).expect("second create");
}

/// A second node taking a unique value is refused, naming the index and the
/// node that holds the value.
#[test]
fn a_taken_unique_value_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    reg.register_in_memory(bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    ));

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("first");
    let err =
        create(&reg, &engine, 2, &[("email", s("alice@test.com"))]).expect_err("duplicate email");

    assert!(
        matches!(&err, IndexWriteError::Unique(v) if v.holder == NodeId::from_raw(1)),
        "expected a unique violation held by node 1, got {err:?}"
    );
    assert!(err.to_string().contains("unique constraint violated"));
    assert!(err.to_string().contains("user_email"));
}

#[test]
fn a_changed_value_moves_its_entry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    );
    reg.register_in_memory(index.clone());

    create(&reg, &engine, 1, &[("email", s("old@test.com"))]).expect("create");
    change(
        &reg,
        &engine,
        1,
        "email",
        &[("email", s("old@test.com"))],
        &[("email", s("new@test.com"))],
    )
    .expect("update");

    assert!(lookup(&engine, &index, &[s("old@test.com")]).is_empty());
    assert_eq!(lookup(&engine, &index, &[s("new@test.com")]), vec![1]);
}

#[test]
fn changing_to_a_taken_unique_value_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    reg.register_in_memory(bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    ));

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("create 1");
    create(&reg, &engine, 2, &[("email", s("bob@test.com"))]).expect("create 2");
    let result = change(
        &reg,
        &engine,
        2,
        "email",
        &[("email", s("bob@test.com"))],
        &[("email", s("alice@test.com"))],
    );
    assert!(matches!(result, Err(IndexWriteError::Unique(_))));
}

/// A compound index reads several properties; changing one of them moves
/// the node's single entry to the new combination.
#[test]
fn a_compound_entry_moves_when_one_of_its_properties_changes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::compound(
            "user_city_age",
            "User",
            vec!["city".into(), "age".into()],
        ),
        1,
    );
    reg.register_in_memory(index.clone());

    let before = [("city", s("Oslo")), ("age", Value::Int(30))];
    let after = [("city", s("Oslo")), ("age", Value::Int(31))];
    create(&reg, &engine, 1, &before).expect("create");
    change(&reg, &engine, 1, "age", &before, &after).expect("change");

    assert!(lookup(&engine, &index, &[s("Oslo"), Value::Int(30)]).is_empty());
    assert_eq!(
        lookup(&engine, &index, &[s("Oslo"), Value::Int(31)]),
        vec![1]
    );
}

#[test]
fn a_deleted_node_leaves_the_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    reg.register_in_memory(index.clone());

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("create");
    let props = props(&[("email", s("alice@test.com"))]);
    reg.on_node_deleted(
        &engine,
        &mut txn(&engine),
        &NodeState {
            node_id: NodeId::from_raw(1),
            valid_from: None,
            label: "User",
            value_of: &props_lookup(&props),
        },
        &no_fields,
    )
    .expect("delete");

    assert!(lookup(&engine, &index, &[s("alice@test.com")]).is_empty());
}

/// Only B-tree indexes have entries: a vector index registered alongside
/// for planning and advice is not maintained here.
#[test]
fn a_vector_index_in_the_registry_gets_no_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let vector = bound(
        crate::index::IndexDescriptor::hnsw(
            "user_vec",
            "User",
            "email",
            crate::index::definition::VectorIndexConfig::default(),
        ),
        1,
    );
    reg.register_in_memory(vector.clone());

    assert!(!reg.has_btree_for("User"));
    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("create");
    assert!(all_ids(&engine, &vector).is_empty());
}

/// A definition dropped between its publication and its registration here
/// is not brought back: the statement that published it registers it only
/// while it is still stored, so a DROP that committed first wins.
#[test]
fn a_definition_dropped_before_its_registration_is_not_registered() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let def = LocalIndexStore::new(&engine)
        .publish_definition_txn(
            &mut txn(&engine),
            crate::index::IndexDescriptor::btree("idx1", "User", "email").unique(),
        )
        .expect("publish");
    engine
        .delete(
            coordinode_storage::engine::partition::Partition::Schema,
            &def.schema_key(),
        )
        .expect("the DROP removes the definition");

    let reg = IndexRegistry::new();
    reg.register_published(&engine, def)
        .expect("registering a dropped definition is not an error");
    assert_eq!(reg.get("idx1"), None, "the dropped index came back");
}

#[test]
fn load_all_from_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let store = LocalIndexStore::new(&engine);
    let first = store
        .publish_definition_txn(
            &mut txn(&engine),
            crate::index::IndexDescriptor::btree("idx1", "User", "email").unique(),
        )
        .expect("publish");
    let second = store
        .publish_definition_txn(
            &mut txn(&engine),
            crate::index::IndexDescriptor::btree("idx2", "User", "name"),
        )
        .expect("publish");

    let reg = IndexRegistry::new();
    reg.load_all(&engine).expect("load");
    assert_eq!(reg.len(), 2);
    assert_eq!(reg.get("idx1"), Some(first));
    assert_eq!(reg.get("idx2"), Some(second));
}

#[test]
fn sparse_index_skips_null_on_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("user_bio", "User", "bio").sparse(),
        1,
    );
    reg.register_in_memory(index.clone());

    create(&reg, &engine, 1, &[("bio", Value::Null)]).expect("create");
    assert!(all_ids(&engine, &index).is_empty());
}

// ====== Partial index ======

#[test]
fn partial_index_filters_on_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("active_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEquals {
                property: "status".into(),
                value: "active".into(),
            },
        ),
        1,
    );
    reg.register_in_memory(index.clone());

    create(
        &reg,
        &engine,
        1,
        &[("email", s("alice@test.com")), ("status", s("active"))],
    )
    .expect("create active");
    create(
        &reg,
        &engine,
        2,
        &[("email", s("bob@test.com")), ("status", s("inactive"))],
    )
    .expect("create inactive");

    assert_eq!(all_ids(&engine, &index), vec![1]);
}

/// Changing the property a partial index filters on moves the node into or
/// out of the index, though no indexed value changed.
#[test]
fn a_node_leaves_a_partial_index_when_its_filter_stops_matching() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("active_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEquals {
                property: "status".into(),
                value: "active".into(),
            },
        ),
        1,
    );
    reg.register_in_memory(index.clone());

    let active = [("email", s("alice@test.com")), ("status", s("active"))];
    let inactive = [("email", s("alice@test.com")), ("status", s("inactive"))];
    create(&reg, &engine, 1, &active).expect("create");
    change(&reg, &engine, 1, "status", &active, &inactive).expect("deactivate");

    assert!(all_ids(&engine, &index).is_empty());
}

#[test]
fn partial_index_bool_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("verified_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEqualsBool {
                property: "verified".into(),
                value: true,
            },
        ),
        1,
    );
    reg.register_in_memory(index.clone());

    create(
        &reg,
        &engine,
        1,
        &[
            ("email", s("alice@test.com")),
            ("verified", Value::Bool(true)),
        ],
    )
    .expect("create");
    create(
        &reg,
        &engine,
        2,
        &[
            ("email", s("bob@test.com")),
            ("verified", Value::Bool(false)),
        ],
    )
    .expect("create");

    assert_eq!(all_ids(&engine, &index), vec![1]);
}

#[test]
fn partial_index_exists_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("user_bio_idx", "User", "bio").with_filter(
            super::super::definition::PartialFilter::PropertyExists {
                property: "bio".into(),
            },
        ),
        1,
    );
    reg.register_in_memory(index.clone());

    create(
        &reg,
        &engine,
        1,
        &[("name", s("Alice")), ("bio", s("Developer"))],
    )
    .expect("create");
    create(
        &reg,
        &engine,
        2,
        &[("name", s("Bob")), ("bio", Value::Null)],
    )
    .expect("create");

    assert_eq!(all_ids(&engine, &index), vec![1]);
}

#[test]
fn partial_filter_matches_function() {
    use super::super::definition::PartialFilter;

    let props = vec![
        ("status".to_string(), Value::String("active".into())),
        ("age".to_string(), Value::Int(30)),
    ];

    assert!(
        PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "active".into(),
        }
        .matches(&props)
    );
    assert!(
        !PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "inactive".into(),
        }
        .matches(&props)
    );
    assert!(
        PartialFilter::PropertyEqualsInt {
            property: "age".into(),
            value: 30,
        }
        .matches(&props)
    );
    assert!(
        PartialFilter::PropertyExists {
            property: "status".into(),
        }
        .matches(&props)
    );
    assert!(
        !PartialFilter::PropertyExists {
            property: "missing".into(),
        }
        .matches(&props)
    );
}
