use std::sync::Arc;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_modality::{LocalNodeStore, NodeStore as _};

use super::*;
use crate::index::lifecycle::test_env::{TestEnv, commit};
use crate::index::{BuildFailure, IndexBuildOutcome, IndexDescriptor, Mismatch};

/// A member with two users and a built `email` index, its findings kept by
/// `registry`, and a built index over a label with no nodes.
struct Member {
    env: Arc<TestEnv>,
    email: IndexDefinition,
    empty: IndexDefinition,
}

fn member() -> Member {
    let env = TestEnv::open();
    let mut fields = FieldInterner::new();
    fields.intern("email");
    env.set_fields(&fields);
    for (id, email) in [(1, "a@x"), (2, "b@x")] {
        let mut record = NodeRecord::new("User");
        let field = env.fields_now().lookup("email").expect("email field");
        record.set(field, Value::String(email.into()));
        let mut txn = env.begin();
        LocalNodeStore
            .put(&mut txn, 1, NodeId::from_raw(id), &record)
            .expect("put");
        commit(&mut txn).expect("commit node");
    }
    let builds = env.service(1);
    let mut built = Vec::new();
    for (name, label) in [("user_email", "User"), ("ghost_email", "Ghost")] {
        let def = admit(&env, IndexDescriptor::btree(name, label, "email"));
        builds.submit(def.generation).expect("submit");
        assert!(matches!(
            builds.wait_here(def.generation),
            Some(IndexBuildOutcome::Published { .. })
        ));
        built.push(
            LocalIndexStore::new(&env.engine)
                .load_definition(def.id)
                .expect("load")
                .expect("published"),
        );
    }
    builds.shutdown();
    let empty = built.pop().expect("empty index");
    let email = built.pop().expect("email index");
    Member { env, email, empty }
}

/// Publish `descriptor` building with its build admitted, as CREATE INDEX
/// does.
fn admit(env: &TestEnv, descriptor: IndexDescriptor) -> IndexDefinition {
    let store = LocalIndexStore::new(&env.engine);
    let mut descriptor = descriptor;
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
            &crate::index::IndexBuildRecord::accepted(
                def.id,
                def.generation,
                BuildFailure::Withdraw,
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

/// A qualifier over `member`'s findings. The test environment's registry is
/// not shared, so its findings are copied into one that is.
fn qualifier(member: &Member) -> (CopyQualifier, Arc<IndexRegistry>) {
    let registry = Arc::new(IndexRegistry::new());
    registry.load_all(&member.env.engine).expect("load");
    (CopyQualifier::new(Arc::clone(&registry)), registry)
}

/// The states `inventory` (as sealed) gives each generation.
fn states(inventory: &[u8]) -> FxHashMap<u64, CopyState> {
    rmp_serde::from_slice::<Vec<Entry>>(inventory)
        .expect("decode")
        .into_iter()
        .map(|e| (e.generation, e.state))
        .collect()
}

fn stray() -> Mismatch {
    Mismatch::Extra {
        node: 7,
        valid_from: None,
        tuple: vec![1],
    }
}

/// A built copy is qualified by its build's proof, an empty index's too;
/// the inventory lists both.
#[test]
fn a_built_copy_is_qualified_and_an_empty_index_is_listed() {
    let member = member();
    let (qualifier, _) = qualifier(&member);
    let engine = &member.env.engine;
    let sealed = qualifier
        .seal(engine, &qualifier.capture(engine).expect("capture"))
        .expect("seal");
    let states = states(&sealed);
    assert_eq!(states.len(), 2, "every B-tree generation: {states:?}");
    for def in [&member.email, &member.empty] {
        assert!(
            matches!(
                states.get(&def.generation.as_raw()),
                Some(CopyState::Qualified(_))
            ),
            "{def}: {states:?}"
        );
    }
}

/// A copy found disagreeing is excluded, whether its finding is stored or
/// held only in memory; one found after the capture is excluded when the
/// capture is sealed; a copy with no proof is unknown.
#[test]
fn findings_exclude_and_missing_proof_is_unknown() {
    let before = member();
    let (before_qualifier, registry) = qualifier(&before);
    let engine = &before.env.engine;
    registry.report_mismatch(engine, &before.email, stray());
    let sealed = before_qualifier
        .seal(engine, &before_qualifier.capture(engine).expect("capture"))
        .expect("seal");
    assert_eq!(
        states(&sealed).get(&before.email.generation.as_raw()),
        Some(&CopyState::Excluded),
        "found before the capture"
    );

    let after = member();
    let (after_qualifier, registry) = qualifier(&after);
    let engine = &after.env.engine;
    let captured = after_qualifier.capture(engine).expect("capture");
    registry.report_mismatch(engine, &after.email, stray());
    assert_eq!(
        states(&after_qualifier.seal(engine, &captured).expect("seal"))
            .get(&after.email.generation.as_raw()),
        Some(&CopyState::Excluded),
        "found after the capture, before the transfer sent it"
    );

    // No proof kept: a copy whose evidence went is unknown, not healthy.
    let unproved = member();
    let store = LocalIndexStore::new(&unproved.env.engine);
    store.record_transfer(&[], &[]).expect("drop every proof");
    let (unproved_qualifier, _) = self::qualifier(&unproved);
    let engine = &unproved.env.engine;
    let sealed = unproved_qualifier
        .seal(
            engine,
            &unproved_qualifier.capture(engine).expect("capture"),
        )
        .expect("seal");
    assert_eq!(
        states(&sealed).get(&unproved.email.generation.as_raw()),
        Some(&CopyState::Unknown)
    );
}

/// A finding made after the capture and lifted again before the seal leaves
/// no mark behind, yet the capture may hold the copy as it was before the
/// repair: the copy is excluded, not sent as qualified.
#[test]
fn a_finding_made_and_lifted_after_the_capture_excludes_the_copy() {
    let member = member();
    let (qualifier, registry) = qualifier(&member);
    let engine = &member.env.engine;
    let captured = qualifier.capture(engine).expect("capture");
    registry.report_mismatch(engine, &member.email, stray());
    let revision = registry.local_revision(member.email.generation);
    assert!(
        registry
            .verified_here(engine, &member.email, revision, 1)
            .expect("lift"),
        "the mark is lifted"
    );
    assert_eq!(registry.local_revision(member.email.generation), 0);
    assert_eq!(
        states(&qualifier.seal(engine, &captured).expect("seal"))
            .get(&member.email.generation.as_raw()),
        Some(&CopyState::Excluded)
    );
}

/// Two readings of a copy's proof bracket a history export: a finding made
/// and lifted between them changes the second, so the export carries no
/// proof for what it may have read while the copy was wrong.
#[test]
fn a_finding_made_and_lifted_between_two_readings_changes_the_proof() {
    let member = member();
    let (qualifier, registry) = qualifier(&member);
    let engine = &member.env.engine;
    let generation = member.email.generation.as_raw();
    let before = qualifier
        .copy_proof(engine, generation)
        .expect("read")
        .expect("a proved copy");
    assert_eq!(
        qualifier.copy_proof(engine, generation).expect("read"),
        Some(before.clone()),
        "nothing happened between the readings"
    );
    registry.report_mismatch(engine, &member.email, stray());
    assert_eq!(
        qualifier.copy_proof(engine, generation).expect("read"),
        None
    );
    let revision = registry.local_revision(member.email.generation);
    assert!(
        registry
            .verified_here(engine, &member.email, revision, 2)
            .expect("lift")
    );
    let after = qualifier
        .copy_proof(engine, generation)
        .expect("read")
        .expect("proved again");
    assert_ne!(after, before, "the readings around the finding differ");
}

/// A copy whose use guard a previous run left open is excluded from the
/// next capture, its proof notwithstanding: the guard says the copy served
/// while its process died, which nothing since has checked.
#[test]
fn a_copy_left_guarded_by_a_dead_run_is_excluded() {
    let member = member();
    let engine = &member.env.engine;
    let store = LocalIndexStore::new(engine);
    let copy = store
        .local_copy(member.email.generation)
        .expect("a copy here");
    store.arm_here(&[copy]).expect("in use when the run died");
    let reopened = Arc::new(IndexRegistry::new());
    reopened.recover_local_guards(engine).expect("recover");
    reopened.load_all(engine).expect("load");
    let qualifier = CopyQualifier::new(reopened);
    let states = states(
        &qualifier
            .seal(engine, &qualifier.capture(engine).expect("capture"))
            .expect("seal"),
    );
    assert_eq!(
        states.get(&member.email.generation.as_raw()),
        Some(&CopyState::Excluded),
        "the guarded copy: {states:?}"
    );
    assert!(
        matches!(
            states.get(&member.empty.generation.as_raw()),
            Some(CopyState::Qualified(_))
        ),
        "the copy no run used keeps its proof: {states:?}"
    );
}

/// The records, catalog and index copies of `sender` installed on
/// `receiver` as a Raft snapshot with `inventory`.
fn install(receiver: &StorageEngine, sender: &StorageEngine, inventory: Option<&[u8]>) {
    use coordinode_storage::engine::partition::Partition::{Idx, Node, Schema};
    receiver
        .install_raft_image(
            &[Node, Schema, Idx],
            |p| sender.replicated_rows(p),
            10,
            b"payload",
            &[],
            inventory,
        )
        .expect("install");
}

/// The generations `engine` keeps a mark of, and a proof of.
fn marked_and_proved(engine: &StorageEngine) -> (Vec<u64>, Vec<u64>) {
    let store = LocalIndexStore::new(engine);
    let mut marked: Vec<u64> = store
        .list_unfit_here()
        .expect("marks")
        .into_iter()
        .map(|c| c.generation.as_raw())
        .collect();
    let mut proved: Vec<u64> = store
        .list_qualified_here()
        .expect("proofs")
        .into_iter()
        .map(|(g, _)| g.as_raw())
        .collect();
    marked.sort_unstable();
    proved.sort_unstable();
    (marked, proved)
}

/// A receiver keeps the proof of each copy the inventory qualifies, rebound
/// to its own installation, and marks the copy the sender excluded; nothing
/// is pending afterwards. Without an inventory every copy is marked.
#[test]
fn a_receiver_keeps_qualified_proofs_and_marks_the_rest() {
    let sender = member();
    let (qualifier, registry) = qualifier(&sender);
    registry.report_mismatch(&sender.env.engine, &sender.email, stray());
    let sealed = qualifier
        .seal(
            &sender.env.engine,
            &qualifier.capture(&sender.env.engine).expect("capture"),
        )
        .expect("seal");

    let receiver = TestEnv::open();
    let receiver_registry = Arc::new(IndexRegistry::new());
    receiver
        .engine
        .set_index_copy_qualifier(Arc::new(CopyQualifier::new(Arc::clone(&receiver_registry))));
    install(&receiver.engine, &sender.env.engine, Some(&sealed));
    let (marked, proved) = marked_and_proved(&receiver.engine);
    assert_eq!(
        marked,
        vec![sender.email.generation.as_raw()],
        "excluded there"
    );
    assert_eq!(
        proved,
        vec![sender.empty.generation.as_raw()],
        "qualified there"
    );
    let proof = LocalIndexStore::new(&receiver.engine)
        .list_qualified_here()
        .expect("proofs")[0]
        .1;
    assert_eq!(
        Some(proof.installation),
        receiver.engine.installation_of(sender.empty.generation),
        "rebound to the receiver's installation"
    );
    assert!(!receiver.engine.index_transfer_pending().expect("read"));

    let bare = TestEnv::open();
    bare.engine
        .set_index_copy_qualifier(Arc::new(CopyQualifier::new(Arc::new(IndexRegistry::new()))));
    install(&bare.engine, &sender.env.engine, None);
    let (marked, proved) = marked_and_proved(&bare.engine);
    let mut both = vec![
        sender.email.generation.as_raw(),
        sender.empty.generation.as_raw(),
    ];
    both.sort_unstable();
    assert_eq!(marked, both, "no inventory: every copy is checked here");
    assert!(proved.is_empty());
}

/// A receiver with its qualifier registered over a fresh registry.
fn receiver() -> Arc<TestEnv> {
    let receiver = TestEnv::open();
    receiver
        .engine
        .set_index_copy_qualifier(Arc::new(CopyQualifier::new(Arc::new(IndexRegistry::new()))));
    receiver
}

/// The sender's inventory, captured and sealed, with nothing found.
fn sealed_inventory(sender: &Member) -> Vec<u8> {
    let (qualifier, _) = qualifier(sender);
    let engine = &sender.env.engine;
    qualifier
        .seal(engine, &qualifier.capture(engine).expect("capture"))
        .expect("seal")
}

/// A proof of another interpretation (another key layout here) does not
/// qualify the receiver's copy: the evidence must describe the definition
/// installed with it.
#[test]
fn a_proof_of_another_interpretation_does_not_qualify() {
    let sender = member();
    let mut entries: Vec<Entry> =
        rmp_serde::from_slice(&sealed_inventory(&sender)).expect("decode");
    for entry in &mut entries {
        if let CopyState::Qualified(proof) = &mut entry.state {
            proof.layout += 1;
        }
    }
    let altered = rmp_serde::to_vec(&entries).expect("encode");
    let receiver = receiver();
    install(&receiver.engine, &sender.env.engine, Some(&altered));
    let (marked, proved) = marked_and_proved(&receiver.engine);
    assert_eq!(marked.len(), 2, "both copies are checked here: {marked:?}");
    assert!(proved.is_empty());
}

/// A finding the receiver stored against its own copy before the transfer
/// is not discharged by the sender's proof: the copy stays unfit here.
#[test]
fn a_receivers_own_finding_outlives_the_senders_proof() {
    let sender = member();
    let inventory = sealed_inventory(&sender);
    let receiver = receiver();
    let store = LocalIndexStore::new(&receiver.engine);
    let copy = store
        .bind_local_copy(sender.email.generation)
        .expect("bind");
    store.mark_unfit_here(copy).expect("the receiver's finding");
    install(&receiver.engine, &sender.env.engine, Some(&inventory));
    let (marked, proved) = marked_and_proved(&receiver.engine);
    assert_eq!(
        marked,
        vec![sender.email.generation.as_raw()],
        "its own finding stands"
    );
    assert_eq!(
        proved.len(),
        2,
        "the proofs are kept for when it is checked"
    );
    let reopened = IndexRegistry::new();
    reopened.load_all(&receiver.engine).expect("load");
    assert!(reopened.is_suspect(sender.email.generation));
    assert!(!reopened.is_suspect(sender.empty.generation));
}

/// A receiver that checked a copy it received unproved keeps the proof of
/// its own check, and a transfer it sends later carries that proof.
#[test]
fn a_receivers_own_proof_travels_with_its_next_transfer() {
    let sender = member();
    let receiver = receiver();
    install(&receiver.engine, &sender.env.engine, None);
    let registry = Arc::new(IndexRegistry::new());
    registry
        .recover_local_guards(&receiver.engine)
        .expect("recover");
    registry.load_all(&receiver.engine).expect("load");
    let def = LocalIndexStore::new(&receiver.engine)
        .load_definition(sender.email.id)
        .expect("load")
        .expect("installed definition");
    let revision = registry.local_revision(def.generation);
    assert!(
        registry
            .verified_here(&receiver.engine, &def, revision, 7)
            .expect("a check here proves it"),
        "lifted"
    );
    let qualifier = CopyQualifier::new(Arc::clone(&registry));
    let engine = &receiver.engine;
    let sealed = qualifier
        .seal(engine, &qualifier.capture(engine).expect("capture"))
        .expect("seal");
    assert!(
        matches!(
            states(&sealed).get(&def.generation.as_raw()),
            Some(CopyState::Qualified(proof)) if proof.verified_at == 7
        ),
        "its own check's proof"
    );
    assert_eq!(
        states(&sealed).get(&sender.empty.generation.as_raw()),
        Some(&CopyState::Excluded),
        "the copy it has not checked yet stays excluded"
    );
}

/// An install interrupted before its inventory was taken leaves the
/// transfer pending: the next open marks every copy, and a copy the
/// receiver had a stale proof of is not taken as proved by it.
#[test]
fn an_interrupted_transfer_marks_every_copy_at_the_next_open() {
    let sender = member();
    let receiver = TestEnv::open();
    // No qualifier registered: the install leaves the transfer pending, as
    // a crash before the inventory is taken does.
    install(&receiver.engine, &sender.env.engine, None);
    assert!(receiver.engine.index_transfer_pending().expect("read"));
    let reopened = IndexRegistry::new();
    reopened
        .recover_local_guards(&receiver.engine)
        .expect("recover");
    reopened.load_all(&receiver.engine).expect("load");
    assert!(reopened.is_suspect(sender.email.generation));
    assert!(reopened.is_suspect(sender.empty.generation));
    assert!(!receiver.engine.index_transfer_pending().expect("read"));
}
