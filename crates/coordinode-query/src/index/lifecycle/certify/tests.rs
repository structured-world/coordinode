use super::*;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_modality::{IndexDescriptor, LocalSchemaStore, PartState};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};

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

fn commit(txn: &mut Transaction<'_>) -> Result<(), CommitError> {
    let wc = WriteConcern::majority();
    txn.commit(&CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
    })
    .map(drop)
}

/// The label lookup keeps parts of two changes: one put in force earlier,
/// whose fold is about to drop its parts, and one being concluded now. The
/// fold commits after the conclusion has listed the definitions and before
/// it reads them. The conclusion's commit is refused as a write conflict,
/// and its retry, as the certification runs it, concludes the record the
/// fold left: the folded parts stay dropped and the concluded ones are in
/// force, nothing else.
#[test]
fn a_conclusion_never_brings_back_parts_a_fold_dropped() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = std::sync::Arc::new(test_engine(dir.path()));
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1)));
    let indexes = LocalIndexStore::new(&engine);
    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    let mut lookup = indexes
        .publish_label_lookup_txn(&mut txn, IndexDescriptor::label_lookup())
        .expect("publish the lookup")
        .expect("a lookup");
    commit(&mut txn).expect("commit the lookup");
    let schemas = LocalSchemaStore::new(&engine);
    let (thing, gadget) = (
        schemas.bind_label("Thing").expect("bind"),
        schemas.bind_label("Gadget").expect("bind"),
    );
    let (folded, concluded) = (GenerationId::from_raw(1_000), GenerationId::from_raw(2_000));
    lookup.propose_parts(
        thing,
        folded,
        GenerationId::from_raw(1_001),
        GenerationId::from_raw(1_002),
    );
    lookup.conclude_parts(folded, PartState::InForce);
    lookup.propose_parts(
        gadget,
        concluded,
        GenerationId::from_raw(2_001),
        GenerationId::from_raw(2_002),
    );
    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    indexes
        .put_definition_txn(&mut txn, &lookup)
        .expect("record the parts");
    commit(&mut txn).expect("commit the parts");

    // The fold of the first change, between the listing and the reads.
    let (fold_engine, fold_oracle) = (
        std::sync::Arc::clone(&engine),
        std::sync::Arc::clone(&oracle),
    );
    let id = lookup.id;
    *AFTER_LISTING.lock() = Some(Box::new(move || {
        let indexes = LocalIndexStore::new(&fold_engine);
        let mut def = indexes
            .load_definition(id)
            .expect("load")
            .expect("the lookup");
        def.drop_part(GenerationId::from_raw(1_001));
        def.drop_part(GenerationId::from_raw(1_002));
        let mut txn = Transaction::begin(&fold_engine, Some(&fold_oracle), fold_oracle.next());
        indexes.put_definition_txn(&mut txn, &def).expect("fold");
        commit(&mut txn).expect("commit the fold");
    }));

    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    stage_selection_conclusion(&engine, &mut txn, concluded, PartState::InForce)
        .expect("stage the conclusion");
    assert!(AFTER_LISTING.lock().is_none(), "the fold ran in the window");
    assert!(
        matches!(commit(&mut txn), Err(CommitError::Conflict(_))),
        "the fold's commit after this one's snapshot refuses it"
    );
    // The retry, as the certification runs it.
    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    stage_selection_conclusion(&engine, &mut txn, concluded, PartState::InForce)
        .expect("stage the conclusion again");
    commit(&mut txn).expect("the retry commits");

    let def = indexes
        .load_definition(id)
        .expect("load")
        .expect("the lookup");
    let mut parts: Vec<(u64, PartState)> = def
        .selection
        .iter()
        .map(|p| (p.part.as_raw(), p.state))
        .collect();
    parts.sort_unstable_by_key(|(part, _)| *part);
    assert_eq!(
        parts,
        [(2_001, PartState::InForce), (2_002, PartState::InForce)],
        "exactly the concluded parts, in force"
    );
}
