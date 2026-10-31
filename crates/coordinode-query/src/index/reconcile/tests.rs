use super::*;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_modality::{IndexDescriptor, LocalSchemaStore, PartState};
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

fn direct(engine: &StorageEngine) -> Transaction<'_> {
    Transaction::new(engine, None, Timestamp::ZERO, None)
}

/// A page's budget holds what the pass carries into it before the page
/// takes anything: carried state that alone does not fit refuses the page
/// at once, and state that fits leaves the rest of the bytes for the page.
#[test]
fn a_page_holds_what_the_pass_carries_first() {
    assert!(matches!(
        page_budget(1024, 2048),
        Err(coordinode_core::budget::BudgetStop::Memory {
            requested: 2048,
            used: 0,
            limit: 1024
        })
    ));
    let budget = page_budget(1024, 1000).expect("fits");
    assert_eq!(budget.memory_used(), 1000);
    assert!(
        budget.reserve(25).is_err(),
        "the page took past what was left"
    );
    assert!(budget.reserve(24).is_ok());
}

/// A change put in force keeps its parts until its fold brings them back to
/// the index's own entries. A change of the same label published meanwhile
/// is refused, as it would be while the first is certified: two changes'
/// parts never stand side by side, so a reader or writer selects at most one
/// gained and one retired part. Before the check covered the concluded
/// change too, the second change was admitted beside the unfolded parts.
#[test]
fn a_change_waits_for_the_previous_one_to_be_folded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let indexes = LocalIndexStore::new(&engine);
    let mut lookup = indexes
        .publish_label_lookup_txn(&mut direct(&engine), IndexDescriptor::label_lookup())
        .expect("publish the lookup")
        .expect("a lookup");
    let thing = LocalSchemaStore::new(&engine)
        .bind_label("Thing")
        .expect("bind label");
    let first = GenerationId::from_raw(1_000);
    lookup.propose_parts(
        thing,
        first,
        GenerationId::from_raw(1_001),
        GenerationId::from_raw(1_002),
    );
    lookup.conclude_parts(first, PartState::InForce);
    indexes
        .put_definition_txn(&mut direct(&engine), &lookup)
        .expect("record the concluded parts");

    let refused = propose_parts_txn(
        &engine,
        &mut direct(&engine),
        GenerationId::from_raw(2_000),
        &[(thing, "Thing".to_string())],
    );

    match refused {
        Err(StoreError::Invariant(why)) => assert!(why.contains("not folded"), "{why}"),
        other => panic!("the second change was admitted beside unfolded parts: {other:?}"),
    }
}
