//! Integration tests: the field dictionary that maps property names to the
//! integer ids stored data is encoded with.
//!
//! A stored property carries an id, not its name, so the binding that gives
//! the id its meaning must exist durably before any data that uses it, must
//! never change afterwards, and must never be invented by a path that only
//! reads. These tests cover each way the binding can go missing or change:
//! a failed binding write after its data, concurrent writers, index rebuilds
//! on open, and a damaged or missing dictionary.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use coordinode_core::graph::types::{Value, VectorMetric};
use coordinode_core::txn::proposal::{
    Mutation, ProposalError, ProposalOutcome, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_query::index::VectorIndexConfig;
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_storage::Guard as _;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// Key prefixes the field dictionary has been kept under.
const DICTIONARY_KEYS: [&[u8]; 2] = [b"meta:field_interner", b"ids:field:"];

/// Whether `m` writes the field dictionary.
fn writes_field_dictionary(m: &Mutation) -> bool {
    use coordinode_core::txn::proposal::MetadataCommand;
    match m {
        Mutation::Put { key, .. } | Mutation::Merge { key, .. } => {
            DICTIONARY_KEYS.iter().any(|p| key.starts_with(p))
        }
        Mutation::Command(
            MetadataCommand::RegisterFields { .. } | MetadataCommand::AdoptFields { .. },
        ) => true,
        _ => false,
    }
}

/// A pipeline that refuses every proposal writing the field dictionary and
/// applies everything else: the state a crash leaves when it lands between a
/// write and the dictionary entry that write needed.
struct RefuseDictionaryWrites {
    inner: OwnedLocalProposalPipeline,
}

impl ProposalPipeline for RefuseDictionaryWrites {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        if proposal.mutations.iter().any(writes_field_dictionary) {
            return Err(ProposalError::Storage(
                "field dictionary write refused".into(),
            ));
        }
        self.inner.propose_and_wait(proposal)
    }
}

/// Whether `m` writes a node record.
fn writes_node(m: &Mutation) -> bool {
    use coordinode_core::txn::proposal::PartitionId;
    matches!(
        m,
        Mutation::Put {
            partition: PartitionId::Node,
            ..
        } | Mutation::Merge {
            partition: PartitionId::Node,
            ..
        }
    )
}

/// A pipeline that applies the field dictionary's writes and refuses every
/// proposal writing a node: the state a crash leaves when it lands after a
/// statement's names were bound and before its data committed.
struct RefuseNodeWrites {
    inner: OwnedLocalProposalPipeline,
}

impl ProposalPipeline for RefuseNodeWrites {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        if proposal.mutations.iter().any(writes_node) {
            return Err(ProposalError::Storage("node write refused".into()));
        }
        self.inner.propose_and_wait(proposal)
    }
}

/// A pipeline that applies the first field dictionary write and then reports
/// it as timed out: the unknown outcome a proposer sees when its connection
/// drops after the write committed.
struct ApplyThenTimeOutOnce {
    inner: OwnedLocalProposalPipeline,
    fired: std::sync::atomic::AtomicBool,
}

impl ProposalPipeline for ApplyThenTimeOutOnce {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        let outcome = self.inner.propose_and_wait(proposal)?;
        if proposal.mutations.iter().any(writes_field_dictionary)
            && !self.fired.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(ProposalError::Timeout { retries: 3 });
        }
        Ok(outcome)
    }
}

fn vector_config() -> VectorIndexConfig {
    VectorIndexConfig {
        dimensions: 2,
        metric: VectorMetric::L2,
        ..VectorIndexConfig::default()
    }
}

/// Every key the dictionary is stored under.
fn dictionary_keys(db: &Database) -> Vec<Vec<u8>> {
    let mut keys = Vec::new();
    for prefix in DICTIONARY_KEYS {
        for guard in db.engine().prefix_scan(Partition::Schema, prefix).unwrap() {
            let (key, _) = guard.into_inner().unwrap();
            keys.push(key.to_vec());
        }
    }
    keys
}

/// A write whose property name could not be recorded commits nothing: data
/// whose ids mean nothing is unreadable, so the binding has to be durable
/// before the data, and a failure to record it fails the whole write.
#[test]
fn a_failed_dictionary_write_commits_no_data() {
    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    let pipeline: Arc<dyn ProposalPipeline> = Arc::new(RefuseDictionaryWrites {
        inner: OwnedLocalProposalPipeline::new(&engine),
    });
    let mut db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();

    let err = db.execute_cypher("CREATE (:Orphan {colour: 'red'})");
    assert!(
        err.is_err(),
        "the write must fail with its binding: {err:?}"
    );

    let rows = db.execute_cypher("MATCH (n:Orphan) RETURN n").unwrap();
    assert!(
        rows.is_empty(),
        "a node whose property name was never recorded was committed: {rows:?}"
    );
}

/// A binding whose statement never committed its data stays bound, with the
/// same id, across a reopen: the next write naming it reuses the id, and a
/// name introduced later takes the next one. A binding is durable on its
/// own, so losing the data after it costs an id, never a meaning.
#[test]
fn a_binding_whose_data_never_committed_is_kept_and_reused() {
    let dir = tempfile::tempdir().unwrap();
    let config = || {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
    };
    let orphan_id = {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = Arc::new(StorageEngine::open_embedded(&config(), oracle.clone()).unwrap());
        let pipeline: Arc<dyn ProposalPipeline> = Arc::new(RefuseNodeWrites {
            inner: OwnedLocalProposalPipeline::new(&engine),
        });
        let mut db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();
        let err = db.execute_cypher("CREATE (:Lost {orphan: 1})");
        assert!(err.is_err(), "the node write was refused: {err:?}");
        let id = db.interner().unwrap().lookup("orphan");
        assert!(id.is_some(), "the name was bound before its data");
        id
    };

    let mut db = Database::open(dir.path()).unwrap();
    assert_eq!(
        db.interner().unwrap().lookup("orphan"),
        orphan_id,
        "the binding survives the reopen unchanged"
    );
    assert!(
        db.execute_cypher("MATCH (n:Lost) RETURN n")
            .unwrap()
            .is_empty(),
        "the refused node was not committed"
    );
    db.execute_cypher("CREATE (:Found {orphan: 7, later: 8})")
        .unwrap();
    let interner = db.interner().unwrap();
    assert_eq!(interner.lookup("orphan"), orphan_id, "the id is reused");
    assert!(
        interner.lookup("later") > orphan_id,
        "a later name takes a later id"
    );
    let rows = db
        .execute_cypher("MATCH (n:Found) RETURN n.orphan AS o, n.later AS l")
        .unwrap();
    assert_eq!(rows[0].get("o"), Some(&Value::Int(7)));
    assert_eq!(rows[0].get("l"), Some(&Value::Int(8)));
}

/// A reader whose dictionary view predates a binding made through another
/// handle on the same engine (a replica applying the leader's log is the
/// same shape) still reads the property: the view catches up before the
/// statement resolves ids, so a stored id it has not seen yet is never read
/// as an absent property.
#[test]
fn a_reader_with_an_older_view_reads_a_name_bound_elsewhere() {
    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_embedded(&config, oracle.clone()).unwrap());
    let open = || {
        let pipeline: Arc<dyn ProposalPipeline> =
            Arc::new(OwnedLocalProposalPipeline::new(&engine));
        Database::from_engine(dir.path(), Arc::clone(&engine), oracle.clone(), pipeline).unwrap()
    };
    let mut reader = open();
    let mut writer = open();

    reader.execute_cypher("CREATE (:Seen {old: 1})").unwrap();
    assert!(
        reader.interner().unwrap().lookup("fresh").is_none(),
        "the reader's view has not seen the name yet"
    );
    writer
        .execute_cypher("CREATE (:Seen {old: 2, fresh: 'new'})")
        .unwrap();

    let rows = reader
        .execute_cypher("MATCH (n:Seen {old: 2}) RETURN n.fresh AS f")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("f"),
        Some(&Value::String("new".into())),
        "the reader resolved the id bound through the other handle"
    );
}

/// A registration whose outcome the proposer never learned (it committed, the
/// reply was lost) fails its statement, and the retried statement finds the
/// names already bound: each keeps the one id the first attempt decided, and
/// no id is spent twice.
#[test]
fn a_retry_after_an_unknown_registration_outcome_reuses_its_ids() {
    use coordinode_storage::engine::metadata::field_frontier;

    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_embedded(&config, oracle.clone()).unwrap());
    let pipeline: Arc<dyn ProposalPipeline> = Arc::new(ApplyThenTimeOutOnce {
        inner: OwnedLocalProposalPipeline::new(&engine),
        fired: std::sync::atomic::AtomicBool::new(false),
    });
    let mut db = Database::from_engine(dir.path(), Arc::clone(&engine), oracle, pipeline).unwrap();

    let first = db.execute_cypher("CREATE (:Retry {alpha: 1, beta: 2})");
    assert!(first.is_err(), "the unknown outcome fails the statement");
    let bound = db.interner().unwrap();
    let (alpha, beta) = (bound.lookup("alpha"), bound.lookup("beta"));
    assert!(
        alpha.is_some() && beta.is_some(),
        "the first attempt did bind"
    );
    assert!(
        db.execute_cypher("MATCH (n:Retry) RETURN n")
            .unwrap()
            .is_empty(),
        "no data went with the failed statement"
    );

    db.execute_cypher("CREATE (:Retry {alpha: 1, beta: 2})")
        .unwrap();
    let after = db.interner().unwrap();
    assert_eq!(after.lookup("alpha"), alpha, "the retry reuses alpha's id");
    assert_eq!(after.lookup("beta"), beta, "the retry reuses beta's id");
    assert_eq!(
        field_frontier(&engine).unwrap(),
        alpha.max(beta).unwrap(),
        "no id was spent a second time"
    );
    let rows = db
        .execute_cypher("MATCH (n:Retry) RETURN n.alpha AS a, n.beta AS b")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("a"), Some(&Value::Int(1)));
    assert_eq!(rows[0].get("b"), Some(&Value::Int(2)));
}

/// Vector indexes rebuilt on open resolve their label and property through
/// the ids the data was written with. Registering names during the rebuild
/// hands them out in the order the index definitions sort, not the order the
/// data used, so a stored vector would be read back as another property.
#[test]
fn vector_indexes_rebuilt_on_open_keep_the_ids_their_data_uses() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        // Defined in the reverse of the order their names sort in.
        db.create_vector_index("z_first", "First", "p_first", vector_config())
            .unwrap();
        db.create_vector_index("a_second", "Second", "p_second", vector_config())
            .unwrap();
        db.execute_cypher("CREATE (:First {p_first: [1.0, 0.0]})")
            .unwrap();
        db.execute_cypher("CREATE (:Second {p_second: [0.0, 1.0]})")
            .unwrap();
    }

    let mut db = Database::open(dir.path()).unwrap();
    let rows = db
        .execute_cypher("MATCH (n:First) RETURN n.p_first AS v, n.p_second AS other")
        .unwrap();
    assert_eq!(rows.len(), 1);
    // No schema declares the column VECTOR, so the list literal is stored as
    // an array of floats.
    assert_eq!(
        rows[0].get("v"),
        Some(&Value::Array(vec![Value::Float(1.0), Value::Float(0.0)])),
        "the vector of :First after a reopen"
    );
    assert_eq!(
        rows[0].get("other"),
        Some(&Value::Null),
        ":First never had p_second"
    );
    let rows = db
        .execute_cypher("MATCH (n:Second) RETURN n.p_second AS v")
        .unwrap();
    assert_eq!(
        rows[0].get("v"),
        Some(&Value::Array(vec![Value::Float(0.0), Value::Float(1.0)]))
    );
}

/// Writers introducing new property names at the same time each keep their
/// binding: none of them overwrites another's, before or after a reopen.
#[test]
fn concurrent_new_property_names_all_survive_a_reopen() {
    const WRITERS: usize = 8;
    const PER_WRITER: usize = 25;
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        std::thread::scope(|scope| {
            for w in 0..WRITERS {
                let db = &db;
                scope.spawn(move || {
                    for i in 0..PER_WRITER {
                        let name = format!("p_{w}_{i}");
                        let query = format!("CREATE (:Conc {{tag: '{name}', {name}: {i}}})");
                        db.execute_cypher_shared(&query, None, None, None, None)
                            .unwrap();
                    }
                });
            }
        });
    }

    let mut db = Database::open(dir.path()).unwrap();
    for w in 0..WRITERS {
        for i in 0..PER_WRITER {
            let name = format!("p_{w}_{i}");
            let rows = db
                .execute_cypher(&format!(
                    "MATCH (n:Conc {{tag: '{name}'}}) RETURN n.{name} AS v"
                ))
                .unwrap();
            assert_eq!(rows.len(), 1, "node {name}");
            assert_eq!(
                rows[0].get("v"),
                Some(&Value::Int(i as i64)),
                "property {name} after a reopen"
            );
        }
    }
}

/// A dictionary that no longer decodes refuses the open. Starting from an
/// empty one would serve every stored property as absent and hand the stored
/// ids out again to new names.
#[test]
fn a_damaged_field_dictionary_refuses_to_open() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.execute_cypher("CREATE (:Kept {name: 'alice'})").unwrap();
        for key in dictionary_keys(&db) {
            db.engine()
                .put(Partition::Schema, &key, &[0xFF, 0x01])
                .unwrap();
        }
        db.persist().unwrap();
    }

    let opened = Database::open(dir.path());
    assert!(
        opened.is_err(),
        "a damaged dictionary opened as if the database were new"
    );
}

/// A dictionary that is gone while stored properties still use it refuses
/// the open, for the same reason a damaged one does.
#[test]
fn a_missing_field_dictionary_with_stored_properties_refuses_to_open() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.execute_cypher("CREATE (:Kept {name: 'alice'})").unwrap();
        for key in dictionary_keys(&db) {
            db.engine().delete(Partition::Schema, &key).unwrap();
        }
        db.persist().unwrap();
    }

    let opened = Database::open(dir.path());
    assert!(
        opened.is_err(),
        "a lost dictionary opened as if the database were new"
    );
}

/// A database whose nodes carry no properties needs no dictionary, and opens
/// with none.
#[test]
fn nodes_without_properties_need_no_dictionary() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.execute_cypher("CREATE (:Bare)").unwrap();
    }
    let mut db = Database::open(dir.path()).unwrap();
    let rows = db.execute_cypher("MATCH (n:Bare) RETURN n").unwrap();
    assert_eq!(rows.len(), 1);
}
