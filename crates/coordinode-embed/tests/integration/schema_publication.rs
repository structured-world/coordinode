//! Integration tests: schema and index definitions created through the
//! embedded API are published through the write pipeline, like any write.
//!
//! A definition put straight into the store is neither replicated nor in the
//! journal that recovers it, so a replica never sees it and a crash can lose
//! it. A label schema's revision and the pointer naming it as current travel
//! in one proposal, so no reader sees a pointer to a revision it lacks, and a
//! definition's persistence failure reaches the caller instead of being
//! logged while the local registration goes ahead.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use coordinode_core::graph::types::VectorMetric;
use coordinode_core::schema::definition::{
    EdgeTypeSchema, LabelSchema, encode_edge_type_current_revision_key,
    encode_label_current_revision_key,
};
use coordinode_core::txn::proposal::{
    MetadataCommand, Mutation, ProposalError, ProposalOutcome, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_query::index::{TextIndexConfig, VectorIndexConfig};
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

/// Applies every proposal and keeps a copy of each, optionally refusing the
/// ones that write a Schema key starting with `refuse`.
struct Recording {
    inner: OwnedLocalProposalPipeline,
    seen: Mutex<Vec<Vec<Mutation>>>,
    refuse: Option<&'static [u8]>,
}

impl ProposalPipeline for Recording {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        let refused = self.refuse.is_some_and(|prefix| {
            proposal
                .mutations
                .iter()
                .any(|m| matches!(m, Mutation::Put { key, .. } if key.starts_with(prefix)))
        });
        if refused {
            return Err(ProposalError::Storage("refused".into()));
        }
        self.seen.lock().unwrap().push(proposal.mutations.clone());
        self.inner.propose_and_wait(proposal)
    }
}

fn open(refuse: Option<&'static [u8]>) -> (Database, Arc<Recording>, tempfile::TempDir) {
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
    let recording = Arc::new(Recording {
        inner: OwnedLocalProposalPipeline::new(&engine),
        seen: Mutex::new(Vec::new()),
        refuse,
    });
    let db = Database::from_engine(
        dir.path(),
        engine,
        oracle,
        Arc::clone(&recording) as Arc<dyn ProposalPipeline>,
    )
    .unwrap();
    (db, recording, dir)
}

fn put_keys(mutations: &[Mutation]) -> Vec<Vec<u8>> {
    mutations
        .iter()
        .filter_map(|m| match m {
            Mutation::Put { key, .. } => Some(key.clone()),
            _ => None,
        })
        .collect()
}

/// The proposals whose puts include a key starting with `prefix`.
fn proposals_writing(recording: &Recording, prefix: &[u8]) -> Vec<Vec<Mutation>> {
    recording
        .seen
        .lock()
        .unwrap()
        .iter()
        .filter(|m| put_keys(m).iter().any(|k| k.starts_with(prefix)))
        .cloned()
        .collect()
}

#[test]
fn a_label_schema_revision_and_its_pointer_publish_together() {
    let (mut db, recording, _dir) = open(None);
    let revision = db
        .create_label_schema(LabelSchema::new_node_id("Person"))
        .unwrap();

    let pointer = encode_label_current_revision_key("Person");
    let published = proposals_writing(&recording, &pointer);
    assert_eq!(
        published.len(),
        1,
        "one proposal names the current revision"
    );
    let keys = put_keys(&published[0]);
    assert_eq!(keys.len(), 2, "the revision and its pointer: {keys:?}");
    assert!(keys.contains(&pointer));
    assert_eq!(
        db.engine()
            .get(
                coordinode_storage::engine::partition::Partition::Schema,
                &pointer
            )
            .unwrap()
            .as_deref(),
        Some(revision.to_be_bytes().as_slice())
    );
}

#[test]
fn an_edge_type_schema_revision_and_its_pointer_publish_together() {
    let (mut db, recording, _dir) = open(None);
    db.create_edge_type_schema(EdgeTypeSchema::new("KNOWS"))
        .unwrap();
    let pointer = encode_edge_type_current_revision_key("KNOWS");
    let published = proposals_writing(&recording, &pointer);
    assert_eq!(published.len(), 1);
    assert_eq!(put_keys(&published[0]).len(), 2);
}

/// The names a vector index is keyed by are registered before its definition
/// is published: every member that sees the definition can resolve them.
#[test]
fn a_vector_index_registers_its_names_before_its_definition() {
    let (mut db, recording, _dir) = open(None);
    db.create_vector_index(
        "doc_vec",
        "Doc",
        "emb",
        VectorIndexConfig {
            dimensions: 2,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    )
    .unwrap();

    let seen = recording.seen.lock().unwrap().clone();
    let registered = seen.iter().position(|m| {
        m.iter().any(|m| {
            matches!(m, Mutation::Command(MetadataCommand::RegisterFields { names })
                if names.iter().any(|n| n == "emb"))
        })
    });
    let defined = seen
        .iter()
        .position(|m| put_keys(m).iter().any(|k| k.starts_with(b"schema:idx:")));
    match (registered, defined) {
        (Some(r), Some(d)) => assert!(r < d, "names at {r}, definition at {d}"),
        other => panic!("both must be published through the pipeline: {other:?}"),
    }
    let view = db.interner().unwrap();
    assert!(view.lookup("Doc").is_some() && view.lookup("emb").is_some());
}

#[test]
fn a_text_index_definition_is_published() {
    let (mut db, recording, _dir) = open(None);
    db.create_text_index("bio_text", "Person", "bio", TextIndexConfig::default())
        .unwrap();
    assert_eq!(proposals_writing(&recording, b"schema:idx:").len(), 1);
}

/// A definition the pipeline refuses is reported, and nothing of it is
/// registered locally: a caller told the index exists would query an index
/// no other member knows and a restart forgets.
#[test]
fn a_refused_definition_fails_the_call_and_registers_nothing() {
    let (mut db, _recording, _dir) = open(Some(b"schema:idx:"));
    let created = db.create_vector_index(
        "doc_vec",
        "Doc",
        "emb",
        VectorIndexConfig {
            dimensions: 2,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    );
    assert!(created.is_err(), "the refusal must reach the caller");
    assert!(
        !db.vector_index_registry().has_index("Doc", "emb"),
        "a refused definition registered an index"
    );

    let (mut db, _recording, _dir) = open(Some(b"schema:"));
    assert!(
        db.create_label_schema(LabelSchema::new_node_id("Person"))
            .is_err()
    );
    let stored = db
        .engine()
        .get(
            coordinode_storage::engine::partition::Partition::Schema,
            &encode_label_current_revision_key("Person"),
        )
        .unwrap();
    assert!(stored.is_none(), "a refused schema left a pointer behind");
}
