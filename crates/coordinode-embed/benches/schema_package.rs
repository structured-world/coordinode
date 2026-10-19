//! Declaring an application's schema at open: one package against the same
//! declarations made one call at a time, on a fresh store and on a store
//! already holding them. Reports the wall time to declare (and, for a
//! fresh store, to have every definition in force), and how many log
//! proposals each way commits, of which how many write the catalog. The
//! shape is an application's: a few scope labels, a few dozen STRICT types,
//! a uniqueness on some of them and a dozen edge types.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::schema::definition::{
    ConstraintKind, EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_core::txn::proposal::{
    Mutation, ProposalError, ProposalOutcome, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::{
    ConstraintDeclaration, Database, EdgeTypeDeclaration, Intent, LabelDeclaration,
    PackageConstraint, SchemaPackage,
};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline;

/// Stores per arm.
const ROUNDS: usize = 20;
/// Scope labels: STRICT, declaring nothing.
const SCOPES: usize = 3;
/// Type labels: STRICT, each with `PROPERTIES` strings.
const TYPES: usize = 30;
const PROPERTIES: usize = 6;
/// Type labels with a uniqueness on their first property.
const UNIQUE: usize = 3;
/// Edge types.
const EDGES: usize = 12;
/// The longest a fresh declaration waits for its definitions to be in force.
const READY: Duration = Duration::from_secs(120);

/// Counts every proposal, and those writing the schema catalog.
struct Counting {
    inner: OwnedLocalProposalPipeline,
    all: AtomicUsize,
    catalog: AtomicUsize,
}

impl ProposalPipeline for Counting {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        self.all.fetch_add(1, Ordering::Relaxed);
        if proposal
            .mutations
            .iter()
            .any(|m| matches!(m, Mutation::Put { key, .. } if key.starts_with(b"schema:")))
        {
            self.catalog.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.propose_and_wait(proposal)
    }
}

fn open(dir: &std::path::Path) -> (Database, Arc<Counting>) {
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).expect("engine"));
    let counting = Arc::new(Counting {
        inner: OwnedLocalProposalPipeline::new(&engine),
        all: AtomicUsize::new(0),
        catalog: AtomicUsize::new(0),
    });
    let db = Database::from_engine(
        dir,
        engine,
        oracle,
        Arc::clone(&counting) as Arc<dyn ProposalPipeline>,
    )
    .expect("open");
    (db, counting)
}

fn labels() -> Vec<LabelSchema> {
    let strict = |name: String, properties: usize| {
        let mut schema = LabelSchema::new_node_id(name);
        schema.set_mode(SchemaMode::Strict);
        for p in 0..properties {
            schema.add_property(PropertyDef::new(format!("f{p}"), PropertyType::String));
        }
        schema
    };
    (0..SCOPES)
        .map(|i| strict(format!("Scope{i}"), 0))
        .chain((0..TYPES).map(|i| strict(format!("Type{i}"), PROPERTIES)))
        .collect()
}

fn edges() -> Vec<EdgeTypeSchema> {
    (0..EDGES)
        .map(|i| EdgeTypeSchema::new(format!("EDGE_{i}")))
        .collect()
}

fn package() -> SchemaPackage {
    SchemaPackage {
        labels: labels()
            .into_iter()
            .map(|schema| LabelDeclaration {
                schema,
                intent: Intent::Ensure,
            })
            .collect(),
        edge_types: edges()
            .into_iter()
            .map(|schema| EdgeTypeDeclaration {
                schema,
                intent: Intent::Ensure,
            })
            .collect(),
        constraints: (0..UNIQUE)
            .map(|i| PackageConstraint {
                name: Some(format!("type{i}_f0")),
                label: format!("Type{i}"),
                properties: vec!["f0".into()],
                kind: ConstraintKind::Unique,
                scope: None,
                on_duplicate_rename: None,
            })
            .collect(),
    }
}

/// The same declarations one call at a time, each waiting for its own
/// work as a statement does.
fn one_by_one(db: &Database) {
    for schema in labels() {
        db.create_label_schema(schema).expect("label");
    }
    for i in 0..UNIQUE {
        db.create_constraint(ConstraintDeclaration {
            name: Some(format!("type{i}_f0")),
            label: format!("Type{i}"),
            properties: vec!["f0".into()],
            kind: ConstraintKind::Unique,
            if_not_exists: true,
            wait: None,
            on_duplicate_rename: None,
            scope: None,
        })
        .expect("constraint");
    }
    for schema in edges() {
        db.create_edge_type_schema(schema).expect("edge type");
    }
}

/// Every declaration in one package, waiting for all of it to be in force.
fn as_package(db: &Database) {
    let published = db.publish_schema(package()).expect("publish");
    let ready = db.await_schema(&published, READY).expect("await");
    assert!(ready.all_ready(), "{ready:?}");
}

/// One way of declaring, on stores fresh or already declared.
struct Arm {
    label: &'static str,
    declare: fn(&Database),
    declared_before: bool,
    times: Vec<Duration>,
    proposals: usize,
    catalog: usize,
}

impl Arm {
    fn new(label: &'static str, declare: fn(&Database), declared_before: bool) -> Self {
        Self {
            label,
            declare,
            declared_before,
            times: Vec::new(),
            proposals: 0,
            catalog: 0,
        }
    }

    fn report(mut self) {
        self.times.sort_unstable();
        let at = |q: usize| self.times[(self.times.len() - 1) * q / 100];
        let rounds = self.times.len();
        println!(
            "{}: n={rounds} p50={:?} p99={:?} proposals/run={} catalog-commits/run={}",
            self.label,
            at(50),
            at(99),
            self.proposals / rounds,
            self.catalog / rounds,
        );
    }
}

fn main() {
    let mut opens = Vec::new();
    let mut arms = vec![
        Arm::new("fresh one by one", one_by_one, false),
        Arm::new("fresh package", as_package, false),
        Arm::new("unchanged one by one", one_by_one, true),
        Arm::new("unchanged package", as_package, true),
    ];
    for _ in 0..ROUNDS {
        // Arms alternate, so none is always measured after another.
        for arm in &mut arms {
            let dir = tempfile::tempdir().expect("tempdir");
            if arm.declared_before {
                let (db, _) = open(dir.path());
                as_package(&db);
            }
            let started = Instant::now();
            let (db, counting) = open(dir.path());
            opens.push(started.elapsed());
            let before = (
                counting.all.load(Ordering::Relaxed),
                counting.catalog.load(Ordering::Relaxed),
            );
            let started = Instant::now();
            (arm.declare)(&db);
            arm.times.push(started.elapsed());
            arm.proposals += counting.all.load(Ordering::Relaxed) - before.0;
            arm.catalog += counting.catalog.load(Ordering::Relaxed) - before.1;
        }
    }
    for arm in arms {
        arm.report();
    }
    opens.sort_unstable();
    println!(
        "open: n={} p50={:?} p99={:?}",
        opens.len(),
        opens[(opens.len() - 1) / 2],
        opens[(opens.len() - 1) * 99 / 100]
    );
}
