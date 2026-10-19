//! Schema packages: label definitions, edge type definitions and separate
//! constraints published together, in one catalog commit.
//!
//! Every declaration is checked against one consistent catalog read plus
//! the package itself, so a constraint may name a label the same package
//! defines, in any order. An absent object is created, an equal one is kept
//! as it is, and a different one is a conflict unless the declaration asks
//! for the change at the revision it was read at; an object the package does
//! not mention is left alone. A package that changes nothing commits
//! nothing. Building the indexes the new constraints own runs after the
//! commit and is awaited separately ([`Database::await_schema`]).

use core::time::Duration;
use std::collections::BTreeMap;

use coordinode_core::schema::definition::{
    ConstraintKind, ConstraintState, EdgeTypeSchema, LabelCandidate, LabelSchema, PropertyDef,
    SchemaMode, encode_edge_type_current_revision_key,
};
use coordinode_modality::LabelCertificationRecord;
use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
use coordinode_query::executor::constraint_plan::{
    ConstraintRequest, OwnedIndex, plan_constraint, stage_owned_index,
};
use coordinode_query::executor::runner::{CatalogObject, ExecutionError};
use coordinode_query::index::definition::PartialFilter;
use coordinode_query::index::{BuildError, GenerationId, IndexBuildOutcome};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use super::catalog::{check_label_declaration, refused, resolve_edge_identity, unchanged_label};
use super::{Database, DatabaseError};

/// The most declarations one package may carry. A larger one is refused
/// before any work, never split into several commits.
pub const MAX_SCHEMA_PACKAGE_DECLARATIONS: usize = 4096;

/// What a declaration asks of an object that already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Intent {
    /// Create it when absent, keep it when equal, refuse it when it differs.
    Ensure,
    /// Change it to this declaration, provided it is still at
    /// `expected_revision`, the revision the caller read it at. An absent
    /// object is refused: a change names something that exists.
    Change {
        /// The revision the object must be at.
        expected_revision: u64,
    },
}

/// A label definition in a package.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LabelDeclaration {
    /// The type facts of the label. It declares no constraints: those are
    /// separate declarations.
    pub schema: LabelSchema,
    /// What to do when the label has a definition.
    pub intent: Intent,
}

/// An edge type definition in a package.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct EdgeTypeDeclaration {
    /// The type facts of the edge type.
    pub schema: EdgeTypeSchema,
    /// What to do when the edge type has a definition.
    pub intent: Intent,
}

/// A constraint in a package. A constraint is created or kept: changing one
/// is dropping it and declaring another.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PackageConstraint {
    /// Its name; `None` derives one from the label, properties and kind.
    pub name: Option<String>,
    /// The label whose nodes it constrains, defined in the catalog, in the
    /// same package, or by neither (then the constraint gives it a FLEXIBLE
    /// definition).
    pub label: String,
    /// The constrained properties, in order.
    pub properties: Vec<String>,
    /// What it requires.
    pub kind: ConstraintKind,
    /// For a uniqueness, the nodes it holds among; `None` is every node of
    /// the label.
    pub scope: Option<PartialFilter>,
    /// The string property a uniqueness build may rewrite to end a stored
    /// duplicate; `None` fails the build on the first one.
    pub on_duplicate_rename: Option<String>,
}

/// Declarations published together. Each name appears at most once per
/// kind of object.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct SchemaPackage {
    /// Label definitions.
    pub labels: Vec<LabelDeclaration>,
    /// Edge type definitions.
    pub edge_types: Vec<EdgeTypeDeclaration>,
    /// Constraints.
    pub constraints: Vec<PackageConstraint>,
}

impl SchemaPackage {
    /// How many declarations the package carries.
    pub fn len(&self) -> usize {
        self.labels.len() + self.edge_types.len() + self.constraints.len()
    }

    /// Whether the package declares nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What publication did with one declared object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// It was absent and is created.
    Created,
    /// It was there, equal to the declaration, and is kept unchanged.
    Retained,
    /// It was changed as the declaration asked.
    Changed,
}

/// One declared object after publication.
#[derive(Debug, Clone, PartialEq)]
pub struct PublishedObject {
    /// What kind of object it is.
    pub object: CatalogObject,
    /// Its name.
    pub name: String,
    /// What publication did with it.
    pub disposition: Disposition,
    /// Its revision: a definition's own, or for a constraint the revision of
    /// its label's definition that holds it.
    pub revision: u64,
    /// For a constraint, where it is in its life. A validating one is
    /// enforced on every write while the build of its index checks the
    /// stored nodes.
    pub state: Option<ConstraintState>,
    /// The background work putting it in force: for a label, the proof of
    /// its candidate definition; for a constraint owning an index, that
    /// index's build. [`Database::await_schema`] waits for it,
    /// [`Database::index_build`] inspects a build and
    /// [`Database::cancel_index_build`] cancels either. `None` when it is in
    /// force already.
    pub operation: Option<GenerationId>,
}

/// The outcome of publishing a package.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaPublication {
    /// Whether a catalog commit published anything. `false` when every
    /// declaration was already in place: nothing was written, scanned or
    /// started.
    pub committed: bool,
    /// Every declared object, labels first, then edge types, then
    /// constraints, each in declaration order.
    pub objects: Vec<PublishedObject>,
}

/// Where a published label definition or constraint is after a wait.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Readiness {
    /// It is in force over the stored data.
    Ready,
    /// The proof of the definition, or the build of the constraint's index,
    /// is still running; the rules in force before still are.
    Pending,
    /// The proof or build ended without putting it in force: a candidate
    /// definition is withdrawn with the rules in force before kept, a
    /// constraint withdrawn with its index. Nothing else the package
    /// published is undone.
    Failed(String),
}

/// The readiness of a publication's label definitions and constraints after
/// a wait.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaReadiness {
    /// Each label and constraint of the publication, in its order.
    pub objects: Vec<(CatalogObject, String, Readiness)>,
}

impl SchemaReadiness {
    /// Whether every one is in force.
    pub fn all_ready(&self) -> bool {
        self.objects
            .iter()
            .all(|(_, _, readiness)| *readiness == Readiness::Ready)
    }

    /// The readiness of the object of `kind` named `name`.
    pub fn of(&self, kind: CatalogObject, name: &str) -> Option<&Readiness> {
        self.objects
            .iter()
            .find(|(k, n, _)| *k == kind && n == name)
            .map(|(_, _, r)| r)
    }
}

/// What staging does with a declared definition that already exists: the
/// two package intents, and the two contracts of the single-object calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Existing {
    /// Keep it when equal, refuse it when it differs.
    Ensure,
    /// Change it, provided it is at this revision.
    Change(u64),
    /// Refuse it, equal or not.
    Refuse,
    /// Change it at whatever revision this staging reads.
    Replace,
}

impl From<Intent> for Existing {
    fn from(intent: Intent) -> Self {
        match intent {
            Intent::Ensure => Self::Ensure,
            Intent::Change { expected_revision } => Self::Change(expected_revision),
        }
    }
}

/// What [`Database::stage_declarations`] staged, for
/// [`Database::finish_publication`] to commit.
pub(super) struct StagedPackage {
    /// Every declared object, with its revision and operation as published.
    pub(super) objects: Vec<PublishedObject>,
    /// Whether anything is staged for a commit.
    changes: bool,
    /// The label definitions the commit publishes, to name a stored node
    /// that refuses one.
    labels: Vec<LabelSchema>,
    /// The indexes the commit publishes building.
    published: Vec<coordinode_query::index::IndexDefinition>,
    /// The certifications the commit admits.
    certifications: Vec<GenerationId>,
}

/// A label as the package stages it.
struct StagedLabel {
    /// Its definition as the commit publishes it; the facts the package
    /// declares are in it as if in force.
    schema: LabelSchema,
    /// Its revision in the catalog before the package, `None` when absent.
    read_revision: Option<u64>,
    /// Whether the package writes a new revision of it.
    dirty: bool,
    /// The facts in force before the package, when the declared ones could
    /// refuse a stored node: they stay in force, and the declared ones are
    /// published as a candidate to prove.
    proved_over: Option<(BTreeMap<String, PropertyDef>, SchemaMode)>,
}

impl Database {
    /// Publish `package` in one catalog commit, or in none when every
    /// declaration is already in place. Returns each declared object with
    /// what publication did with it. Constraints that own an index are
    /// published validating with their builds started; [`Self::await_schema`]
    /// waits for them.
    ///
    /// # Errors
    ///
    /// Nothing is published when any declaration is refused: the package is
    /// larger than [`MAX_SCHEMA_PACKAGE_DECLARATIONS`] or names an object
    /// twice; a declaration differs from the object in place without asking
    /// for the change, or asks for one at a revision the object is no longer
    /// at; a definition or constraint is invalid or cannot hold; or the
    /// catalog could not be read or written, a concurrent change of the
    /// same objects included.
    pub fn publish_schema(
        &self,
        package: SchemaPackage,
    ) -> Result<SchemaPublication, DatabaseError> {
        self.publish_package(package, None)
    }

    /// [`Self::publish_schema`] as the client operation `operation`: the
    /// catalog commit takes effect at most once however the call is
    /// retried. A retry after a lost reply of a publication that committed
    /// is refused with [`DatabaseError::OperationSettled`], and
    /// [`Self::resolve_operation`] tells how it stands without publishing
    /// again. A package that changes nothing commits nothing and records
    /// nothing under the operation.
    ///
    /// # Errors
    ///
    /// As [`Self::publish_schema`], and the operation already settled or
    /// not admissible.
    pub fn publish_schema_as(
        &self,
        package: SchemaPackage,
        operation: &crate::operation::OperationDescriptor,
    ) -> Result<SchemaPublication, DatabaseError> {
        self.publish_package(package, Some(operation))
    }

    fn publish_package(
        &self,
        package: SchemaPackage,
        operation: Option<&crate::operation::OperationDescriptor>,
    ) -> Result<SchemaPublication, DatabaseError> {
        check_package(&package)?;
        let operation = match operation {
            Some(descriptor) => {
                let encoded = rmp_serde::to_vec(&package)
                    .map_err(|e| DatabaseError::Other(format!("encode the schema package: {e}")))?;
                let intent = crate::operation::package_intent(&encoded);
                // Before the catalog is read: a retry of a publication that
                // committed is answered by its outcome, not by the conflict
                // its own commit now makes for it.
                self.admit_operation(descriptor, &intent)?;
                Some((descriptor, intent))
            }
            None => None,
        };
        let SchemaPackage {
            labels,
            edge_types,
            constraints,
        } = package;
        let mut txn = self.begin_catalog_txn();
        let staged = self.stage_declarations(
            &mut txn,
            labels
                .into_iter()
                .map(|d| (d.schema, Existing::from(d.intent)))
                .collect(),
            edge_types
                .into_iter()
                .map(|d| (d.schema, Existing::from(d.intent)))
                .collect(),
            constraints,
        )?;
        // The commit records the operation; a concurrent attempt of the same
        // one that committed first refuses this commit there.
        if let Some((descriptor, intent)) = operation {
            if staged.changes {
                txn.bind_operation(descriptor.op, intent);
            }
        }
        self.finish_publication(txn, staged)
    }

    /// The publication core every schema entry point stages through: one
    /// catalog read, one commit or none. `check_package` has passed.
    pub(super) fn publish_declarations(
        &self,
        labels: Vec<(LabelSchema, Existing)>,
        edge_types: Vec<(EdgeTypeSchema, Existing)>,
        constraints: Vec<PackageConstraint>,
    ) -> Result<SchemaPublication, DatabaseError> {
        let mut txn = self.begin_catalog_txn();
        let staged = self.stage_declarations(&mut txn, labels, edge_types, constraints)?;
        self.finish_publication(txn, staged)
    }

    /// Stage every declaration in `txn` against the catalog it reads, plus
    /// the declarations staged before it: the definitions, name claims and
    /// build admissions a commit of `txn` publishes. Nothing commits here.
    pub(super) fn stage_declarations(
        &self,
        txn: &mut Transaction<'_>,
        labels: Vec<(LabelSchema, Existing)>,
        edge_types: Vec<(EdgeTypeSchema, Existing)>,
        constraints: Vec<PackageConstraint>,
    ) -> Result<StagedPackage, DatabaseError> {
        let store = LocalSchemaStore::new(&self.engine);
        let declared = labels.len() + edge_types.len() + constraints.len();
        let mut objects = Vec::with_capacity(declared);
        // The label whose new revision each created or changed label or
        // constraint is published at, known once every label is staged.
        let mut revision_of: Vec<Option<String>> = Vec::with_capacity(declared);

        let mut staged: BTreeMap<String, StagedLabel> = BTreeMap::new();
        for (schema, existing) in labels {
            let (label, object) = stage_label(&store, txn, schema, existing)?;
            revision_of
                .push((object.disposition != Disposition::Retained).then(|| object.name.clone()));
            objects.push(object);
            staged.insert(label.schema.name.clone(), label);
        }

        let mut edge_writes = Vec::new();
        for (schema, existing) in edge_types {
            let (write, object) = stage_edge_type(&store, txn, schema, existing)?;
            revision_of.push(None);
            objects.push(object);
            edge_writes.extend(write);
        }

        let mut owned_indexes: Vec<(usize, OwnedIndex)> = Vec::new();
        for declaration in constraints {
            let label = match staged.entry(declaration.label.clone()) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let current = store.load_label_for_update_txn(txn, &declaration.label)?;
                    let read_revision = current.as_ref().map(|s| s.schema_revision);
                    entry.insert(StagedLabel {
                        schema: current.unwrap_or_else(|| {
                            // A label without a definition is enforced as
                            // FLEXIBLE; the definition the constraint gives
                            // it keeps that.
                            let mut s = LabelSchema::new_node_id(&declaration.label);
                            s.set_mode(SchemaMode::Flexible);
                            s
                        }),
                        read_revision,
                        dirty: read_revision.is_none(),
                        proved_over: None,
                    })
                }
            };
            let holder = label.schema.name.clone();
            let (object, owned) = self.stage_constraint(&store, txn, label, declaration)?;
            if let Some(owned) = owned {
                owned_indexes.push((objects.len(), owned));
            }
            revision_of.push((object.disposition == Disposition::Created).then_some(holder));
            objects.push(object);
        }

        let changes = staged.values().any(|l| l.dirty) || !edge_writes.is_empty();
        if !changes {
            return Ok(StagedPackage {
                objects,
                changes,
                labels: Vec::new(),
                published: Vec::new(),
                certifications: Vec::new(),
            });
        }

        // Each label the package changes gets one new revision, however many
        // of its declarations changed it.
        let indexes = coordinode_modality::LocalIndexStore::new(&self.engine);
        let mut certifications = Vec::new();
        for label in staged.values_mut().filter(|l| l.dirty) {
            if let Some(read) = label.read_revision {
                label.schema.schema_revision = read.checked_add(1).ok_or_else(|| {
                    refused(format!(
                        "label '{}' has no schema revision left",
                        label.schema.name
                    ))
                })?;
            }
            // Declared facts that could refuse a stored node are published
            // as a candidate, proved in the background; the facts in force
            // stay until the proof puts it in force.
            if let Some((properties, mode)) = label.proved_over.take() {
                use coordinode_modality::IndexStore as _;
                let operation = indexes.allocate_generation_txn(txn)?;
                let declared = core::mem::replace(&mut label.schema.properties, properties);
                let declared_mode = core::mem::replace(&mut label.schema.mode, mode);
                label.schema.set_candidate(LabelCandidate {
                    properties: declared,
                    mode: declared_mode,
                    operation,
                });
                indexes.put_certification_txn(
                    txn,
                    &LabelCertificationRecord::accepted(operation, label.schema.name.clone()),
                    None,
                )?;
                certifications.push((label.schema.name.clone(), operation));
            }
            store.save_label_txn(txn, &label.schema)?;
        }
        for object in &mut objects {
            if object.object == CatalogObject::Label && object.disposition != Disposition::Retained
            {
                if let Some((_, operation)) = certifications.iter().find(|(l, _)| *l == object.name)
                {
                    object.operation = Some(*operation);
                }
            }
        }
        for (object, label) in objects.iter_mut().zip(&revision_of) {
            if let Some(staged) = label.as_ref().and_then(|name| staged.get(name)) {
                object.revision = staged.schema.schema_revision;
            }
        }
        for schema in &edge_writes {
            store.save_edge_type_txn(txn, schema)?;
        }
        let mut published = Vec::with_capacity(owned_indexes.len());
        for (at, owned) in owned_indexes {
            let def = stage_owned_index(&self.engine, txn, owned)?;
            objects[at].operation = Some(def.generation);
            published.push(def);
        }
        Ok(StagedPackage {
            objects,
            changes,
            labels: staged
                .into_values()
                .filter(|l| l.dirty)
                .map(|l| l.schema)
                .collect(),
            published,
            certifications: certifications.into_iter().map(|(_, op)| op).collect(),
        })
    }

    /// Commit what [`Self::stage_declarations`] staged in `txn`, then
    /// register the indexes it published and start their builds. A package
    /// that changed nothing commits nothing.
    pub(super) fn finish_publication(
        &self,
        txn: Transaction<'_>,
        staged: StagedPackage,
    ) -> Result<SchemaPublication, DatabaseError> {
        let StagedPackage {
            objects,
            changes,
            labels,
            published,
            certifications,
        } = staged;
        if !changes {
            // Every object was in place: nothing to commit, scan or start.
            // The answer reads as "published", so what it reports is made
            // as durable as a publishing call's answer would be.
            drop(txn);
            if self.write_concern.journal == coordinode_core::txn::write_concern::Journal::Journal {
                self.engine.sync_applied()?;
            }
            return Ok(SchemaPublication {
                committed: false,
                objects,
            });
        }
        if let Err(refused) = self.commit_catalog_txn(txn) {
            return Err(self.name_refusing_node(refused, &labels));
        }
        for def in published {
            let generation = def.generation;
            self.index_registry
                .register_published(&self.engine, def)
                .map_err(ExecutionError::from)?;
            self.index_builds
                .submit(generation)
                .map_err(DatabaseError::Other)?;
        }
        for operation in certifications {
            self.index_builds
                .submit_certification(operation)
                .map_err(DatabaseError::Other)?;
        }
        Ok(SchemaPublication {
            committed: true,
            objects,
        })
    }

    /// Wait up to `wait` for every label definition and constraint
    /// `publication` reports to be in force: a candidate definition proved
    /// over the stored nodes, a constraint's index built. Waiting cancels
    /// nothing: work outwaited goes on, and a failed one leaves every other
    /// published object in place.
    ///
    /// # Errors
    ///
    /// A record could not be read, or the answer could not be made durable.
    pub fn await_schema(
        &self,
        publication: &SchemaPublication,
        wait: Duration,
    ) -> Result<SchemaReadiness, DatabaseError> {
        let deadline = std::time::Instant::now() + wait;
        let mut objects = Vec::new();
        let mut any_published = false;
        for object in &publication.objects {
            if object.object == CatalogObject::EdgeType {
                continue;
            }
            let readiness = match (object.state, object.operation) {
                (Some(ConstraintState::Active), _) | (_, None) => Readiness::Ready,
                (_, Some(operation)) => {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    match self.index_builds.wait(operation, Some(left))? {
                        Some(IndexBuildOutcome::Published { .. }) => {
                            any_published = true;
                            Readiness::Ready
                        }
                        Some(IndexBuildOutcome::Failed(BuildError::Duplicate(v))) => {
                            Readiness::Failed(ExecutionError::from(v).to_string())
                        }
                        Some(IndexBuildOutcome::Failed(BuildError::Other(reason))) => {
                            Readiness::Failed(reason)
                        }
                        Some(IndexBuildOutcome::Cancelled) => {
                            Readiness::Failed("the build was cancelled".into())
                        }
                        None => Readiness::Pending,
                    }
                }
            };
            objects.push((object.object, object.name.clone(), readiness));
        }
        // Background work commits unsynced; an answer naming it in force
        // waits for one sync covering every such commit.
        if any_published
            && self.write_concern.journal == coordinode_core::txn::write_concern::Journal::Journal
        {
            self.engine.sync_applied()?;
        }
        Ok(SchemaReadiness { objects })
    }

    /// The error a refused publication reports: when the commit refused it
    /// because stored nodes break a definition it publishes, which node and
    /// why; any other refusal as it is. Only a refused publication looks
    /// for the node, so an accepted one pays no second pass over the data.
    fn name_refusing_node(&self, refused: DatabaseError, labels: &[LabelSchema]) -> DatabaseError {
        if !matches!(
            refused,
            DatabaseError::Execution(ExecutionError::InvariantRefused(_))
        ) {
            return refused;
        }
        let none = std::collections::HashMap::new();
        for schema in labels {
            match coordinode_storage::engine::claims::evaluate::first_label_schema_violation(
                &self.engine,
                schema,
                &none,
            ) {
                Ok(Some(violation)) => {
                    return ExecutionError::SchemaViolation(format!(
                        "the definition of :{} is not published: node {} breaks it ({})",
                        schema.name,
                        violation.node.to_element_id(),
                        violation.reason
                    ))
                    .into();
                }
                Ok(None) => {}
                // The refusal stands; failing to explain it does not replace
                // it.
                Err(_) => return refused,
            }
        }
        refused
    }

    /// Stage constraint `declaration` on `label`, as the package has it so
    /// far. Returns the object and, for a new constraint owning an index,
    /// that index to publish building.
    fn stage_constraint(
        &self,
        store: &LocalSchemaStore<'_>,
        txn: &mut Transaction<'_>,
        label: &mut StagedLabel,
        declaration: PackageConstraint,
    ) -> Result<(PublishedObject, Option<OwnedIndex>), DatabaseError> {
        let request = ConstraintRequest {
            name: declaration.name,
            label: declaration.label,
            properties: declaration.properties,
            kind: declaration.kind,
            scope: declaration.scope,
            on_duplicate_rename: declaration.on_duplicate_rename,
            sparse: None,
            maintenance: None,
        };
        let name = request.resolved_name();
        let requirement = request.requirement();
        let retained = |existing: &coordinode_core::schema::definition::NodeConstraint,
                        label: &StagedLabel| PublishedObject {
            object: CatalogObject::Constraint,
            name: existing.name.clone(),
            disposition: Disposition::Retained,
            revision: label.read_revision.unwrap_or(label.schema.schema_revision),
            state: Some(existing.state),
            operation: self
                .index_registry
                .get(&existing.name)
                .filter(|d| d.owner.as_deref() == Some(existing.name.as_str()))
                .map(|d| d.generation),
        };
        if let Some(holder) = store.constraint_label_txn(txn, &name)? {
            let existing = (holder == request.label)
                .then(|| label.schema.constraint(&name))
                .flatten()
                .filter(|existing| existing.same_requirement(&requirement));
            return match existing {
                Some(existing) => Ok((retained(existing, label), None)),
                None => Err(ExecutionError::CatalogObjectDiffers {
                    object: CatalogObject::Constraint,
                    name,
                    revision: label.read_revision.unwrap_or(0),
                }
                .into()),
            };
        }
        // An unnamed declaration is answered by an equal constraint under
        // another name; a named one asks for that name.
        if let Some(existing) = label
            .schema
            .constraints()
            .iter()
            .find(|c| c.same_requirement(&requirement))
        {
            if request.name.is_none() || existing.name == name {
                return Ok((retained(existing, label), None));
            }
            return Err(ExecutionError::CatalogObjectExists {
                object: CatalogObject::Constraint,
                name: existing.name.clone(),
            }
            .into());
        }
        let planned = plan_constraint(&request, Some(&label.schema))?;
        if planned.owned_index.is_some() && self.index_registry.get(&name).is_some() {
            return Err(ExecutionError::CatalogObjectExists {
                object: CatalogObject::Index,
                name,
            }
            .into());
        }
        store.claim_constraint_name_txn(txn, &name, &request.label)?;
        label.schema.add_constraint(planned.constraint.clone());
        label.dirty = true;
        Ok((
            PublishedObject {
                object: CatalogObject::Constraint,
                name,
                disposition: Disposition::Created,
                revision: 0,
                state: Some(planned.constraint.state),
                operation: None,
            },
            planned.owned_index,
        ))
    }
}

/// Refuse a package too large to publish, or naming one object twice,
/// before reading the catalog.
fn check_package(package: &SchemaPackage) -> Result<(), DatabaseError> {
    if package.len() > MAX_SCHEMA_PACKAGE_DECLARATIONS {
        return Err(refused(format!(
            "a schema package carries at most {MAX_SCHEMA_PACKAGE_DECLARATIONS} declarations, \
             this one {}",
            package.len()
        )));
    }
    let twice =
        |kind: &str, name: &str| refused(format!("the package declares {kind} '{name}' twice"));
    let mut seen = std::collections::BTreeSet::new();
    for declaration in &package.labels {
        if !seen.insert(declaration.schema.name.as_str()) {
            return Err(twice("label", &declaration.schema.name));
        }
        check_label_declaration(&declaration.schema)?;
    }
    seen.clear();
    for declaration in &package.edge_types {
        if !seen.insert(declaration.schema.name.as_str()) {
            return Err(twice("edge type", &declaration.schema.name));
        }
    }
    let mut names = std::collections::BTreeSet::new();
    for declaration in &package.constraints {
        let name = declaration.name.clone().unwrap_or_else(|| {
            coordinode_query::executor::constraint_plan::derived_constraint_name(
                &declaration.label,
                &declaration.properties,
                &declaration.kind,
            )
        });
        if !names.insert(name.clone()) {
            return Err(twice("constraint", &name));
        }
    }
    Ok(())
}

/// Refuse a definition at revision `read` that `existing` does not let be
/// kept or changed: a call that creates only, or a change expecting
/// another revision.
fn check_existing(
    object: CatalogObject,
    name: &str,
    existing: Existing,
    read: u64,
) -> Result<(), DatabaseError> {
    match existing {
        Existing::Refuse => Err(ExecutionError::CatalogObjectExists {
            object,
            name: name.to_string(),
        }
        .into()),
        Existing::Change(expected) if expected != read => {
            Err(ExecutionError::CatalogRevisionMismatch {
                object,
                name: name.to_string(),
                expected,
                current: Some(read),
            }
            .into())
        }
        _ => Ok(()),
    }
}

/// Stage label `schema` against the catalog `txn` reads, conditioning
/// the commit on the label's definition still being the one read.
fn stage_label(
    store: &LocalSchemaStore<'_>,
    txn: &mut Transaction<'_>,
    mut schema: LabelSchema,
    existing: Existing,
) -> Result<(StagedLabel, PublishedObject), DatabaseError> {
    let name = schema.name.clone();
    let current = store.load_label_for_update_txn(txn, &name)?;
    let object = |disposition, revision| PublishedObject {
        object: CatalogObject::Label,
        name: name.clone(),
        disposition,
        revision,
        state: None,
        operation: None,
    };
    let Some(current) = current else {
        if let Existing::Change(expected) = existing {
            return Err(ExecutionError::CatalogRevisionMismatch {
                object: CatalogObject::Label,
                name,
                expected,
                current: None,
            }
            .into());
        }
        let revision = schema.schema_revision;
        // Until proved, the label is held to nothing more than it was
        // without a definition.
        let proved_over = schema
            .facts_can_refuse_nodes()
            .then(|| (BTreeMap::new(), SchemaMode::Flexible));
        return Ok((
            StagedLabel {
                schema,
                read_revision: None,
                dirty: true,
                proved_over,
            },
            object(Disposition::Created, revision),
        ));
    };
    let read = current.schema_revision;
    check_existing(CatalogObject::Label, &name, existing, read)?;
    // A candidate being proved is what the label is declared as.
    let declared = current
        .candidate_schema()
        .unwrap_or_else(|| current.clone());
    if !current.is_table() && unchanged_label(&schema, &declared) {
        let operation = current.candidate().map(|c| c.operation);
        let mut retained = object(Disposition::Retained, read);
        retained.operation = operation;
        return Ok((
            StagedLabel {
                schema: current,
                read_revision: Some(read),
                dirty: false,
                proved_over: None,
            },
            retained,
        ));
    }
    if existing == Existing::Ensure {
        return Err(ExecutionError::CatalogObjectDiffers {
            object: CatalogObject::Label,
            name,
            revision: read,
        }
        .into());
    }
    if current.is_table() {
        return Err(refused(format!(
            ":{name} is a TABLE; its definition changes through table DDL"
        )));
    }
    if current.temporal != schema.temporal {
        return Err(refused(format!(
            "whether :{name} is temporal is fixed when it is created"
        )));
    }
    if let Some(candidate) = current.candidate() {
        return Err(refused(format!(
            "the definition of :{name} declared by operation {} is still being certified; \
             wait for it or cancel it before changing the label",
            candidate.operation.as_raw()
        )));
    }
    schema.placement = current.placement.clone();
    schema.shard_keys = current.shard_keys.clone();
    for constraint in current.constraints() {
        if let Some(conflict) = schema.constraint_conflict(constraint) {
            return Err(refused(format!(
                "constraint '{}' cannot hold under the new definition: {conflict}",
                constraint.name
            )));
        }
        schema.add_constraint(constraint.clone());
    }
    let proved_over = schema
        .facts_can_refuse_nodes()
        .then(|| (current.properties.clone(), current.mode));
    Ok((
        StagedLabel {
            schema,
            read_revision: Some(read),
            dirty: true,
            proved_over,
        },
        object(Disposition::Changed, read),
    ))
}

/// Stage edge type `declaration` against the catalog `txn` reads,
/// conditioning the commit on its definition still being the one read.
/// Returns the definition to write, if any.
fn stage_edge_type(
    store: &LocalSchemaStore<'_>,
    txn: &mut Transaction<'_>,
    mut schema: EdgeTypeSchema,
    existing: Existing,
) -> Result<(Option<EdgeTypeSchema>, PublishedObject), DatabaseError> {
    resolve_edge_identity(&mut schema)?;
    let name = schema.name.clone();
    // Refused also when only stored edges carry the name: defining it then
    // would declare a shape those edges never had to fit.
    if existing == Existing::Refuse && store.edge_type_exists(txn, &name)? {
        return Err(ExecutionError::CatalogObjectExists {
            object: CatalogObject::EdgeType,
            name,
        }
        .into());
    }
    let pointer = encode_edge_type_current_revision_key(&name);
    // Version first, then the definition: a change landing between the two
    // refuses this commit instead of passing under it.
    let version = txn.record_version(Partition::Schema, &pointer)?;
    let current = store.load_edge_type(&name)?;
    txn.expect_version(Partition::Schema, &pointer, version)?;
    let object = |disposition, revision| PublishedObject {
        object: CatalogObject::EdgeType,
        name: name.clone(),
        disposition,
        revision,
        state: None,
        operation: None,
    };
    let Some(current) = current else {
        if let Existing::Change(expected) = existing {
            return Err(ExecutionError::CatalogRevisionMismatch {
                object: CatalogObject::EdgeType,
                name,
                expected,
                current: None,
            }
            .into());
        }
        let revision = schema.schema_revision;
        return Ok((Some(schema), object(Disposition::Created, revision)));
    };
    let read = current.schema_revision;
    check_existing(CatalogObject::EdgeType, &name, existing, read)?;
    let mut carried = schema.clone();
    carried.placement = current.placement;
    carried.schema_revision = read;
    if carried == current {
        return Ok((None, object(Disposition::Retained, read)));
    }
    if existing == Existing::Ensure {
        return Err(ExecutionError::CatalogObjectDiffers {
            object: CatalogObject::EdgeType,
            name,
            revision: read,
        }
        .into());
    }
    if current.temporal != schema.temporal {
        return Err(refused(format!(
            "whether edge type '{name}' is temporal is fixed when it is created"
        )));
    }
    if current.discriminator() != schema.discriminator() {
        return Err(refused(format!(
            "what identifies an edge of type '{name}' is fixed when it is created"
        )));
    }
    schema.placement = current.placement;
    schema.schema_revision = read
        .checked_add(1)
        .ok_or_else(|| refused(format!("edge type '{name}' has no schema revision left")))?;
    let revision = schema.schema_revision;
    Ok((Some(schema), object(Disposition::Changed, revision)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
