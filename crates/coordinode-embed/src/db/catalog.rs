//! Schema management through typed calls: label and edge type definitions,
//! and named constraints over them.
//!
//! A definition carries the type facts of a record only; uniqueness and the
//! other constraints are separate objects. Constraint calls run as the same
//! statement `CREATE CONSTRAINT` and `DROP CONSTRAINT` run as, so both reach
//! one catalog through one admission path.

use coordinode_core::graph::types::VectorConsistencyMode;
use coordinode_core::schema::definition::{
    ConstraintKind, EdgeTypeSchema, LabelSchema, NodeConstraint,
};
use coordinode_core::txn::read_consistency::ReadConsistencyMode;
use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
use coordinode_query::executor::runner::{CatalogObject, ExecutionError};
use coordinode_query::planner::logical::{LogicalOp, LogicalPlan};

use super::{Database, DatabaseError, QuerySession, Statement, TxnMode};

/// A constraint to create, as `CREATE CONSTRAINT` declares one.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintDeclaration {
    /// The constraint's name; `None` lets the engine derive one from the
    /// label, properties and kind.
    pub name: Option<String>,
    /// The label whose nodes are constrained.
    pub label: String,
    /// The constrained properties, in order.
    pub properties: Vec<String>,
    /// What the constraint requires.
    pub kind: ConstraintKind,
    /// A constraint of the same name, or an equivalent one, already in place
    /// is returned instead of refused.
    pub if_not_exists: bool,
}

/// A constraint as the catalog holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelConstraint {
    /// The label whose nodes it constrains.
    pub label: String,
    /// The constraint, with its state.
    pub constraint: NodeConstraint,
    /// The index it owns and is enforced through, for a uniqueness or key
    /// constraint whose index is published.
    pub backing_index: Option<String>,
}

/// What a label definition does to a label that already has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExistingDefinition {
    /// Refuse it.
    Refuse,
    /// Replace its type facts, keeping its constraints, placement and
    /// temporal flag.
    Replace,
}

impl Database {
    /// Define label `schema`. A label that already has a definition is
    /// refused. Returns the published schema revision.
    ///
    /// # Errors
    ///
    /// The label already has a definition; the definition declares a
    /// constraint or a unique property (constraints are created with
    /// [`Self::create_constraint`]); stored nodes of the label break it; or
    /// the catalog could not be written.
    pub fn define_label(&self, schema: LabelSchema) -> Result<u64, DatabaseError> {
        self.publish_label_definition(schema, ExistingDefinition::Refuse)
    }

    /// Define label `schema`, or replace the type facts of its definition.
    /// A replaced definition keeps the label's constraints, placement and
    /// temporal flag; each kept constraint must still be able to hold under
    /// the new type facts. Returns the published schema revision.
    ///
    /// # Errors
    ///
    /// The definition declares a constraint or a unique property; it changes
    /// the temporal flag or the definition of a TABLE; a kept constraint
    /// cannot hold under it; stored nodes of the label break it; or the
    /// catalog could not be written.
    pub fn create_label_schema(&self, schema: LabelSchema) -> Result<u64, DatabaseError> {
        self.publish_label_definition(schema, ExistingDefinition::Replace)
    }

    /// Publish a label definition in one catalog transaction that reads the
    /// label's current definition, so a constraint created or dropped
    /// concurrently is neither lost nor brought back: that transaction's
    /// commit refuses this one instead.
    fn publish_label_definition(
        &self,
        mut schema: LabelSchema,
        existing: ExistingDefinition,
    ) -> Result<u64, DatabaseError> {
        if !schema.constraints().is_empty() {
            return Err(refused(format!(
                "the definition of :{} declares constraints; a type definition carries none, \
                 create them with create_constraint",
                schema.name
            )));
        }
        if let Some(p) = schema.properties.values().find(|p| {
            coordinode_core::schema::definition::TEMPORAL_ENGINE_FIELDS.contains(&p.name.as_str())
        }) {
            return Err(refused(format!(
                "property `{}` of :{} is reserved: the engine writes it into every version of a \
                 temporal node",
                p.name, schema.name
            )));
        }
        self.commit_catalog(|txn| self.stage_label_definition(txn, &mut schema, existing))?;
        Ok(schema.schema_revision)
    }

    /// Stage label definition `schema` in `txn`, against the label's current
    /// definition as `txn` reads it: a replacement takes over the label's
    /// constraints, placement and the next revision from there.
    fn stage_label_definition(
        &self,
        txn: &mut coordinode_storage::engine::transaction::Transaction<'_>,
        schema: &mut LabelSchema,
        existing: ExistingDefinition,
    ) -> Result<(), DatabaseError> {
        let store = LocalSchemaStore::new(&self.engine);
        let name = schema.name.clone();
        if let Some(current) = store.load_label_for_update_txn(txn, &name)? {
            if existing == ExistingDefinition::Refuse {
                return Err(ExecutionError::CatalogObjectExists {
                    object: CatalogObject::Label,
                    name,
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
            schema.schema_revision = current
                .schema_revision
                .checked_add(1)
                .ok_or_else(|| refused(format!("label '{name}' has no schema revision left")))?;
        }
        // The commit decides this authoritatively; checking here as well
        // names the node that refuses it.
        let staged = std::collections::HashMap::new();
        if let Some(violation) =
            coordinode_storage::engine::claims::evaluate::first_label_schema_violation(
                &self.engine,
                schema,
                &staged,
            )?
        {
            return Err(ExecutionError::SchemaViolation(format!(
                "the definition of :{name} is not published: node {} breaks it ({})",
                violation.node.to_element_id(),
                violation.reason
            ))
            .into());
        }
        store.save_label_txn(txn, schema)?;
        Ok(())
    }

    /// Define edge type `schema`. An edge type that already has a
    /// definition, or that stored edges already carry, is refused. Returns
    /// the published schema revision.
    ///
    /// # Errors
    ///
    /// The edge type already exists, or the catalog could not be written.
    pub fn define_edge_type(&self, schema: EdgeTypeSchema) -> Result<u64, DatabaseError> {
        let name = schema.name.clone();
        self.commit_catalog(|txn| -> Result<(), DatabaseError> {
            let store = LocalSchemaStore::new(&self.engine);
            if store.edge_type_exists(txn, &name)? {
                return Err(ExecutionError::CatalogObjectExists {
                    object: CatalogObject::EdgeType,
                    name: name.clone(),
                }
                .into());
            }
            store.save_edge_type_txn(txn, &schema)?;
            Ok(())
        })?;
        Ok(schema.schema_revision)
    }

    /// Define edge type `schema`, or replace its definition at the next
    /// revision, keeping its placement and temporal flag. The replacement is
    /// conditioned on the definition it read, so a concurrent change of the
    /// same edge type refuses it rather than being overwritten. Returns the
    /// published schema revision.
    ///
    /// # Errors
    ///
    /// The definition changes the temporal flag, or the catalog could not be
    /// written.
    pub fn create_edge_type_schema(
        &self,
        mut schema: EdgeTypeSchema,
    ) -> Result<u64, DatabaseError> {
        use coordinode_core::schema::definition::encode_edge_type_current_revision_key;
        use coordinode_storage::engine::partition::Partition;
        self.commit_catalog(|txn| -> Result<(), DatabaseError> {
            let store = LocalSchemaStore::new(&self.engine);
            let pointer = encode_edge_type_current_revision_key(&schema.name);
            // Version first, then the definition: a change landing between
            // the two refuses this commit instead of passing under it.
            let version = txn.record_version(Partition::Schema, &pointer)?;
            if let Some(current) = store.load_edge_type(&schema.name)? {
                if current.temporal != schema.temporal {
                    return Err(refused(format!(
                        "whether edge type '{}' is temporal is fixed when it is created",
                        schema.name
                    )));
                }
                schema.placement = current.placement;
                schema.schema_revision =
                    current.schema_revision.checked_add(1).ok_or_else(|| {
                        refused(format!(
                            "edge type '{}' has no schema revision left",
                            schema.name
                        ))
                    })?;
            }
            txn.expect_version(Partition::Schema, &pointer, version)?;
            store.save_edge_type_txn(txn, &schema)?;
            Ok(())
        })?;
        Ok(schema.schema_revision)
    }

    /// Every label with a definition, ordered by name.
    ///
    /// # Errors
    ///
    /// The catalog could not be read.
    pub fn label_schemas(&self) -> Result<Vec<LabelSchema>, DatabaseError> {
        let mut labels = LocalSchemaStore::new(&self.engine)
            .list_labels()
            .map_err(ExecutionError::from)?;
        labels.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(labels)
    }

    /// Every edge type with a definition, ordered by name.
    ///
    /// # Errors
    ///
    /// The catalog could not be read.
    pub fn edge_type_schemas(&self) -> Result<Vec<EdgeTypeSchema>, DatabaseError> {
        let mut types = LocalSchemaStore::new(&self.engine)
            .list_edge_types()
            .map_err(ExecutionError::from)?;
        types.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(types)
    }

    /// The name of every edge type the catalog knows, with a definition or
    /// only carried by stored edges, ordered by name.
    ///
    /// # Errors
    ///
    /// The catalog could not be read.
    pub fn edge_type_names(&self) -> Result<Vec<String>, DatabaseError> {
        let mut names = LocalSchemaStore::new(&self.engine)
            .list_edge_type_names_engine()
            .map_err(ExecutionError::from)?;
        names.sort();
        names.dedup();
        Ok(names)
    }

    /// Create a constraint, as `CREATE CONSTRAINT` does: a uniqueness or key
    /// constraint returns once the index it owns is validated against the
    /// stored nodes, or fails with nothing left behind. Returns the
    /// constraint as the catalog holds it.
    ///
    /// # Errors
    ///
    /// A constraint or index of the name, or an equivalent constraint,
    /// already exists (unless `if_not_exists`); the constraint cannot hold
    /// under the label's definition; stored nodes break it; or the catalog
    /// could not be written.
    pub fn create_constraint(
        &self,
        declaration: ConstraintDeclaration,
    ) -> Result<LabelConstraint, DatabaseError> {
        let ConstraintDeclaration {
            name,
            label,
            properties,
            kind,
            if_not_exists,
        } = declaration;
        let rows = self.run_catalog_statement(LogicalOp::CreateConstraint {
            name,
            if_not_exists,
            label,
            properties,
            kind,
        })?;
        let created = rows
            .first()
            .and_then(|row| match row.get("constraint") {
                Some(coordinode_core::graph::types::Value::String(name)) => Some(name.clone()),
                _ => None,
            })
            .ok_or_else(|| {
                DatabaseError::Other("CREATE CONSTRAINT returned no constraint name".into())
            })?;
        self.constraints()?
            .into_iter()
            .find(|c| c.constraint.name == created)
            .ok_or_else(|| {
                DatabaseError::Execution(ExecutionError::CatalogObjectMissing {
                    object: CatalogObject::Constraint,
                    name: created,
                })
            })
    }

    /// Drop constraint `name` and the index it owns in one catalog change,
    /// as `DROP CONSTRAINT` does. Returns whether a constraint was dropped.
    ///
    /// # Errors
    ///
    /// No constraint has the name (unless `if_exists`), or the catalog could
    /// not be written.
    pub fn drop_constraint(&self, name: &str, if_exists: bool) -> Result<bool, DatabaseError> {
        let rows = self.run_catalog_statement(LogicalOp::DropConstraint {
            name: name.to_string(),
            if_exists,
        })?;
        Ok(rows.first().is_some_and(|row| {
            matches!(
                row.get("dropped"),
                Some(coordinode_core::graph::types::Value::Bool(true))
            )
        }))
    }

    /// Every constraint, ordered by name.
    ///
    /// # Errors
    ///
    /// The catalog could not be read.
    pub fn constraints(&self) -> Result<Vec<LabelConstraint>, DatabaseError> {
        let indexes = self.index_registry.all();
        let mut out: Vec<LabelConstraint> = self
            .label_schemas()?
            .into_iter()
            .flat_map(|schema| {
                let label = schema.name.clone();
                schema
                    .constraints()
                    .iter()
                    .map(|constraint| LabelConstraint {
                        label: label.clone(),
                        backing_index: indexes
                            .iter()
                            .find(|d| d.owner.as_deref() == Some(constraint.name.as_str()))
                            .and_then(|d| d.name.clone()),
                        constraint: constraint.clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        out.sort_by(|a, b| a.constraint.name.cmp(&b.constraint.name));
        Ok(out)
    }

    /// Run one catalog operation as an auto-commit statement, the path every
    /// statement takes.
    fn run_catalog_statement(
        &self,
        root: LogicalOp,
    ) -> Result<Vec<coordinode_query::executor::row::Row>, DatabaseError> {
        let plan = LogicalPlan {
            root,
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::Current,
            read_consistency: ReadConsistencyMode::default(),
        };
        let canonical = plan.explain();
        let fingerprint = coordinode_query::advisor::fingerprint::fingerprint(&canonical);
        // A catalog change reads the current state: it neither uses nor
        // consumes a snapshot timestamp pinned for the next query.
        let session = QuerySession {
            read_concern: self.read_concern,
            snapshot_read_ts: None,
            write_concern: self.write_concern,
            vector_consistency: self.vector_consistency,
            vector_build_wait: self.vector_build_wait,
            after_commit_generation: 0,
        };
        self.run_plan(
            Statement {
                plan,
                canonical,
                fingerprint,
                build_wait: None,
            },
            None,
            None,
            &session,
            TxnMode::AutoCommit,
            &mut None,
        )
        .map(|(rows, _, _)| rows)
    }
}

/// A catalog change refused by the catalog's current state.
fn refused(reason: String) -> DatabaseError {
    DatabaseError::Execution(ExecutionError::CatalogRefused(reason))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
