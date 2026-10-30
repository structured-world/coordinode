//! Admission of a label declaration's kind: what it may declare, what the
//! declarations implying it require of it, and the stored nodes it must leave
//! valid. Every surface that declares a label runs it before publishing.

use coordinode_core::schema::definition::{LabelKind, LabelSchema};
use coordinode_core::schema::labels::check_label_declaration;
use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;

use super::runner::ExecutionError;

/// Admit `schema` as a declaration of its label in `txn`.
///
/// - It may declare only what its kind allows, and its `IMPLIES` reaches
///   neither itself nor an auxiliary label, with every property of an
///   implied label declared with the same type.
/// - Every other declared type implying it, directly or not, still declares
///   every property it declares.
/// - When `created`, no stored node it types is left invalid: an auxiliary
///   label types none, and a type gains implied labels only while it types
///   none, since its stored nodes would carry them without the indexes and
///   constraints of those labels having covered them.
///
/// `staged` answers for labels the same publication declares, ahead of the
/// catalog. Each label read from the catalog is read in `txn`, so a change
/// of it committed meanwhile refuses this commit.
///
/// # Errors
///
/// `CatalogRefused` for a declaration the rules refuse, `SchemaViolation`
/// naming a stored node it would leave invalid, or a catalog read failure.
pub fn admit_label_kind(
    engine: &StorageEngine,
    txn: &mut Transaction<'_>,
    schema: &LabelSchema,
    created: bool,
    staged: &dyn Fn(&str) -> Option<LabelSchema>,
) -> Result<(), ExecutionError> {
    let store = LocalSchemaStore::new(engine);
    // The declared labels as this publication leaves them: `schema` itself,
    // the other staged ones, then the catalog read in `txn`.
    let lookup = |txn: &mut Transaction<'_>, label: &str| -> Result<_, ExecutionError> {
        if label == schema.name {
            return Ok(Some(schema.clone()));
        }
        if let Some(staged) = staged(label) {
            return Ok(Some(staged));
        }
        Ok(store.load_label_txn(txn, label)?)
    };

    check_label_declaration(schema, |label| lookup(txn, label))?
        .map_err(|e| ExecutionError::CatalogRefused(e.to_string()))?;

    // A type implying this label, directly or through others, must still
    // declare its properties; checking each implying type whole covers both.
    for other in store.list_labels()? {
        if other.name == schema.name || other.implies().is_empty() {
            continue;
        }
        let Some(other) = lookup(txn, &other.name)? else {
            continue;
        };
        check_label_declaration(&other, |label| lookup(txn, label))?
            .map_err(|e| ExecutionError::CatalogRefused(e.to_string()))?;
    }

    // A label without an identity yet types no stored node.
    if created
        && (schema.is_auxiliary() || !schema.implies().is_empty())
        && let Some(id) = store.label_id_txn(txn, &schema.name)?
    {
        let staged_writes = std::collections::HashMap::new();
        if let Some(node) = coordinode_storage::engine::claims::evaluate::first_node_typed_by(
            engine,
            id,
            &staged_writes,
        )? {
            let why = match schema.kind() {
                LabelKind::Auxiliary => "an auxiliary label cannot be a node's type".to_string(),
                LabelKind::Type => format!(
                    "a type that has stored nodes cannot gain implied labels ({})",
                    schema.implies().join(", ")
                ),
            };
            return Err(ExecutionError::SchemaViolation(format!(
                ":{} cannot be declared so: node {} has it as its type, and {why}",
                schema.name,
                node.as_raw()
            )));
        }
    }
    // Every implied label is a type a node can carry, so each has an
    // identity from the declaration on: lookup and indexes reach a node
    // through it before any node stores it.
    for implied in schema.implies() {
        store.bind_label_txn(txn, implied)?;
    }
    Ok(())
}
