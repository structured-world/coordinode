//! A change-feed interest given by name, bound to what its names are bound
//! to now: the label identities a reader follows and the relationship types.

use coordinode_storage::engine::change_feed::Interest;
use coordinode_storage::engine::core::StorageEngine;

use crate::error::StoreResult;
use crate::schema::{LocalSchemaStore, SchemaStore as _};

/// Why an interest given by name is not one the feed follows.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InterestRefusal {
    /// No label is bound to the name.
    #[error("label :{0} is not bound")]
    Unbound(String),
}

/// The interest that follows the nodes of `labels` and the relationships of
/// `edge_types`, `None` for either following all of that kind, in the
/// namespace `engine` is bound to and no other. A label name
/// is bound to the label it names now: a label dropped and created again
/// later is another label, whose nodes this interest does not follow. A
/// label a type implies is followed through the nodes of every type
/// implying it, as each change was committed under.
///
/// # Errors
///
/// The catalog could not be read. A name the feed cannot follow is the
/// inner `Err`.
pub fn resolve_change_interest(
    engine: &StorageEngine,
    labels: Option<&[String]>,
    edge_types: Option<&[String]>,
) -> StoreResult<Result<Interest, InterestRefusal>> {
    resolve_with(engine, labels, edge_types, |store, name| {
        store.label_id(name)
    })
}

/// [`resolve_change_interest`] with each name bound to the label it named as
/// `snapshot` reads the catalog: a reader starting from a cut at that
/// snapshot follows the labels in force there. A name with no live binding
/// there is bound as the catalog binds it now.
///
/// # Errors
///
/// As [`resolve_change_interest`].
pub fn resolve_change_interest_at(
    engine: &StorageEngine,
    snapshot: coordinode_storage::engine::StorageSnapshot,
    labels: Option<&[String]>,
    edge_types: Option<&[String]>,
) -> StoreResult<Result<Interest, InterestRefusal>> {
    resolve_with(engine, labels, edge_types, |store, name| {
        store.label_id_at(snapshot, name)
    })
}

fn resolve_with(
    engine: &StorageEngine,
    labels: Option<&[String]>,
    edge_types: Option<&[String]>,
    mut bind: impl FnMut(
        &LocalSchemaStore<'_>,
        &str,
    ) -> StoreResult<Option<coordinode_core::schema::label_id::LabelId>>,
) -> StoreResult<Result<Interest, InterestRefusal>> {
    let store = LocalSchemaStore::new(engine);
    let labels = match labels {
        None => None,
        Some(names) => {
            let mut ids = Vec::with_capacity(names.len());
            for name in names {
                match bind(&store, name)? {
                    Some(id) => ids.push(id),
                    None => return Ok(Err(InterestRefusal::Unbound(name.clone()))),
                }
            }
            Some(ids)
        }
    };
    let edge_types = edge_types.map(|names| names.iter().map(|n| n.as_bytes().to_vec()).collect());
    Ok(Ok(
        Interest::new(labels, edge_types).in_namespace(engine.namespace())
    ))
}
