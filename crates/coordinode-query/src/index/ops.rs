//! Index catalog operations over [`coordinode_modality::LocalIndexStore`].
//!
//! Entries are maintained through the statement transaction by
//! [`super::IndexRegistry`]; these helpers read and write the definition
//! catalog.

use coordinode_modality::{IndexStore as _, LocalIndexStore, StoreError};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::error::StorageError;

use super::definition::{IndexDefinition, IndexState};

/// Convert [`coordinode_modality::StoreError`] back into the
/// [`StorageError`] vocabulary callers of this module use.
fn map_store_err(e: StoreError) -> StorageError {
    match e {
        StoreError::Storage(se) => se,
        other => StorageError::PartitionNotFound {
            name: format!("index store: {other}"),
        },
    }
}

/// Save an index definition to the catalog directly, outside the log. For
/// state a member keeps about itself; a definition the deployment shares is
/// written by the statement that creates it.
pub fn save_index_definition(
    engine: &StorageEngine,
    index: &IndexDefinition,
) -> Result<(), StorageError> {
    LocalIndexStore::new(engine)
        .put_definition(index)
        .map_err(map_store_err)
}

/// Update only the `state` field of a persisted index definition.
///
/// Returns `Ok(false)` if the index has no persisted definition (the
/// caller's race to handle).
pub fn save_index_state(
    engine: &StorageEngine,
    name: &str,
    state: IndexState,
) -> Result<bool, StorageError> {
    LocalIndexStore::new(engine)
        .set_definition_state(name, state)
        .map_err(map_store_err)
}

/// Load index definition from the index-store catalog.
pub fn load_index_definition(
    engine: &StorageEngine,
    name: &str,
) -> Result<Option<IndexDefinition>, StorageError> {
    LocalIndexStore::new(engine)
        .load_definition(name)
        .map_err(map_store_err)
}

/// List every persisted index definition in `schema:idx:` order.
///
/// Skips entries whose body fails to decode rather than aborting the
/// whole list (a corrupt index def shouldn't take out the registry).
pub fn list_index_definitions(
    engine: &StorageEngine,
) -> Result<Vec<IndexDefinition>, StorageError> {
    LocalIndexStore::new(engine)
        .list_definitions()
        .map_err(map_store_err)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
