//! Index catalog operations over [`coordinode_modality::LocalIndexStore`].
//!
//! Entries are maintained through the statement transaction by
//! [`super::IndexRegistry`]; these helpers read the definition catalog,
//! which only the log writes.

use coordinode_modality::{IndexStore as _, LocalIndexStore, StoreError};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::error::StorageError;

use super::definition::{IndexDefinition, IndexId};

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

/// Load index definition from the index-store catalog.
pub fn load_index_definition(
    engine: &StorageEngine,
    id: IndexId,
) -> Result<Option<IndexDefinition>, StorageError> {
    LocalIndexStore::new(engine)
        .load_definition(id)
        .map_err(map_store_err)
}

/// List every persisted index definition in identity order.
///
/// # Errors
///
/// [`StorageError::UnreadableCatalog`] for a definition this build cannot
/// decode: the list never leaves one out.
pub fn list_index_definitions(
    engine: &StorageEngine,
) -> Result<Vec<IndexDefinition>, StorageError> {
    LocalIndexStore::new(engine)
        .list_definitions()
        .map_err(map_store_err)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
