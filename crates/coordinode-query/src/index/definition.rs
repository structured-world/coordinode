//! Index metadata definitions.
//!
//! The definition types are catalog records owned by Layer 4
//! (`coordinode-modality`), alongside the [`IndexStore`] that persists
//! them. This module re-exports them so query-layer callers keep their
//! existing `crate::index::definition::*` paths while the storage of
//! these definitions lives below the query engine.
//!
//! [`IndexStore`]: coordinode_modality::IndexStore

pub use coordinode_modality::index_def::{
    IndexDefinition, IndexState, IndexType, OnlineDuringBuild, PartialFilter, TextFieldConfig,
    TextIndexConfig, VectorIndexConfig,
};
