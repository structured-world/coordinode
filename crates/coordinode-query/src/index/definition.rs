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
    BuildFailure, BuildState, CheckPhase, CheckState, DuplicateRepair, DuplicateRepairRecord,
    GenerationId, IndexBuildRecord, IndexCheck, IndexDefinition, IndexDescriptor, IndexId,
    IndexIntegrityRecord, IndexMaintenance, IndexProfile, IndexState, IndexType, Integrity,
    Mismatch, NamespaceIndexPolicy, OnlineDuringBuild, PartialFilter, ProfileSource,
    TextFieldConfig, TextIndexConfig, VectorIndexConfig,
};
