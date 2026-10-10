//! Index system: B-tree property indexes whose entries live in the index
//! partition and are maintained through the writing transaction, plus the
//! vector and full-text registries.

pub mod build;
pub mod definition;
pub mod lifecycle;
pub mod ops;
pub mod registry;
mod repair;
pub mod ttl_reaper;

pub mod vector_build;
pub mod vector_registry;

pub mod coverage;
pub mod text_registry;

pub use crate::planner::logical::{NumericCmp, VectorPredicate};
pub use coverage::{IndexCoverage, IndexDelta};
pub use definition::{
    BuildFailure, BuildState, CheckPhase, CheckState, DuplicateRepair, DuplicateRepairRecord,
    GenerationId, IndexBuildRecord, IndexCheck, IndexDefinition, IndexDescriptor, IndexId,
    IndexIntegrityRecord, IndexMaintenance, IndexProfile, IndexState, IndexType, Integrity,
    Mismatch, NamespaceIndexPolicy, OnlineDuringBuild, ProfileSource, TextFieldConfig,
    TextIndexConfig, VectorIndexConfig,
};
pub use lifecycle::{
    BuildEnvironment, BuildError, BuildIndex, BuildPhase, BuildStatus, CheckOutcome,
    CheckRequestError, CheckStatus, DEFAULT_CHECK_INTERVAL, DEFAULT_CHECK_MAX_REPAIRS,
    DEFAULT_CHECK_PAGE, DEFAULT_STATEMENT_WAIT, DEFAULT_UNIQUE_ADMISSION_READ_LIMIT,
    IndexBuildConfig, IndexBuildOutcome, IndexBuildService,
};

/// How a maintenance operation names the index it acts on: by its stable
/// identity, or by the name bound to it now. The two never stand for each
/// other: a name that looks like a number is a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexSelector {
    /// The index with this identity.
    Id(IndexId),
    /// The index this name binds.
    Name(String),
}

impl IndexSelector {
    /// The index this selector names now, with its definition.
    ///
    /// # Errors
    ///
    /// A storage failure, or no index matches.
    pub fn resolve(
        &self,
        engine: &coordinode_storage::engine::core::StorageEngine,
    ) -> Result<IndexDefinition, IndexSelectorError> {
        use coordinode_modality::{IndexStore as _, LocalIndexStore};
        let store = LocalIndexStore::new(engine);
        let id = match self {
            Self::Id(id) => *id,
            Self::Name(name) => store
                .resolve_name(name)
                .map_err(|e| IndexSelectorError::Storage(e.to_string()))?
                .ok_or_else(|| IndexSelectorError::NotFound(self.to_string()))?,
        };
        store
            .load_definition(id)
            .map_err(|e| IndexSelectorError::Storage(e.to_string()))?
            .ok_or_else(|| IndexSelectorError::NotFound(self.to_string()))
    }
}

impl core::fmt::Display for IndexSelector {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Id(id) => write!(f, "index id {}", id.as_raw()),
            Self::Name(name) => write!(f, "index '{name}'"),
        }
    }
}

/// Why an [`IndexSelector`] did not resolve.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IndexSelectorError {
    /// No index matches.
    #[error("no {0}")]
    NotFound(String),
    /// The catalog could not be read.
    #[error("read the index catalog: {0}")]
    Storage(String),
}
pub use registry::{
    IndexRegistry, IndexWriteError, MarkNotDurable, PropertyChange, UniqueClaim, UniqueViolation,
    props_lookup,
};
pub use text_registry::TextIndexRegistry;
pub use vector_build::{BuildOutcome, BuildTarget, VectorBuild};
pub use vector_registry::{BuildToken, VectorIndexRegistry};
