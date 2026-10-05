//! Index system: B-tree property indexes whose entries live in the index
//! partition and are maintained through the writing transaction, plus the
//! vector and full-text registries.

pub mod build;
pub mod definition;
pub mod lifecycle;
pub mod ops;
pub mod registry;
pub mod ttl_reaper;

pub mod vector_build;
pub mod vector_registry;

pub mod coverage;
pub mod text_registry;

pub use crate::planner::logical::{NumericCmp, VectorPredicate};
pub use coverage::{IndexCoverage, IndexDelta};
pub use definition::{
    BuildFailure, BuildState, GenerationId, IndexBuildRecord, IndexDefinition, IndexDescriptor,
    IndexId, IndexMaintenance, IndexProfile, IndexState, IndexType, NamespaceIndexPolicy,
    OnlineDuringBuild, ProfileSource, TextFieldConfig, TextIndexConfig, VectorIndexConfig,
};
pub use lifecycle::{BuildEnvironment, BuildError, IndexBuildOutcome, IndexBuildService};
pub use registry::{
    IndexRegistry, IndexWriteError, PropertyChange, UniqueClaim, UniqueViolation, props_lookup,
};
pub use text_registry::TextIndexRegistry;
pub use vector_build::{BuildOutcome, BuildTarget, VectorBuild};
pub use vector_registry::{BuildToken, VectorIndexRegistry};
