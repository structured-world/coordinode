//! B-tree index system: single-field property indexes stored in the `idx:` partition.
//!
//! Indexes accelerate property lookups from O(N) full scan to O(log N) B-tree lookup.
//! Key format: `idx:<index_name>:<encoded_value>:<node_id>`

pub mod build;
pub mod definition;
pub mod ops;
pub mod registry;
pub mod ttl;
pub mod ttl_reaper;

pub mod vector_registry;

pub mod text_registry;

pub use definition::{
    IndexDefinition, IndexState, IndexType, OnlineDuringBuild, TextFieldConfig, TextIndexConfig,
    VectorIndexConfig,
};
pub use ops::{
    create_index_entries, create_index_entry, delete_index_entries, delete_index_entry, index_scan,
    index_scan_exact,
};
pub use registry::{IndexRegistry, UniqueViolation};
pub use text_registry::TextIndexRegistry;
pub use vector_registry::VectorIndexRegistry;
