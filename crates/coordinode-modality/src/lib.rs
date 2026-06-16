//! Layer 4 — typed modality stores over the multimodal coordinator.
//!
//! This crate sits between [`coordinode_storage`] (Layer 3 coordinator +
//! Layer 2 tier-aware partitions + Layer 1 endpoints) and
//! `coordinode-query` (Layer 5 query engine). It exposes one trait per
//! modality, each with typed get / put / scan / delete plus
//! modality-specific operations. The traits hide partition keys,
//! encoders, and physical placement from the query layer.
//!
//! ## Why a separate crate
//!
//! - **Crate-level isolation of physical layout.** Once the query layer
//!   migrates to these traits (R165), `coordinode-query` no longer
//!   imports `Partition` or any `encode_*` key builder. The
//!   partition-key surface drops to `pub(in store impl)` and the
//!   physical layout becomes an internal detail.
//! - **Monomorphization for the planner.** Callers above are generic
//!   over the trait (`fn execute<E: EdgeStore>(...)`) so the compiler
//!   specializes per concrete impl — same zero-cost dispatch as direct
//!   storage calls, no `dyn` indirection on hot paths.
//! - **CE/EE seam.** The trait surface is identical in CE and EE; only
//!   the concrete impls differ (`LocalXStore` in CE, `ShardedXStore`
//!   in EE). The query layer compiles against the trait — picking the
//!   right impl is a binary build choice, not a runtime branch.
//!
//! ## Store inventory
//!
//! | Modality | Trait | CE impl |
//! |----------|-------|---------|
//! | Schema (label / edge-type / migration / chunk-assignment DDL state) | [`SchemaStore`] | [`LocalSchemaStore`] |
//! | Blob (binary chunks + blob references) | [`BlobStore`] | [`LocalBlobStore`] |
//! | Index (secondary indexes — btree, hash, fulltext term postings) | [`IndexStore`] | [`LocalIndexStore`] |
//! | Node (incl. temporal versioning, ADR-027) | [`NodeStore`] | [`LocalNodeStore`] |
//! | Edge (adjacency + properties, non-temporal) | [`EdgeStore`] | [`LocalEdgeStore`] |
//! | Vector | VectorStore | — (next PR) |
//! | Document | DocumentStore | — (next PR) |
//! | TimeSeries | TimeSeriesStore | — (next PR) |
//! | Spatial | SpatialStore | — (next PR) |
//!
//! Remaining six stores land in follow-up commits grouped by modality
//! family (graph, vectors, specialty). See the storage stack
//! architecture document for the full Layer 4 contract.
//!
//! ## Error model
//!
//! All store operations return [`StoreError`], a thin classification
//! wrapper over [`coordinode_storage::error::StorageError`]. Storage
//! errors propagate verbatim through the [`StoreError::Storage`]
//! variant so capacity-exhausted, checksum-mismatch, and other typed
//! engine errors are preserved end-to-end (the gRPC layer drills into
//! the chain to surface `RESOURCE_EXHAUSTED` correctly).

#![deny(missing_docs)]

pub mod blob;
pub mod document;
pub mod edge;
pub mod encrypted_index;
pub mod error;
pub mod index;
pub mod index_def;
pub mod node;
pub mod schema;
pub mod spatial;
pub mod timeseries;
pub mod trigger;
pub mod vector;

pub use blob::{BlobStore, LocalBlobStore};
pub use document::{DocumentStore, LocalDocumentStore};
pub use edge::{EdgeStore, LocalEdgeStore};
pub use encrypted_index::{
    EncryptedIndexDefinition, EncryptedIndexStore, LocalEncryptedIndexStore,
};
pub use error::{StoreError, StoreResult};
pub use index::{IndexStore, LocalIndexStore};
pub use index_def::{
    IndexDefinition, IndexState, IndexType, OnlineDuringBuild, PartialFilter, TextFieldConfig,
    TextIndexConfig, VectorIndexConfig,
};
pub use node::{LocalNodeStore, NodeStore};
pub use schema::{LocalSchemaStore, SchemaStore};
pub use spatial::{distance, Bbox, Crs, LocalSpatialStore, Point, SpatialStore};
pub use timeseries::{
    Bucket, BucketControl, FieldStats, LocalTimeSeriesStore, Measurement, OverflowEntry,
    TimeSeriesStore,
};
pub use trigger::{LocalTriggerStore, TriggerStore};
pub use vector::{LocalVectorStore, VectorStore};
