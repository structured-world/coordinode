# coordinode-modality

Layer 4 of the CoordiNode storage stack: **typed per-modality APIs**
over the multimodal storage coordinator.

This crate sits between `coordinode-storage` (Layer 3 coordinator +
Layer 2 tier-aware partitions + Layer 1 endpoints) and
`coordinode-query` (Layer 5 query engine). It exposes one trait per
modality, each with typed get / put / scan / delete plus
modality-specific operations. The traits hide partition keys,
encoders, and physical placement from the query layer.

## Store inventory

| Modality | Trait | CE impl |
|----------|-------|---------|
| Schema (label / edge-type / migration DDL) | `SchemaStore` | `LocalSchemaStore` |
| Blob (content-addressed chunks + blob refs) | `BlobStore` | `LocalBlobStore` |
| Index (secondary B-tree, compound, full-text) | `IndexStore` | `LocalIndexStore` |
| Node (CRUD + temporal versioning ADR-027) | `NodeStore` | `LocalNodeStore` |
| Edge (adjacency + properties; temporal ADR-027) | `EdgeStore` | `LocalEdgeStore` |
| Document (path-targeted partial updates ADR-015) | `DocumentStore` | `LocalDocumentStore` |
| Vector (HNSW approximate nearest neighbour) | `VectorStore` | `LocalVectorStore` |
| TimeSeries (bucket persistence + overflow) | `TimeSeriesStore` | `LocalTimeSeriesStore` |
| Spatial (4 CRS: WGS-84 2D/3D, Cartesian 2D/3D) | `SpatialStore` | `LocalSpatialStore` |

## Why a separate crate

- **Crate-level isolation of physical layout.** Once the query layer
  migrates to these traits, `coordinode-query` no longer imports
  `Partition` or any `encode_*` key builder.
- **Monomorphization for the planner.** Callers above are generic
  over the trait (`fn execute<E: EdgeStore>(...)`) so the compiler
  specialises per concrete impl — zero-cost dispatch, no `dyn`
  indirection on hot paths.
- **CE/EE seam.** The trait surface is identical in CE and EE; only
  the concrete impls differ (`LocalXStore` in CE, `ShardedXStore`
  in EE). The query layer compiles against the trait — picking the
  right impl is a binary build choice, not a runtime branch.

## Quick example

```rust,no_run
use coordinode_modality::{LocalNodeStore, NodeStore};
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_storage::engine::config::{
    Durability, EndpointConfig, Media, StorageConfig, Tier,
};
use coordinode_storage::engine::core::StorageEngine;

let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    "ep",
    std::path::Path::new("/tmp/store"),
    Media::Hdd,
    Durability::Durable,
    Tier::Warm,
)]);
let engine = StorageEngine::open(&config).unwrap();
let store = LocalNodeStore::new(&engine);

let id = NodeId::from_raw(1);
store.put(0, id, &NodeRecord::new("User")).unwrap();
let loaded = store.get(0, id).unwrap();
```

## Error model

All store operations return `StoreError`, a thin classification
wrapper over `coordinode_storage::error::StorageError`. Storage
errors propagate verbatim through the `StoreError::Storage`
variant — capacity-exhausted, checksum-mismatch, and other typed
engine errors are preserved end-to-end, so a gRPC layer above can
drill into the chain and surface `RESOURCE_EXHAUSTED` correctly.

## Test coverage

127 tests across 9 stores: per-store unit tests, cross-store
integration tests (`tests/cross_store_flow.rs`), and property-based
invariants (`tests/proptest_invariants.rs`). Run with:

```sh
cargo nextest run -p coordinode-modality
```

Edge cases covered include: corrupt bytes, capacity propagation,
i64 boundary timestamps, Morton Z-shape false positives, stale-row
contract on point moves, concurrent merge-operator stress on
super-nodes, heterogeneous TimeSeries field schemas, and HNSW
in-place id dedup.

## License

AGPL-3.0 (CE). See workspace `LICENSE`.
