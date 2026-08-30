# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## 0.5.1 - 2026-08-29

#### Added

- *(query)* DROP TABLE for the relational TABLE modality
- *(session)* keyset-resumable server-side cursor
- *(triggers)* [**breaking**] execute AFTER COMMIT triggers via durable event journal
- *(storage)* batched multi_get for known-key sets
- *(modality)* DROP INDEX clears the index keyspace with one range tombstone
- *(modality)* typed direct edge-prop write, close restore encoder residuals
- *(modality)* add snapshot-aware EdgeStore::get_props_snapshot
- *(modality)* add per-label shard strategy to the vector index config
- *(query)* expose ef_search and rerank_candidates as vector index options
- *(spatial)* S2 geometry for WGS-84, Hilbert for Cartesian-2D
- *(edge)* discriminator-aware edge property keys and EdgeStore API
- *(modality)* add IndexStore clear + delete_raw, route index maintenance through the store
- *(modality)* VectorStore::knn_search_with_mode for exact path
- *(modality,timeseries)* G103 sub-system #3 - bitemporal __ingestion_ts__ axis
- *(modality,timeseries)* G103 sub-system #4 - overflow compactor primitives
- *(modality,query)* SchemaStore::list_labels / list_edge_types + ttl_reaper migration
- *(modality/node)* add get_at_seqno + scan_shard + migrate build.rs (R165 slice 2)
- *(modality/timeseries)* reopen_bucket + late-write flow test
- *(modality)* temporal edge methods (ADR-027)
- *(modality)* add SpatialStore + LocalSpatialStore
- *(modality)* add TimeSeriesStore + LocalTimeSeriesStore
- *(modality)* add VectorStore + LocalVectorStore
- *(modality)* add DocumentStore + LocalDocumentStore (ADR-015)
- *(modality)* add EdgeStore + LocalEdgeStore (non-temporal)
- *(modality)* add NodeStore + LocalNodeStore (temporal-aware)
- *(modality)* introduce coordinode-modality crate with Schema/Blob/Index stores

#### Documentation

- *(modality)* 100% method doctest coverage + remaining bench groups
- *(modality)* # Examples doctest on every non-trivial public method
- *(modality)* README + doctests + concurrency tests + benches

#### Fixed

- *(modality,vector)* pass &StorageEngine in doctests + add LockFreeNeighbours::is_empty
- *(modality/spatial)* real curve windowing in scan_within_bbox

#### Performance

- *(spatial)* Z-curve skip-scan via seekable range iterator
- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix
- *(modality/spatial)* G101 infrastructure - adaptive bailout disabled pending upstream lsm-tree seek primitive
- *(modality/spatial)* Z-curve subrange decomposition (G101)

#### Refactored

- extract unit tests into sibling test files
- *(modality)* split blob data plane from metadata plane
- *(modality)* thread Transaction through spatial, blob, time-series stores
- *(query)* own encrypted-index metadata in a typed store
- *(query)* persist index definitions through the index store
- *(query)* read TTL reaper state through typed stores
- *(modality)* own index definitions in the index store
- thread storage transaction through stores
- *(storage/coordinator)* extract MultiModalCoordinator trait (G105)

#### Testing

- *(storage,modality)* G101 audit close - range_scan API + CRS dispatch + stronger exclusion
- *(modality,storage)* reduce proptest cases for faster regression runs
- *(modality)* proptest harness + remaining edge cases + docs hygiene
- *(modality)* contract clarifications + cross-store integration
- *(modality/edge)* concurrency stress on adjacency merge operators
- *(modality)* edge-case coverage for Blob/Index/Vector stores
- *(modality/document)* edge-case coverage for all 7 DocDelta variants

#### Revert

- move per-label vector shard routing out of CE
- *(modality/spatial)* G101 reverted - naive decomposition regressed bench

---

## 0.5.0 - 2026-06-27

#### Added

- *(triggers)* [**breaking**] execute AFTER COMMIT triggers via durable event journal
- *(storage)* batched multi_get for known-key sets
- *(modality)* DROP INDEX clears the index keyspace with one range tombstone
- *(modality)* typed direct edge-prop write, close restore encoder residuals
- *(modality)* add snapshot-aware EdgeStore::get_props_snapshot
- *(modality)* add per-label shard strategy to the vector index config
- *(query)* expose ef_search and rerank_candidates as vector index options
- *(spatial)* S2 geometry for WGS-84, Hilbert for Cartesian-2D
- *(edge)* discriminator-aware edge property keys and EdgeStore API
- *(modality)* add IndexStore clear + delete_raw, route index maintenance through the store
- *(modality)* VectorStore::knn_search_with_mode for exact path
- *(modality,timeseries)* G103 sub-system #3 - bitemporal __ingestion_ts__ axis
- *(modality,timeseries)* G103 sub-system #4 - overflow compactor primitives
- *(modality,query)* SchemaStore::list_labels / list_edge_types + ttl_reaper migration
- *(modality/node)* add get_at_seqno + scan_shard + migrate build.rs (R165 slice 2)
- *(modality/timeseries)* reopen_bucket + late-write flow test
- *(modality)* temporal edge methods (ADR-027)
- *(modality)* add SpatialStore + LocalSpatialStore
- *(modality)* add TimeSeriesStore + LocalTimeSeriesStore
- *(modality)* add VectorStore + LocalVectorStore
- *(modality)* add DocumentStore + LocalDocumentStore (ADR-015)
- *(modality)* add EdgeStore + LocalEdgeStore (non-temporal)
- *(modality)* add NodeStore + LocalNodeStore (temporal-aware)
- *(modality)* introduce coordinode-modality crate with Schema/Blob/Index stores

#### Documentation

- *(modality)* 100% method doctest coverage + remaining bench groups
- *(modality)* # Examples doctest on every non-trivial public method
- *(modality)* README + doctests + concurrency tests + benches

#### Fixed

- *(modality,vector)* pass &StorageEngine in doctests + add LockFreeNeighbours::is_empty
- *(modality/spatial)* real curve windowing in scan_within_bbox

#### Performance

- *(spatial)* Z-curve skip-scan via seekable range iterator
- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix
- *(modality/spatial)* G101 infrastructure - adaptive bailout disabled pending upstream lsm-tree seek primitive
- *(modality/spatial)* Z-curve subrange decomposition (G101)

#### Refactored

- extract unit tests into sibling test files
- *(modality)* split blob data plane from metadata plane
- *(modality)* thread Transaction through spatial, blob, time-series stores
- *(query)* own encrypted-index metadata in a typed store
- *(query)* persist index definitions through the index store
- *(query)* read TTL reaper state through typed stores
- *(modality)* own index definitions in the index store
- thread storage transaction through stores
- *(storage/coordinator)* extract MultiModalCoordinator trait (G105)

#### Testing

- *(storage,modality)* G101 audit close - range_scan API + CRS dispatch + stronger exclusion
- *(modality,storage)* reduce proptest cases for faster regression runs
- *(modality)* proptest harness + remaining edge cases + docs hygiene
- *(modality)* contract clarifications + cross-store integration
- *(modality/edge)* concurrency stress on adjacency merge operators
- *(modality)* edge-case coverage for Blob/Index/Vector stores
- *(modality/document)* edge-case coverage for all 7 DocDelta variants

#### Revert

- move per-label vector shard routing out of CE
- *(modality/spatial)* G101 reverted - naive decomposition regressed bench
