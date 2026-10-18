# Changelog

All notable changes to this crate are documented in this file.

## v0.7.3 - 2026-10-18

### Added

- *(embed)* replicated embedded database behind a feature

### Performance

- *(storage)* flush the device only where recovery needs it
- *(storage)* defer the log sync of volatile writes, not the write

## v0.7.1 - 2026-10-13

### Added

- *(index)* maintain and provision the built-in label lookup
- *(index)* label posting codec and lookup catalog record
- *(index)* carry index copy proofs through transfers
- *(index)* guard each local index copy before it serves
- *(storage)* store index generations under local installations
- *(index)* check B-tree indexes against their records and repair them
- *(storage)* keep exact edge cardinality counts at commit
- *(schema)* resolve edge identity apart from temporality
- *(schema)* declare a partial unique index as a scoped constraint
- *(schema)* repair stored duplicates during a unique build

### Fixed

- *(index)* bind builds and checks to the definition, not a held version
- *(index)* bind a check's repair to the check its executor holds
- *(query)* store index unfitness when found and lift it only as proved
- *(query)* keep a member's unfit index copy excluded until proved here
- *(storage)* fold node deltas into a whole write of the same node
- *(reads)* reuse the statement schema view
- *(query)* search temporal labels at their valid state
- *(storage)* hide rows a transaction deleted from its scans
- *(modality)* refuse an unresolved temporal edge type on every write
- refuse to open over unreadable index and consensus state
- *(index)* refuse to open over an index catalog record it cannot read

### Performance

- *(query)* charge index lookups to the statement budget
- *(query)* stream node scans within the statement budget
- *(query)* bound property delta chains with whole writes
- *(query)* take TTL candidates from a ready anchor index
- *(query)* write a top-level SET as a property delta
- *(storage)* borrow node bytes during decode

## v0.7.0 - 2026-10-06

### Added

- *(index)* build full-text and vector indexes as member-owned builds
- *(index)* run B-tree index builds as durable engine-owned operations
- *(search)* answer text and vector searches at a named timestamp
- *(schema)* [**breaking**] replace the schema service with v2 and named constraints
- *(query)* CREATE/DROP CONSTRAINT for node labels
- *(query)* read temporal nodes at the statement's current time
- *(vector)* remove nodes and reuse their slots beside live search
- *(modality)* insert vectors under the shared index lock
- schema claims, event-driven background work
- journal index writes, speed up the Raft path
- *(query)* let the caller bound the wait for a building vector index

### Documentation

- drop internal ids from core, modality, site theme and examples
- *(modality)* drop internal ids and document paths

### Fixed

- *(schema)* read a schema's pointer and body from one snapshot
- *(restore)* [**breaking**] keep every NodeId, refuse issued ones, restore schema and indexes
- *(query)* finish a vector build under writes that never pause
- *(vector)* build indexes from a tap of applied writes
- *(deps)* require the lsm-tree release the code needs

### Refactored

- *(index)* name indexes by stable identity and generation
- *(query)* pair the vector registry with its build engine

## v0.6.0 - 2026-09-23

### Added

- *(txn)* protect a MERGE that acted on an absence it observed
- *(txn)* writers state what their result depends on

### Documentation

- fix broken links and stray tags in the API docs

### Fixed

- *(storage)* never let the watermark pass a snapshot just handed out
- *(storage)* keep adjacency operands in the order they were staged

### Performance

- *(txn)* decide a live endpoint without reading its record
- *(storage)* let a counter chain fold before its base is known

### Refactored

- *(txn)* install a commit's protections in one place

## 0.5.7 - 2026-09-01

#### Added

- *(vector)* drain writes into an index while it builds

---

## 0.5.6 - 2026-09-01

#### Performance

- *(txn)* coalesce counter deltas per key in the transaction buffer
- *(stats)* incremental node/label counters replace the statistics scan

---

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

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
