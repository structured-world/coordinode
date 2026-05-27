# Changelog

## v0.4.4 — 2026-05-27

### coordinode-auth
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-auth-v0.4.3...coordinode-auth-v0.4.4) - 2026-05-27

#### Added

- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

### coordinode-bench
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-bench-v0.4.3...coordinode-bench-v0.4.4) - 2026-05-27

#### Added

- *(bench)* R700+R704 — coordinode-bench harness + ann-benchmarks SIFT1M adapter (Stage 1)

### coordinode-bench-vector-ann
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-bench-vector-ann-v0.4.3...coordinode-bench-vector-ann-v0.4.4) - 2026-05-27

#### Added

- *(bench-vector-ann)* --quantization {none,sq8,rabitq} CLI flag
- *(bench-vector-ann)* record process RSS in BenchReport metrics
- *(bench)* R700+R704 — coordinode-bench harness + ann-benchmarks SIFT1M adapter (Stage 1)

#### Testing

- *(vector)* add hnswlib SIFT1M single-thread baseline + thread pinning

### coordinode-client
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-client-v0.4.3...coordinode-client-v0.4.4) - 2026-05-27

#### Added

- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(query)* [**breaking**] add rrf_score Cypher function with RankFuse operator
- *(client)* causal session API — CausalToken, execute_causal_write/read (G089)
- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)
- *(consistency)* implement R142 causal consistency sessions
- *(client)* add coordinode-client crate with source location tracking

#### Fixed

- *(ci)* resolve release-plz cargo package failures for coordinode-client
- *(client)* add replication proto module and new ExecuteCypherRequest fields
- *(client)* use publish.workspace = true (consistent with other crates)
- *(client)* add tokio-test dev-dep; remove stale execute_cypher_annotated reference

#### Testing

- *(client,server)* cover params+source gRPC branch and invalid endpoint

### coordinode-cluster
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-cluster-v0.4.3...coordinode-cluster-v0.4.4) - 2026-05-27

#### Added

- *(cluster)* Layer 6 ClusterTopology + ShardRouting traits + CE impls

#### Testing

- *(cluster)* doctests + edge cases + ADR-028 helpers + benches + proptest

### coordinode-core
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-core-v0.4.3...coordinode-core-v0.4.4) - 2026-05-27

#### Added

- *(modality)* introduce coordinode-modality crate with Schema/Blob/Index stores
- *(storage)* per-version node key + __ingestion_ts__ for TEMPORAL labels
- *(cypher)* CREATE NODE TYPE DDL with TEMPORAL flag (bitemporal nodes scaffold)
- *(triggers)* storage layout + DDL executors + probe helper (R190 part 2)
- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(temporal)* bitemporal edge types with valid-time semantics
- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(txn)* add per-shard MaxAssignedWatermark + WaitForTs primitive
- *(query)* ATTACH DOCUMENT — demote graph node to nested DOCUMENT property
- *(core)* implement HybridLogicalClock for CE timestamps (R143)
- *(storage)* implement standalone WAL for crash durability
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(raft)* add WaitForMajorityService for batched proposal coalescing (G047)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Documentation

- *(triggers)* scrub internal task / ADR references + SHOW filter bug fix

#### Fixed

- *(storage)* gate every write path + typed propagation to gRPC client
- *(query)* TTL scope=Subtree now deletes target_field, not anchor

#### Performance

- *(codec)* switch UidEncoder/Decoder to StreamVByte Coder1234
- *(query)* reuse adjacency key buffer in graph traversal hot path

#### Refactored

- *(core,query)* R165 last raw encoder — Mutation::delete_edge_props typed constructor

#### Testing

- *(core)* add roundtrip test for ComputedSpec::Ttl with target_field=Some
- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)

### coordinode-embed
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-embed-v0.4.3...coordinode-embed-v0.4.4) - 2026-05-27

#### Added

- *(storage)* cascade also fires at Full + metric/concurrency tests
- *(storage)* background capacity scanner + hard-limit-strategy edges
- *(storage)* per-endpoint capacity tracking + hard-limit enforcement
- *(storage)* page-checksum wire-through + ECC policy config surface
- *(storage)* per-LSM-level endpoint routing + cascade eviction
- *(storage)* R156 + R157 — multi-endpoint storage placement
- *(temporal)* R172d — pattern predicate into temporal target
- *(temporal)* R172d initial slice — traversal into temporal target
- *(temporal)* R172c Phase 3c — DETACH/ATTACH on temporal nodes (partial)
- *(temporal)* R172c Phase 3b — nested PropertyPath / doc_* fns on temporal
- *(temporal)* R172c Phase 3 — REMOVE on temporal as close+open new version
- *(temporal)* R172c Phase 3 — DELETE on temporal as positive bitemporal fact
- *(storage)* R172c Phase 2 — temporal node SET close-current + open-new
- *(storage)* R172c Phase 1 — temporal node SET valid_to + valid_from immutability
- *(storage)* per-version node key + __ingestion_ts__ for TEMPORAL labels
- *(cypher)* CREATE NODE TYPE DDL with TEMPORAL flag (bitemporal nodes scaffold)
- *(triggers)* expand BEFORE COMMIT firing to edge SET/MERGE/DELETE
- *(triggers)* BEFORE COMMIT firing on SET / DELETE / CREATE-edge
- *(triggers)* BEFORE COMMIT firing on node CREATE (R191 first cut)
- *(triggers)* validate body source at DDL time + WITH-passthrough coverage
- *(triggers)* storage layout + DDL executors + probe helper (R190 part 2)
- *(cypher)* trigger DDL grammar + AST + parser + L1/L2 cycle tracking (R190 part 1)
- *(cypher)* native MERGE NODES (a, b) INTO target
- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(query)* snapshot API contract tests + fix modality_count over-promotion
- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(query)* expose applied_watermark handle on ExecutionContext
- *(storage)* implement standalone WAL for crash durability
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL
- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer
- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement
- *(query)* implement standalone MERGE relationship (G074)
- *(query)* HNSW-accelerated vector top-K via planner optimization
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(server)* wire DrainBuffer with RaftProposalPipeline in cluster mode (G063)
- *(query)* COMPUTED VECTOR_DECAY planner pattern detection (R084)
- *(query)* SSE encrypted search via Cypher DDL + encrypted_match() (G017)
- *(query)* adaptive parallel traversal via rayon (G010) + varlen edge props fix (G066)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Documentation

- *(temporal)* R172c/R172d temporal-node mutation surface + NOT predicate test
- *(triggers)* scrub internal task / ADR references + SHOW filter bug fix
- *(merge-nodes)* close coverage of reference, index, compatibility, README + 3 tests

#### Fixed

- *(storage)* capacity scanner counts every endpoint file, not only SSTs
- *(storage)* R172b safe-reject for UPSERT ON CREATE + pattern predicates
- *(storage)* R172b safe-reject for temporal labels in REMOVE / MERGE / ATTACH / DETACH
- *(triggers)* MERGE NODES fires source DELETE + target UPDATE + cascade
- *(triggers)* REMOVE / UPSERT ON CREATE / DETACH DOCUMENT firing
- *(triggers)* fire DELETE triggers from ATTACH DOCUMENT cascade path
- *(triggers)* tighten edge UPDATE firing + cover temporal/MERGE/docs
- *(executor)* propagate variable-bound property columns through WITH projection
- *(executor)* Cypher three-valued logic for NULL comparisons + edge-case audit
- *(query)* text_match() hard-fails on missing FT-index
- *(executor)* RETURN must not expose SET value when write was not applied
- *(query)* support query parameters in percentileCont/percentileDisc
- *(query)* clean up B-tree index entry on REMOVE property
- *(query)* update B-tree index on SET property
- *(query)* clean up B-tree index entries on node DELETE/DETACH DELETE
- *(embed)* add missing target_field to ComputedSpec::Ttl in integration tests

#### Performance

- *(query)* HNSW writes from CREATE row-stream are batched per statement
- *(embed)* plan cache — skip parse + analyze + build_logical_plan on repeats
- *(executor)* cache schema label per node per statement (R-API6)

#### Refactored

- *(vector)* migrate quantization config from bool to QuantizationCodec enum
- *(embed)* execute_cypher_impl is now &self; add shared entry point
- *(embed)* per-call QuerySession replaces self.* save/restore dance
- *(embed)* wrap FieldInterner in Arc<RwLock> on the Database side
- *(tests)* embed + storage migration to in-memory fixtures (Database::open_in_memory)
- *(query/tests)* R166 migration — 4 query test files on dual-FS fixture
- *(embed)* sweep raw encoder usage to LocalNodeStore
- *(storage,query)* move OCC tracking to Layer 3 Coordinator (G104)

#### Testing

- *(capacity)* trigger fail-fast on CapacityExhausted, no retry loop
- *(storage)* compaction-driven capacity recovery — writes resume automatically
- *(storage)* regression tests for ungated write paths + propagation
- *(storage)* per-LSM-level routing — WAL replay + primary-evict edges
- *(storage)* per-LSM-level routing edge cases
- *(temporal)* edge cases — doc_pull/add_to_set, Merge/Replace, varlen, multi-label, multi-segment ATTACH
- *(storage)* cover R172b audit gaps — valid_to type-mismatch + pre-epoch valid_from
- *(triggers)* BEFORE COMMIT CREATE — multi-label, $after Map, multi-trigger
- WITH passthrough composability + disabled-trigger persistence + empty-body reject
- *(merge-nodes)* STRICT happy-path + mixed self-loop and peers
- *(merge-nodes)* cover no-transfer drop, temporal edges, multi-type, composability
- *(merge-nodes)* close STRICT extra-map gap + composability/index coverage
- *(embed)* full integration coverage for CREATE/DROP VECTOR INDEX (R-API3)
- *(embed)* complete R-API3 integration test suite for CREATE/DROP VECTOR INDEX
- *(schema)* add reopen test — unique constraint enforced after load_all
- *(embed)* add integration test for TTL Subtree+target_field (G068)

### coordinode-gold-bench
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-gold-bench-v0.4.3...coordinode-gold-bench-v0.4.4) - 2026-05-27

#### Added

- *(benches)* L2 gold suite foundation — YCSB Workload A + C + composite report

#### Documentation

- *(benches)* compression-aware methodology — zstd × zstd and none × none only
- *(benches)* rewrite methodology — multi-model competitor matrix replaces Redis tier

### coordinode-integration
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-integration-v0.4.3...coordinode-integration-v0.4.4) - 2026-05-27

#### Added

- *(temporal)* bitemporal edge types with valid-time semantics
- *(query)* [**breaking**] add rrf_score Cypher function with RankFuse operator
- *(client)* causal session API — CausalToken, execute_causal_write/read (G089)

#### Fixed

- *(tests)* add at_timestamp field to ReadConcern callsites in workspace integration crate
- *(integration)* isolate test harness processes from nextest group tracking
- *(executor)* RETURN must not expose SET value when write was not applied
- *(integration)* allow expect_used in cluster test (clippy gate)
- *(integration)* add description, proto_gen fallback, and build.rs fix

#### Testing

- *(integration)* bug5 SIGKILL restart + wait_for_leader harness helper
- *(integration)* R150 server startup and NodeInfoLayer response headers
- *(integration)* add G088 gRPC integration tests + wire write_concern
- *(integration)* add ClusterService gRPC integration tests for R091c
- *(integration)* add FLEXIBLE-mode MATCH visibility regression test after restart
- *(integration)* add G082 regression test — SET on vector property

### coordinode-modality
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-modality-v0.4.3...coordinode-modality-v0.4.4) - 2026-05-27

#### Added

- *(modality,timeseries)* G103 sub-system #3 — bitemporal __ingestion_ts__ axis
- *(modality,timeseries)* G103 sub-system #4 — overflow compactor primitives
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

- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix
- *(modality/spatial)* G101 infrastructure — adaptive bailout disabled pending upstream lsm-tree seek primitive
- *(modality/spatial)* Z-curve subrange decomposition (G101)

#### Refactored

- *(storage/coordinator)* extract MultiModalCoordinator trait (G105)

#### Testing

- *(storage,modality)* G101 audit close — range_scan API + CRS dispatch + stronger exclusion
- *(modality,storage)* reduce proptest cases for faster regression runs
- *(modality)* proptest harness + remaining edge cases + docs hygiene
- *(modality)* contract clarifications + cross-store integration
- *(modality/edge)* concurrency stress on adjacency merge operators
- *(modality)* edge-case coverage for Blob/Index/Vector stores
- *(modality/document)* edge-case coverage for all 7 DocDelta variants

#### Revert

- *(modality/spatial)* G101 reverted — naive decomposition regressed bench

### coordinode-query
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-query-v0.4.3...coordinode-query-v0.4.4) - 2026-05-27

#### Added

- *(vector)* C1 day 6 — HnswConfig::max_elements drives pre-allocation
- *(query/index)* list_index_definitions helper + registry::load_all migration
- *(modality,query)* SchemaStore::list_labels / list_edge_types + ttl_reaper migration
- *(storage,query)* OccScope typed contains helpers + audit-test migration
- *(modality/node)* add get_at_seqno + scan_shard + migrate build.rs (R165 slice 2)
- *(modality)* introduce coordinode-modality crate with Schema/Blob/Index stores
- *(storage)* R156 + R157 — multi-endpoint storage placement
- *(temporal)* R172d — pattern predicate into temporal target
- *(temporal)* R172d initial slice — traversal into temporal target
- *(temporal)* R172c Phase 3c — DETACH/ATTACH on temporal nodes (partial)
- *(temporal)* R172c Phase 3b — nested PropertyPath / doc_* fns on temporal
- *(temporal)* R172c Phase 3 — REMOVE on temporal as close+open new version
- *(temporal)* R172c Phase 3 — DELETE on temporal as positive bitemporal fact
- *(storage)* R172c Phase 2 — temporal node SET close-current + open-new
- *(storage)* R172c Phase 1 — temporal node SET valid_to + valid_from immutability
- *(storage)* per-version node key + __ingestion_ts__ for TEMPORAL labels
- *(cypher)* CREATE NODE TYPE DDL with TEMPORAL flag (bitemporal nodes scaffold)
- *(triggers)* expand BEFORE COMMIT firing to edge SET/MERGE/DELETE
- *(triggers)* BEFORE COMMIT firing on SET / DELETE / CREATE-edge
- *(triggers)* BEFORE COMMIT firing on node CREATE (R191 first cut)
- *(triggers)* validate body source at DDL time + WITH-passthrough coverage
- *(triggers)* storage layout + DDL executors + probe helper (R190 part 2)
- *(cypher)* trigger DDL grammar + AST + parser + L1/L2 cycle tracking (R190 part 1)
- *(cypher)* native MERGE NODES (a, b) INTO target
- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(temporal)* bitemporal edge types with valid-time semantics
- *(query)* snapshot API contract tests + fix modality_count over-promotion
- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(query)* expose applied_watermark handle on ExecutionContext
- *(query)* add doc_score Cypher function for document-level aggregate
- *(query)* [**breaking**] add rrf_score Cypher function with RankFuse operator
- *(query)* hybrid_score() scoring helper (R-HYB2 part 1/3)
- *(query)* text_score() composition + guard against silent-0 on missing FT index
- *(query)* ATTACH DOCUMENT — demote graph node to nested DOCUMENT property
- *(query)* DETACH DOCUMENT — promote nested property to graph node + edge
- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)
- *(schema)* complete R-API5 schema modes enforcement
- *(schema)* enforce required fields at CREATE + multi-update tests
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(schema)* implement SchemaMode enforcement in executor (R-API5)
- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL
- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer
- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement
- *(query)* add MERGE ALL — Cartesian-product relationship upsert
- *(query)* implement standalone MERGE relationship (G074)
- *(query)* implement pattern predicates in WHERE clause
- *(query)* implement type(r) and labels(n) scalar functions
- *(query)* HNSW-accelerated vector top-K via planner optimization
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(query)* COMPUTED VECTOR_DECAY planner pattern detection (R084)
- *(query)* SSE encrypted search via Cypher DDL + encrypted_match() (G017)
- *(query)* adaptive parallel traversal via rayon (G010) + varlen edge props fix (G066)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Documentation

- *(triggers)* scrub internal task / ADR references + SHOW filter bug fix

#### Fixed

- *(query/server)* preserve CapacityExhausted type through Cypher pipeline → gRPC
- *(storage)* R172b safe-reject for B-tree index scan + log HNSW snapshot gap
- *(storage)* R172b safe-reject for UPSERT ON CREATE + pattern predicates
- *(storage)* R172b safe-reject for temporal labels in REMOVE / MERGE / ATTACH / DETACH
- *(triggers)* MERGE NODES fires source DELETE + target UPDATE + cascade
- *(triggers)* REMOVE / UPSERT ON CREATE / DETACH DOCUMENT firing
- *(triggers)* fire DELETE triggers from ATTACH DOCUMENT cascade path
- *(triggers)* tighten edge UPDATE firing + cover temporal/MERGE/docs
- *(executor)* propagate variable-bound property columns through WITH projection
- *(executor)* Cypher three-valued logic for NULL comparisons + edge-case audit
- *(query)* text_match() hard-fails on missing FT-index
- *(clippy)* resolve 11 warnings on Rust 1.95
- *(executor)* RETURN must not expose SET value when write was not applied
- *(query)* wire parameter substitution into execute() + expand tests
- *(query)* support query parameters in percentileCont/percentileDisc
- *(query)* implement percentileCont/percentileDisc with correct percentile arg
- *(query)* clean up B-tree index entry on REMOVE property
- *(query)* update B-tree index on SET property
- *(query)* clean up B-tree index entries on node DELETE/DETACH DELETE
- *(executor)* schema enforcement for PropertyPath, DocFunction, map SET ops
- *(vector)* fill labels/properties in VectorResult, respect distance metric
- *(query)* skip Subtree removal when target_field already absent
- *(query)* short-circuit reap_label when Subtree target_field_id unresolved
- *(query)* log error when Subtree target_field_id unresolved
- *(query)* skip Subtree deletion when target_field_id unresolved
- *(query)* TTL scope=Subtree now deletes target_field, not anchor
- *(query)* store and check edge properties in MERGE relationship (G075)
- *(executor)* support MERGE relationship patterns (G069, G070, G072)
- *(query)* track OCC read-set in parallel traversal path (G067)

#### Performance

- *(query)* HNSW writes from CREATE row-stream are batched per statement
- *(query)* reuse adjacency key buffer in graph traversal hot path
- *(executor)* cache schema label per node per statement (R-API6)

#### Refactored

- *(vector)* migrate quantization config from bool to QuantizationCodec enum
- *(query/tests)* R166 migration — 4 query test files on dual-FS fixture
- *(core,query)* R165 last raw encoder — Mutation::delete_edge_props typed constructor
- *(query/tests)* migrate integration-test fixtures to LocalNodeStore
- *(query/tests)* R166 finish — ttl_reaper fixtures migrated to LocalNodeStore
- *(query/tests)* R166 test fixture migration to LocalNodeStore
- *(query/index/ops)* route through LocalIndexStore (R165 slice 12)
- *(query/runner)* typed edge-property delete + transfer/update migration (R165 slice 11)
- *(query/runner)* typed temporal edge-property helpers + 3 more sites (R165 slice 10)
- *(query/runner)* typed edge-property helpers + first EdgeStore sites (R165 slice 9)
- *(query/runner)* SET / REMOVE / schema-peek migration (R165 slice 8)
- *(query/runner)* delete redundant byte-CAS in execute_merge (R165 slice 6)
- *(query/runner)* DELETE + DETACH/ATTACH branching migration (R165 slice 7)
- *(query/runner)* temporal-node typed helpers + 4-block migration (R165 slice 5)
- *(query/runner)* typed node helpers + 7-site migration (R165 slice 4)
- *(storage,query)* move OCC tracking to Layer 3 Coordinator (G104)
- *(query/runner)* migrate label-index node-fetch loop to LocalNodeStore (R165 slice 3)
- *(query/ttl)* migrate ttl.rs node ops to LocalNodeStore (R165 slice)

#### Testing

- *(query)* R165 audit close — EdgeStore OCC invariants + lockdown coverage gate
- *(query)* R165 encoder lockdown regression gate (slice 13)
- *(query)* R165 slice 4 second audit — OCC invariant + doc fix
- *(query)* R165 slice 4 audit — edge cases + 2 more SET sites
- *(query)* mvcc_flush idempotency + read-only short-circuit
- *(query)* RYOW + legacy-mode OCC invariants, scrub task IDs
- *(storage,query)* edge cases + dyn dispatch for G104/G105
- *(merge-nodes)* close STRICT extra-map gap + composability/index coverage
- *(query)* R-SNAP1 exact mode + AS OF target + hint docs
- *(query)* freeze hybrid scoring API surface with contract tests
- *(query)* cover R-HYB2c edge cases missed in the initial PR
- *(query)* cover R-HYB1b empty-input shortcut and document text_match guard
- *(query)* add regression tests for text_match hard-fail on missing FT-index
- *(query)* cover rrf_score edge cases missed in the initial PR
- *(query)* verify is_write() classifies all DDL clause variants
- *(query)* add unit tests for Query::is_write() AST predicate
- *(semantic)* add WITH * regression tests for analyze_with fix
- *(executor)* add two-MERGE, self-loop, G069+G072 integration tests
- *(executor)* add ON MATCH SET, incoming direction, edge-property tests (G072/G075)
- *(executor)* add G072 edge-case tests + document G074 gap

### coordinode-raft
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-raft-v0.4.3...coordinode-raft-v0.4.4) - 2026-05-27

#### Added

- *(storage)* per-LSM-level endpoint routing + cascade eviction
- *(storage)* R156 + R157 — multi-endpoint storage placement
- *(raft)* wire MaxAssignedWatermark into apply_proposal path
- *(server)* R150 — monolithic binary --mode=full, shared :7080, NodeInfoLayer
- *(cluster)* node decommission protocol + unified Raft write path
- *(cluster)* implement cluster join protocol (R091b)
- *(storage)* implement standalone WAL for crash durability
- *(raft)* R141 follower reads — ReadFence, SyncPerBatch persist fix
- *(raft)* chunked gRPC snapshot transfer to prevent OOM (G046)
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(raft)* add retry with exponential backoff to batch drain loop (G047b)
- *(raft)* add WaitForMajorityService for batched proposal coalescing (G047)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Fixed

- *(storage)* gate every write path + typed propagation to gRPC client
- *(storage)* gate oplog purge on cross-partition flush watermark
- *(raft)* recover last_log_id from oplog on unclean shutdown restart
- *(cluster)* rollback Learner on change_membership failure in monitor_and_promote
- *(server)* resolve proto submodule and clippy::panic in tests
- *(raft)* reduce chunk size to 2MB, add multi-chunk integration test
- *(ci)* update raft build.rs proto path and deny.toml format

#### Testing

- *(raft)* add 3-node pruning decommission test as final R091c entry
- *(cluster)* R091c decommission protocol test suite
- *(raft)* R141 complete test coverage — follower scenarios + StaleReplica
- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)

### coordinode-s3
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-s3-v0.4.3...coordinode-s3-v0.4.4) - 2026-05-27

#### Added

- *(storage)* R156 + R157 — multi-endpoint storage placement
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

### coordinode-search
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-search-v0.4.3...coordinode-search-v0.4.4) - 2026-05-27

#### Added

- *(storage)* R156 + R157 — multi-endpoint storage placement
- *(search)* FTS MVCC snapshot filter via per-doc commit_ts + segment registry
- *(text-search)* implement TextService gRPC with fuzzy + language-aware search
- *(search)* external CJK dictionary loading from filesystem (G014)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Fixed

- *(ci)* replace cargo test with nextest, fix CJK env var race

#### Testing

- *(search)* direct unit tests for search_with_highlights_fuzzy and search_with_highlights_and_language
- *(text-search)* Ukrainian e2e + multi-property merge coverage

### coordinode-server
## [0.4.4](https://github.com/structured-world/coordinode/compare/v0.4.3...v0.4.4) - 2026-05-27

#### Added

- *(server)* wire CreateNodesBatch handler via UNWIND $rows AS r CREATE …
- *(storage)* R156 + R157 — multi-endpoint storage placement

#### Fixed

- *(query/server)* preserve CapacityExhausted type through Cypher pipeline → gRPC
- *(server)* migrate remaining gRPC services to capacity-aware error mapping
- *(storage)* gate every write path + typed propagation to gRPC client

#### Performance

- *(server)* route read-only handlers through .read() + execute_cypher_shared

#### Refactored

- *(vector)* migrate quantization config from bool to QuantizationCodec enum
- *(server)* swap std::sync::Mutex<Database> for parking_lot::RwLock<Database>
- *(embed)* wrap FieldInterner in Arc<RwLock> on the Database side

### coordinode-storage
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-storage-v0.4.3...coordinode-storage-v0.4.4) - 2026-05-27

#### Added

- *(storage,query)* OccScope typed contains helpers + audit-test migration
- *(storage)* cascade also fires at Full + metric/concurrency tests
- *(storage)* background capacity scanner + hard-limit-strategy edges
- *(storage)* per-endpoint capacity tracking + hard-limit enforcement
- *(storage)* page-checksum wire-through + ECC policy config surface
- *(storage)* per-LSM-level endpoint routing + cascade eviction
- *(storage)* R156 + R157 — multi-endpoint storage placement
- *(temporal)* R172c Phase 3b — nested PropertyPath / doc_* fns on temporal
- *(storage)* time-based memtable flush trigger to bound oplog retention
- *(storage)* implement standalone WAL for crash durability
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(storage)* add MemFs in-memory test backend support
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Fixed

- *(storage)* defer capacity-scanner first tick by interval to close warm-load race
- *(storage)* capacity scanner counts every endpoint file, not only SSTs
- *(storage)* gate every write path + typed propagation to gRPC client
- *(storage)* gate oplog purge on cross-partition flush watermark
- *(raft)* recover last_log_id from oplog on unclean shutdown restart

#### Performance

- *(modality/spatial)* Z-curve subrange decomposition (G101)
- *(storage)* batch Extra-targeting deltas in DocumentMerge
- *(storage)* parallel memtable writes within write batch (R091)

#### Refactored

- *(tests)* embed + storage migration to in-memory fixtures (Database::open_in_memory)
- *(storage,query)* move OCC tracking to Layer 3 Coordinator (G104)
- *(storage/coordinator)* extract MultiModalCoordinator trait (G105)
- *(storage/coordinator)* trim doctests to internal-crate scope
- *(storage)* extract Layer 3 Coordinator sub-module (R164)

#### Testing

- *(storage)* regression test for capacity-scanner warm-load race
- *(storage,modality)* G101 audit close — range_scan API + CRS dispatch + stronger exclusion
- *(modality,storage)* reduce proptest cases for faster regression runs
- *(query)* RYOW + legacy-mode OCC invariants, scrub task IDs
- *(storage,query)* edge cases + dyn dispatch for G104/G105
- *(storage)* final R164 coverage round + rustdoc cleanup
- *(storage/coordinator)* edge cases + doctests + concurrency
- *(storage)* page-ECC policy — builder + serde back-compat + Volatile edge

### coordinode-test-fixtures
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-test-fixtures-v0.4.3...coordinode-test-fixtures-v0.4.4) - 2026-05-27

#### Added

- *(test-fixtures)* new crate — engine_for_logic / engine_for_disk / engine_for_memory dual-FS test fixture

#### Performance

- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix

#### Refactored

- *(query/tests)* R166 migration — 4 query test files on dual-FS fixture

#### Testing

- *(test-fixtures)* audit closure — edge cases + doctest + CI matrix verification

### coordinode-timeseries
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-timeseries-v0.4.3...coordinode-timeseries-v0.4.4) - 2026-05-27

#### Added

- *(timeseries)* ε-policy — opt-in WITH BITEMPORAL via split write entry points (β: Cypher paused)
- *(timeseries)* close G103 #3 Gap #4 — PersistentMonotonicHlcClock with engine-backed restart monotonicity
- *(modality,timeseries)* G103 sub-system #3 — bitemporal __ingestion_ts__ axis
- *(modality,timeseries)* G103 sub-system #4 — overflow compactor primitives
- *(timeseries)* G103 slice C — Tier 3 overflow routing + background compactor
- *(timeseries)* G103 slice B — Tier 2 recently-closed LRU + reopen path
- *(timeseries)* new crate coordinode-timeseries (G103 slice A — BucketCatalog + Tier 1 buffer)

#### Testing

- *(timeseries)* G103 #3 audit closure — backfill on compact, edge case tests, restart-monotonicity gap documented

### coordinode-vector
## [0.4.4](https://github.com/structured-world/coordinode/compare/coordinode-vector-v0.4.3...coordinode-vector-v0.4.4) - 2026-05-27

#### Added

- *(vector)* set_rabitq_params for segment reload + serde round-trip tests
- *(vector)* wire RaBitQ codec into HNSW search hot path
- *(vector)* RaBitQ codec foundation + popcount distance kernel
- *(vector)* AtomicU64-packed entry-point + CAS-loop promotion
- *(vector)* C3 day 4 — prune-pass restores recall, insert_batch wired to parallel apply
- *(vector)* C3 day 3 — apply_insert_plans_parallel (opt-in, lossy)
- *(vector)* C3 day 2 — &self write helpers + cas_add_neighbour_to
- *(vector)* C3 day 1 — cas_append + replace primitives
- *(vector)* C2 day 3 — insert_batch vs serial criterion bench
- *(vector)* C2 day 2 — insert_batch with rayon parallel planning
- *(vector)* C2 day 1 — split insert into compute_insert_plan + apply
- *(vector)* C1 day 7 — parallel-search QPS bench
- *(vector)* C1 day 6 — HnswConfig::max_elements drives pre-allocation
- *(vector)* C1 day 5 — atomic mirror is the sole storage
- *(vector)* C1 day 4 — granular dual-write helpers for atomic mirror
- *(vector)* C1 day 3b — search read path now lock-free
- *(vector)* C1 day 3a — auto-sync atomic mirror after every insert/update
- *(vector)* C1 day 2 — atomic-mirror field + manual sync helper
- *(vector)* IndexHealthState + HnswBuildScheduler (rebalance prereqs)
- *(vector)* C1 day 1 — AtomicNeighbourList<N> scaffold for lock-free HNSW
- *(bench)* R700+R704 — coordinode-bench harness + ann-benchmarks SIFT1M adapter (Stage 1)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Documentation

- *(vector)* C3 day 6 — record measured 14.6× speedup in bench doc

#### Fixed

- *(vector)* HNSW search beam must be at least k for any caller
- *(modality,vector)* pass &StorageEngine in doctests + add LockFreeNeighbours::is_empty
- *(vector)* preserve back-edges when neighbour list is at M_MAX0 cap
- *(vector)* dedupe duplicate ids within insert_batch + proptest stress
- *(vector)* update HNSW graph position when node vector is overwritten (G082)

#### Performance

- *(vector)* criterion harness for RaBitQ popcount kernel
- *(vector)* SQ8 dequantize into reusable scratch + SIMD
- *(vector)* cache query L2 norm per HNSW search (cosine path)
- *(vector)* C3 day 5b — parallel prune-pass via rayon
- *(vector)* C3 day 5a — dedupe backfill before prune-pass

#### Refactored

- *(vector)* migrate quantization config from bool to QuantizationCodec enum

#### Testing

- *(vector)* regression tests for HNSW recall when ef_search < k
- *(vector)* wire loom interleaving suite for AtomicNeighbourList
- *(vector)* stress AtomicNeighbourList cas_append vs concurrent snapshot
- *(vector)* add proptest stress for multi-batch + concurrent search

---

## v0.4.3 — 2026-05-17

### coordinode-core
## [0.4.3](https://github.com/structured-world/coordinode/compare/v0.4.2...v0.4.3) - 2026-05-17

#### Added

- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(temporal)* bitemporal edge types with valid-time semantics

### coordinode-embed
## [0.4.3](https://github.com/structured-world/coordinode/compare/v0.4.2...v0.4.3) - 2026-05-17

#### Added

- *(cypher)* native MERGE NODES (a, b) INTO target
- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through

#### Documentation

- *(merge-nodes)* close coverage of reference, index, compatibility, README + 3 tests

#### Fixed

- *(executor)* Cypher three-valued logic for NULL comparisons + edge-case audit

#### Testing

- *(merge-nodes)* STRICT happy-path + mixed self-loop and peers
- *(merge-nodes)* cover no-transfer drop, temporal edges, multi-type, composability
- *(merge-nodes)* close STRICT extra-map gap + composability/index coverage

### coordinode-query
## [0.4.3](https://github.com/structured-world/coordinode/compare/v0.4.2...v0.4.3) - 2026-05-17

#### Added

- *(cypher)* native MERGE NODES (a, b) INTO target
- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(temporal)* bitemporal edge types with valid-time semantics

#### Fixed

- *(executor)* Cypher three-valued logic for NULL comparisons + edge-case audit

#### Testing

- *(merge-nodes)* close STRICT extra-map gap + composability/index coverage

### coordinode-server
## [0.4.3](https://github.com/structured-world/coordinode/compare/v0.4.2...v0.4.3) - 2026-05-17

#### Added

- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through

---

## v0.4.2 — 2026-05-11

### coordinode-raft
## [0.4.2](https://github.com/structured-world/coordinode/compare/v0.4.1...v0.4.2) - 2026-05-11

#### Fixed

- *(storage)* gate oplog purge on cross-partition flush watermark

### coordinode-storage
## [0.4.2](https://github.com/structured-world/coordinode/compare/v0.4.1...v0.4.2) - 2026-05-11

#### Added

- *(storage)* time-based memtable flush trigger to bound oplog retention

#### Fixed

- *(storage)* gate oplog purge on cross-partition flush watermark

---

## v0.4.1 — 2026-04-18

### coordinode-core
## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(txn)* add per-shard MaxAssignedWatermark + WaitForTs primitive

### coordinode-embed
## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(query)* snapshot API contract tests + fix modality_count over-promotion
- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(query)* expose applied_watermark handle on ExecutionContext

### coordinode-query
## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(query)* snapshot API contract tests + fix modality_count over-promotion
- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(query)* expose applied_watermark handle on ExecutionContext

#### Testing

- *(query)* R-SNAP1 exact mode + AS OF target + hint docs

### coordinode-raft
## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(raft)* wire MaxAssignedWatermark into apply_proposal path

### coordinode-search
## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(search)* FTS MVCC snapshot filter via per-doc commit_ts + segment registry

---

## v0.4.0 — 2026-04-17

### coordinode-embed
## [0.4.0](https://github.com/structured-world/coordinode/compare/v0.3.20...v0.4.0) - 2026-04-17

#### Fixed

- *(query)* text_match() hard-fails on missing FT-index

### coordinode-query
## [0.4.0](https://github.com/structured-world/coordinode/compare/v0.3.20...v0.4.0) - 2026-04-17

#### Added

- *(query)* add doc_score Cypher function for document-level aggregate
- *(query)* [**breaking**] add rrf_score Cypher function with RankFuse operator
- *(query)* hybrid_score() scoring helper (R-HYB2 part 1/3)
- *(query)* text_score() composition + guard against silent-0 on missing FT index

#### Fixed

- *(query)* text_match() hard-fails on missing FT-index

#### Testing

- *(query)* freeze hybrid scoring API surface with contract tests
- *(query)* cover R-HYB2c edge cases missed in the initial PR
- *(query)* cover R-HYB1b empty-input shortcut and document text_match guard
- *(query)* add regression tests for text_match hard-fail on missing FT-index
- *(query)* cover rrf_score edge cases missed in the initial PR

### coordinode-server
## [0.4.0](https://github.com/structured-world/coordinode/compare/v0.3.20...v0.4.0) - 2026-04-17

#### Added

- *(query)* [**breaking**] add rrf_score Cypher function with RankFuse operator

---

## Unreleased

### coordinode-query

#### Added

- *(query)* `rrf_score([method_exprs…], {vector: …, text: …})` — Reciprocal Rank Fusion Cypher function. N-method rank fusion with competition ranks, `k=60` (IR standard, non-tunable), per-method direction from HNSW metric config. Supports node vectors, edge vectors (brute-force), and BM25 text methods.

### coordinode-server

#### Removed (BREAKING)

- *(proto)* `TextService.HybridTextVectorSearch` RPC, `HybridTextVectorSearchRequest` / `HybridTextVectorSearchResponse` / `HybridResult` messages, `POST /v1/query/text/hybrid` HTTP endpoint. Superseded by the general-purpose Cypher function `rrf_score([methods…], {vector, text})` invoked via `CypherService.ExecuteCypher`. The Cypher form supports N methods (not 2), edge vectors, configurable HNSW metrics, and composes with MATCH / WHERE / ORDER BY / LIMIT in a single plan. Callers: replace the RPC with an equivalent Cypher query.

## v0.3.20 — 2026-04-17

### coordinode-core
## [0.3.20](https://github.com/structured-world/coordinode/compare/v0.3.19...v0.3.20) - 2026-04-17

#### Added

- *(query)* ATTACH DOCUMENT — demote graph node to nested DOCUMENT property

### coordinode-query
## [0.3.20](https://github.com/structured-world/coordinode/compare/v0.3.19...v0.3.20) - 2026-04-17

#### Added

- *(query)* ATTACH DOCUMENT — demote graph node to nested DOCUMENT property
- *(query)* DETACH DOCUMENT — promote nested property to graph node + edge

---

## v0.3.19 — 2026-04-17

### coordinode-query
## [0.3.19](https://github.com/structured-world/coordinode/compare/v0.3.18...v0.3.19) - 2026-04-17

#### Fixed

- *(clippy)* resolve 11 warnings on Rust 1.95

---

## v0.3.18 — 2026-04-16

### coordinode-raft
## [0.3.18](https://github.com/structured-world/coordinode/compare/v0.3.17...v0.3.18) - 2026-04-16

#### Added

- *(server)* R150 — monolithic binary --mode=full, shared :7080, NodeInfoLayer

#### Fixed

- *(raft)* recover last_log_id from oplog on unclean shutdown restart

### coordinode-server
## [0.3.18](https://github.com/structured-world/coordinode/compare/v0.3.17...v0.3.18) - 2026-04-16

#### Added

- *(server)* R150 — monolithic binary --mode=full, shared :7080, NodeInfoLayer

### coordinode-storage
## [0.3.18](https://github.com/structured-world/coordinode/compare/v0.3.17...v0.3.18) - 2026-04-16

#### Fixed

- *(raft)* recover last_log_id from oplog on unclean shutdown restart

---

## v0.3.17 — 2026-04-15

### coordinode-core
## [0.3.17](https://github.com/structured-world/coordinode/compare/v0.3.16...v0.3.17) - 2026-04-15

#### Added

- *(core)* implement HybridLogicalClock for CE timestamps (R143)

### coordinode-embed
## [0.3.17](https://github.com/structured-world/coordinode/compare/v0.3.16...v0.3.17) - 2026-04-15

#### Fixed

- *(executor)* RETURN must not expose SET value when write was not applied

### coordinode-query
## [0.3.17](https://github.com/structured-world/coordinode/compare/v0.3.16...v0.3.17) - 2026-04-15

#### Added

- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)

#### Fixed

- *(executor)* RETURN must not expose SET value when write was not applied

#### Testing

- *(query)* verify is_write() classifies all DDL clause variants
- *(query)* add unit tests for Query::is_write() AST predicate

### coordinode-server
## [0.3.17](https://github.com/structured-world/coordinode/compare/v0.3.16...v0.3.17) - 2026-04-15

#### Added

- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)
- *(consistency)* implement R142 causal consistency sessions

---

## v0.3.16 — 2026-04-15

### coordinode-server
## [0.3.16](https://github.com/structured-world/coordinode/compare/v0.3.15...v0.3.16) - 2026-04-15

#### Added

- *(server)* gate REST proxy behind rest-proxy feature flag
- *(server)* embed REST proxy in coordinode binary

---

## v0.3.15 — 2026-04-15

### coordinode-core
## [0.3.15](https://github.com/structured-world/coordinode/compare/v0.3.14...v0.3.15) - 2026-04-15

#### Performance

- *(codec)* switch UidEncoder/Decoder to StreamVByte Coder1234
- *(query)* reuse adjacency key buffer in graph traversal hot path

### coordinode-embed
## [0.3.15](https://github.com/structured-world/coordinode/compare/v0.3.14...v0.3.15) - 2026-04-15

#### Fixed

- *(query)* support query parameters in percentileCont/percentileDisc

### coordinode-query
## [0.3.15](https://github.com/structured-world/coordinode/compare/v0.3.14...v0.3.15) - 2026-04-15

#### Fixed

- *(query)* wire parameter substitution into execute() + expand tests
- *(query)* support query parameters in percentileCont/percentileDisc
- *(query)* implement percentileCont/percentileDisc with correct percentile arg

#### Performance

- *(query)* reuse adjacency key buffer in graph traversal hot path

### coordinode-storage
## [0.3.15](https://github.com/structured-world/coordinode/compare/v0.3.14...v0.3.15) - 2026-04-15

#### Performance

- *(storage)* batch Extra-targeting deltas in DocumentMerge

---

## v0.3.13 — 2026-04-14

### coordinode-embed
## [0.3.13](https://github.com/structured-world/coordinode/compare/v0.3.12...v0.3.13) - 2026-04-14

#### Fixed

- *(query)* clean up B-tree index entry on REMOVE property
- *(query)* update B-tree index on SET property
- *(query)* clean up B-tree index entries on node DELETE/DETACH DELETE

### coordinode-query
## [0.3.13](https://github.com/structured-world/coordinode/compare/v0.3.12...v0.3.13) - 2026-04-14

#### Fixed

- *(query)* clean up B-tree index entry on REMOVE property
- *(query)* update B-tree index on SET property
- *(query)* clean up B-tree index entries on node DELETE/DETACH DELETE

---

## v0.3.12 — 2026-04-14

### coordinode-raft
## [0.3.12](https://github.com/structured-world/coordinode/compare/v0.3.11...v0.3.12) - 2026-04-14

#### Added

- *(cluster)* node decommission protocol + unified Raft write path

#### Testing

- *(raft)* add 3-node pruning decommission test as final R091c entry
- *(cluster)* R091c decommission protocol test suite

### coordinode-server
## [0.3.12](https://github.com/structured-world/coordinode/compare/v0.3.11...v0.3.12) - 2026-04-14

#### Added

- *(cluster)* node decommission protocol + unified Raft write path

---

## v0.3.11 — 2026-04-14

### coordinode-core
## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(storage)* implement standalone WAL for crash durability

### coordinode-embed
## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(storage)* implement standalone WAL for crash durability

### coordinode-raft
## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(cluster)* implement cluster join protocol (R091b)
- *(storage)* implement standalone WAL for crash durability

#### Fixed

- *(cluster)* rollback Learner on change_membership failure in monitor_and_promote

### coordinode-server
## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(cluster)* implement cluster join protocol (R091b)
- *(storage)* implement standalone WAL for crash durability

#### Testing

- *(server)* add CLI unit tests for AdminNodeJoin parsing

### coordinode-storage
## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(storage)* implement standalone WAL for crash durability

### coordinode-vector
## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Fixed

- *(vector)* update HNSW graph position when node vector is overwritten (G082)

---

## v0.3.10 — 2026-04-14

### coordinode-raft
## [0.3.10](https://github.com/structured-world/coordinode/compare/v0.3.9...v0.3.10) - 2026-04-14

#### Added

- *(raft)* R141 follower reads — ReadFence, SyncPerBatch persist fix

#### Fixed

- *(server)* resolve proto submodule and clippy::panic in tests

#### Testing

- *(raft)* R141 complete test coverage — follower scenarios + StaleReplica

### coordinode-server
## [0.3.10](https://github.com/structured-world/coordinode/compare/v0.3.9...v0.3.10) - 2026-04-14

#### Added

- *(raft)* R141 follower reads — ReadFence, SyncPerBatch persist fix

#### Testing

- *(client,server)* cover params+source gRPC branch and invalid endpoint
- *(server)* add gRPC source tracking round-trip test

### coordinode-storage
## [0.3.10](https://github.com/structured-world/coordinode/compare/v0.3.9...v0.3.10) - 2026-04-14

#### Performance

- *(storage)* parallel memtable writes within write batch (R091)

---

## v0.3.9 — 2026-04-13

### coordinode-embed
## [0.3.9](https://github.com/structured-world/coordinode/compare/v0.3.8...v0.3.9) - 2026-04-13

#### Performance

- *(executor)* cache schema label per node per statement (R-API6)

### coordinode-query
## [0.3.9](https://github.com/structured-world/coordinode/compare/v0.3.8...v0.3.9) - 2026-04-13

#### Performance

- *(executor)* cache schema label per node per statement (R-API6)

### coordinode-search
## [0.3.9](https://github.com/structured-world/coordinode/compare/v0.3.8...v0.3.9) - 2026-04-13

#### Added

- *(text-search)* implement TextService gRPC with fuzzy + language-aware search

#### Testing

- *(search)* direct unit tests for search_with_highlights_fuzzy and search_with_highlights_and_language
- *(text-search)* Ukrainian e2e + multi-property merge coverage

### coordinode-server
## [0.3.9](https://github.com/structured-world/coordinode/compare/v0.3.8...v0.3.9) - 2026-04-13

#### Added

- *(text)* HybridTextVectorSearch with RRF (Reciprocal Rank Fusion)
- *(text-search)* implement TextService gRPC with fuzzy + language-aware search

#### Fixed

- *(graph)* traverse and get_node return full labels and properties
- *(traverse)* respect direction field in Traverse RPC

#### Performance

- *(executor)* cache schema label per node per statement (R-API6)

#### Testing

- *(e2e)* LangChain gRPC API correctness — all search modalities
- *(text-search)* Ukrainian e2e + multi-property merge coverage
- *(text-search)* verify explicit language search routes to Path C
- *(schema)* add DocFunction cache test + fix clippy in R-API6

---

## v0.3.8 — 2026-04-13

### coordinode-core
## [0.3.8](https://github.com/structured-world/coordinode/compare/v0.3.7...v0.3.8) - 2026-04-13

#### Added

- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE

### coordinode-embed
## [0.3.8](https://github.com/structured-world/coordinode/compare/v0.3.7...v0.3.8) - 2026-04-13

#### Added

- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE

### coordinode-query
## [0.3.8](https://github.com/structured-world/coordinode/compare/v0.3.7...v0.3.8) - 2026-04-13

#### Added

- *(schema)* complete R-API5 schema modes enforcement
- *(schema)* enforce required fields at CREATE + multi-update tests
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(schema)* implement SchemaMode enforcement in executor (R-API5)

#### Fixed

- *(executor)* schema enforcement for PropertyPath, DocFunction, map SET ops

### coordinode-server
## [0.3.8](https://github.com/structured-world/coordinode/compare/v0.3.7...v0.3.8) - 2026-04-13

#### Added

- *(schema)* complete R-API5 schema modes enforcement
- *(schema)* enforce required fields at CREATE + multi-update tests
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(schema)* implement SchemaMode enforcement in executor (R-API5)
- *(schema)* wire ComputedPropertyDefinition in CreateLabel gRPC API

#### Fixed

- *(executor)* schema enforcement for PropertyPath, DocFunction, map SET ops

#### Testing

- *(schema)* add thoroughness integration tests for R-API5 schema modes
- *(schema)* add validated_mode_set_extra_accepted_mismatch_rejected

---

## v0.3.6 — 2026-04-13

### coordinode-embed
## [0.3.6](https://github.com/structured-world/coordinode/compare/v0.3.5...v0.3.6) - 2026-04-13

#### Added

- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL

#### Testing

- *(embed)* full integration coverage for CREATE/DROP VECTOR INDEX (R-API3)
- *(embed)* complete R-API3 integration test suite for CREATE/DROP VECTOR INDEX

### coordinode-query
## [0.3.6](https://github.com/structured-world/coordinode/compare/v0.3.5...v0.3.6) - 2026-04-13

#### Added

- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL

---

## v0.3.5 — 2026-04-13

### coordinode-embed
## [0.3.5](https://github.com/structured-world/coordinode/compare/v0.3.4...v0.3.5) - 2026-04-13

#### Added

- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer

### coordinode-query
## [0.3.5](https://github.com/structured-world/coordinode/compare/v0.3.4...v0.3.5) - 2026-04-13

#### Added

- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer

---

## v0.3.4 — 2026-04-12

### coordinode-core
## [0.3.4](https://github.com/structured-world/coordinode/compare/v0.3.3...v0.3.4) - 2026-04-12

#### Fixed

- *(query)* TTL scope=Subtree now deletes target_field, not anchor

#### Testing

- *(core)* add roundtrip test for ComputedSpec::Ttl with target_field=Some

### coordinode-embed
## [0.3.4](https://github.com/structured-world/coordinode/compare/v0.3.3...v0.3.4) - 2026-04-12

#### Added

- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement

#### Fixed

- *(embed)* add missing target_field to ComputedSpec::Ttl in integration tests

#### Testing

- *(schema)* add reopen test — unique constraint enforced after load_all
- *(embed)* add integration test for TTL Subtree+target_field (G068)

### coordinode-query
## [0.3.4](https://github.com/structured-world/coordinode/compare/v0.3.3...v0.3.4) - 2026-04-12

#### Added

- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement
- *(query)* add MERGE ALL — Cartesian-product relationship upsert

#### Fixed

- *(vector)* fill labels/properties in VectorResult, respect distance metric
- *(query)* skip Subtree removal when target_field already absent
- *(query)* short-circuit reap_label when Subtree target_field_id unresolved
- *(query)* log error when Subtree target_field_id unresolved
- *(query)* skip Subtree deletion when target_field_id unresolved
- *(query)* TTL scope=Subtree now deletes target_field, not anchor

#### Testing

- *(semantic)* add WITH * regression tests for analyze_with fix

### coordinode-server
## [0.3.4](https://github.com/structured-world/coordinode/compare/v0.3.3...v0.3.4) - 2026-04-12

#### Added

- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement

#### Fixed

- *(vector)* fill labels/properties in VectorResult, respect distance metric
- *(vector)* honour distance metric parameter in VectorService

#### Testing

- *(vector)* strengthen R-FTS2 regression tests

