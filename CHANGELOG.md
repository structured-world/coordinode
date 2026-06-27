# Changelog

## v0.5.0 — 2026-06-27

### coordinode-auth
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-auth-v0.4.3...coordinode-auth-v0.5.0) - 2026-06-27

#### Added

- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

### coordinode-bench
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-bench-v0.4.3...coordinode-bench-v0.5.0) - 2026-06-27

#### Added

- *(bench)* HNSW M sweep (16, 24, 32) + UI filter
- *(bench)* R700+R704 — coordinode-bench harness + ann-benchmarks SIFT1M adapter (Stage 1)

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)

### coordinode-client
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-client-v0.4.3...coordinode-client-v0.5.0) - 2026-06-27

#### Added

- *(server)* gRPC RPCs for interactive transactions
- *(client)* expose read preference for cypher reads
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(query)* [**breaking**] add rrf_score Cypher function with RankFuse operator
- *(client)* causal session API — CausalToken, execute_causal_write/read (G089)
- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)
- *(consistency)* implement R142 causal consistency sessions
- *(client)* add coordinode-client crate with source location tracking

#### Fixed

- *(client)* generate proto bindings at build time, drop stale proto_gen
- *(ci)* resolve release-plz cargo package failures for coordinode-client
- *(client)* add replication proto module and new ExecuteCypherRequest fields
- *(client)* use publish.workspace = true (consistent with other crates)
- *(client)* add tokio-test dev-dep; remove stale execute_cypher_annotated reference

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)

#### Testing

- *(client,server)* cover params+source gRPC branch and invalid endpoint

### coordinode-cluster
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-cluster-v0.4.3...coordinode-cluster-v0.5.0) - 2026-06-27

#### Added

- *(cluster)* chunk-assignment table for shard routing
- *(cluster)* crash-recovery replay for LocalStateMachine
- *(cluster)* CE LocalStateMachine state-machine backend
- *(cluster)* node state-machine backend trait + operation types
- *(cluster)* add VectorShardRouter trait + single-partition default
- *(cluster)* migration plan explain string
- *(cluster)* plumbed online-during-rebuild policy in planner
- *(cluster)* online-during-rebuild policy enum
- *(cluster)* local migration planner picks lowest-cost target
- *(cluster)* migration cost model with hnsw rebuild line
- *(cluster)* migration plan and cost types
- *(cluster)* Layer 6 ClusterTopology + ShardRouting traits + CE impls

#### Documentation

- *(cluster)* document online-during-rebuild policy
- *(cluster)* document the migration planner

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)

#### Testing

- *(cluster)* online-during-rebuild policy threading
- *(cluster)* planner picks remote endpoint on full source
- *(cluster)* doctests + edge cases + ADR-028 helpers + benches + proptest

### coordinode-core
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-core-v0.4.3...coordinode-core-v0.5.0) - 2026-06-27

#### Added

- *(triggers)* [**breaking**] execute AFTER COMMIT triggers via durable event journal
- *(core)* coalesce delete runs at the proposal producer
- *(storage)* MVCC range-delete apply path + partition cache invalidation
- *(edge)* discriminator-aware edge property keys and EdgeStore API
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(core)* unify edge-property value on one canonical sorted-array codec
- *(query)* add Path value type and nodes/relationships/length
- *(core)* add MultiVector value variant
- *(storage)* VectorF32 + VectorRerank partitions (ADR-033 revised)
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

- *(core)* move delete coalescing to core, operate on mutations
- extract unit tests into sibling test files
- *(core)* hoist try_extract_vector to a single canonical helper
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)
- *(core,query)* R165 last raw encoder — Mutation::delete_edge_props typed constructor

#### Testing

- *(core)* add roundtrip test for ComputedSpec::Ttl with target_field=Some
- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)

### coordinode-embed
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-embed-v0.4.3...coordinode-embed-v0.5.0) - 2026-06-27

#### Added

- *(triggers)* [**breaking**] execute AFTER COMMIT triggers via durable event journal
- *(query)* REDIRECT EDGES on temporal edge types
- *(query)* CLONE NODE on temporal labels with AS OF
- *(storage)* retained oplog journal + single-node repair for embedded
- *(query)* REDIRECT EDGES native procedure
- *(query)* CLONE NODE native procedure
- *(query)* support FOREACH update loop
- *(modality)* typed direct edge-prop write, close restore encoder residuals
- *(embed)* thread an extension-op registry through Database
- *(query)* extension-op seam for engine extensions
- *(modality)* add per-label shard strategy to the vector index config
- *(query)* expose ef_search and rerank_candidates as vector index options
- *(storage)* io_uring filesystem backend behind --features io-uring
- *(storage)* multi-endpoint topology config
- *(embed)* bound interactive transaction buffered writes
- *(server)* gRPC RPCs for interactive transactions
- *(embed)* interactive multi-statement transactions
- *(vector)* serving health + HLC freshness watermark for indexes
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(core)* unify edge-property value on one canonical sorted-array codec
- *(backup)* validate binary dump compatibility on restore
- *(restore)* selective restore via --only-labels filter
- *(restore)* transparent decompression and Hetionet hetnet-JSON import
- *(backup)* raft-snapshot backup and restore via the CLI
- *(query)* add Path value type and nodes/relationships/length
- *(backup)* restore Neo4j APOC json and cypher dumps
- *(backup)* implement cypher restore + align edge-prop wire format
- *(embed)* wire the vector oplog worker into Database
- *(embed)* vector index worker tails the oplog
- *(cluster)* replicate vector index DDL to followers
- *(query)* plan HnswScan for pure vector top-k
- *(planner)* maxsim_score top-k as a dedicated operator
- *(core)* add MultiVector value variant
- *(vector-index)* recover stale build state on engine reopen
- *(vector-index)* online-during-build policy on reads
- *(vector-index)* background backfill on CREATE VECTOR INDEX
- *(executor)* add Arc engine handle to ExecutionContext
- *(cypher)* CREATE VECTOR INDEX OPTIONS {quantization}
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

- *(cluster)* refresh follower interner on entry apply
- *(embed)* replicate field interner through the pipeline
- *(embed)* route cypher writes through the injected pipeline
- *(test)* rewrite rabitq-2bit wiring test on dense clustered data
- *(vector)* wire LsmVectorTier without re-entering interner lock
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

- *(query)* dedup variable-length traversal target emission
- *(query)* HNSW writes from CREATE row-stream are batched per statement
- *(embed)* plan cache — skip parse + analyze + build_logical_plan on repeats
- *(executor)* cache schema label per node per statement (R-API6)

#### Refactored

- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)
- *(embed)* read backup edge props via get_props_snapshot
- *(core)* hoist try_extract_vector to a single canonical helper
- *(search)* thread Transaction through the SSE token index
- thread storage transaction through stores
- *(vector)* migrate quantization config from bool to QuantizationCodec enum
- *(embed)* execute_cypher_impl is now &self; add shared entry point
- *(embed)* per-call QuerySession replaces self.* save/restore dance
- *(embed)* wrap FieldInterner in Arc<RwLock> on the Database side
- *(tests)* embed + storage migration to in-memory fixtures (Database::open_in_memory)
- *(query/tests)* R166 migration — 4 query test files on dual-FS fixture
- *(embed)* sweep raw encoder usage to LocalNodeStore
- *(storage,query)* move OCC tracking to Layer 3 Coordinator (G104)

#### Testing

- *(embed)* corrupt all post-checkpoint tables at several offsets
- *(embed)* target the SST tables dir for repair corruption
- *(embed)* scope repair corruption victim to the Node partition
- *(embed)* corrupt only post-checkpoint data in repair tests
- *(embed)* cover auto-on-open repair through Database::open
- *(embed)* gate integration suite against raw data-plane encoders
- *(embed)* load edge-type schema via store in reopen test
- *(embed)* probe trigger state via store in drop test
- *(embed)* pin multi-vector property round-trip with maxsim
- *(backup)* full data-equality roundtrip for binary dump/restore
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

#### Revert

- move per-label vector shard routing out of CE

### coordinode-integration
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-integration-v0.4.3...coordinode-integration-v0.5.0) - 2026-06-27

#### Added

- *(embed)* bound interactive transaction buffered writes
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

- *(cluster)* integration test for leader self-decommission
- *(integration)* bug5 SIGKILL restart + wait_for_leader harness helper
- *(integration)* R150 server startup and NodeInfoLayer response headers
- *(integration)* add G088 gRPC integration tests + wire write_concern
- *(integration)* add ClusterService gRPC integration tests for R091c
- *(integration)* add FLEXIBLE-mode MATCH visibility regression test after restart
- *(integration)* add G082 regression test — SET on vector property

### coordinode-modality
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-modality-v0.4.3...coordinode-modality-v0.5.0) - 2026-06-27

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

- *(spatial)* Z-curve skip-scan via seekable range iterator
- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix
- *(modality/spatial)* G101 infrastructure — adaptive bailout disabled pending upstream lsm-tree seek primitive
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

- *(storage,modality)* G101 audit close — range_scan API + CRS dispatch + stronger exclusion
- *(modality,storage)* reduce proptest cases for faster regression runs
- *(modality)* proptest harness + remaining edge cases + docs hygiene
- *(modality)* contract clarifications + cross-store integration
- *(modality/edge)* concurrency stress on adjacency merge operators
- *(modality)* edge-case coverage for Blob/Index/Vector stores
- *(modality/document)* edge-case coverage for all 7 DocDelta variants

#### Revert

- move per-label vector shard routing out of CE
- *(modality/spatial)* G101 reverted — naive decomposition regressed bench

### coordinode-query
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-query-v0.4.3...coordinode-query-v0.5.0) - 2026-06-27

#### Added

- *(triggers)* [**breaking**] execute AFTER COMMIT triggers via durable event journal
- *(query)* REDIRECT EDGES on temporal edge types
- *(query)* CLONE NODE on temporal labels with AS OF
- *(query)* REDIRECT EDGES native procedure
- *(query)* CLONE NODE native procedure
- *(query)* COUNT and COLLECT subquery expressions
- *(query)* emit push-down decision in EXPLAIN
- *(query)* support CALL { subquery }
- *(query)* support FOREACH update loop
- *(query)* support UNION and UNION ALL
- *(query)* add IS :: TYPE type predicate
- *(query)* =~ regex match operator
- *(query)* pattern comprehension with ctx-aware projection
- *(query)* list comprehension
- *(query)* route non-trivial pattern predicates through EXISTS
- *(query)* EXISTS { MATCH … } correlated subquery
- *(query)* list quantifier predicates all/any/none/single
- *(query)* reduce() list fold expression
- *(query)* Cypher list functions head, last, tail, range, isEmpty, keys
- *(query)* Cypher scalar functions
- *(query)* batch HNSW result hydration via multi_get
- *(storage)* batched multi_get for known-key sets
- *(query)* Cypher trigonometric functions
- *(query)* Cypher math functions
- *(query)* Cypher string functions
- *(query)* expose a vector-index-definition builder for extensions
- *(query)* capture a trailing extension clause on CREATE VECTOR INDEX
- *(query)* extension-op seam for engine extensions
- *(query)* route filtered vector search through the sharded layout
- *(query)* similarity-partitioned vector index layout in the registry
- *(modality)* add per-label shard strategy to the vector index config
- *(query)* expose ef_search and rerank_candidates as vector index options
- *(vector)* serving health + HLC freshness watermark for indexes
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(modality)* add IndexStore clear + delete_raw, route index maintenance through the store
- *(cypher)* bind a path for named single-relationship traversals
- *(cypher)* wire shortestPath() through grammar and planner
- *(query)* shortestPath returns a Path instead of a hop count
- *(query)* add Path value type and nodes/relationships/length
- *(embed)* vector index worker tails the oplog
- *(cluster)* replicate vector index DDL to followers
- *(query)* plan HnswScan for pure vector top-k
- *(query)* HnswScan index access path executor
- *(planner)* maxsim_score top-k as a dedicated operator
- *(query)* maxsim_score scalar in cypher evaluator
- *(vector-index)* numeric range predicates in pushdown
- *(planner)* parse cc_score and dbsf_score
- *(executor)* convex-combination and dbsf score fusion
- *(planner)* fusion strategy enum on rank-fuse op
- *(planner)* build predicate from match+where for vector top-k
- *(vector-index)* dispatch to filtered hnsw search on acorn strategy
- *(executor)* predicate evaluator for vector top-k
- *(planner)* predicate descriptor on vector top-k
- *(vector-index)* online-during-build policy on reads
- *(vector-index)* background backfill on CREATE VECTOR INDEX
- *(executor)* add Arc engine handle to ExecutionContext
- *(vector-index)* add IndexState enum with persisted state-only updates
- *(cypher)* CREATE VECTOR INDEX OPTIONS {quantization}
- *(storage)* VectorF32 + VectorRerank partitions (ADR-033 revised)
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

- *(query)* preserve edge whitespace in string literals
- *(query)* continue a later MATCH from an already-bound node
- *(query)* resolve inline node-property filters against outer bindings
- *(query)* default new HnswConfig fields in vector_registry
- *(vector)* wire LsmVectorTier without re-entering interner lock
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

- *(query)* demote per-super-node traversal log to debug
- *(query)* dedup variable-length traversal target emission
- *(query)* index-scan a lifted correlated endpoint lookup
- *(traverse)* avoid per-edge row clone at the source frontier
- *(traverse)* skip write-buffer probe key when buffer is empty
- *(traverse)* drop per-read copy and pre-size fan-out buffer
- *(query)* expand each node once in variable-length traversal
- *(query)* faster hashing and frontier dedup on graph traversal
- *(query)* index point-lookup for correlated equality keys
- *(query)* HNSW writes from CREATE row-stream are batched per statement
- *(query)* reuse adjacency key buffer in graph traversal hot path
- *(executor)* cache schema label per node per statement (R-API6)

#### Refactored

- extract multi-module test files (planner, runner, engine core/merge)
- extract unit tests into sibling files (query, storage, vector, search)
- extract unit tests into sibling test files
- *(core)* hoist try_extract_vector to a single canonical helper
- *(search)* thread Transaction through the SSE token index
- *(query)* split execute into commit and no-commit entry points
- *(query)* own encrypted-index metadata in a typed store
- *(query)* scan index backfill through the node store
- *(query)* persist index definitions through the index store
- *(query)* read TTL reaper state through typed stores
- *(storage)* apply proposal mutations through the engine
- *(query)* read nodes through the node store in vector predicate
- *(modality)* own index definitions in the index store
- *(query)* type the parallel OCC read-set accumulator
- thread storage transaction through stores
- *(query)* route adjacency + schema access through typed Layer-5 helpers
- *(traverse)* batch frontier expansion behind one step
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)
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

- *(query)* freeze push-down EXPLAIN contract and plan invariant
- *(query)* register edge-type schema via store in fixtures
- *(query)* SET map merge and replace semantics
- *(query)* xor truth table and null propagation
- *(query)* simple and nested CASE expression tests
- *(query)* seed fixtures through Layer-4 stores, not raw key encoders
- *(traverse)* distributed frontier-exchange matches single engine
- *(query)* pin index access path plan for pure vector top-k
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

#### Revert

- move per-label vector shard routing out of CE

### coordinode-raft
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-raft-v0.4.3...coordinode-raft-v0.5.0) - 2026-06-27

#### Added

- *(storage)* retained oplog journal + single-node repair for embedded
- *(server)* fall back to WAL-replay repair when no replica serves
- *(raft)* expose committed oplog entries since an index
- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(server)* serve gRPC over TLS and mTLS
- *(raft)* compress RaftService wire traffic with the zstd codec
- *(raft)* add zstd transport codec for inter-node gRPC
- *(raft)* runtime voter and learner role transitions
- *(storage)* MVCC range-delete apply path + partition cache invalidation
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(storage)* VectorF32 + VectorRerank partitions (ADR-033 revised)
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

- *(raft)* set default wire zstd level to 1 and measure the wire
- *(raft)* gate snapshot trigger on log progress
- *(raft)* advance follower oracle during entry apply
- *(storage)* gate every write path + typed propagation to gRPC client
- *(storage)* gate oplog purge on cross-partition flush watermark
- *(raft)* recover last_log_id from oplog on unclean shutdown restart
- *(cluster)* rollback Learner on change_membership failure in monitor_and_promote
- *(server)* resolve proto submodule and clippy::panic in tests
- *(raft)* reduce chunk size to 2MB, add multi-chunk integration test
- *(ci)* update raft build.rs proto path and deny.toml format

#### Performance

- *(raft)* O(delta) incremental snapshot via changed-keys scan

#### Refactored

- extract shared wire codec, compress segment transfer too
- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)

#### Testing

- *(raft)* widen cluster-test election timeout under CI load
- *(raft)* add linearizability checker and clock-skew nemesis
- *(raft)* read_oplog_since returns post-checkpoint ops only
- *(raft)* inter-node mutual-TLS cluster replication
- *(raft)* snapshot trigger must skip idle intervals
- *(raft)* add 3-node pruning decommission test as final R091c entry
- *(cluster)* R091c decommission protocol test suite
- *(raft)* R141 complete test coverage — follower scenarios + StaleReplica
- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)

### coordinode-replicate
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-replicate-v0.4.3...coordinode-replicate-v0.5.0) - 2026-06-27

#### Added

- *(storage)* retained oplog journal + single-node repair for embedded
- *(replicate)* rebuild a partition from checkpoint plus oplog replay
- *(server)* repair corrupt partitions from peers on scrub detection
- *(replicate)* repair a partition from healthy peers via swarm pull
- *(replicate)* gRPC piece source for the swarm pull
- *(replicate)* serve the receiver-driven swarm piece-exchange
- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(replicate)* add segment drain client for peer push
- *(server)* register segment-transfer service in cluster mode
- *(replicate)* self-describing segment blob and dispatching installer
- *(replicate)* storage-backed segment export and install
- *(replicate)* source-side frame gather for segment transfer
- *(replicate)* segment-transfer gRPC receive handler
- *(replicate)* wire SegmentTransferService gRPC codegen
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)

#### Fixed

- *(storage)* clear corrupt partition physically before repair reinstall
- *(replicate)* physically replace corrupt tables on repair

#### Refactored

- extract shared wire codec, compress segment transfer too
- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)

### coordinode-s3
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-s3-v0.4.3...coordinode-s3-v0.5.0) - 2026-06-27

#### Added

- *(storage)* io_uring filesystem backend behind --features io-uring
- *(storage)* R156 + R157 — multi-endpoint storage placement
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)

### coordinode-search
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-search-v0.4.3...coordinode-search-v0.5.0) - 2026-06-27

#### Added

- *(storage)* R156 + R157 — multi-endpoint storage placement
- *(search)* FTS MVCC snapshot filter via per-doc commit_ts + segment registry
- *(text-search)* implement TextService gRPC with fuzzy + language-aware search
- *(search)* external CJK dictionary loading from filesystem (G014)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Fixed

- *(ci)* replace cargo test with nextest, fix CJK env var race

#### Refactored

- extract unit tests into sibling files (query, storage, vector, search)
- extract unit tests into sibling test files
- *(search)* thread Transaction through the SSE token index

#### Testing

- *(search)* direct unit tests for search_with_highlights_fuzzy and search_with_highlights_and_language
- *(text-search)* Ukrainian e2e + multi-property merge coverage

### coordinode-server
## [0.5.0](https://github.com/structured-world/coordinode/compare/v0.4.3...v0.5.0) - 2026-06-27

#### Added

- *(triggers)* [**breaking**] execute AFTER COMMIT triggers via durable event journal
- *(server)* make the CDC consumer TTL configurable
- *(server)* register CDC streams as oplog consumers for retention
- *(server)* fall back to WAL-replay repair when no replica serves
- *(server)* periodic local checkpoints
- *(server)* repair corrupt partitions from peers on scrub detection
- *(server)* run a periodic per-node background integrity scrub
- *(server)* wire verify --deep to the block-checksum scrub
- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(server)* serve gRPC over TLS and mTLS
- *(server)* make inter-node wire compression level configurable
- *(server)* register segment-transfer service in cluster mode
- *(storage)* OplogOp::RemoveRange wire type + CDC mapping
- *(modality)* add per-label shard strategy to the vector index config
- *(query)* expose ef_search and rerank_candidates as vector index options
- *(storage)* io_uring filesystem backend behind --features io-uring
- *(storage)* multi-endpoint topology config
- *(server)* unified config surface with YAML file and CLI overrides
- *(server)* gRPC RPCs for interactive transactions
- *(vector)* serving health + HLC freshness watermark for indexes
- *(server)* expose consumer-retention registry tuning via serve config
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(server)* add offline compact subcommand
- *(server)* operator config for fd, network, and storage limits
- *(backup)* validate binary dump compatibility on restore
- *(restore)* selective restore via --only-labels filter
- *(restore)* transparent decompression and Hetionet hetnet-JSON import
- *(backup)* incremental raft-snapshot via --since seqno
- *(backup)* raft-snapshot backup and restore via the CLI
- *(query)* add Path value type and nodes/relationships/length
- *(backup)* restore Neo4j APOC json and cypher dumps
- *(storage)* hard-link checkpoint of the whole database
- *(backup)* implement cypher restore + align edge-prop wire format
- *(embed)* wire the vector oplog worker into Database
- *(cluster)* replicate vector index DDL to followers
- *(core)* add MultiVector value variant
- *(server)* wire CreateNodesBatch handler via UNWIND $rows AS r CREATE …
- *(storage)* R156 + R157 — multi-endpoint storage placement

#### Fixed

- *(wire)* default to zstd level 3 to avoid the Fast-strategy panic path
- *(cluster)* refresh follower interner on entry apply
- *(query/server)* preserve CapacityExhausted type through Cypher pipeline → gRPC
- *(server)* migrate remaining gRPC services to capacity-aware error mapping
- *(storage)* gate every write path + typed propagation to gRPC client

#### Performance

- *(server)* route read-only handlers through .read() + execute_cypher_shared

#### Refactored

- extract shared wire codec, compress segment transfer too
- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)
- *(vector)* migrate quantization config from bool to QuantizationCodec enum
- *(server)* swap std::sync::Mutex<Database> for parking_lot::RwLock<Database>
- *(embed)* wrap FieldInterner in Arc<RwLock> on the Database side

#### Testing

- *(server)* follower must answer vector search like leader

#### Revert

- move per-label vector shard routing out of CE

### coordinode-storage
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-storage-v0.4.3...coordinode-storage-v0.5.0) - 2026-06-27

#### Added

- *(storage)* retained oplog journal + single-node repair for embedded
- *(replicate)* self-describing segment blob and dispatching installer
- *(storage)* placement-segment descriptor and per-partition map
- *(storage)* descending range and prefix scans
- *(storage)* batched multi_get for known-key sets
- *(storage)* coalesce delete runs on the durable commit path
- *(storage)* MVCC range-delete apply path + partition cache invalidation
- *(storage)* OplogOp::RemoveRange wire type + CDC mapping
- *(storage)* run-length coalesce delete sets into point + range deletes
- *(storage)* io_uring filesystem backend behind --features io-uring
- *(storage)* multi-endpoint topology config
- *(embed)* bound interactive transaction buffered writes
- *(embed)* interactive multi-statement transactions
- *(storage)* park and resume transaction state across statements
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(storage)* GC watermark driver from live snapshot pins
- *(storage)* hard-link checkpoint of the whole database
- *(embed)* wire the vector oplog worker into Database
- *(storage)* oplog tailer reads the active segment [skip bench]
- *(storage)* VectorF32 + VectorRerank partitions (ADR-033 revised)
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

- *(storage)* clear corrupt partition physically before repair reinstall
- *(storage)* scrub collects block corruption instead of aborting
- *(backup)* flush memtables before creating a checkpoint
- *(raft)* advance follower oracle during entry apply
- *(storage)* defer capacity-scanner first tick by interval to close warm-load race
- *(storage)* capacity scanner counts every endpoint file, not only SSTs
- *(storage)* gate every write path + typed propagation to gRPC client
- *(storage)* gate oplog purge on cross-partition flush watermark
- *(raft)* recover last_log_id from oplog on unclean shutdown restart

#### Performance

- *(spatial)* Z-curve skip-scan via seekable range iterator
- *(raft)* O(delta) incremental snapshot via changed-keys scan
- *(storage)* fold adjacency operands in force_compaction, time-travel safe
- *(storage)* collapse adjacency merge operands into single values
- *(modality/spatial)* Z-curve subrange decomposition (G101)
- *(storage)* batch Extra-targeting deltas in DocumentMerge
- *(storage)* parallel memtable writes within write batch (R091)

#### Refactored

- *(core)* move delete coalescing to core, operate on mutations
- extract multi-module test files (planner, runner, engine core/merge)
- extract unit tests into sibling files (query, storage, vector, search)
- *(query)* read TTL reaper state through typed stores
- *(storage)* apply proposal mutations through the engine
- *(query)* type the parallel OCC read-set accumulator
- thread storage transaction through stores
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)
- *(tests)* embed + storage migration to in-memory fixtures (Database::open_in_memory)
- *(storage,query)* move OCC tracking to Layer 3 Coordinator (G104)
- *(storage/coordinator)* extract MultiModalCoordinator trait (G105)
- *(storage/coordinator)* trim doctests to internal-crate scope
- *(storage)* extract Layer 3 Coordinator sub-module (R164)

#### Testing

- *(storage)* pin oplog segment filename contract
- *(backup)* assert checkpoint dirs by real partition name
- *(storage)* regression test for capacity-scanner warm-load race
- *(storage,modality)* G101 audit close — range_scan API + CRS dispatch + stronger exclusion
- *(modality,storage)* reduce proptest cases for faster regression runs
- *(query)* RYOW + legacy-mode OCC invariants, scrub task IDs
- *(storage,query)* edge cases + dyn dispatch for G104/G105
- *(storage)* final R164 coverage round + rustdoc cleanup
- *(storage/coordinator)* edge cases + doctests + concurrency
- *(storage)* page-ECC policy — builder + serde back-compat + Volatile edge

### coordinode-swarm
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-swarm-v0.4.3...coordinode-swarm-v0.5.0) - 2026-06-27

#### Added

- *(replicate)* serve the receiver-driven swarm piece-exchange
- *(swarm)* multi-source rarest-first segment download driver
- *(replicate)* segment-transfer gRPC receive handler
- *(swarm)* source-selection scoring for swarm transfer
- *(swarm)* single-source segment transfer driver
- *(swarm)* streaming piece decode + zstd transfer encoding
- *(swarm)* rarest-first piece scheduling state
- *(swarm)* segment piece model for swarm transfer

#### Fixed

- *(swarm)* record per-piece transfer encoding in segment manifest

#### Refactored

- extract unit tests into sibling files (swarm)
- extract unit tests into sibling test files

### coordinode-test-fixtures
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-test-fixtures-v0.4.3...coordinode-test-fixtures-v0.5.0) - 2026-06-27

#### Added

- *(test-fixtures)* new crate — engine_for_logic / engine_for_disk / engine_for_memory dual-FS test fixture

#### Performance

- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)
- *(query/tests)* R166 migration — 4 query test files on dual-FS fixture

#### Testing

- *(test-fixtures)* audit closure — edge cases + doctest + CI matrix verification

### coordinode-timeseries
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-timeseries-v0.4.3...coordinode-timeseries-v0.5.0) - 2026-06-27

#### Added

- *(timeseries)* ε-policy — opt-in WITH BITEMPORAL via split write entry points (β: Cypher paused)
- *(timeseries)* close G103 #3 Gap #4 — PersistentMonotonicHlcClock with engine-backed restart monotonicity
- *(modality,timeseries)* G103 sub-system #3 — bitemporal __ingestion_ts__ axis
- *(modality,timeseries)* G103 sub-system #4 — overflow compactor primitives
- *(timeseries)* G103 slice C — Tier 3 overflow routing + background compactor
- *(timeseries)* G103 slice B — Tier 2 recently-closed LRU + reopen path
- *(timeseries)* new crate coordinode-timeseries (G103 slice A — BucketCatalog + Tier 1 buffer)

#### Refactored

- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)
- extract unit tests into sibling test files
- *(modality)* thread Transaction through spatial, blob, time-series stores

#### Testing

- *(timeseries)* G103 #3 audit closure — backfill on compact, edge case tests, restart-monotonicity gap documented

### coordinode-vector
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-vector-v0.4.3...coordinode-vector-v0.5.0) - 2026-06-27

#### Added

- *(vector)* opt-in cache-locality reorder on bulk build
- *(vector)* apply BFS reorder permutation across HNSW stores
- *(vector)* BFS visit-order permutation for HNSW cache-locality reorder
- *(vector)* add the centroid shard router (closure replication + adaptive fan-out)
- *(vector)* serving health + HLC freshness watermark for indexes
- *(vector)* distance-kernel call counter behind a feature
- *(vector)* brute-force cluster assignment for bulk_build
- *(vector)* seed leader sample in bulk_build
- *(vector)* bulk_build skeleton with insert_batch fallback
- *(vector)* maxsim scoring kernel for late-interaction
- *(vector)* add SearchMode::Exact for recall=1.0 brute-force kNN
- *(vector)* add f32-only contiguous layer-0 block
- *(vector)* add reusable SearchScratch buffers + pool
- *(vector)* mirror RaBitQ code and scalars to InlineLayer0
- *(vector)* InlineLayer0 reserves RaBitQ scalars + bit-aware code
- *(vector)* mirror layer-0 neighbours to InlineLayer0 on every write
- *(vector)* mirror f32 vector and label to InlineLayer0 on insert
- *(vector)* InlineLayer0 store for HNSW layer-0 contiguous payload
- *(vector)* RaBitQ EndOfSearch oversampling knob
- *(vector)* RerankMode knob (Inline | EndOfSearch | None) for RaBitQ
- *(vector)* R863 RobustPrune α-pruning neighbour selector
- *(vector)* K-cluster IVF via K-means Lloyd, default K=16 for RaBitQ
- *(vector)* K=1 IVF centering for RaBitQ cosine reconstruction
- *(vector)* RaBitQ paper Equation 20 asymmetric kernel (4-bit query)
- *(vector)* RaBitQ supports any dim via internal padding to next mult of 64
- *(vector)* wire Extended-RaBitQ 2/3/4-bit through HnswIndex
- *(vector)* Extended-RaBitQ 2/3/4-bit codec primitive
- *(vector)* LsmVectorTier — production binding of VectorTierStorage
- *(vector)* VectorTierStorage trait + write hooks in HnswIndex (ADR-033)
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

- *(vector)* bulk_build path and follow-up boundary
- *(vector)* C3 day 6 — record measured 14.6× speedup in bench doc

#### Fixed

- *(vector)* silence unused_unsafe in distance kernel bench
- *(vector)* require results full before terminating HNSW search
- *(vector)* reconstruct ‖x‖ from IVF code header for cosine rerank
- *(vector)* track f32 distance in HNSW results-heap, RaBitQ in frontier
- *(vector)* RaBitQ per-vector correction scaling (chroma-style)
- *(vector)* build HNSW on exact f32, only use RaBitQ at search time
- *(vector)* apply-phase backfill keeps closer candidates over incumbents
- *(vector)* HNSW search beam must be at least k for any caller
- *(modality,vector)* pass &StorageEngine in doctests + add LockFreeNeighbours::is_empty
- *(vector)* preserve back-edges when neighbour list is at M_MAX0 cap
- *(vector)* dedupe duplicate ids within insert_batch + proptest stress
- *(vector)* update HNSW graph position when node vector is overwritten (G082)

#### Performance

- *(vector)* prune HNSW back-edges with the diversity heuristic
- *(vector)* size the layer-0 block to the effective neighbour degree
- *(vector)* co-locate neighbours and f32 in one contiguous block, drop SoA neighbour list
- *(vector)* free f32 from contiguous blocks on offload, drop redundant SoA store
- *(vector)* default RobustPrune alpha to 1.15 for cosine builds
- *(vector)* walk inline layer-0 neighbour rows in place during search
- *(vector)* stop eager RaBitQ code indexing on unquantized visits
- *(vector)* cache inverse node norms, multiply instead of divide in cosine
- *(vector)* prefetch next frontier candidate's neighbour ids
- *(vector)* skip per-node RaBitQ lookup in prefetch when codec inactive
- *(vector)* prefetch the full vector span, not one cache line
- *(vector)* bind SIMD kernel pointer once on x86_64
- *(vector)* skip exact re-distance when no quantizer is active
- *(vector)* both-norms fast path for insert pruning distances
- *(vector)* pre-normalise cosine vectors at insert to drop divide
- *(vector)* cache per-node L2 norm to skip per-visit norm pass
- *(vector)* skip RaBitQ+SQ8 chain when neither active
- *(vector)* prefetch top candidate vector after push
- *(metrics)* mark distance kernel implementations inline
- *(vector)* thread-local VisitedPool storage
- *(vector)* wire data_level0 into search prefetch + f32 read
- *(vector)* prefer SoA over inline for f32 vector reads
- *(vector)* read RaBitQ code from InlineLayer0 in cosine search
- *(vector)* read f32 vector from InlineLayer0 in compute_exact_distance
- *(vector)* read layer-0 neighbours from InlineLayer0 in search
- *(vector)* hoist nodes.len() out of search inner loop
- *(vector)* unchecked visited.check_and_mark on search hot path
- *(vector)* SIMD-ify FHT butterfly via AVX2 / NEON with runtime detect
- *(vector)* inline RaBitQQuery bit-planes via SmallVec
- *(vector)* skip prefetch_node_vector entirely on cosine + RaBitQ
- *(vector)* flat rabitq_code_ptr cache eliminates SoA load on prefetch
- *(vector)* split layer-0 neighbours flat + skip f32 prefetch on RaBitQ
- *(vector)* inline RaBitQ code u64 words via SmallVec (D ≤ 256 = no heap)
- *(vector)* prefetch next neighbour's visited counter byte (hnswlib pattern)
- *(vector)* replace results-heap push+pop with peek_mut+swap
- *(vector)* prefetch RaBitQ code in addition to f32 vector
- *(vector)* fused 4-plane AND+popcount kernel for RaBitQ asymmetric path
- *(vector)* cache node norm in cosine rerank to drop per-call norm pass
- *(vector)* replace Gram-Schmidt rotation with FHT-Kac (O(D²)→O(D log D))
- *(vector)* flat contiguous vector store for HNSW distance hot path
- *(vector)* prefetch full vector range (8 cache lines @ d=128)
- *(vector)* pack Candidate/FarCandidate to 8 bytes (u32 idx)
- *(vector)* hoist alloc + cache farthest in HNSW search inner loop
- *(vector)* inline distance dispatch chain end-to-end
- *(vector)* store internal indices in HNSW neighbour lists
- *(vector)* revert x86_64 AVX2 L2/dot/L1 to single-acc FMA shape
- *(vector)* match hnswlib AVX2/512 kernel shape (single-acc, mul+add)
- *(vector)* multi-accumulator SIMD distance kernels + runtime AVX-512
- *(vector)* bit-pack RaBitQExtCode to paper-quoted sizes
- *(vector)* criterion harness for RaBitQ popcount kernel
- *(vector)* SQ8 dequantize into reusable scratch + SIMD
- *(vector)* cache query L2 norm per HNSW search (cosine path)
- *(vector)* C3 day 5b — parallel prune-pass via rayon
- *(vector)* C3 day 5a — dedupe backfill before prune-pass

#### Refactored

- extract unit tests into sibling files (query, storage, vector, search)
- extract unit tests into sibling test files
- *(vector)* SoA split of HnswNode payload arrays
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)
- *(vector)* migrate quantization config from bool to QuantizationCodec enum

#### Testing

- *(vector)* stabilize the reordered bulk-build self-recall check
- *(hnsw)* assert batch recall vs ground truth, not serial topology
- *(vector)* bulk_build vs insert_batch criterion arm
- *(vector)* RaBitQ cosine dim=100 reproducer narrows bug to scale
- *(vector)* isolate RaBitQ recall bug + cap rayon to 4 threads in CI
- *(vector)* end-to-end RaBitQ + LSM tier wiring
- *(vector)* regression tests for HNSW recall when ef_search < k
- *(vector)* wire loom interleaving suite for AtomicNeighbourList
- *(vector)* stress AtomicNeighbourList cas_append vs concurrent snapshot
- *(vector)* add proptest stress for multi-batch + concurrent search

#### Revert

- move per-label vector shard routing out of CE
- *(vector)* undo "flat contiguous vector store" — bench regressed
- *(vector)* undo "prefetch full vector range" — bench regressed

### coordinode-wire
## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-wire-v0.4.3...coordinode-wire-v0.5.0) - 2026-06-27

#### Added

- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(wire)* TLS/mTLS config foundation with pure-Rust crypto provider

#### Fixed

- *(wire)* default to zstd level 3 to avoid the Fast-strategy panic path

#### Refactored

- *(wire)* migrate PEM parsing off unmaintained rustls-pemfile
- extract shared wire codec, compress segment transfer too

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

