# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## 0.5.7 - 2026-09-01

#### Added

- *(server)* tell a client its write went to the wrong node
- *(vector)* drain writes into an index while it builds

#### Fixed

- *(vector)* give the index ownership of its background build

---

## 0.5.6 - 2026-09-01

#### Added

- *(storage+server)* compaction-debt backpressure with stop-only admission
- *(txn+server)* write-set conflicts by default; machine-readable error reasons

#### Fixed

- *(query+server)* query faults answer INVALID_ARGUMENT; sort keys go flat
- *(query)* report a call to an unknown function
- *(query)* fail arithmetic that has no answer instead of guessing

#### Performance

- *(stats)* incremental node/label counters replace the statistics scan

---

## 0.5.5 - 2026-08-31

#### Fixed

- *(query)* persist SET on a relationship bound by MERGE
- *(query)* stop a path element borrowing the previous one's properties

---

## 0.5.4 - 2026-08-30

#### Fixed

- *(query)* search every relationship type in an untyped shortestPath
- *(query)* give path elements their properties in comprehensions
- *(query)* keep a computed grouping key through the projection
- *(query)* write relationship properties on SET r += and SET r =

---

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

## 0.5.1 - 2026-08-29

#### Added

- *(query)* SQL CREATE TABLE and DROP TABLE in the frontend
- *(storage)* journal STORAGE COLUMNAR writes to the retained oplog
- *(query)* SQL UPDATE and DELETE complete the CRUD surface
- *(query)* SQL frontend lowering SELECT and INSERT to the neutral IR
- *(query)* columnar table row writes and scan-back
- *(query)* primary-key identity and row inserts for relational tables
- *(query)* DROP TABLE for the relational TABLE modality
- *(query)* CREATE TABLE DDL for the relational TABLE modality
- *(query)* add QueryFrontend trait and route execution through it
- *(query)* parameter substitution on the neutral expression IR
- *(query)* neutral expression evaluator
- *(query)* lower Cypher expressions into the neutral IR
- *(query)* introduce language-neutral expression IR kernel
- *(session)* SHOW SESSIONS / SHOW TRANSACTIONS introspection
- *(session)* keyset-resumable server-side cursor
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
- *(vector)* C1 day 6 - HnswConfig::max_elements drives pre-allocation
- *(query/index)* list_index_definitions helper + registry::load_all migration
- *(modality,query)* SchemaStore::list_labels / list_edge_types + ttl_reaper migration
- *(storage,query)* OccScope typed contains helpers + audit-test migration
- *(modality/node)* add get_at_seqno + scan_shard + migrate build.rs (R165 slice 2)
- *(modality)* introduce coordinode-modality crate with Schema/Blob/Index stores
- *(storage)* R156 + R157 - multi-endpoint storage placement
- *(temporal)* R172d - pattern predicate into temporal target
- *(temporal)* R172d initial slice - traversal into temporal target
- *(temporal)* R172c Phase 3c - DETACH/ATTACH on temporal nodes (partial)
- *(temporal)* R172c Phase 3b - nested PropertyPath / doc_* fns on temporal
- *(temporal)* R172c Phase 3 - REMOVE on temporal as close+open new version
- *(temporal)* R172c Phase 3 - DELETE on temporal as positive bitemporal fact
- *(storage)* R172c Phase 2 - temporal node SET close-current + open-new
- *(storage)* R172c Phase 1 - temporal node SET valid_to + valid_from immutability
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
- *(query)* ATTACH DOCUMENT - demote graph node to nested DOCUMENT property
- *(query)* DETACH DOCUMENT - promote nested property to graph node + edge
- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)
- *(schema)* complete R-API5 schema modes enforcement
- *(schema)* enforce required fields at CREATE + multi-update tests
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(schema)* implement SchemaMode enforcement in executor (R-API5)
- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL
- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer
- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement
- *(query)* add MERGE ALL - Cartesian-product relationship upsert
- *(query)* implement standalone MERGE relationship (G074)
- *(query)* implement pattern predicates in WHERE clause
- *(query)* implement type(r) and labels(n) scalar functions
- *(query)* HNSW-accelerated vector top-K via planner optimization
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(query)* COMPUTED VECTOR_DECAY planner pattern detection (R084)
- *(query)* SSE encrypted search via Cypher DDL + encrypted_match() (G017)
- *(query)* adaptive parallel traversal via rayon (G010) + varlen edge props fix (G066)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

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

- *(query)* route remaining plan-build and tracking paths through the frontend
- *(query)* remove the cypher expression evaluator
- *(query)* neutralize graph-pattern AST in the logical layer
- *(query)* neutralize DDL clause descriptors
- *(query)* neutralize graph-mutation clause descriptors
- *(query)* neutralize SET/REMOVE items and MERGE NODES strategy
- *(query)* neutralize Sort items and remove dead cypher rewrite passes
- *(query)* neutralize the entire LogicalOp expression surface
- *(query)* predicate and inline-filter expressions use the neutral IR
- *(query)* UNWIND expression uses the neutral expression IR
- *(query)* LIMIT/SKIP counts use the neutral expression IR
- *(query)* extract scalar-function dispatch from arg evaluation
- *(query)* evaluate binary/unary ops on the neutral operator IR
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
- *(query/tests)* R166 migration - 4 query test files on dual-FS fixture
- *(core,query)* R165 last raw encoder - Mutation::delete_edge_props typed constructor
- *(query/tests)* migrate integration-test fixtures to LocalNodeStore
- *(query/tests)* R166 finish - ttl_reaper fixtures migrated to LocalNodeStore
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

- *(query)* cover every expression variant in neutral-IR lowering
- *(query)* freeze push-down EXPLAIN contract and plan invariant
- *(query)* register edge-type schema via store in fixtures
- *(query)* SET map merge and replace semantics
- *(query)* xor truth table and null propagation
- *(query)* simple and nested CASE expression tests
- *(query)* seed fixtures through Layer-4 stores, not raw key encoders
- *(traverse)* distributed frontier-exchange matches single engine
- *(query)* pin index access path plan for pure vector top-k
- *(query)* R165 audit close - EdgeStore OCC invariants + lockdown coverage gate
- *(query)* R165 encoder lockdown regression gate (slice 13)
- *(query)* R165 slice 4 second audit - OCC invariant + doc fix
- *(query)* R165 slice 4 audit - edge cases + 2 more SET sites
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

---

## 0.5.0 - 2026-06-27

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
- *(vector)* C1 day 6 - HnswConfig::max_elements drives pre-allocation
- *(query/index)* list_index_definitions helper + registry::load_all migration
- *(modality,query)* SchemaStore::list_labels / list_edge_types + ttl_reaper migration
- *(storage,query)* OccScope typed contains helpers + audit-test migration
- *(modality/node)* add get_at_seqno + scan_shard + migrate build.rs (R165 slice 2)
- *(modality)* introduce coordinode-modality crate with Schema/Blob/Index stores
- *(storage)* R156 + R157 - multi-endpoint storage placement
- *(temporal)* R172d - pattern predicate into temporal target
- *(temporal)* R172d initial slice - traversal into temporal target
- *(temporal)* R172c Phase 3c - DETACH/ATTACH on temporal nodes (partial)
- *(temporal)* R172c Phase 3b - nested PropertyPath / doc_* fns on temporal
- *(temporal)* R172c Phase 3 - REMOVE on temporal as close+open new version
- *(temporal)* R172c Phase 3 - DELETE on temporal as positive bitemporal fact
- *(storage)* R172c Phase 2 - temporal node SET close-current + open-new
- *(storage)* R172c Phase 1 - temporal node SET valid_to + valid_from immutability
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
- *(query)* ATTACH DOCUMENT - demote graph node to nested DOCUMENT property
- *(query)* DETACH DOCUMENT - promote nested property to graph node + edge
- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)
- *(schema)* complete R-API5 schema modes enforcement
- *(schema)* enforce required fields at CREATE + multi-update tests
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(schema)* implement SchemaMode enforcement in executor (R-API5)
- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL
- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer
- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement
- *(query)* add MERGE ALL - Cartesian-product relationship upsert
- *(query)* implement standalone MERGE relationship (G074)
- *(query)* implement pattern predicates in WHERE clause
- *(query)* implement type(r) and labels(n) scalar functions
- *(query)* HNSW-accelerated vector top-K via planner optimization
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(query)* COMPUTED VECTOR_DECAY planner pattern detection (R084)
- *(query)* SSE encrypted search via Cypher DDL + encrypted_match() (G017)
- *(query)* adaptive parallel traversal via rayon (G010) + varlen edge props fix (G066)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

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
- *(query/tests)* R166 migration - 4 query test files on dual-FS fixture
- *(core,query)* R165 last raw encoder - Mutation::delete_edge_props typed constructor
- *(query/tests)* migrate integration-test fixtures to LocalNodeStore
- *(query/tests)* R166 finish - ttl_reaper fixtures migrated to LocalNodeStore
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
- *(query)* R165 audit close - EdgeStore OCC invariants + lockdown coverage gate
- *(query)* R165 encoder lockdown regression gate (slice 13)
- *(query)* R165 slice 4 second audit - OCC invariant + doc fix
- *(query)* R165 slice 4 audit - edge cases + 2 more SET sites
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

---

## 0.4.3 - 2026-05-17

#### Added

- *(cypher)* native MERGE NODES (a, b) INTO target
- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(temporal)* bitemporal edge types with valid-time semantics

#### Fixed

- *(executor)* Cypher three-valued logic for NULL comparisons + edge-case audit

#### Testing

- *(merge-nodes)* close STRICT extra-map gap + composability/index coverage

---

## 0.4.1 - 2026-04-18

#### Added

- *(query)* snapshot API contract tests + fix modality_count over-promotion
- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(query)* expose applied_watermark handle on ExecutionContext

#### Testing

- *(query)* R-SNAP1 exact mode + AS OF target + hint docs

---

## 0.4.0 - 2026-04-17

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

---

## Unreleased

#### Added

- *(query)* `rrf_score([method_exprs…], {vector: …, text: …})` - Reciprocal Rank Fusion Cypher function. N-method rank fusion with competition ranks, `k=60` (IR standard, non-tunable), per-method direction from HNSW metric config. Supports node vectors, edge vectors (brute-force), and BM25 text methods.

---

## 0.3.20 - 2026-04-17

#### Added

- *(query)* ATTACH DOCUMENT - demote graph node to nested DOCUMENT property
- *(query)* DETACH DOCUMENT - promote nested property to graph node + edge

---

## 0.3.19 - 2026-04-17

#### Fixed

- *(clippy)* resolve 11 warnings on Rust 1.95

---

## 0.3.17 - 2026-04-15

#### Added

- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)

#### Fixed

- *(executor)* RETURN must not expose SET value when write was not applied

#### Testing

- *(query)* verify is_write() classifies all DDL clause variants
- *(query)* add unit tests for Query::is_write() AST predicate

---

## 0.3.15 - 2026-04-15

#### Fixed

- *(query)* wire parameter substitution into execute() + expand tests
- *(query)* support query parameters in percentileCont/percentileDisc
- *(query)* implement percentileCont/percentileDisc with correct percentile arg

#### Performance

- *(query)* reuse adjacency key buffer in graph traversal hot path

---

## 0.3.13 - 2026-04-14

#### Fixed

- *(query)* clean up B-tree index entry on REMOVE property
- *(query)* update B-tree index on SET property
- *(query)* clean up B-tree index entries on node DELETE/DETACH DELETE

---

## 0.3.9 - 2026-04-13

#### Performance

- *(executor)* cache schema label per node per statement (R-API6)

---

## 0.3.8 - 2026-04-13

#### Added

- *(schema)* complete R-API5 schema modes enforcement
- *(schema)* enforce required fields at CREATE + multi-update tests
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(schema)* implement SchemaMode enforcement in executor (R-API5)

#### Fixed

- *(executor)* schema enforcement for PropertyPath, DocFunction, map SET ops

---

## 0.3.6 - 2026-04-13

#### Added

- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL

---

## 0.3.5 - 2026-04-13

#### Added

- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer

---

## 0.3.4 - 2026-04-12

#### Added

- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement
- *(query)* add MERGE ALL - Cartesian-product relationship upsert

#### Fixed

- *(vector)* fill labels/properties in VectorResult, respect distance metric
- *(query)* skip Subtree removal when target_field already absent
- *(query)* short-circuit reap_label when Subtree target_field_id unresolved
- *(query)* log error when Subtree target_field_id unresolved
- *(query)* skip Subtree deletion when target_field_id unresolved
- *(query)* TTL scope=Subtree now deletes target_field, not anchor

#### Testing

- *(semantic)* add WITH * regression tests for analyze_with fix
