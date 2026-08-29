# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.5.1](https://github.com/structured-world/coordinode/compare/coordinode-embed-v0.5.0...coordinode-embed-v0.5.1) - 2026-08-29

#### Added

- *(storage)* journal STORAGE COLUMNAR writes to the retained oplog
- *(query)* SQL UPDATE and DELETE complete the CRUD surface
- *(embed)* execute SQL through the shared dialect-agnostic path
- *(query)* SQL frontend lowering SELECT and INSERT to the neutral IR
- *(query)* columnar table row writes and scan-back
- *(query)* primary-key identity and row inserts for relational tables
- *(query)* DROP TABLE for the relational TABLE modality
- *(query)* CREATE TABLE DDL for the relational TABLE modality
- *(query)* add QueryFrontend trait and route execution through it
- *(session)* SHOW SESSIONS / SHOW TRANSACTIONS introspection
- *(session)* keyset-resumable server-side cursor
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

- *(query)* route remaining plan-build and tracking paths through the frontend
- *(query)* remove the cypher expression evaluator
- *(query)* neutralize DDL clause descriptors
- *(query)* neutralize the entire LogicalOp expression surface
- *(query)* predicate and inline-filter expressions use the neutral IR
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

- *(query)* error-path and reopen-recovery coverage for CREATE/DROP TABLE
- *(session)* cover keyset cursor params, transport, and doc fields
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

---

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

---

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

---

## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(query)* snapshot API contract tests + fix modality_count over-promotion
- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(query)* expose applied_watermark handle on ExecutionContext

---

## [0.4.0](https://github.com/structured-world/coordinode/compare/v0.3.20...v0.4.0) - 2026-04-17

#### Fixed

- *(query)* text_match() hard-fails on missing FT-index

---

## [0.3.17](https://github.com/structured-world/coordinode/compare/v0.3.16...v0.3.17) - 2026-04-15

#### Fixed

- *(executor)* RETURN must not expose SET value when write was not applied

---

## [0.3.15](https://github.com/structured-world/coordinode/compare/v0.3.14...v0.3.15) - 2026-04-15

#### Fixed

- *(query)* support query parameters in percentileCont/percentileDisc

---

## [0.3.13](https://github.com/structured-world/coordinode/compare/v0.3.12...v0.3.13) - 2026-04-14

#### Fixed

- *(query)* clean up B-tree index entry on REMOVE property
- *(query)* update B-tree index on SET property
- *(query)* clean up B-tree index entries on node DELETE/DETACH DELETE

---

## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(storage)* implement standalone WAL for crash durability

---

## [0.3.9](https://github.com/structured-world/coordinode/compare/v0.3.8...v0.3.9) - 2026-04-13

#### Performance

- *(executor)* cache schema label per node per statement (R-API6)

---

## [0.3.8](https://github.com/structured-world/coordinode/compare/v0.3.7...v0.3.8) - 2026-04-13

#### Added

- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE

---

## [0.3.6](https://github.com/structured-world/coordinode/compare/v0.3.5...v0.3.6) - 2026-04-13

#### Added

- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL

#### Testing

- *(embed)* full integration coverage for CREATE/DROP VECTOR INDEX (R-API3)
- *(embed)* complete R-API3 integration test suite for CREATE/DROP VECTOR INDEX

---

## [0.3.5](https://github.com/structured-world/coordinode/compare/v0.3.4...v0.3.5) - 2026-04-13

#### Added

- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer

---

## [0.3.4](https://github.com/structured-world/coordinode/compare/v0.3.3...v0.3.4) - 2026-04-12

#### Added

- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement

#### Fixed

- *(embed)* add missing target_field to ComputedSpec::Ttl in integration tests

#### Testing

- *(schema)* add reopen test — unique constraint enforced after load_all
- *(embed)* add integration test for TTL Subtree+target_field (G068)
