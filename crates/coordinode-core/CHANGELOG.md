# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## 0.5.6 - 2026-09-01

#### Performance

- *(stats)* incremental node/label counters replace the statistics scan

---

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

## 0.5.1 - 2026-08-29

#### Added

- *(query)* primary-key identity and row inserts for relational tables
- *(core)* table label schema - primary key and storage layout
- *(session)* SHOW SESSIONS / SHOW TRANSACTIONS introspection
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
- *(query)* ATTACH DOCUMENT - demote graph node to nested DOCUMENT property
- *(core)* implement HybridLogicalClock for CE timestamps (R143)
- *(storage)* implement standalone WAL for crash durability
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(raft)* add WaitForMajorityService for batched proposal coalescing (G047)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

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
- *(core,query)* R165 last raw encoder - Mutation::delete_edge_props typed constructor

#### Testing

- *(core)* add roundtrip test for ComputedSpec::Ttl with target_field=Some
- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)

---

## 0.5.0 - 2026-06-27

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
- *(query)* ATTACH DOCUMENT - demote graph node to nested DOCUMENT property
- *(core)* implement HybridLogicalClock for CE timestamps (R143)
- *(storage)* implement standalone WAL for crash durability
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(raft)* add WaitForMajorityService for batched proposal coalescing (G047)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

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
- *(core,query)* R165 last raw encoder - Mutation::delete_edge_props typed constructor

#### Testing

- *(core)* add roundtrip test for ComputedSpec::Ttl with target_field=Some
- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)

---

## 0.4.3 - 2026-05-17

#### Added

- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(temporal)* bitemporal edge types with valid-time semantics

---

## 0.4.1 - 2026-04-18

#### Added

- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(txn)* add per-shard MaxAssignedWatermark + WaitForTs primitive

---

## 0.3.20 - 2026-04-17

#### Added

- *(query)* ATTACH DOCUMENT - demote graph node to nested DOCUMENT property

---

## 0.3.17 - 2026-04-15

#### Added

- *(core)* implement HybridLogicalClock for CE timestamps (R143)

---

## 0.3.15 - 2026-04-15

#### Performance

- *(codec)* switch UidEncoder/Decoder to StreamVByte Coder1234
- *(query)* reuse adjacency key buffer in graph traversal hot path

---

## 0.3.11 - 2026-04-14

#### Added

- *(storage)* implement standalone WAL for crash durability

---

## 0.3.8 - 2026-04-13

#### Added

- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE

---

## 0.3.4 - 2026-04-12

#### Fixed

- *(query)* TTL scope=Subtree now deletes target_field, not anchor

#### Testing

- *(core)* add roundtrip test for ComputedSpec::Ttl with target_field=Some
