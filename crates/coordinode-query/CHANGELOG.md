# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

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

---

## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(query)* snapshot API contract tests + fix modality_count over-promotion
- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(query)* expose applied_watermark handle on ExecutionContext

#### Testing

- *(query)* R-SNAP1 exact mode + AS OF target + hint docs

---

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

---

#### Added

- *(query)* `rrf_score([method_exprs…], {vector: …, text: …})` — Reciprocal Rank Fusion Cypher function. N-method rank fusion with competition ranks, `k=60` (IR standard, non-tunable), per-method direction from HNSW metric config. Supports node vectors, edge vectors (brute-force), and BM25 text methods.

---

## [0.3.20](https://github.com/structured-world/coordinode/compare/v0.3.19...v0.3.20) - 2026-04-17

#### Added

- *(query)* ATTACH DOCUMENT — demote graph node to nested DOCUMENT property
- *(query)* DETACH DOCUMENT — promote nested property to graph node + edge

---

## [0.3.19](https://github.com/structured-world/coordinode/compare/v0.3.18...v0.3.19) - 2026-04-17

#### Fixed

- *(clippy)* resolve 11 warnings on Rust 1.95

---

## [0.3.17](https://github.com/structured-world/coordinode/compare/v0.3.16...v0.3.17) - 2026-04-15

#### Added

- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)

#### Fixed

- *(executor)* RETURN must not expose SET value when write was not applied

#### Testing

- *(query)* verify is_write() classifies all DDL clause variants
- *(query)* add unit tests for Query::is_write() AST predicate

---

## [0.3.15](https://github.com/structured-world/coordinode/compare/v0.3.14...v0.3.15) - 2026-04-15

#### Fixed

- *(query)* wire parameter substitution into execute() + expand tests
- *(query)* support query parameters in percentileCont/percentileDisc
- *(query)* implement percentileCont/percentileDisc with correct percentile arg

#### Performance

- *(query)* reuse adjacency key buffer in graph traversal hot path

---

## [0.3.13](https://github.com/structured-world/coordinode/compare/v0.3.12...v0.3.13) - 2026-04-14

#### Fixed

- *(query)* clean up B-tree index entry on REMOVE property
- *(query)* update B-tree index on SET property
- *(query)* clean up B-tree index entries on node DELETE/DETACH DELETE

---

## [0.3.9](https://github.com/structured-world/coordinode/compare/v0.3.8...v0.3.9) - 2026-04-13

#### Performance

- *(executor)* cache schema label per node per statement (R-API6)

---

## [0.3.8](https://github.com/structured-world/coordinode/compare/v0.3.7...v0.3.8) - 2026-04-13

#### Added

- *(schema)* complete R-API5 schema modes enforcement
- *(schema)* enforce required fields at CREATE + multi-update tests
- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE
- *(schema)* implement SchemaMode enforcement in executor (R-API5)

#### Fixed

- *(executor)* schema enforcement for PropertyPath, DocFunction, map SET ops

---

## [0.3.6](https://github.com/structured-world/coordinode/compare/v0.3.5...v0.3.6) - 2026-04-13

#### Added

- *(query)* use planner hnsw_index annotation in executor for index-name lookup
- *(query)* CREATE/DROP VECTOR INDEX Cypher DDL

---

## [0.3.5](https://github.com/structured-world/coordinode/compare/v0.3.4...v0.3.5) - 2026-04-13

#### Added

- *(query)* implement CREATE/DROP INDEX Cypher DDL with IndexScan optimizer

---

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
