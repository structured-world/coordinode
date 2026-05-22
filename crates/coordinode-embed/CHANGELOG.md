# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

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
