# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

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
