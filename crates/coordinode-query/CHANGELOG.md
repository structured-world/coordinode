# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.3.20](https://github.com/structured-world/coordinode/compare/coordinode-query-v0.3.19...coordinode-query-v0.3.20) - 2026-04-17

### Added

- *(query)* ATTACH DOCUMENT — demote graph node to nested DOCUMENT property
- *(query)* DETACH DOCUMENT — promote nested property to graph node + edge

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
