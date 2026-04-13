# Changelog

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

