# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.4.3](https://github.com/structured-world/coordinode/compare/v0.4.2...v0.4.3) - 2026-05-17

#### Added

- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through

---

## [0.4.0](https://github.com/structured-world/coordinode/compare/v0.3.20...v0.4.0) - 2026-04-17

#### Added

- *(query)* [**breaking**] add rrf_score Cypher function with RankFuse operator

---

## Unreleased

---

#### Removed (BREAKING)

- *(proto)* `TextService.HybridTextVectorSearch` RPC, `HybridTextVectorSearchRequest` / `HybridTextVectorSearchResponse` / `HybridResult` messages, `POST /v1/query/text/hybrid` HTTP endpoint. Superseded by the general-purpose Cypher function `rrf_score([methods…], {vector, text})` invoked via `CypherService.ExecuteCypher`. The Cypher form supports N methods (not 2), edge vectors, configurable HNSW metrics, and composes with MATCH / WHERE / ORDER BY / LIMIT in a single plan. Callers: replace the RPC with an equivalent Cypher query.

---

## [0.3.18](https://github.com/structured-world/coordinode/compare/v0.3.17...v0.3.18) - 2026-04-16

#### Added

- *(server)* R150 — monolithic binary --mode=full, shared :7080, NodeInfoLayer

---

## [0.3.17](https://github.com/structured-world/coordinode/compare/v0.3.16...v0.3.17) - 2026-04-15

#### Added

- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)
- *(consistency)* implement R142 causal consistency sessions

---

## [0.3.16](https://github.com/structured-world/coordinode/compare/v0.3.15...v0.3.16) - 2026-04-15

#### Added

- *(server)* gate REST proxy behind rest-proxy feature flag
- *(server)* embed REST proxy in coordinode binary

---

## [0.3.12](https://github.com/structured-world/coordinode/compare/v0.3.11...v0.3.12) - 2026-04-14

#### Added

- *(cluster)* node decommission protocol + unified Raft write path

---

## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(cluster)* implement cluster join protocol (R091b)
- *(storage)* implement standalone WAL for crash durability

#### Testing

- *(server)* add CLI unit tests for AdminNodeJoin parsing

---

## [0.3.10](https://github.com/structured-world/coordinode/compare/v0.3.9...v0.3.10) - 2026-04-14

#### Added

- *(raft)* R141 follower reads — ReadFence, SyncPerBatch persist fix

#### Testing

- *(client,server)* cover params+source gRPC branch and invalid endpoint
- *(server)* add gRPC source tracking round-trip test

---

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

## [0.3.4](https://github.com/structured-world/coordinode/compare/v0.3.3...v0.3.4) - 2026-04-12

#### Added

- *(schema)* wire create_label/create_edge_type to persist schemas with unique index enforcement

#### Fixed

- *(vector)* fill labels/properties in VectorResult, respect distance metric
- *(vector)* honour distance metric parameter in VectorService

#### Testing

- *(vector)* strengthen R-FTS2 regression tests
