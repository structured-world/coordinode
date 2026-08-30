# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

#### Removed (BREAKING)

- *(proto)* `TextService.HybridTextVectorSearch` RPC, `HybridTextVectorSearchRequest` / `HybridTextVectorSearchResponse` / `HybridResult` messages, `POST /v1/query/text/hybrid` HTTP endpoint. Superseded by the general-purpose Cypher function `rrf_score([methods…], {vector, text})` invoked via `CypherService.ExecuteCypher`. The Cypher form supports N methods (not 2), edge vectors, configurable HNSW metrics, and composes with MATCH / WHERE / ORDER BY / LIMIT in a single plan. Callers: replace the RPC with an equivalent Cypher query.
