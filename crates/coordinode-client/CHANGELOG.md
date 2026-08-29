# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-client-v0.4.3...coordinode-client-v0.5.0) - 2026-06-27

#### Added

- *(server)* gRPC RPCs for interactive transactions
- *(client)* expose read preference for cypher reads
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(query)* [**breaking**] add rrf_score Cypher function with RankFuse operator
- *(client)* causal session API — CausalToken, execute_causal_write/read (G089)
- *(causal)* enforce writeConcern=MAJORITY in causal write sessions (G088)
- *(consistency)* implement R142 causal consistency sessions
- *(client)* add coordinode-client crate with source location tracking

#### Fixed

- *(client)* generate proto bindings at build time, drop stale proto_gen
- *(ci)* resolve release-plz cargo package failures for coordinode-client
- *(client)* add replication proto module and new ExecuteCypherRequest fields
- *(client)* use publish.workspace = true (consistent with other crates)
- *(client)* add tokio-test dev-dep; remove stale execute_cypher_annotated reference

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)

#### Testing

- *(client,server)* cover params+source gRPC branch and invalid endpoint
