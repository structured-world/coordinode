# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1](https://github.com/structured-world/coordinode/releases/tag/v0.3.1) - 2026-04-11

### Fixed

- *(executor)* support MERGE relationship patterns (G069, G070, G072)
- *(packaging)* align manifest.json with repo build-docs.py format
- *(ci)* use GitHub App token for repo publish dispatch (RELEASER_APP_ID + KEY)
- *(ci)* continue-on-error for repo publish dispatch (non-blocking)
- *(ci)* use fixed binary path for nfpm (env vars not expanded in contents.src)

### Testing

- *(executor)* add two-MERGE, self-loop, G069+G072 integration tests
- *(executor)* add ON MATCH SET, incoming direction, edge-property tests (G072/G075)
- *(executor)* add G072 edge-case tests + document G074 gap
- *(server)* add cross-service regression test for create_node persistence

### Documentation

- *(sdk)* add LangChain and LlamaIndex integration guides

## [0.3.0](https://github.com/structured-world/coordinode/releases/tag/v0.3.0) - 2026-04-09

### Added

- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(query)* COMPUTED VECTOR_DECAY planner pattern detection (R084)
- *(query)* SSE encrypted search via Cypher DDL + encrypted_match() (G017)
- *(query)* adaptive parallel traversal via rayon (G010) + varlen edge props fix (G066)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

### Fixed

- *(query)* track OCC read-set in parallel traversal path (G067)

## [0.1.0-alpha.1](https://github.com/structured-world/coordinode/releases/tag/v0.1.0-alpha.1) - 2026-04-08

### Added

- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(query)* COMPUTED VECTOR_DECAY planner pattern detection (R084)
- *(query)* SSE encrypted search via Cypher DDL + encrypted_match() (G017)
- *(query)* adaptive parallel traversal via rayon (G010) + varlen edge props fix (G066)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

### Fixed

- *(query)* track OCC read-set in parallel traversal path (G067)
