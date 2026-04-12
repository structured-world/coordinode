# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3](https://github.com/structured-world/coordinode/compare/v0.3.2...v0.3.3) - 2026-04-12

### Added

- *(query)* implement standalone MERGE relationship (G074)
- *(query)* implement pattern predicates in WHERE clause

### Fixed

- *(query)* store and check edge properties in MERGE relationship (G075)

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
