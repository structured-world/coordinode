# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/structured-world/coordinode/releases/tag/v0.3.0) - 2026-04-09

### Added

- *(server)* wire SchemaService to database; fix graph/vector stubs
- *(server)* implement GraphService RPC with real database persistence
- *(server)* wire VectorServiceImpl to database for real vector search
- *(server)* add backup/restore CLI subcommands (G018)
- *(server)* wire DrainBuffer with RaftProposalPipeline in cluster mode (G063)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

## [0.1.0-alpha.1](https://github.com/structured-world/coordinode/releases/tag/v0.1.0-alpha.1) - 2026-04-08

### Added

- *(server)* add backup/restore CLI subcommands (G018)
- *(server)* wire DrainBuffer with RaftProposalPipeline in cluster mode (G063)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine
