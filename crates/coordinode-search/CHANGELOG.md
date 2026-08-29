# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-search-v0.4.3...coordinode-search-v0.5.0) - 2026-06-27

#### Added

- *(storage)* R156 + R157 — multi-endpoint storage placement
- *(search)* FTS MVCC snapshot filter via per-doc commit_ts + segment registry
- *(text-search)* implement TextService gRPC with fuzzy + language-aware search
- *(search)* external CJK dictionary loading from filesystem (G014)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Fixed

- *(ci)* replace cargo test with nextest, fix CJK env var race

#### Refactored

- extract unit tests into sibling files (query, storage, vector, search)
- extract unit tests into sibling test files
- *(search)* thread Transaction through the SSE token index

#### Testing

- *(search)* direct unit tests for search_with_highlights_fuzzy and search_with_highlights_and_language
- *(text-search)* Ukrainian e2e + multi-property merge coverage

---

## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(search)* FTS MVCC snapshot filter via per-doc commit_ts + segment registry

---

## [0.3.9](https://github.com/structured-world/coordinode/compare/v0.3.8...v0.3.9) - 2026-04-13

#### Added

- *(text-search)* implement TextService gRPC with fuzzy + language-aware search

#### Testing

- *(search)* direct unit tests for search_with_highlights_fuzzy and search_with_highlights_and_language
- *(text-search)* Ukrainian e2e + multi-property merge coverage
