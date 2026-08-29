# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-test-fixtures-v0.4.3...coordinode-test-fixtures-v0.5.0) - 2026-06-27

#### Added

- *(test-fixtures)* new crate — engine_for_logic / engine_for_disk / engine_for_memory dual-FS test fixture

#### Performance

- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)
- *(query/tests)* R166 migration — 4 query test files on dual-FS fixture

#### Testing

- *(test-fixtures)* audit closure — edge cases + doctest + CI matrix verification
