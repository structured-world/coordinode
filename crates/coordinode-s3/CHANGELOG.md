# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-s3-v0.4.3...coordinode-s3-v0.5.0) - 2026-06-27

#### Added

- *(storage)* io_uring filesystem backend behind --features io-uring
- *(storage)* R156 + R157 — multi-endpoint storage placement
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)
