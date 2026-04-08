# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.1](https://github.com/structured-world/coordinode/releases/tag/v0.1.0-alpha.1) - 2026-04-08

### Added

- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(raft)* add WaitForMajorityService for batched proposal coalescing (G047)
- CoordiNode v0.1.0-alpha.1 — graph + vector + full-text engine

### Testing

- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)
