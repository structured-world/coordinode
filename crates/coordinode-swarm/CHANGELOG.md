# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-swarm-v0.4.3...coordinode-swarm-v0.5.0) - 2026-06-27

#### Added

- *(replicate)* serve the receiver-driven swarm piece-exchange
- *(swarm)* multi-source rarest-first segment download driver
- *(replicate)* segment-transfer gRPC receive handler
- *(swarm)* source-selection scoring for swarm transfer
- *(swarm)* single-source segment transfer driver
- *(swarm)* streaming piece decode + zstd transfer encoding
- *(swarm)* rarest-first piece scheduling state
- *(swarm)* segment piece model for swarm transfer

#### Fixed

- *(swarm)* record per-piece transfer encoding in segment manifest

#### Refactored

- extract unit tests into sibling files (swarm)
- extract unit tests into sibling test files
