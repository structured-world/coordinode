# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.4.2](https://github.com/structured-world/coordinode/compare/v0.4.1...v0.4.2) - 2026-05-11

#### Added

- *(storage)* time-based memtable flush trigger to bound oplog retention

#### Fixed

- *(storage)* gate oplog purge on cross-partition flush watermark

---

## [0.3.18](https://github.com/structured-world/coordinode/compare/v0.3.17...v0.3.18) - 2026-04-16

#### Fixed

- *(raft)* recover last_log_id from oplog on unclean shutdown restart

---

## [0.3.15](https://github.com/structured-world/coordinode/compare/v0.3.14...v0.3.15) - 2026-04-15

#### Performance

- *(storage)* batch Extra-targeting deltas in DocumentMerge

---

## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(storage)* implement standalone WAL for crash durability

---

## [0.3.10](https://github.com/structured-world/coordinode/compare/v0.3.9...v0.3.10) - 2026-04-14

#### Performance

- *(storage)* parallel memtable writes within write batch (R091)
