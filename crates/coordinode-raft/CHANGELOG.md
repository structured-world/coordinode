# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.4.2](https://github.com/structured-world/coordinode/compare/v0.4.1...v0.4.2) - 2026-05-11

#### Fixed

- *(storage)* gate oplog purge on cross-partition flush watermark

---

## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(raft)* wire MaxAssignedWatermark into apply_proposal path

---

## [0.3.18](https://github.com/structured-world/coordinode/compare/v0.3.17...v0.3.18) - 2026-04-16

#### Added

- *(server)* R150 — monolithic binary --mode=full, shared :7080, NodeInfoLayer

#### Fixed

- *(raft)* recover last_log_id from oplog on unclean shutdown restart

---

## [0.3.12](https://github.com/structured-world/coordinode/compare/v0.3.11...v0.3.12) - 2026-04-14

#### Added

- *(cluster)* node decommission protocol + unified Raft write path

#### Testing

- *(raft)* add 3-node pruning decommission test as final R091c entry
- *(cluster)* R091c decommission protocol test suite

---

## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(cluster)* implement cluster join protocol (R091b)
- *(storage)* implement standalone WAL for crash durability

#### Fixed

- *(cluster)* rollback Learner on change_membership failure in monitor_and_promote

---

## [0.3.10](https://github.com/structured-world/coordinode/compare/v0.3.9...v0.3.10) - 2026-04-14

#### Added

- *(raft)* R141 follower reads — ReadFence, SyncPerBatch persist fix

#### Fixed

- *(server)* resolve proto submodule and clippy::panic in tests

#### Testing

- *(raft)* R141 complete test coverage — follower scenarios + StaleReplica
