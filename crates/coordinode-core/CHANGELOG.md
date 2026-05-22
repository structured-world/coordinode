# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.4.3](https://github.com/structured-world/coordinode/compare/v0.4.2...v0.4.3) - 2026-05-17

#### Added

- *(planner)* graph predicate push-down rule (R-PUSH1)
- *(identity,placement,consistency)* u20/u44 NodeId, schema_revision, gRPC concern wire-through
- *(temporal)* bitemporal edge types with valid-time semantics

---

## [0.4.1](https://github.com/structured-world/coordinode/compare/v0.4.0...v0.4.1) - 2026-04-18

#### Added

- *(query)* add read_consistency knob + planner auto-promotion (R-SNAP1)
- *(txn)* add per-shard MaxAssignedWatermark + WaitForTs primitive

---

## [0.3.20](https://github.com/structured-world/coordinode/compare/v0.3.19...v0.3.20) - 2026-04-17

#### Added

- *(query)* ATTACH DOCUMENT — demote graph node to nested DOCUMENT property

---

## [0.3.17](https://github.com/structured-world/coordinode/compare/v0.3.16...v0.3.17) - 2026-04-15

#### Added

- *(core)* implement HybridLogicalClock for CE timestamps (R143)

---

## [0.3.15](https://github.com/structured-world/coordinode/compare/v0.3.14...v0.3.15) - 2026-04-15

#### Performance

- *(codec)* switch UidEncoder/Decoder to StreamVByte Coder1234
- *(query)* reuse adjacency key buffer in graph traversal hot path

---

## [0.3.11](https://github.com/structured-world/coordinode/compare/v0.3.10...v0.3.11) - 2026-04-14

#### Added

- *(storage)* implement standalone WAL for crash durability

---

## [0.3.8](https://github.com/structured-world/coordinode/compare/v0.3.7...v0.3.8) - 2026-04-13

#### Added

- *(schema)* R-API5 schema modes STRICT/VALIDATED/FLEXIBLE

---

## [0.3.4](https://github.com/structured-world/coordinode/compare/v0.3.3...v0.3.4) - 2026-04-12

#### Fixed

- *(query)* TTL scope=Subtree now deletes target_field, not anchor

#### Testing

- *(core)* add roundtrip test for ComputedSpec::Ttl with target_field=Some
