# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.3.20](https://github.com/structured-world/coordinode/compare/coordinode-core-v0.3.19...coordinode-core-v0.3.20) - 2026-04-17

### Added

- *(query)* ATTACH DOCUMENT — demote graph node to nested DOCUMENT property

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
