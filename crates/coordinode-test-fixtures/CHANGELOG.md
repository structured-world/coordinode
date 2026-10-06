# Changelog

All notable changes to this crate are documented in this file.

## v0.7.0 - 2026-10-06

### Documentation

- drop internal ids from search, replicate, swarm, wire and timeseries

### Fixed

- *(raft)* settle membership and shutdown under load
- *(raft)* resume and purge the raft log by per-tree apply coverage
- *(deps)* require the lsm-tree release the code needs

### Testing

- reserve test ports across processes and catch failed starts
- make the suites pass on any target dir and as root
- *(storage,raft)* cut power at every sync and write, and upgrade a v0.6.0 store

## 0.5.8 - 2026-09-05

#### Testing

- *(cluster)* stop handing two nodes the same port

---

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

## 0.5.1 - 2026-08-29

#### Added

- *(test-fixtures)* new crate - engine_for_logic / engine_for_disk / engine_for_memory dual-FS test fixture

#### Performance

- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)
- *(query/tests)* R166 migration - 4 query test files on dual-FS fixture

#### Testing

- *(test-fixtures)* audit closure - edge cases + doctest + CI matrix verification

---

## 0.5.0 - 2026-06-27

#### Added

- *(test-fixtures)* new crate - engine_for_logic / engine_for_disk / engine_for_memory dual-FS test fixture

#### Performance

- *(tests)* modality src + proptest + cross_store_flow migrated to in-memory matrix

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)
- *(query/tests)* R166 migration - 4 query test files on dual-FS fixture

#### Testing

- *(test-fixtures)* audit closure - edge cases + doctest + CI matrix verification
