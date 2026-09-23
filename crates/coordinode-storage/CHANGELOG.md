# Changelog

All notable changes to this crate are documented in this file.

## v0.6.0 - 2026-09-23

### Added

- *(embed)* expected-revision writes on the embedded handle
- *(txn)* write a record only at the version it was read at
- *(txn)* make the snapshot wait an operator setting
- *(txn)* make a read at a named timestamp wait for what it covers
- *(txn)* bound the commits a node admits at once
- *(txn)* decide the remaining condition classes and measure the guard
- *(txn)* protect a MERGE that acted on an absence it observed
- *(txn)* writers state what their result depends on
- *(txn)* refuse a commit that breaks a declared condition
- *(storage)* reserve and decide invariant claims
- *(cluster)* [**breaking**] refuse a join from a node that holds data
- *(txn)* [**breaking**] write concern as two axes, w and journal
- *(storage)* compress cold levels with zstd by default
- *(storage)* measure retained MVCC history per partition
- *(storage)* [**breaking**] engine-owned MVCC retention window
- *(txn)* [**breaking**] commit receipt, one seqno per proposal

### Documentation

- fix broken and private intra-doc links

### Fixed

- *(storage)* open a partition whose L0 holds more than 255 runs
- *(storage)* recover the journal segment a killed process left open
- *(core)* encode node records canonically
- *(storage)* report damaged planner statistics
- *(txn)* validate from the first write a view may have missed
- *(storage)* never let the watermark pass a snapshot just handed out
- *(txn)* validate against the view a transaction read, not its timestamp
- *(txn)* close the window between validating a write and applying it
- *(txn)* count an instance bound over the attempt's own entries
- *(storage)* refuse a counter overflow where the caller can still hear it
- *(storage)* keep adjacency operands in the order they were staged
- *(embed)* rebuild the planner's counters after a restore
- *(storage)* refuse a counter that leaves i64 or arrives corrupt
- *(storage)* stop compaction losing merged writes
- *(storage)* commit w:0 writes through the log
- *(storage)* open when the clock restarts behind the floor
- *(storage)* adopt per-key mvcc retention
- *(storage)* [**breaking**] honour the retention boundary, and stop maintenance compaction destroying history

### Performance

- *(txn)* wait for a complete snapshot instead of reading behind one
- *(txn)* decide a live endpoint without reading its record
- *(storage)* let a counter chain fold before its base is known

### Refactored

- *(txn)* install a commit's protections in one place
- *(storage)* drop a saturating add the invariant makes unnecessary

### Testing

- *(txn)* a reader never sees part of a batch
- *(txn)* a transfer and a new edge into its scope exclude each other
- *(storage)* measure what declining to fold costs a document
- *(storage)* resolve a merge chain on every read path
- *(storage)* state the composition contract as executable facts
- *(storage)* time travel over a merge chain keeps its steps
- *(storage)* a counter fold is the sum an independent model makes
- *(storage)* a document fold means the same however it is split
- *(storage)* wait for replaced tables to be unlinked

## 0.5.7 - 2026-09-01

#### Added

- *(server)* tell a client its write went to the wrong node
- *(vector)* drain writes into an index while it builds

#### Fixed

- *(vector)* give the index ownership of its background build

---

## 0.5.6 - 2026-09-01

#### Added

- *(storage+server)* compaction-debt backpressure with stop-only admission
- *(storage)* range-scoped WAL-replay-repair for scopable salvage losses
- *(storage)* structural self-repair of partition trees at engine open
- *(txn+server)* write-set conflicts by default; machine-readable error reasons

#### Fixed

- *(storage)* oplog trim guarded by durability and checkpoint floors
- *(storage)* handle weak tombstones in changed-keys scan

#### Performance

- *(txn)* coalesce counter deltas per key in the transaction buffer
- *(stats)* incremental node/label counters replace the statistics scan
- *(storage)* columnar table readback through the tree-level projected scan
- *(storage)* bench the transactional read wrapper, with numbers

#### Testing

- *(storage)* power-loss recovery through the crash-simulator filesystem
- *(storage)* failure-path coverage for open-time structural repair

---

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout
- *(storage)* recover from a segment left behind by an interrupted write

---

## 0.5.1 - 2026-08-29

#### Added

- *(storage)* journal STORAGE COLUMNAR writes to the retained oplog
- *(query)* columnar table row writes and scan-back
- *(storage)* wire the columnar table registry into the engine
- *(storage)* per-table columnar tree registry
- *(storage)* native columnar block seam for STORAGE COLUMNAR tables
- *(storage)* keyset-resumable paged prefix scan
- *(storage)* retained oplog journal + single-node repair for embedded
- *(replicate)* self-describing segment blob and dispatching installer
- *(storage)* placement-segment descriptor and per-partition map
- *(storage)* descending range and prefix scans
- *(storage)* batched multi_get for known-key sets
- *(storage)* coalesce delete runs on the durable commit path
- *(storage)* MVCC range-delete apply path + partition cache invalidation
- *(storage)* OplogOp::RemoveRange wire type + CDC mapping
- *(storage)* run-length coalesce delete sets into point + range deletes
- *(storage)* io_uring filesystem backend behind --features io-uring
- *(storage)* multi-endpoint topology config
- *(embed)* bound interactive transaction buffered writes
- *(embed)* interactive multi-statement transactions
- *(storage)* park and resume transaction state across statements
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(storage)* GC watermark driver from live snapshot pins
- *(storage)* hard-link checkpoint of the whole database
- *(embed)* wire the vector oplog worker into Database
- *(storage)* oplog tailer reads the active segment [skip bench]
- *(storage)* VectorF32 + VectorRerank partitions (ADR-033 revised)
- *(storage,query)* OccScope typed contains helpers + audit-test migration
- *(storage)* cascade also fires at Full + metric/concurrency tests
- *(storage)* background capacity scanner + hard-limit-strategy edges
- *(storage)* per-endpoint capacity tracking + hard-limit enforcement
- *(storage)* page-checksum wire-through + ECC policy config surface
- *(storage)* per-LSM-level endpoint routing + cascade eviction
- *(storage)* R156 + R157 - multi-endpoint storage placement
- *(temporal)* R172c Phase 3b - nested PropertyPath / doc_* fns on temporal
- *(storage)* time-based memtable flush trigger to bound oplog retention
- *(storage)* implement standalone WAL for crash durability
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(storage)* add MemFs in-memory test backend support
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

#### Fixed

- *(storage)* columnar table registry uses the engine filesystem backend
- *(storage)* clear corrupt partition physically before repair reinstall
- *(storage)* scrub collects block corruption instead of aborting
- *(backup)* flush memtables before creating a checkpoint
- *(raft)* advance follower oracle during entry apply
- *(storage)* defer capacity-scanner first tick by interval to close warm-load race
- *(storage)* capacity scanner counts every endpoint file, not only SSTs
- *(storage)* gate every write path + typed propagation to gRPC client
- *(storage)* gate oplog purge on cross-partition flush watermark
- *(raft)* recover last_log_id from oplog on unclean shutdown restart

#### Performance

- *(spatial)* Z-curve skip-scan via seekable range iterator
- *(raft)* O(delta) incremental snapshot via changed-keys scan
- *(storage)* fold adjacency operands in force_compaction, time-travel safe
- *(storage)* collapse adjacency merge operands into single values
- *(modality/spatial)* Z-curve subrange decomposition (G101)
- *(storage)* batch Extra-targeting deltas in DocumentMerge
- *(storage)* parallel memtable writes within write batch (R091)

#### Refactored

- *(core)* move delete coalescing to core, operate on mutations
- extract multi-module test files (planner, runner, engine core/merge)
- extract unit tests into sibling files (query, storage, vector, search)
- *(query)* read TTL reaper state through typed stores
- *(storage)* apply proposal mutations through the engine
- *(query)* type the parallel OCC read-set accumulator
- thread storage transaction through stores
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)
- *(tests)* embed + storage migration to in-memory fixtures (Database::open_in_memory)
- *(storage,query)* move OCC tracking to Layer 3 Coordinator (G104)
- *(storage/coordinator)* extract MultiModalCoordinator trait (G105)
- *(storage/coordinator)* trim doctests to internal-crate scope
- *(storage)* extract Layer 3 Coordinator sub-module (R164)

#### Testing

- *(storage)* pin oplog segment filename contract
- *(backup)* assert checkpoint dirs by real partition name
- *(storage)* regression test for capacity-scanner warm-load race
- *(storage,modality)* G101 audit close - range_scan API + CRS dispatch + stronger exclusion
- *(modality,storage)* reduce proptest cases for faster regression runs
- *(query)* RYOW + legacy-mode OCC invariants, scrub task IDs
- *(storage,query)* edge cases + dyn dispatch for G104/G105
- *(storage)* final R164 coverage round + rustdoc cleanup
- *(storage/coordinator)* edge cases + doctests + concurrency
- *(storage)* page-ECC policy - builder + serde back-compat + Volatile edge

---

## 0.5.0 - 2026-06-27

#### Added

- *(storage)* retained oplog journal + single-node repair for embedded
- *(replicate)* self-describing segment blob and dispatching installer
- *(storage)* placement-segment descriptor and per-partition map
- *(storage)* descending range and prefix scans
- *(storage)* batched multi_get for known-key sets
- *(storage)* coalesce delete runs on the durable commit path
- *(storage)* MVCC range-delete apply path + partition cache invalidation
- *(storage)* OplogOp::RemoveRange wire type + CDC mapping
- *(storage)* run-length coalesce delete sets into point + range deletes
- *(storage)* io_uring filesystem backend behind --features io-uring
- *(storage)* multi-endpoint topology config
- *(embed)* bound interactive transaction buffered writes
- *(embed)* interactive multi-statement transactions
- *(storage)* park and resume transaction state across statements
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(storage)* GC watermark driver from live snapshot pins
- *(storage)* hard-link checkpoint of the whole database
- *(embed)* wire the vector oplog worker into Database
- *(storage)* oplog tailer reads the active segment [skip bench]
- *(storage)* VectorF32 + VectorRerank partitions (ADR-033 revised)
- *(storage,query)* OccScope typed contains helpers + audit-test migration
- *(storage)* cascade also fires at Full + metric/concurrency tests
- *(storage)* background capacity scanner + hard-limit-strategy edges
- *(storage)* per-endpoint capacity tracking + hard-limit enforcement
- *(storage)* page-checksum wire-through + ECC policy config surface
- *(storage)* per-LSM-level endpoint routing + cascade eviction
- *(storage)* R156 + R157 - multi-endpoint storage placement
- *(temporal)* R172c Phase 3b - nested PropertyPath / doc_* fns on temporal
- *(storage)* time-based memtable flush trigger to bound oplog retention
- *(storage)* implement standalone WAL for crash durability
- *(computed)* R085 decay interpolation tests and NVMe write buffer for w:cache
- *(storage)* add MemFs in-memory test backend support
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

#### Fixed

- *(storage)* clear corrupt partition physically before repair reinstall
- *(storage)* scrub collects block corruption instead of aborting
- *(backup)* flush memtables before creating a checkpoint
- *(raft)* advance follower oracle during entry apply
- *(storage)* defer capacity-scanner first tick by interval to close warm-load race
- *(storage)* capacity scanner counts every endpoint file, not only SSTs
- *(storage)* gate every write path + typed propagation to gRPC client
- *(storage)* gate oplog purge on cross-partition flush watermark
- *(raft)* recover last_log_id from oplog on unclean shutdown restart

#### Performance

- *(spatial)* Z-curve skip-scan via seekable range iterator
- *(raft)* O(delta) incremental snapshot via changed-keys scan
- *(storage)* fold adjacency operands in force_compaction, time-travel safe
- *(storage)* collapse adjacency merge operands into single values
- *(modality/spatial)* Z-curve subrange decomposition (G101)
- *(storage)* batch Extra-targeting deltas in DocumentMerge
- *(storage)* parallel memtable writes within write batch (R091)

#### Refactored

- *(core)* move delete coalescing to core, operate on mutations
- extract multi-module test files (planner, runner, engine core/merge)
- extract unit tests into sibling files (query, storage, vector, search)
- *(query)* read TTL reaper state through typed stores
- *(storage)* apply proposal mutations through the engine
- *(query)* type the parallel OCC read-set accumulator
- thread storage transaction through stores
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)
- *(tests)* embed + storage migration to in-memory fixtures (Database::open_in_memory)
- *(storage,query)* move OCC tracking to Layer 3 Coordinator (G104)
- *(storage/coordinator)* extract MultiModalCoordinator trait (G105)
- *(storage/coordinator)* trim doctests to internal-crate scope
- *(storage)* extract Layer 3 Coordinator sub-module (R164)

#### Testing

- *(storage)* pin oplog segment filename contract
- *(backup)* assert checkpoint dirs by real partition name
- *(storage)* regression test for capacity-scanner warm-load race
- *(storage,modality)* G101 audit close - range_scan API + CRS dispatch + stronger exclusion
- *(modality,storage)* reduce proptest cases for faster regression runs
- *(query)* RYOW + legacy-mode OCC invariants, scrub task IDs
- *(storage,query)* edge cases + dyn dispatch for G104/G105
- *(storage)* final R164 coverage round + rustdoc cleanup
- *(storage/coordinator)* edge cases + doctests + concurrency
- *(storage)* page-ECC policy - builder + serde back-compat + Volatile edge

---

## 0.4.2 - 2026-05-11

#### Added

- *(storage)* time-based memtable flush trigger to bound oplog retention

#### Fixed

- *(storage)* gate oplog purge on cross-partition flush watermark

---

## 0.3.18 - 2026-04-16

#### Fixed

- *(raft)* recover last_log_id from oplog on unclean shutdown restart

---

## 0.3.15 - 2026-04-15

#### Performance

- *(storage)* batch Extra-targeting deltas in DocumentMerge

---

## 0.3.11 - 2026-04-14

#### Added

- *(storage)* implement standalone WAL for crash durability

---

## 0.3.10 - 2026-04-14

#### Performance

- *(storage)* parallel memtable writes within write batch (R091)
