# Changelog

All notable changes to this crate are documented in this file.

## v0.8.0 - 2026-10-23

### Added

- *(embed)* [**breaking**] publish embedded Rust libraries ([#70](https://github.com/structured-world/coordinode/pull/70))
- *(storage)* wire the remaining lsm-tree 6.0 controls
- *(storage)* one I/O budget for compaction and scrub
- *(raft)* send snapshots from pinned trees, compact on flush
- *(schema)* prove presence and type constraints in the background
- *(schema)* prove label definitions in the background

### Documentation

- *(storage)* describe a lost frame without linking a private item

### Fixed

- *(storage)* recover lost writes inside a sealed oplog segment
- *(storage)* answer a blob write only once it is durable
- *(storage)* tell a group member's store by its bound consensus log
- *(storage)* make the journal's retirement into a group restartable
- *(storage)* refuse a group member's store an own journal

### Performance

- *(index)* project, checksum and bind index check passes
- *(storage)* flush columnar tables in the background
- *(storage)* keep vector truth values in the block cache
- *(storage)* order tree syncs, flush devices at promises
- *(schema)* prove a package's labels in one certification

### Testing

- *(storage)* match synced tree paths with either separator
- *(storage)* cut the power at data syncs as well as full ones
- *(storage)* a persist flushes the device of every endpoint
- *(storage)* stop leaning on the 5.x file layout
- *(storage)* retirement segments brought back by a cut are retired again

## v0.7.3 - 2026-10-18

### Added

- *(operation)* apply each client write once
- *(embed)* replicated embedded database behind a feature

### Fixed

- *(storage)* let a catalog commit wait for a claim in flight
- *(storage)* scrub blob files and tune local journal compaction
- *(raft)* keep snapshot captures on the engine's filesystem
- *(storage)* never copy a checkpoint over a file it holds
- *(storage)* apply each recovered cache write once
- *(storage)* land a volatile write once, through the log
- *(storage)* keep a node's own records across a rebuild

### Performance

- *(storage)* flush the device only where recovery needs it
- *(storage)* defer the log sync of volatile writes, not the write
- *(storage)* index registered commits by key

### Testing

- *(storage)* cover crashes inside log truncation and compaction

## v0.7.2 - 2026-10-15

### Added

- *(cluster)* authenticate every inter-node call by node identity
- *(raft)* record each member's node identity

### Fixed

- *(storage)* recover oplog segments that lost writes mid-log
- *(cdc)* age a consumer's lag from committed work only
- *(storage)* keep oplog segments inside the retention window

## v0.7.1 - 2026-10-13

### Added

- *(index)* label posting codec and lookup catalog record
- *(index)* carry the peer's proof with a generation's history
- *(index)* carry index copy proofs through transfers
- *(index)* guard each local index copy before it serves
- *(replicate)* fill a generation's replacement from a peer
- *(storage)* replace a generation's local installation
- *(storage)* store index generations under local installations
- *(storage)* charge staged writes to the statement budget
- *(config)* make the index feed capacity a setting
- *(query)* count a label's nodes from its counter
- *(storage)* keep exact edge cardinality counts at commit
- *(core)* count edge cardinality over logical identities

### Fixed

- *(storage)* hold the retention window from the first watermark
- *(storage)* refuse a write over a moved index binding as a conflict
- *(storage)* admit a replacement's writes like any other write
- *(storage)* bound a Raft generation history by its applied entries
- *(storage)* mark a partition capture with the engine format
- *(storage)* read a record's latest value at the present
- *(storage)* announce an applied entry before a reader can see it
- *(storage)* fold node deltas into a whole write of the same node
- *(reads)* reuse the statement schema view
- *(storage)* release commit claims where the writes land
- *(raft)* fence follower cuts with a closed bound
- *(query)* search temporal labels at their valid state
- *(storage)* hide rows a transaction deleted from its scans
- *(storage)* let a catalog change wait for admitted writers
- *(server)* list labels from the counters, under a read lock
- *(storage)* flush memtables sealed while the queue was full
- *(storage)* wait for every older transaction in an index build
- drop the upgrade path from 0.6 stores
- *(storage)* keep the timeline docs free of private links
- *(index)* refuse to open over an index catalog record it cannot read

### Performance

- *(storage)* finish a generation import without flushing
- *(query)* stream node scans within the statement budget
- *(storage)* keep five minutes of MVCC history by default
- *(query)* bound property delta chains with whole writes
- *(query)* take TTL candidates from a ready anchor index
- *(query)* write a top-level SET as a property delta
- *(storage)* hard-link sealed journal segments into checkpoints
- *(storage)* borrow node bytes during decode
- *(storage)* admit bounds by the way writes move the count
- *(cdc)* answer a consumer's progress age without rereading the log

### Testing

- two checks that failed on loaded CI runners
- *(storage)* keep generation-tagged keys out of the round-trip property
- *(storage)* keep a transaction's view and conflicts across a switch
- *(storage)* drop a replacement when a Raft snapshot replaces the index
- *(storage)* cover a replacement's registration, readers and rebuilds
- *(storage)* cover a replacement against concurrent effects
- *(storage)* keep each test's doc comment with its test
- *(storage)* prove a persist after a failed one keeps its write
- *(storage)* replay kept cardinality counts once after a power cut
- *(storage)* race edge commits through the whole pipeline
- *(storage)* first edges into different pairs share an upper bound
- *(storage)* cover kept counts on every commit path

## v0.7.0 - 2026-10-06

### Added

- *(index)* enforce unique values while an index builds
- *(storage)* pause writes when the disk runs low, keep serving reads
- *(cluster)* match group members by version and move a group by majority
- *(query)* CREATE/DROP CONSTRAINT for node labels
- *(server)* configure storage block compression from the config file
- *(query)* read temporal nodes at the statement's current time
- *(cdc)* [**breaking**] explicit retention policy and acknowledged progress per consumer
- *(storage)* report local commits on the applied feed
- schema claims, event-driven background work
- journal index writes, speed up the Raft path
- *(server)* [**breaking**] drop seqno-bounded incremental backups

### Documentation

- *(storage)* drop a public link to a private constant
- *(storage)* state the durability invariants instead of their ids
- *(storage)* drop internal ids and document paths

### Fixed

- *(index)* release the build config lock before backfilling
- *(storage)* sample the partition footprint without capacity limits
- *(storage)* bound the time-travel horizon by data partitions only
- *(server)* record the consensus and compaction metrics the server describes
- *(storage)* read the oplog tail without decoding the last sealed segment
- *(storage)* never serve a tiered-cache value a later write replaced
- *(search)* answer unfolded writes exactly instead of waiting
- *(search)* maintain text indexes from committed writes
- *(cluster)* restart a member that holds its group's data
- *(schema)* hold a strict temporal label's schema on every version
- *(raft)* keep acknowledged writes across a restart over a snapshot
- *(storage)* record a directory's migration only when its open succeeds
- *(storage)* return a transaction's prefix scan in key order
- *(cdc)* refuse a resume past purged log instead of skipping
- *(storage)* enforce version conditions on unwritten keys
- *(storage)* skip delete-pending files in retention stats on Windows
- *(server)* report an unusable storage topology instead of panicking
- *(cdc)* stream applied Raft log entries from the log's own files
- *(vector)* maintain replica indexes from applied entries
- *(vector)* build indexes from a tap of applied writes
- *(storage)* [**breaking**] name the upgrade path from 0.6 stores
- *(replicate)* repair partitions at an exact raft position
- *(raft)* build snapshots from a capture at their log id
- *(storage)* rebuild from what the checkpoint trees hold
- *(storage)* survive a crash inside a partition rebuild
- *(storage)* keep the raft coverage reset tombstone below its base
- *(raft)* resume and purge the raft log by per-tree apply coverage
- *(storage)* cover columnar tables in recovery and in snapshots
- *(storage)* replay the embedded journal by per-partition coverage
- *(deps)* require the lsm-tree release the code needs

### Performance

- *(index)* read uncovered nodes once per index per commit
- *(storage)* keep no time-travel history of system partitions
- *(storage)* publish where the oplog starts instead of listing it
- *(storage)* tail the oplog from where the last read stopped
- *(storage)* measure the apply pauses of copies and installs
- *(storage)* measure what apply coverage costs

### Refactored

- *(index)* name indexes by stable identity and generation

### Testing

- *(storage)* name the recorded metric type in the compaction metrics test
- make the suites pass on any target dir and as root
- *(storage,raft)* cut power at every sync and write, and upgrade a v0.6.0 store

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
