# Changelog

All notable changes to this crate are documented in this file.

## v0.10.0 - 2026-11-19

### Added

- *(replication)* deliver a dropped label as the end of its scope
- *(storage)* give a cut's continuation its logical events
- *(embed)* keep a projection's effects and progress in one commit
- *(replication)* continue a change read inside a large transaction
- *(replication)* route change events by implied labels
- *(server)* stream committed changes as logical events
- *(embed)* read committed changes as events from a cursor
- *(storage)* project committed units into logical change events
- *(core)* name coalesced delete runs in the effect context
- *(storage)* seal the change-feed context at the writers
- *(core)* carry the logical effect context in the unit frame
- *(server)* time a statement's stages in diagnostic builds
- *(embed)* rebuild a consumer's state through a fixed cut
- *(storage)* fixed cut of a group member
- *(storage)* name a fixed cut's history by its deployment
- *(storage)* fixed-cut source handoff for a standalone store
- *(storage)* reclaim covered journal as flushes land
- *(storage)* count compactions by what they did
- *(storage)* trace what a footprint refusal was judged on
- *(storage)* bounded material fold in commit order
- *(storage)* material accounting and log-ordered registry endings
- *(raft)* client writes take room granted by a majority
- *(storage)* keep a store's own journal for its log consumers
- *(replication)* typed causal bounds and receipts
- *(storage)* use the io_uring backend by default
- *(storage)* hold a temporal unique value per instant of valid time
- *(storage)* bind history to the store's deployment before it lands
- *(query)* answer AS OF past the retention from temporal history
- *(storage)* keep node history and write records keyed by their commit
- *(index)* bound the bytes of implication placement and folding
- *(index)* key unique entries by their owner
- *(search)* configure the text index worker's step and retry
- *(index)* answer a text search with its view's interpretation
- *(schema)* give labels a catalog identity records store
- *(schema)* declare type, implied and auxiliary labels
- *(storage)* collect orphan blobs and return their space
- *(storage)* give blobs owners, a catalog and a guarded lifetime
- *(storage)* per-table pressure series that go with the table
- *(storage)* judge, repair and drop columnar tables as partitions
- *(storage)* hold the database to a strict total footprint limit

### Documentation

- *(storage)* link a table's creation stamp to where it comes from
- *(storage)* drop links to private items and stray table text
- *(readme)* drop the archived Python SDK
- *(storage)* describe the node history partition without a dash
- *(storage)* name a private helper without linking it

### Fixed

- *(storage)* a cursor at the end follows every entry applied before it
- address the code analysis findings on main
- *(storage)* leave the open allowance to a store full of client writes
- *(storage)* copy kept labels and label names only within their room
- *(server)* hold one change batch per stream and name labels by identity
- *(storage)* take a change read's working memory before allocating it
- *(storage)* bound a change read's memory where it is held
- *(storage)* deliver blob owner changes and refuse unknown components
- *(storage)* bind every node change's labels and type interpretation
- *(storage)* route a node's later change by the labels its earlier left
- *(embed)* refuse a projection batch that does not continue it
- *(storage)* bind a stored ceiling read to its ledger installation
- *(storage)* cover a visible raft write in the receipt
- *(storage)* keep the stamp a table creation got
- *(storage)* end a journal copy at a whole entry, keep table stamps
- *(storage)* end a checkpoint's journal copy at its pins
- *(storage)* hold a table operation whole against a checkpoint
- *(storage)* hold the table set still at a checkpoint's cut
- *(storage)* capture a checkpoint at one cut of the commit clock
- *(storage)* take a checkpoint from pinned trees, without a flush
- *(storage)* read the stored closure for an offline group cut
- *(storage)* charge a cut's clipped bounds and real scan state
- *(storage)* bound a cut entry's memory as a whole
- *(storage)* bound a cut's reads by the caller's budget
- *(storage)* clip a cut's range deletes to its scope
- *(storage)* qualify a group member's cut by its scope's tree
- *(storage)* bound a state cut by scope, settled entries and bytes
- *(storage)* refuse a cut whose view the store does not hold
- *(storage)* wake a failed journal reclamation when room returns
- *(storage)* rerun journal reclamation on real grounds only
- *(storage)* keep charging what a failed checkpoint left
- *(storage)* remove a checkpoint that failed partway
- *(storage)* wake a flush refused for room without a lost wakeup
- *(raft)* a follower below its disk reserve takes no committed entries
- *(storage)* keep the recovery reserve's margin beside two starts
- *(storage)* hold collection at the group's closed bound
- *(storage)* answer no commit ceiling while the Schema tree is owed
- *(storage)* seal the replay floor and the records a rebuild keeps
- *(storage)* charge the staged Schema copy before it is written
- *(raft)* read a stored commit ceiling off the async runtime
- *(storage)* read a checkpoint's replay floor without opening it
- *(storage)* settle the Schema tree before the open reads it
- *(storage)* material accounting in a tree of its own
- *(storage)* a refused open removes the lock files it created
- one-cut admin backup, exclusive bounded restore
- *(storage)* closure ledger in the reserved keyspace
- *(query)* charge evaluator memory before it is held
- *(storage)* a statement's deadline and cancel bound every admission wait
- *(replicate)* bound consumer progress, heartbeats and breakpoints
- *(storage)* owe the manifest write a deferred flush leaves to its barrier
- *(storage)* read history before the binding when founding
- *(storage)* end a wait's waker registration with the wait
- *(embed)* wait for a standalone causal bound without a thread
- *(storage)* keep each log prefix's commit ceiling exactly
- *(raft)* hand over again a sealed commit its own entry closed
- *(raft)* let an entry close the commits it carries
- *(query)* read only views every commit below is applied at
- *(raft)* close the commits a lost leader left uncovered
- *(storage)* keep the memtable start debt crate-private
- *(storage)* admit a write into an empty memtable for its flush
- *(storage)* count a flush allowance per memtable in the debt
- *(storage)* keep member records from making closing entries due
- *(storage)* report a write of unknown outcome as such
- *(storage)* hold a value per instant while its unique index is built
- *(storage)* stop the log at its first failed write
- *(storage)* never found a new domain over retained history
- *(storage)* admit the commit timestamp a stamped key carries
- *(index)* keep a claim's charge only on what is kept
- *(index)* charge unique claims and checks for as long as they live
- *(index)* count a unit's derived fan-out before expanding it
- *(index)* bound the fan-out everywhere and type every index refusal
- *(index)* refuse keys past the store's limit, bound entry fan-out
- *(storage)* map a bound too short for a slot past the whole slot
- *(index)* charge implication placement memory before taking it
- *(storage)* keep the engine and dump format versions
- *(storage)* refuse only a change of what a typed label implies
- *(schema)* bind a label's first use through the ordered log
- *(storage)* decide a schema read from the current state
- *(storage)* save the consensus state below half the space reserve
- *(storage)* name the key a pending-commit conflict is about
- *(storage)* close a dropped table's tree before removing its folder
- *(storage)* admit promise copy writes as recovery work
- *(raft)* bind promises to a member, write copy durably
- *(raft)* keep the purge point and a loss with the vote; replace a member
- *(raft)* keep a second copy of the member's vote
- *(storage)* let blob collection run below the free-space reserve
- *(storage)* flush and capture a table only while it lives
- *(storage)* no table file is made in a folder being dropped
- *(storage)* a second drop of a table drops the same table
- *(storage)* hold a table name until its drop finishes
- *(storage)* serve a rebuilt tree's history above its copy's floor
- *(storage)* open a checkpoint on its clock, never compacted
- *(storage)* refuse a rebuild from a damaged checkpoint copy
- *(storage)* rebuild a partition at its original seqnos
- *(storage)* rebuild a table only from provable state
- *(storage)* roll back a torn manifest tail instead of repairing
- *(storage)* capture the journal tail after a checkpoint's trees
- *(storage)* refuse a hard limit below the supported minimum
- *(storage)* sync a new journal segment's directory

### Performance

- *(server)* share one log reader across change streams again
- *(storage)* take a statement's receipt at its own view
- *(storage)* attribute complete-snapshot calls to their caller
- *(storage)* time the complete-snapshot wait under stage-timing
- *(embed)* open a checkpointed store without scrubbing it
- *(storage)* read free space off the write path
- *(storage)* check snapshot retention of the tree read only
- *(storage)* copy a transaction's snapshot read once
- *(storage)* copy a cut's page into budget-charged blocks
- *(storage)* search a batch entry's seals, not scan them
- *(storage)* keep a transaction's writes in key order
- *(storage)* read a unique value's staged owners by range
- *(txn)* restate bound trends by moving the claims
- *(query)* read a node's schemas once per write
- *(storage)* leave a journaled tree's flush syncs to a barrier
- *(storage)* export a generation's history by its own key range
- *(storage)* bench flush and compaction of a table
- index partition trees by discriminant, share index definitions

### Refactored

- *(storage)* [**breaking**] withdraw prepaid flush room and member grants
- *(storage)* [**breaking**] remove per-mutation material accounting
- *(schema)* store canonical labels and judge RETYPE by its result
- *(storage)* keep the table repair record crate-private

### Testing

- *(storage)* a backup at the limit may see room given back meanwhile
- *(storage)* capture the store's log in the full-store cleanup test
- *(storage)* name the charges a barrier finds no room beside
- *(storage)* ask a refused barrier again under the raised limit
- *(storage)* wait for flushes by deadline, not a fixed budget
- *(embed)* cover instances, dropped labels and a failed last fragment
- *(storage)* hold a settled persist to its own success
- *(storage)* fail the installation of a persist on its own
- *(storage)* show a persist sweep reaching an unknown installation
- *(storage)* sweep a background flush's failures across a power cut
- *(storage)* a compaction stopped by the limit holds no room
- *(storage)* fill a capped store until a one-byte write is refused
- *(storage)* assert shared cut page block
- *(embed)* wait for the idle reaper's end, not a fixed sleep
- *(storage)* keep unit retirement out of the replay multiplicity check
- *(raft)* bench the closure ledger's window, lookups and load
- *(storage)* the recovery floor's limit as a compile-time check
- *(storage)* bench what the material accounting costs the writes
- *(storage)* give the queued-flush test room on a loaded host
- *(integration)* prove a member's disk stall and check every attempt
- stall a server's Raft log on real disk I/O for the move test
- *(raft)* a stalled log sync keeps the leader
- *(storage)* use is_multiple_of in the mixed load test
- *(storage)* mixed reads, writes and compactions on the default file system
- *(storage)* a power cut never leaves history without its domain
- *(storage)* name the candidate implication in the admission test
- *(storage)* build the candidate test schema with its own fields
- *(storage)* fail every matching op of a persist
- *(storage)* fail only the persisting partition's files
- *(storage)* wait for the drop to close the tree, not for a sleep
- *(storage)* measure the copy's peak from a steady store
- *(storage)* model clean-stale pages and torn files
- *(storage)* fail each step of a promise copy write
- *(storage)* compare table tree paths absolute, as the tree config does
- *(storage)* compare table tree paths against the endpoints as configured
- *(storage)* hold a slowed table's rows to the write rate
- *(storage)* judge each table into its own verdict and one metrics series
- *(storage)* check the tail-captured segment's bytes on every platform

### Bench

- *(storage)* separate foreground and full accounting cost
- *(storage)* measure statement flushes on a fresh store per series
- *(raft)* separate leadership seen from confirmed

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
