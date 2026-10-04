# Changelog

All notable changes to this crate are documented in this file.

## v0.7.0 - 2026-10-04

### Added

- *(storage)* pause writes when the disk runs low, keep serving reads
- *(cluster)* serve a read-only member's reads as of its last commit
- *(cluster)* report each member's version and the group's write pause
- *(cluster)* match group members by version and move a group by majority
- *(query)* read temporal nodes at the statement's current time
- schema claims, event-driven background work
- journal index writes, speed up the Raft path
- *(server)* [**breaking**] drop seqno-bounded incremental backups

### Documentation

- state design reasons instead of citing other codebases
- *(raft)* drop internal ids and document paths

### Fixed

- *(cluster)* a member at an older version than its group is behind
- *(cluster)* restart a member that holds its group's data
- *(cluster)* count only reachable voters toward a version majority
- *(raft)* keep acknowledged writes across a restart over a snapshot
- *(storage)* record a directory's migration only when its open succeeds
- *(server)* stop reporting ready once consensus has stopped
- *(raft)* let a purge below what is already purged do nothing
- *(raft)* settle membership and shutdown under load
- *(raft)* keep snapshots in a file, not in memory or the store
- *(raft)* wait for leadership transfers and checkpoints to finish
- *(raft)* bound the linearizable lease read by the fence timeout
- *(raft)* honour the limiter bypass for every write concern
- *(raft)* publish the applied index when a snapshot is installed
- *(cdc)* stream applied Raft log entries from the log's own files
- *(raft)* stop without waiting on a group that is gone
- *(raft)* wait out snapshot work on shutdown
- *(storage)* [**breaking**] name the upgrade path from 0.6 stores
- *(replicate)* repair partitions at an exact raft position
- *(raft)* build snapshots from a capture at their log id
- *(raft)* resume and purge the raft log by per-tree apply coverage
- *(storage)* cover columnar tables in recovery and in snapshots
- *(storage)* replay the embedded journal by per-partition coverage

### Testing

- *(cluster)* abandon a version move by re-adding the updated member
- time the watermark advance from the read it unblocks
- *(raft)* poll for snapshot builds instead of sleeping on them
- *(raft)* pin the snapshot install's ordering guarantees
- *(raft)* cut open replication streams in the partition nemesis
- make the suites pass on any target dir and as root
- *(raft)* cover log retention, failed rebuilds and interrupted installs
- *(storage,raft)* cut power at every sync and write, and upgrade a v0.6.0 store

## v0.6.0 - 2026-09-23

### Added

- *(cluster)* [**breaking**] refuse a join from a node that holds data
- *(server)* name the field in a refused write concern
- *(txn)* [**breaking**] write concern as two axes, w and journal
- *(storage)* [**breaking**] engine-owned MVCC retention window
- *(txn)* [**breaking**] commit receipt, one seqno per proposal

### Documentation

- fix broken links and stray tags in the API docs

### Fixed

- *(raft)* wait for the previous membership change to commit
- *(storage)* refuse a counter overflow where the caller can still hear it
- *(cluster)* publish existing data as the group's base state
- *(raft)* publish own address before adding a peer
- *(storage)* commit w:0 writes through the log

### Testing

- *(raft)* log shutdown steps in the partition test
- *(raft)* a node without a quorum still stops

## 0.5.8 - 2026-09-05

#### Testing

- *(cluster)* stop handing two nodes the same port

---

## 0.5.7 - 2026-09-01

#### Added

- *(session)* tell a client what its connection can do, and let it configure one
- *(server)* carry a write to the leader instead of refusing it

#### Fixed

- *(raft)* bind gRPC listener eagerly and free the port on shutdown

#### Testing

- *(raft)* wait for the cluster instead of guessing how long it takes
- *(raft)* stop the consensus suite from testing election timing

---

## 0.5.2 - 2026-08-30

#### Added

- *(wire)* let the TLS crypto provider be chosen at startup

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout
- *(raft)* stop the volatile-write drain panicking off-runtime

#### Testing

- *(raft)* wait for the leader instead of sleeping past the election

---

## 0.5.1 - 2026-08-29

#### Added

- *(storage)* retained oplog journal + single-node repair for embedded
- *(server)* fall back to WAL-replay repair when no replica serves
- *(raft)* expose committed oplog entries since an index
- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(server)* serve gRPC over TLS and mTLS
- *(raft)* compress RaftService wire traffic with the zstd codec
- *(raft)* add zstd transport codec for inter-node gRPC
- *(raft)* runtime voter and learner role transitions
- *(storage)* MVCC range-delete apply path + partition cache invalidation
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(storage)* VectorF32 + VectorRerank partitions (ADR-033 revised)
- *(storage)* per-LSM-level endpoint routing + cascade eviction
- *(storage)* R156 + R157 - multi-endpoint storage placement
- *(raft)* wire MaxAssignedWatermark into apply_proposal path
- *(server)* R150 - monolithic binary --mode=full, shared :7080, NodeInfoLayer
- *(cluster)* node decommission protocol + unified Raft write path
- *(cluster)* implement cluster join protocol (R091b)
- *(storage)* implement standalone WAL for crash durability
- *(raft)* R141 follower reads - ReadFence, SyncPerBatch persist fix
- *(raft)* chunked gRPC snapshot transfer to prevent OOM (G046)
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(raft)* add retry with exponential backoff to batch drain loop (G047b)
- *(raft)* add WaitForMajorityService for batched proposal coalescing (G047)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

#### Fixed

- *(raft)* set default wire zstd level to 1 and measure the wire
- *(raft)* gate snapshot trigger on log progress
- *(raft)* advance follower oracle during entry apply
- *(storage)* gate every write path + typed propagation to gRPC client
- *(storage)* gate oplog purge on cross-partition flush watermark
- *(raft)* recover last_log_id from oplog on unclean shutdown restart
- *(cluster)* rollback Learner on change_membership failure in monitor_and_promote
- *(server)* resolve proto submodule and clippy::panic in tests
- *(raft)* reduce chunk size to 2MB, add multi-chunk integration test
- *(ci)* update raft build.rs proto path and deny.toml format

#### Performance

- *(raft)* O(delta) incremental snapshot via changed-keys scan

#### Refactored

- extract shared wire codec, compress segment transfer too
- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)

#### Testing

- *(raft)* widen cluster-test election timeout under CI load
- *(raft)* add linearizability checker and clock-skew nemesis
- *(raft)* read_oplog_since returns post-checkpoint ops only
- *(raft)* inter-node mutual-TLS cluster replication
- *(raft)* snapshot trigger must skip idle intervals
- *(raft)* add 3-node pruning decommission test as final R091c entry
- *(cluster)* R091c decommission protocol test suite
- *(raft)* R141 complete test coverage - follower scenarios + StaleReplica
- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)

---

## 0.5.0 - 2026-06-27

#### Added

- *(storage)* retained oplog journal + single-node repair for embedded
- *(server)* fall back to WAL-replay repair when no replica serves
- *(raft)* expose committed oplog entries since an index
- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(server)* serve gRPC over TLS and mTLS
- *(raft)* compress RaftService wire traffic with the zstd codec
- *(raft)* add zstd transport codec for inter-node gRPC
- *(raft)* runtime voter and learner role transitions
- *(storage)* MVCC range-delete apply path + partition cache invalidation
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)
- *(storage)* VectorF32 + VectorRerank partitions (ADR-033 revised)
- *(storage)* per-LSM-level endpoint routing + cascade eviction
- *(storage)* R156 + R157 - multi-endpoint storage placement
- *(raft)* wire MaxAssignedWatermark into apply_proposal path
- *(server)* R150 - monolithic binary --mode=full, shared :7080, NodeInfoLayer
- *(cluster)* node decommission protocol + unified Raft write path
- *(cluster)* implement cluster join protocol (R091b)
- *(storage)* implement standalone WAL for crash durability
- *(raft)* R141 follower reads - ReadFence, SyncPerBatch persist fix
- *(raft)* chunked gRPC snapshot transfer to prevent OOM (G046)
- *(raft)* true async wtimeout via propose_with_timeout (G048)
- *(raft)* add retry with exponential backoff to batch drain loop (G047b)
- *(raft)* add WaitForMajorityService for batched proposal coalescing (G047)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

#### Fixed

- *(raft)* set default wire zstd level to 1 and measure the wire
- *(raft)* gate snapshot trigger on log progress
- *(raft)* advance follower oracle during entry apply
- *(storage)* gate every write path + typed propagation to gRPC client
- *(storage)* gate oplog purge on cross-partition flush watermark
- *(raft)* recover last_log_id from oplog on unclean shutdown restart
- *(cluster)* rollback Learner on change_membership failure in monitor_and_promote
- *(server)* resolve proto submodule and clippy::panic in tests
- *(raft)* reduce chunk size to 2MB, add multi-chunk integration test
- *(ci)* update raft build.rs proto path and deny.toml format

#### Performance

- *(raft)* O(delta) incremental snapshot via changed-keys scan

#### Refactored

- extract shared wire codec, compress segment transfer too
- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)

#### Testing

- *(raft)* widen cluster-test election timeout under CI load
- *(raft)* add linearizability checker and clock-skew nemesis
- *(raft)* read_oplog_since returns post-checkpoint ops only
- *(raft)* inter-node mutual-TLS cluster replication
- *(raft)* snapshot trigger must skip idle intervals
- *(raft)* add 3-node pruning decommission test as final R091c entry
- *(cluster)* R091c decommission protocol test suite
- *(raft)* R141 complete test coverage - follower scenarios + StaleReplica
- *(raft)* add tests for propose_with_timeout and WriteConcernTimeout (G048)

---

## 0.4.2 - 2026-05-11

#### Fixed

- *(storage)* gate oplog purge on cross-partition flush watermark

---

## 0.4.1 - 2026-04-18

#### Added

- *(raft)* wire MaxAssignedWatermark into apply_proposal path

---

## 0.3.18 - 2026-04-16

#### Added

- *(server)* R150 - monolithic binary --mode=full, shared :7080, NodeInfoLayer

#### Fixed

- *(raft)* recover last_log_id from oplog on unclean shutdown restart

---

## 0.3.12 - 2026-04-14

#### Added

- *(cluster)* node decommission protocol + unified Raft write path

#### Testing

- *(raft)* add 3-node pruning decommission test as final R091c entry
- *(cluster)* R091c decommission protocol test suite

---

## 0.3.11 - 2026-04-14

#### Added

- *(cluster)* implement cluster join protocol (R091b)
- *(storage)* implement standalone WAL for crash durability

#### Fixed

- *(cluster)* rollback Learner on change_membership failure in monitor_and_promote

---

## 0.3.10 - 2026-04-14

#### Added

- *(raft)* R141 follower reads - ReadFence, SyncPerBatch persist fix

#### Fixed

- *(server)* resolve proto submodule and clippy::panic in tests

#### Testing

- *(raft)* R141 complete test coverage - follower scenarios + StaleReplica
