# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

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
