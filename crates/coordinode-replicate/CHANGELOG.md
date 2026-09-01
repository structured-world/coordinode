# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## 0.5.7 - 2026-09-01

#### Added

- *(session)* tell a client what its connection can do, and let it configure one

---

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

## 0.5.1 - 2026-08-29

#### Added

- *(storage)* retained oplog journal + single-node repair for embedded
- *(replicate)* rebuild a partition from checkpoint plus oplog replay
- *(server)* repair corrupt partitions from peers on scrub detection
- *(replicate)* repair a partition from healthy peers via swarm pull
- *(replicate)* gRPC piece source for the swarm pull
- *(replicate)* serve the receiver-driven swarm piece-exchange
- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(replicate)* add segment drain client for peer push
- *(server)* register segment-transfer service in cluster mode
- *(replicate)* self-describing segment blob and dispatching installer
- *(replicate)* storage-backed segment export and install
- *(replicate)* source-side frame gather for segment transfer
- *(replicate)* segment-transfer gRPC receive handler
- *(replicate)* wire SegmentTransferService gRPC codegen
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)

#### Fixed

- *(storage)* clear corrupt partition physically before repair reinstall
- *(replicate)* physically replace corrupt tables on repair

#### Refactored

- extract shared wire codec, compress segment transfer too
- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)

---

## 0.5.0 - 2026-06-27

#### Added

- *(storage)* retained oplog journal + single-node repair for embedded
- *(replicate)* rebuild a partition from checkpoint plus oplog replay
- *(server)* repair corrupt partitions from peers on scrub detection
- *(replicate)* repair a partition from healthy peers via swarm pull
- *(replicate)* gRPC piece source for the swarm pull
- *(replicate)* serve the receiver-driven swarm piece-exchange
- *(wire)* encrypt outbound inter-node gRPC with client TLS
- *(replicate)* add segment drain client for peer push
- *(server)* register segment-transfer service in cluster mode
- *(replicate)* self-describing segment blob and dispatching installer
- *(replicate)* storage-backed segment export and install
- *(replicate)* source-side frame gather for segment transfer
- *(replicate)* segment-transfer gRPC receive handler
- *(replicate)* wire SegmentTransferService gRPC codegen
- *(replicate)* replication-orchestration crate (replicated writes + retention registry)

#### Fixed

- *(storage)* clear corrupt partition physically before repair reinstall
- *(replicate)* physically replace corrupt tables on repair

#### Refactored

- extract shared wire codec, compress segment transfer too
- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)
