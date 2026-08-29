# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-timeseries-v0.4.3...coordinode-timeseries-v0.5.0) - 2026-06-27

#### Added

- *(timeseries)* ε-policy — opt-in WITH BITEMPORAL via split write entry points (β: Cypher paused)
- *(timeseries)* close G103 #3 Gap #4 — PersistentMonotonicHlcClock with engine-backed restart monotonicity
- *(modality,timeseries)* G103 sub-system #3 — bitemporal __ingestion_ts__ axis
- *(modality,timeseries)* G103 sub-system #4 — overflow compactor primitives
- *(timeseries)* G103 slice C — Tier 3 overflow routing + background compactor
- *(timeseries)* G103 slice B — Tier 2 recently-closed LRU + reopen path
- *(timeseries)* new crate coordinode-timeseries (G103 slice A — BucketCatalog + Tier 1 buffer)

#### Refactored

- extract unit tests into sibling files (server, raft, replicate, embed, timeseries)
- extract unit tests into sibling test files
- *(modality)* thread Transaction through spatial, blob, time-series stores

#### Testing

- *(timeseries)* G103 #3 audit closure — backfill on compact, edge case tests, restart-monotonicity gap documented
