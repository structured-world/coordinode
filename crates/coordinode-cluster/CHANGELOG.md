# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## [0.5.0](https://github.com/structured-world/coordinode/compare/coordinode-cluster-v0.4.3...coordinode-cluster-v0.5.0) - 2026-06-27

#### Added

- *(cluster)* chunk-assignment table for shard routing
- *(cluster)* crash-recovery replay for LocalStateMachine
- *(cluster)* CE LocalStateMachine state-machine backend
- *(cluster)* node state-machine backend trait + operation types
- *(cluster)* add VectorShardRouter trait + single-partition default
- *(cluster)* migration plan explain string
- *(cluster)* plumbed online-during-rebuild policy in planner
- *(cluster)* online-during-rebuild policy enum
- *(cluster)* local migration planner picks lowest-cost target
- *(cluster)* migration cost model with hnsw rebuild line
- *(cluster)* migration plan and cost types
- *(cluster)* Layer 6 ClusterTopology + ShardRouting traits + CE impls

#### Documentation

- *(cluster)* document online-during-rebuild policy
- *(cluster)* document the migration planner

#### Refactored

- extract unit tests into sibling files (client, bench, cluster, s3, test-fixtures)

#### Testing

- *(cluster)* online-during-rebuild policy threading
- *(cluster)* planner picks remote endpoint on full source
- *(cluster)* doctests + edge cases + ADR-028 helpers + benches + proptest
