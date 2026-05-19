# coordinode-cluster

Layer 6 of the CoordiNode storage stack: **cluster topology + shard
routing** traits with their CE single-node implementations.

This crate sits at the top of the storage stack (see
`arch/core/storage-stack.md` §Layer 6). It owns the 5-level
failure-domain tree (`geo → dc → rack → server → endpoint`) and the
shard-to-node map. Two traits drive every consumer:

| Trait | Consumed by | Purpose |
|-------|-------------|---------|
| `ClusterTopology` | Layer 5 (planner), Layer 2 (placement) | Topology tree + shard descriptors + placement candidate sets |
| `ShardRouting` | Layer 5 (query engine) | Routing key → shard id resolution |

The trait surface is **identical CE/EE** — only the impls differ. CE
ships `SingleNodeTopology` + `SingleShardRouting`; future Phase 2 CE
multi-node and Phase 3 EE `CrushTopology` plug into the same traits.

## Quick start

```rust
use coordinode_cluster::{
    ClusterTopology, CrushRule, Modality, ShardId, SingleNodeTopology,
};
use coordinode_storage::engine::config::{
    Durability, EndpointConfig, Media, StorageConfig, Tier,
};

let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    "ep-warm", std::path::Path::new("/tmp/store"),
    Media::Hdd, Durability::Durable, Tier::Warm,
)]);
let topology = SingleNodeTopology::from_storage(&cfg);

// One shard, leader is the local node.
assert_eq!(topology.shards().len(), 1);
let leader = topology.shard_leader(ShardId::ZERO).unwrap();
assert_eq!(leader.server, "local");

// Warm-tier candidates are the warm endpoints on the local server.
let candidates = topology
    .placement_candidates(&CrushRule::local_tier(), Modality::Node, Tier::Warm)
    .unwrap();
```

## Trait contract notes

- **Topology mutation is out of scope.** Adding nodes / changing
  CRUSH rules is a separate admin path; this crate's trait is
  read-only on the hot path.
- **EE-only rules in a CE binary surface as `TopologyError::EeOnly`.**
  Phase 3 EE `CrushTopology` parses the full CRUSH rule grammar
  (`crush.md`); CE only handles `CrushRule::LocalTier`.
- **Modality enum is the Layer 4 inventory.** Adding a store
  modality means adding a `Modality` variant and bumping the
  `Modality::all()` count; a regression test pins the count.

## Failure-domain hierarchy

Per `arch/placement/crush.md` the tree has exactly 5 levels:

```
geo → dc → rack → server → endpoint
```

CE single-node deployments collapse the upper four levels to
`"local"`; the leaf endpoint still carries its real id and tier so
placement candidate filtering by tier works out of the box. Phase 2
CE and Phase 3 EE will populate the upper levels from the cluster
config.

## Status

- **CE Phase 1:** ✅ `SingleNodeTopology`, `SingleShardRouting`
- **CE Phase 2 (multi-node HA):** ❌ — same trait, new impl
- **EE Phase 3 (CRUSH multi-DC):** ❌ — same trait, `CrushTopology`
  + `MultiShardRouting`

## License

AGPL-3.0 (CE). See workspace `LICENSE`.
