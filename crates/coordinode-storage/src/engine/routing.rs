//! Per-LSM-level → endpoint routing (storage-stack Layer 2).
//!
//! Each storage partition (except `Schema`, which always stays single-tier
//! for bootstrap reasons) is assigned a `PartitionRouting` — a deterministic
//! mapping from LSM level to the endpoint that hosts SST files for that
//! level. Defaults follow `arch/placement/tiered-storage.md`:
//!
//! - **L0-L1** → first endpoint with `tier ∈ {Hot, HotCache, Memory}`
//! - **L2-L3** → first endpoint with `tier == Warm`
//! - **L4-L6** → first endpoint with `tier == Cold`
//!
//! Volatile endpoints are excluded from persistent SST placement: SSTs are
//! the authoritative store of LSM data and must survive process restart.
//! HotCache endpoints qualify for routing only when no Durable/Degraded Hot
//! endpoint exists — and even then the placement engine emits a warning.
//!
//! When a tier band has no matching endpoint, the resolver falls back along
//! a tier chain (preferred → next-best → final fallback = primary endpoint)
//! and emits a `tracing::warn!` so operators see degraded placement.
//!
//! The routing is computed once on the partition's first open against a
//! given endpoint set, then persisted in the Schema partition under key
//! `meta:routing:<partition_name>` (MessagePack-encoded). On reopen the
//! persisted routing wins so that lsm-tree's recovery scan covers exactly
//! the directories that hold existing SSTs — auto-rederiving on every open
//! would let endpoint-topology changes orphan SSTs.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use lsm_tree::config::LevelRoute;
use lsm_tree::fs::Fs;
use serde::{Deserialize, Serialize};

use crate::engine::config::{Durability, EndpointConfig, Tier};

/// Maximum LSM level the routing covers (lsm-tree uses 0..7 by default).
pub(crate) const MAX_ROUTED_LEVEL: u8 = 6;

/// Mapping from LSM level (0..=6) to the endpoint id that hosts SST files
/// at that level for a single partition.
///
/// `BTreeMap` (not `HashMap` per the original ROADMAP wording) so that
/// MessagePack serialisation is deterministic byte-for-byte across opens —
/// makes diff-based debugging of routing state straightforward.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionRouting {
    /// Level → endpoint_id. Every level in `0..=MAX_ROUTED_LEVEL` must
    /// have an entry.
    pub levels: BTreeMap<u8, String>,
}

/// Errors from routing operations.
#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    /// Persisted routing references an endpoint id that is no longer in
    /// the StorageConfig. The operator removed an endpoint that still
    /// holds SST data; reopen is unsafe.
    #[error(
        "persisted routing references endpoint id {endpoint_id:?} which is \
         not present in the current StorageConfig — operator must restore \
         the endpoint or run a documented drain procedure before removing it"
    )]
    UnknownEndpoint { endpoint_id: String },

    /// Persisted routing is missing a level. Either the file is corrupt
    /// or it was written by an incompatible engine version.
    #[error("persisted routing is missing level {level} (expected 0..={MAX_ROUTED_LEVEL})")]
    MissingLevel { level: u8 },
}

impl PartitionRouting {
    /// Compute the default routing for a partition against a given
    /// endpoint set per the canonical tier mapping (L0-L1 → hot, L2-L3 →
    /// warm, L4+ → cold) with fallback to next-best tier when a band
    /// has no matching endpoint.
    ///
    /// **Volatile filter:** endpoints with `durability == Volatile` are
    /// excluded from the candidate pool — SSTs must survive restart by
    /// definition. If the pool ends up empty (e.g., `with_endpoints_no_persistence`
    /// MemFs config), the Volatile endpoints are used as the last resort.
    pub fn default_for_endpoints(endpoints: &[EndpointConfig]) -> Self {
        // Persistent SST placement excludes Volatile by INV-D1 spirit
        // (oplog/WAL invariant generalises: any byte that must survive
        // restart cannot live solely on Volatile media).
        let persistent: Vec<&EndpointConfig> = endpoints
            .iter()
            .filter(|e| e.durability != Durability::Volatile)
            .collect();
        let pool: Vec<&EndpointConfig> = if persistent.is_empty() {
            endpoints.iter().collect()
        } else {
            persistent
        };

        // Tier preference per level band:
        //   L0-L1 → Hot (then HotCache, Memory, Warm, Cold)
        //   L2-L3 → Warm (then Hot, Cold, HotCache, Memory)
        //   L4-L6 → Cold (then Warm, Hot, HotCache, Memory)
        let hot_ep = pick_endpoint(&pool, &[Tier::Hot, Tier::HotCache, Tier::Memory], "L0-L1");
        let warm_ep = pick_endpoint(&pool, &[Tier::Warm], "L2-L3");
        let cold_ep = pick_endpoint(&pool, &[Tier::Cold], "L4-L6");

        let mut levels = BTreeMap::new();
        for lvl in 0..=1 {
            levels.insert(lvl, hot_ep.id.clone());
        }
        for lvl in 2..=3 {
            levels.insert(lvl, warm_ep.id.clone());
        }
        for lvl in 4..=MAX_ROUTED_LEVEL {
            levels.insert(lvl, cold_ep.id.clone());
        }
        Self { levels }
    }

    /// Validate that every endpoint id referenced by this routing exists
    /// in the current `StorageConfig.endpoints`. Returns
    /// [`RoutingError::UnknownEndpoint`] for the first missing id.
    pub fn validate(&self, endpoints: &[EndpointConfig]) -> Result<(), RoutingError> {
        let known: std::collections::HashSet<&str> =
            endpoints.iter().map(|e| e.id.as_str()).collect();
        for lvl in 0..=MAX_ROUTED_LEVEL {
            let id = self
                .levels
                .get(&lvl)
                .ok_or(RoutingError::MissingLevel { level: lvl })?;
            if !known.contains(id.as_str()) {
                return Err(RoutingError::UnknownEndpoint {
                    endpoint_id: id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Build the `Vec<LevelRoute>` to pass to `lsm_tree::Config::level_routes`.
    ///
    /// `primary_endpoint_id` is the endpoint chosen as the lsm-tree
    /// `Config.path` base. Levels that route to the primary endpoint are
    /// omitted from the LevelRoute list — they implicitly fall through
    /// to `Config.path`. Consecutive levels mapped to the same non-primary
    /// endpoint are coalesced into one route (lsm-tree requires
    /// non-overlapping ranges).
    pub fn to_level_routes(
        &self,
        endpoints: &[EndpointConfig],
        partition_name: &str,
        primary_endpoint_id: &str,
        fs: Arc<dyn Fs>,
    ) -> Vec<LevelRoute> {
        let id_to_path: std::collections::HashMap<&str, &PathBuf> =
            endpoints.iter().map(|e| (e.id.as_str(), &e.path)).collect();

        // Group consecutive levels mapping to the same non-primary endpoint.
        let mut routes: Vec<LevelRoute> = Vec::new();
        let mut current_start: Option<u8> = None;
        let mut current_id: Option<&str> = None;

        let flush = |routes: &mut Vec<LevelRoute>,
                     start: u8,
                     end_exclusive: u8,
                     id: &str,
                     id_to_path: &std::collections::HashMap<&str, &PathBuf>,
                     fs: &Arc<dyn Fs>| {
            if let Some(path) = id_to_path.get(id) {
                routes.push(LevelRoute {
                    levels: start..end_exclusive,
                    path: path.join(partition_name),
                    fs: Arc::clone(fs),
                });
            }
        };

        for lvl in 0..=MAX_ROUTED_LEVEL {
            let target = self.levels.get(&lvl).map(String::as_str).unwrap_or("");
            if target == primary_endpoint_id || target.is_empty() {
                // Close any open run.
                if let (Some(start), Some(id)) = (current_start, current_id) {
                    flush(&mut routes, start, lvl, id, &id_to_path, &fs);
                    current_start = None;
                    current_id = None;
                }
                continue;
            }
            match current_id {
                Some(id) if id == target => {
                    // Continue the run.
                }
                _ => {
                    // Close previous run (if any) and start new one.
                    if let (Some(start), Some(id)) = (current_start, current_id) {
                        flush(&mut routes, start, lvl, id, &id_to_path, &fs);
                    }
                    current_start = Some(lvl);
                    current_id = Some(target);
                }
            }
        }
        // Final flush.
        if let (Some(start), Some(id)) = (current_start, current_id) {
            flush(
                &mut routes,
                start,
                MAX_ROUTED_LEVEL + 1,
                id,
                &id_to_path,
                &fs,
            );
        }
        routes
    }

    /// Return all endpoint ids referenced by this routing — used for
    /// per-endpoint usage tracking and cascade-eviction targeting.
    pub fn endpoints_used(&self) -> std::collections::BTreeSet<&str> {
        self.levels.values().map(String::as_str).collect()
    }

    /// First level routed to the given endpoint, if any. Used by cascade
    /// eviction to identify the "topmost" level on a saturated endpoint.
    pub fn first_level_on(&self, endpoint_id: &str) -> Option<u8> {
        self.levels
            .iter()
            .find(|(_, id)| id.as_str() == endpoint_id)
            .map(|(lvl, _)| *lvl)
    }
}

/// Pick an endpoint matching the preferred tier chain. Falls back through
/// the rest of the canonical tier order (Hot → HotCache → Memory → Warm →
/// Cold) if none of the preferred tiers have a candidate. As a last
/// resort returns `pool[0]` — guaranteed non-empty by `StorageConfig`
/// validation. Emits a warn if the choice falls outside the preferred set.
fn pick_endpoint<'a>(
    pool: &'a [&EndpointConfig],
    preferred: &[Tier],
    band_label: &str,
) -> &'a EndpointConfig {
    debug_assert!(!pool.is_empty(), "endpoint pool is non-empty");

    for &t in preferred {
        if let Some(ep) = pool.iter().find(|e| e.tier == t) {
            return ep;
        }
    }
    // Walk the full tier order as fallback, in degradation-friendly sequence.
    let fallback_order = [
        Tier::Hot,
        Tier::HotCache,
        Tier::Memory,
        Tier::Warm,
        Tier::Cold,
    ];
    for &t in &fallback_order {
        if preferred.contains(&t) {
            continue;
        }
        if let Some(ep) = pool.iter().find(|e| e.tier == t) {
            tracing::warn!(
                band = band_label,
                preferred = ?preferred,
                chosen_tier = ?ep.tier,
                chosen_endpoint = %ep.id,
                "no endpoint matches preferred tier for level band; falling back to next-best tier",
            );
            return ep;
        }
    }
    // Pool non-empty per debug_assert; first element is last-resort.
    let last_resort = pool[0];
    tracing::warn!(
        band = band_label,
        preferred = ?preferred,
        chosen_tier = ?last_resort.tier,
        chosen_endpoint = %last_resort.id,
        "no endpoint matches any standard tier for band; using first endpoint as last resort",
    );
    last_resort
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
