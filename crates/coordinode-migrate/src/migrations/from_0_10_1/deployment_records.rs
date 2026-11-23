//! The deployment's records move from the Schema tree to the Deployment
//! tree.
//!
//! 0.10.1 kept the records that hold for the whole deployment (its
//! identity, the node identifier leases, where each partition's levels
//! lie, the group's version pairs, operation incarnations, the Raft state
//! machine's applied position and membership, its snapshot and what the
//! member enrolled in) in the Schema tree among the namespace's schema.
//! This build reads them from the Deployment tree of the system set, and a
//! store missing them there starts over: a new identity, and node
//! identifiers issued again from the first. Each record moves with its
//! value as it is; where this build already wrote one of them on an earlier
//! open, the 0.10.1 record wins, since it is the one the store's data was
//! written under.
//!
//! The store opens for this, so the trees have moved and the journal's
//! units name their namespace before this runs: the run applies the
//! migrations in order.

use std::path::Path;

use anyhow::Context as _;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::scope::TreeId;

use crate::{Backup, Finding, Migration, stores};

pub(super) const MIGRATION: Migration = Migration {
    name: "deployment-records",
    summary: "the deployment's records move from the Schema tree to the Deployment tree",
    survey,
    apply,
};

/// The deployment's records 0.10.1 kept in the Schema tree: each a key, or
/// a prefix when it ends in ':'.
const RECORDS: &[&[u8]] = &[
    coordinode_core::group::DEPLOYMENT_KEY,
    coordinode_core::group::ENROLLED_DEPLOYMENT_KEY,
    b"raft:enrolled_authorities",
    coordinode_core::graph::node::NODE_LEASE_KEY_PREFIX,
    b"meta:routing:",
    coordinode_core::version::GROUP_PAIR_KEY_PREFIX,
    coordinode_core::txn::operation::INCARNATION_KEY_PREFIX,
    b"raft:sm:",
    b"raft:snapshot:",
];

/// A store 0.10.1 wrote has no Deployment tree: this build makes it on the
/// first open, which this migration's own run is.
fn survey(data: &Path) -> anyhow::Result<Vec<Finding>> {
    let mut findings = Vec::new();
    for store in stores(data)? {
        if !store
            .join(TreeId::system(Partition::Deployment).dir())
            .exists()
        {
            findings.push(Finding {
                path: store,
                detail: "deployment records in the Schema tree".into(),
            });
        }
    }
    Ok(findings)
}

/// Open the store at `dir` as this build opens a database: its journal
/// replayed, and a store holding no catalog given its first namespace, the
/// one its data moved into.
fn open(dir: &Path) -> anyhow::Result<StorageEngine> {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open_with_oracle(&config, std::sync::Arc::new(TimestampOracle::new()))
        .with_context(|| format!("open the store {}", dir.display()))
}

fn apply(finding: &Finding, _backup: &Backup) -> anyhow::Result<()> {
    let engine = open(&finding.path)?;
    let mut moved = Vec::new();
    for &record in RECORDS {
        let exact = !record.ends_with(b":");
        for guard in engine
            .prefix_scan(Partition::Schema, record)
            .with_context(|| format!("scan {}", finding.path.display()))?
        {
            let (key, value) = guard
                .into_inner()
                .with_context(|| format!("scan {}", finding.path.display()))?;
            if exact && key.as_ref() != record {
                continue;
            }
            moved.push((key.to_vec(), value.to_vec()));
        }
    }
    for (key, value) in &moved {
        let printable = coordinode_storage::error::printable_key(key);
        engine
            .put(Partition::Deployment, key, value)
            .with_context(|| format!("write {printable} in {}", finding.path.display()))?;
        engine
            .delete(Partition::Schema, key)
            .with_context(|| format!("remove {printable} in {}", finding.path.display()))?;
    }
    engine
        .persist()
        .with_context(|| format!("flush {}", finding.path.display()))
}
