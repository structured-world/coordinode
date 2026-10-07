//! Duplicate repair: what a unique index build with `ON DUPLICATE RENAME p`
//! does about a stored node whose value another node already holds.
//!
//! The repair is a write of user data like any other: one statement,
//! `SET n.p = <old value + random suffix>`, conditioned on the node still
//! holding the old value, run through the executor and committed through
//! the deployment's log at majority. Every index of the label, the
//! constraints and the derived structures follow it as they follow any
//! write. The record of the repair and the build's repair count commit in
//! the same transaction, conditioned on the build still being this
//! executor's: a cancellation or a takeover that lands first refuses it, and
//! one that lands after it finds the repair committed and keeps it.
//!
//! The new value is a claim like any other: a suffix another node already
//! holds is refused by the same admission and another suffix is drawn. A
//! repair that committed leaves no duplicate behind, so a retry or a resumed
//! build never renames the node again.

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_core::graph::node::{
    IdLease, IdLeaseError, IdLeaseReserver, NodeId, NodeIdAllocator,
};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_modality::{BuildState, DuplicateRepairRecord, IndexStore as _, LocalIndexStore};
use coordinode_storage::engine::transaction::Transaction;

use super::build::BackfillError;
use super::definition::{GenerationId, IndexDefinition};
use super::lifecycle::{BuildEnvironment, IndexBuildService};
use crate::executor::runner::{
    AdaptiveConfig, ExecutionContext, ExecutionError, WriteStats, execute_no_commit, wall_clock_us,
};

/// Suffixes drawn for one repair before it gives up: each one another node
/// already holds is a collision, which a random suffix of
/// [`SUFFIX_LEN`] base-36 digits makes rare.
const SUFFIX_ATTEMPTS: usize = 8;

/// Base-36 digits in a suffix: 36^8, about 2.8e12 values.
const SUFFIX_LEN: usize = 8;

/// How a repair attempt ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Repaired {
    /// The node holds a new value, and the repair is recorded.
    Done,
    /// The node no longer holds the value (a writer changed or deleted it
    /// meanwhile): nothing was written, and the page is read again.
    Changed,
}

/// The build a repair belongs to.
pub(crate) struct RepairBuild<'a> {
    /// The generation the build fills.
    pub generation: GenerationId,
    /// The token the build record holds while this executor runs it.
    pub token: u64,
    /// The index being built.
    pub index: &'a IndexDefinition,
    /// The property the build may change.
    pub property: &'a str,
}

/// Grants no lease: a repair changes a stored node and creates none, so an
/// attempt to create one fails instead of taking identifiers.
struct NoNewNodes;

impl IdLeaseReserver for NoNewNodes {
    fn reserve(&self) -> Result<IdLease, IdLeaseError> {
        Err(IdLeaseError::NotGranted(
            "a duplicate repair creates no node".into(),
        ))
    }
}

/// Give `node`, whose value `old` of the repaired property another node
/// holds, a value of its own.
///
/// # Errors
///
/// [`BackfillError::Superseded`] when the build is no longer this
/// executor's (cancelled, taken over); [`BackfillError::Repair`] when the
/// value is not a string, every suffix drawn was taken, or the statement
/// failed.
pub(crate) fn rename_duplicate(
    service: &IndexBuildService,
    env: &dyn BuildEnvironment,
    build: &RepairBuild<'_>,
    node: NodeId,
    old: Option<&Value>,
) -> Result<Repaired, BackfillError> {
    let Some(Value::String(old)) = old else {
        return Err(BackfillError::Repair(format!(
            "node {} holds {:?} in `{}`, which ON DUPLICATE RENAME cannot rename: only a string \
             value takes a suffix",
            node.to_element_id(),
            old.unwrap_or(&Value::Null),
            build.property
        )));
    };
    let statement = format!(
        "MATCH (n:`{label}`) WHERE id(n) = $node AND n.`{property}` = $old \
         SET n.`{property}` = $new RETURN id(n) AS node",
        label = build.index.label,
        property = build.property,
    );
    let query = crate::cypher::parse(&statement)
        .map_err(|e| BackfillError::Repair(format!("repair statement: {e}")))?;
    let template = crate::planner::build_logical_plan(&query)
        .map_err(|e| BackfillError::Repair(format!("repair plan: {e}")))?;
    let node_value =
        Value::Int(i64::try_from(node.as_raw()).map_err(|_| {
            BackfillError::Repair(format!("node id {} out of range", node.as_raw()))
        })?);

    for _ in 0..SUFFIX_ATTEMPTS {
        let new = format!("{old}_{}", suffix());
        let params = HashMap::from([
            ("node".to_string(), node_value.clone()),
            ("old".to_string(), Value::String(old.clone())),
            ("new".to_string(), Value::String(new.clone())),
        ]);
        let mut plan = template.clone();
        plan.substitute_params(&params);
        match attempt(service, env, build, &plan, params, node, old, &new)? {
            Attempt::Committed => return Ok(Repaired::Done),
            Attempt::Changed => return Ok(Repaired::Changed),
            Attempt::Taken => continue,
        }
    }
    Err(BackfillError::Repair(format!(
        "no free value of `{}` was found for node {} after {SUFFIX_ATTEMPTS} suffixes of {old:?}",
        build.property,
        node.to_element_id()
    )))
}

/// How one attempt with one suffix ended.
enum Attempt {
    Committed,
    Changed,
    /// Another node holds the new value: draw another suffix.
    Taken,
}

/// Run the repair statement with one suffix, record it, and commit.
#[allow(clippy::too_many_arguments)]
fn attempt(
    service: &IndexBuildService,
    env: &dyn BuildEnvironment,
    build: &RepairBuild<'_>,
    plan: &crate::planner::logical::LogicalPlan,
    params: HashMap<String, Value>,
    node: NodeId,
    old: &str,
    new: &str,
) -> Result<Attempt, BackfillError> {
    let engine = env.engine();
    let mut fields = env.fields().map_err(BackfillError::Repair)?;
    let allocator = NodeIdAllocator::leased(0, Arc::new(NoNewNodes));
    let oracle = env.oracle();
    let read_ts = oracle.map_or(coordinode_core::txn::timestamp::Timestamp::ZERO, |o| {
        o.next()
    });
    let txn = match oracle {
        Some(o) => Transaction::begin(engine, Some(o), read_ts),
        None => Transaction::new(engine, None, read_ts, None),
    };
    let log = env.statement_log();
    let mut ctx = ExecutionContext {
        engine,
        interner: &mut fields,
        field_registrar: None,
        id_allocator: &allocator,
        shard_id: env.shard_id(),
        scan_paging: None,
        operations: None,
        adaptive: AdaptiveConfig::default(),
        dedup_varlen_targets: false,
        snapshot_ts: None,
        valid_now: wall_clock_us(),
        temporal_instants: Vec::new(),
        snapshot_pin: None,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        key_claims: Default::default(),
        text_index: None,
        text_index_registry: env.text_registry(),
        vector_indexes: None,
        btree_index_registry: Some(env.registry()),
        index_builds: Some(service),
        extensions: None,
        vector_loader: None,
        mvcc_oracle: oracle,
        mvcc_read_ts: read_ts,
        procedures: None,
        advisor: None,
        txn,
        vector_consistency: plan.vector_consistency,
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: log.map(|(pipeline, _)| pipeline),
        proposal_id_gen: log.map(|(_, ids)| ids),
        read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::default(),
        // A build's writes are durable whatever a session writes at.
        write_concern: WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        mvcc_snapshot: None,
        cascade_depth: 0,
        cascade_depth_limit: 10,
        cascade_fire_counts: HashMap::new(),
        cascade_fanout_limit: 100,
        cascade_chain: Vec::new(),
        after_commit_generation: 0,
        correlated_row: None,
        foreach_scope: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        label_schema_cache: HashMap::new(),
        applied_watermark: None,
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        read_timeout: std::time::Duration::from_secs(2),
        params,
    };

    let rows = match execute_no_commit(plan, &mut ctx) {
        Ok(rows) => rows,
        Err(ExecutionError::UniqueViolation { .. }) => return Ok(Attempt::Taken),
        Err(e) => return Err(BackfillError::Repair(format!("repair statement: {e}"))),
    };
    if rows.is_empty() {
        return Ok(Attempt::Changed);
    }

    // The record and the count commit with the change, and only while the
    // build is still this executor's.
    let store = LocalIndexStore::new(engine);
    let Some((mut record, version)) = store.load_build(build.generation)? else {
        return Err(BackfillError::Superseded);
    };
    if record.state
        != (BuildState::Running {
            executor: build.token,
        })
    {
        return Err(BackfillError::Superseded);
    }
    record.repaired = record
        .repaired
        .checked_add(1)
        .ok_or_else(|| BackfillError::Repair("the repair count overflowed".into()))?;
    store.put_build_txn(&mut ctx.txn, &record, Some(version))?;
    store.put_repair_txn(
        &mut ctx.txn,
        &DuplicateRepairRecord {
            generation: build.generation,
            node: node.as_raw(),
            property: build.property.to_string(),
            old: old.to_string(),
            new: new.to_string(),
        },
    )?;

    match ctx.mvcc_flush() {
        Ok(_) => Ok(Attempt::Committed),
        Err(ExecutionError::UniqueViolation { .. }) => Ok(Attempt::Taken),
        Err(ExecutionError::Conflict(_)) => {
            // A cancellation or a takeover moved the build record; anything
            // else that conflicted changed the node, which the page reads
            // again.
            let still_ours = store.load_build(build.generation)?.is_some_and(|(r, _)| {
                r.state
                    == BuildState::Running {
                        executor: build.token,
                    }
            });
            if still_ours {
                Ok(Attempt::Changed)
            } else {
                Err(BackfillError::Superseded)
            }
        }
        Err(e) => Err(BackfillError::Repair(format!("repair commit: {e}"))),
    }
}

/// A random suffix of [`SUFFIX_LEN`] base-36 digits.
fn suffix() -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut n = rand::random::<u64>();
    let mut out = String::with_capacity(SUFFIX_LEN);
    for _ in 0..SUFFIX_LEN {
        out.push(char::from(DIGITS[(n % 36) as usize]));
        n /= 36;
    }
    out
}
