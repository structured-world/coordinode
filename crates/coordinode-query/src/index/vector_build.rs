//! Building HNSW vector indexes beside live writes.
//!
//! One procedure serves every place an index is built: a `CREATE VECTOR
//! INDEX`, the rebuild when a database opens, and a replica bringing up an
//! index another member defined. Each has live writes landing while it runs,
//! from local statements or from replicated applies, and each must end with
//! every vector of the label in the graph. Several indexes over one shard
//! share a build: one tap, one scan, one fold.

use std::sync::RwLock;
use std::time::Duration;

use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::try_extract_vector;
use coordinode_modality::{LocalNodeStore, NodeStore as _};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::tap::Tapped;
use coordinode_vector::health::HealthSignal;
use coordinode_vector::hnsw::HnswIndex;

use crate::index::BuildToken;

/// How long the build waits, between folds, for the transactions opened
/// before the writers took over.
const OLDER_TRANSACTIONS_SLICE: Duration = Duration::from_millis(50);

/// How often that wait checks whether they have ended.
const OLDER_TRANSACTIONS_POLL: Duration = Duration::from_millis(2);

/// Nodes folded per hold of the graphs' write locks, bounding how long a fold
/// keeps searches waiting.
const TAP_FOLD_CHUNK: usize = 1024;

/// Nodes scanned between two progress reports and cancellation checks.
const PROGRESS_INTERVAL: u64 = 1000;

/// How a build ended: it finished, or it was asked to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildOutcome {
    /// Every index holds every vector of its label; `scanned` of them came
    /// from the scan, the rest from the writes folded in after it.
    Complete {
        /// Vectors the (last) scan inserted, over all the indexes.
        scanned: u64,
    },
    /// Stopped on a cancellation. The graphs keep whatever was inserted:
    /// HNSW insert is an upsert, so a later build re-covers the same nodes
    /// without duplicating them.
    Cancelled,
}

/// One index a build fills.
pub struct BuildTarget<'a> {
    /// The graph being built.
    pub hnsw: &'a RwLock<HnswIndex>,
    /// Where progress, readiness and freshness are published.
    pub health: &'a HealthSignal,
    /// The indexed label.
    pub label: &'a str,
    /// The interned id of the indexed property.
    pub field_id: u32,
}

impl BuildTarget<'_> {
    /// The vector `record` contributes to this index, if it is a member.
    fn member(&self, record: &NodeRecord) -> Option<Vec<f32>> {
        if record.primary_label() != self.label {
            return None;
        }
        record
            .props
            .get(&self.field_id)
            .and_then(try_extract_vector)
    }
}

/// A build of one or more indexes over one shard.
pub struct VectorBuild<'a> {
    /// The store the labels live in.
    pub engine: &'a StorageEngine,
    /// Checked between batches; a cancelled build writes nothing further.
    pub token: &'a BuildToken,
    /// The shard whose node rows are indexed.
    pub shard_id: u16,
    /// The indexes filled.
    pub targets: &'a [BuildTarget<'a>],
}

impl VectorBuild<'_> {
    /// Build the indexes beside live writes: scan, fold what the write tap
    /// delivered, hand maintenance to the writers, wait out the transactions
    /// that did not see the handover, fold again.
    ///
    /// Writes are selected by where they landed, not by timestamp: an entry
    /// applies at its own commit timestamp, which can sit below any snapshot
    /// the build took before it (a commit that reserved its timestamp early,
    /// a leader's entry reaching this node after a later one). The tap opened
    /// before the scan delivers every write that lands after it, and the
    /// scan's snapshot holds every one that landed before.
    ///
    /// While an index rebuilds, writers leave it to the build, which inserts
    /// the scan in batches far cheaper per vector than a write at a time.
    /// Once the scan is in, the indexes are marked ready and writers insert
    /// for themselves; the build folds what landed during the scan, and a
    /// transaction opened before the handover may still leave its vector to
    /// the build, so the tap stays open until every transaction opened at or
    /// before the handover's boundary has ended (the wait PostgreSQL's
    /// concurrent index build makes for older snapshots), and one more fold
    /// takes what they wrote. Nothing waits for the tap to run dry, which
    /// under writes that never pause it would not. The writers are never
    /// paused, and the freshness watermark is published only at the end.
    ///
    /// Blocks until done, and on a Raft store pauses the applies for an
    /// instant at the start: call it off the async runtime, and never from a
    /// thread whose own transaction is open (it would wait for itself).
    ///
    /// # Errors
    ///
    /// A storage failure, or a build snapshot already outside retention.
    pub fn run(&self) -> Result<BuildOutcome, String> {
        let (tap, at) = LocalNodeStore
            .tap_writes(self.engine)
            .map_err(|e| format!("open the write tap: {e}"))?;
        tracing::debug!(
            snapshot = at,
            indexes = self.targets.len(),
            "vector build: tap open"
        );
        let mut outcome = self.scan(at)?;
        tracing::debug!(?outcome, "vector build: scanned");
        if outcome == BuildOutcome::Cancelled {
            return Ok(outcome);
        }
        let mut boundary = self.hand_over();
        // Set once the transactions opened before the handover have ended:
        // the snapshot every write at or below which is in the graphs after
        // one more fold.
        let mut settled: Option<u64> = None;
        loop {
            if outcome == BuildOutcome::Cancelled || self.token.is_cancelled() {
                return Ok(BuildOutcome::Cancelled);
            }
            // Fold what the tap holds. It never has to run dry: under writes
            // that do not pause it would not, and the writers insert for
            // themselves.
            match tap.take() {
                // The partition was cleared or range-deleted: what it lost is
                // not listed, so start over from a fresh snapshot.
                Tapped::Replaced => {
                    let at = self
                        .engine
                        .rebase_tap(&tap)
                        .map_err(|e| format!("rebase the write tap: {e}"))?;
                    outcome = self.scan(at)?;
                    // The scan put the indexes back to rebuilding, so the
                    // writers left what they wrote meanwhile to the build:
                    // hand over again, and wait for the transactions opened
                    // before this handover instead.
                    if outcome != BuildOutcome::Cancelled {
                        boundary = self.hand_over();
                    }
                    settled = None;
                    continue;
                }
                Tapped::Keys(keys) => {
                    if !keys.is_empty() {
                        tracing::debug!(keys = keys.len(), "vector build: folding tapped writes");
                        self.fold(&keys)?;
                    }
                }
            }
            // The fold after the older transactions ended took everything
            // they wrote: whatever lands from here on, its writer inserts.
            if let Some(fresh) = settled {
                drop(tap);
                for target in self.targets {
                    target.health.advance_indexed_hlc(fresh);
                }
                return Ok(outcome);
            }
            if self
                .engine
                .await_transactions_through(
                    boundary,
                    0,
                    OLDER_TRANSACTIONS_POLL,
                    OLDER_TRANSACTIONS_SLICE,
                )
                .is_ok()
            {
                settled = Some(self.engine.snapshot());
            }
        }
    }

    /// Hand the scanned graphs to the writers, who insert what they write
    /// from here on, and return the boundary past which a transaction does
    /// so for itself.
    ///
    /// What landed during the scan, and what the transactions opened before
    /// this point leave, stays with the build through the tap. Handing over
    /// later would not make the index more complete (writes keep landing) but
    /// would leave every write in between to the build's folds, which under
    /// steady load grow with each other. This is the only point a build marks
    /// its indexes ready: once it has handed over, the indexes may already
    /// belong to a newer build, which a later mark would overrule.
    fn hand_over(&self) -> u64 {
        for target in self.targets {
            target.health.mark_ready();
        }
        let boundary = self.engine.snapshot_boundary();
        tracing::debug!(boundary, "vector build: handed over to the writers");
        boundary
    }

    /// Scan the shard at `snapshot` and insert every member of every index.
    ///
    /// Progress is published to each index's [`HealthSignal`], never to the
    /// persisted definition: the definition key belongs to DDL, and a
    /// background writer on it turns any later statement touching that index
    /// into a write conflict against a write the statement could not have
    /// seen. The denominator is the partition's approximate item count, so
    /// the fraction is an under-estimate that snaps to 1.0 at the end: honest
    /// for a progress bar, never a row count.
    fn scan(&self, snapshot: u64) -> Result<BuildOutcome, String> {
        let _pin = self
            .engine
            .pin_snapshot_at(snapshot)
            .ok_or_else(|| format!("the build snapshot {snapshot} is already outside retention"))?;
        let started = std::time::Instant::now();
        // Upper bound over every version and tombstone the levels hold, for
        // the whole partition rather than these labels: deliberately
        // generous, since a progress bar that overshoots and stalls at 100%
        // reads as a hang.
        let estimated_total = LocalNodeStore
            .approximate_count(self.engine)
            .unwrap_or(0)
            .max(1) as f64;
        let mut written = 0u64;
        let mut since_report = 0u64;
        let mut scanned = 0u64;
        let mut cancelled = false;
        LocalNodeStore
            .for_each_in_shard_at_snapshot(
                self.engine,
                Some(snapshot),
                self.shard_id,
                &mut |node_id, _key, record| {
                    scanned += 1;
                    for target in self.targets {
                        if let Some(vec_data) = target.member(record) {
                            super::vector_registry::insert_one(
                                target.hnsw,
                                node_id.as_raw(),
                                &vec_data,
                            );
                            written += 1;
                        }
                    }
                    since_report += 1;
                    if since_report >= PROGRESS_INTERVAL {
                        since_report = 0;
                        if self.token.is_cancelled() {
                            cancelled = true;
                            return Ok(std::ops::ControlFlow::Break(()));
                        }
                        let fraction = ((scanned as f64) / estimated_total).min(0.99) as f32;
                        let elapsed_ms = started.elapsed().as_millis() as u64;
                        let eta_ms = if fraction > 0.0 {
                            ((elapsed_ms as f64) * ((1.0 - fraction as f64) / fraction as f64))
                                as u64
                        } else {
                            0
                        };
                        for target in self.targets {
                            target.health.report_rebuild_progress(fraction, eta_ms);
                        }
                    }
                    Ok(std::ops::ControlFlow::Continue(()))
                },
            )
            .map_err(|e| format!("shard scan: {e}"))?;
        if cancelled || self.token.is_cancelled() {
            return Ok(BuildOutcome::Cancelled);
        }
        Ok(BuildOutcome::Complete { scanned: written })
    }

    /// Fold the nodes behind `keys`, Node-partition keys the write tap
    /// delivered, into the graphs.
    ///
    /// Each node is re-read and reconciled rather than inserted as written:
    /// membership can be LOST while the node lives (its vector property
    /// removed, its label changed), and that is decidable from the current
    /// record alone. A node that is still a member is upserted; one that is
    /// not is removed from the graph, which repairs the links around it and
    /// later reuses its slot.
    ///
    /// The read and the insert happen under the graphs' write locks. A writer
    /// that inserts its own vector after its record landed is ordered by the
    /// same lock; one that inserted before (an interactive statement, ahead
    /// of its commit) is delivered again by the tap when the commit lands,
    /// and that later fold reads the committed record.
    fn fold(&self, keys: &[Vec<u8>]) -> Result<(), String> {
        // Other shards' rows and the per-version keys of temporal nodes
        // decode to nothing these indexes hold.
        let ids: Vec<NodeId> = keys
            .iter()
            .filter_map(|key| coordinode_core::graph::node::decode_node_key(key))
            .filter(|(shard, _)| *shard == self.shard_id)
            .map(|(_, id)| id)
            .collect();
        for chunk in ids.chunks(TAP_FOLD_CHUNK) {
            let mut graphs = self
                .targets
                .iter()
                .map(|t| t.hnsw.write())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "vector index lock poisoned".to_string())?;
            let read_txn = coordinode_storage::engine::transaction::Transaction::new(
                self.engine,
                None,
                coordinode_core::txn::timestamp::Timestamp::ZERO,
                None,
            );
            let records = LocalNodeStore
                .get_many(&read_txn, self.shard_id, chunk)
                .map_err(|e| format!("fold read: {e}"))?;
            let mut batches: Vec<Vec<(u64, Vec<f32>)>> = vec![Vec::new(); self.targets.len()];
            for (node_id, record) in chunk.iter().zip(records) {
                for ((target, graph), batch) in self.targets.iter().zip(&graphs).zip(&mut batches) {
                    match record.as_ref().and_then(|r| target.member(r)) {
                        Some(vec_data) => batch.push((node_id.as_raw(), vec_data)),
                        // Deleted, relabelled or stripped of its vector: out of
                        // the graph, if the graph holds it.
                        None => {
                            graph.remove(node_id.as_raw());
                        }
                    }
                }
            }
            for (graph, batch) in graphs.iter_mut().zip(batches) {
                if !batch.is_empty() {
                    graph.insert_batch(batch);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
