//! State recovery for a reader following nodes by label: a fixed cut of the
//! nodes its interest selects, and the change cursor it reads on from.
//!
//! [`Database::node_cut`] pins a read view below a bound and binds the
//! interest's label names to the labels in force there. A page of the view
//! ([`Database::node_cut_scan`]) holds the nodes that carry one of those
//! labels in the cut's catalog: stored as the node's type or one of its own
//! labels, or implied by its type. The continuation
//! ([`Database::node_cut_tail`]) is read by the change feed itself, under the
//! same interest: the units the view does not hold, with their events and
//! how each reaches the interest (entering, applying to or leaving it), a
//! change of what a type implies included, under their own identities. Once
//! the join is complete ([`NodeCut::joined`]) the reader goes on with the
//! cursor it is handed, which follows the identities bound at the cut, not
//! whatever their names name later.
//!
//! The consumer publishes its replacement, the absence of every node the
//! view lacks and the cursor together, once the join is complete. A change
//! of what a type implies in the continuation moves nodes into or out of the
//! interest without a node event: the consumer reconciles the affected
//! scope from a new cut before it publishes, or reports that it needs one.

use std::time::Duration;

use coordinode_core::budget::{MemoryCharge, QueryBudget};
use coordinode_query::index::membership::{CatalogMembership, LabelCarriers, label_carriers_of};
use coordinode_storage::engine::change_feed::{ChangeCursor, Interest, UnitChanges};
use coordinode_storage::engine::fixed_cut::{CutHistory, CutRows, CutScope, FixedCut};
use coordinode_storage::engine::partition::Partition;

use super::{ChangeFollower, Database, DatabaseError};

/// The prefix every node record of a namespace is stored under.
const NODE_KEYS: &[u8] = b"node:";

/// A pinned read view of the nodes an interest selects below
/// [`Self::bound`], and the change cursor its continuation has reached.
/// Dropping it releases the view.
#[derive(Debug)]
pub struct NodeCut {
    cut: FixedCut,
    /// For each label of the interest, the nodes carrying it in the cut's
    /// catalog; `None` when the interest takes every node.
    carriers: Option<Vec<LabelCarriers>>,
    cursor: ChangeCursor,
}

impl NodeCut {
    /// The view holds every commit stamped below this bound and none at or
    /// above it.
    #[must_use]
    pub fn bound(&self) -> u64 {
        self.cut.bound()
    }

    /// The end of the join: the continuation is complete once read to here.
    #[must_use]
    pub fn join_end(&self) -> u64 {
        self.cut.join_end()
    }

    /// The history the cut is of.
    #[must_use]
    pub fn history(&self) -> CutHistory {
        self.cut.history()
    }

    /// The interest the cut selects nodes by and its continuation is read
    /// under: label identities bound at the cut.
    #[must_use]
    pub fn interest(&self) -> &Interest {
        self.cursor.interest()
    }

    /// Where the continuation's next read starts.
    #[must_use]
    pub fn cursor(&self) -> &ChangeCursor {
        &self.cursor
    }

    /// The change cursor the reader goes on with, once the join is complete;
    /// `None` before. Publishing the replacement earlier would certify a
    /// prefix the view already passed.
    #[must_use]
    pub fn joined(&self) -> Option<ChangeCursor> {
        (self.cursor.position() >= self.cut.join_end()).then(|| self.cursor.clone())
    }

    /// Whether a node storing `labels` is one the interest selects.
    fn selects(&self, labels: &[coordinode_core::schema::label_id::LabelId]) -> bool {
        self.carriers
            .as_ref()
            .is_none_or(|carriers| carriers.iter().any(|c| c.carried_by(labels)))
    }
}

/// One page of a node cut's view. Its rows stay charged to the budget the
/// page was read within until it is dropped.
#[derive(Debug)]
pub struct NodeCutPage<'b> {
    rows: CutRows,
    after: Option<Vec<u8>>,
    _charge: MemoryCharge<'b>,
}

impl NodeCutPage<'_> {
    /// The nodes the interest selects, in key order: their keys and records.
    #[must_use]
    pub fn rows(&self) -> &CutRows {
        &self.rows
    }

    /// Where the next page goes on from: the last node read, selected or
    /// not. `None` only once the view is read to its end; a page selecting
    /// no node is not the end.
    #[must_use]
    pub fn next_after(&self) -> Option<&[u8]> {
        self.after.as_deref()
    }
}

/// One read of a node cut's continuation. Its events stay charged to the
/// budget the read was made within until it is dropped.
#[derive(Debug)]
pub struct NodeCutBatch<'b> {
    units: Vec<UnitChanges>,
    joined: Option<ChangeCursor>,
    _charge: MemoryCharge<'b>,
}

impl NodeCutBatch<'_> {
    /// The units the view does not hold, in log order, each with the events
    /// the interest takes, as a reader of the feed receives them: a unit
    /// with none is there too, so the reader sees every unit it passed.
    #[must_use]
    pub fn units(&self) -> &[UnitChanges] {
        &self.units
    }

    /// See [`NodeCut::joined`], after this read.
    #[must_use]
    pub fn joined(&self) -> Option<&ChangeCursor> {
        self.joined.as_ref()
    }
}

impl Database {
    /// Take a fixed cut of the nodes carrying one of `labels` (every node,
    /// for `None`): a view pinned below a fresh bound, the names bound to the
    /// labels in force there, and the continuation that joins it. A reader
    /// that follows these labels already rebuilds with
    /// [`Self::node_cut_for`], which keeps the labels its cursor is bound to.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Semantic`] for a name no label is bound to;
    /// `CutNotReady` when a commit below the cut is still in flight or not
    /// durable after `timeout`; a catalog that cannot be read.
    pub fn node_cut(
        &self,
        labels: Option<&[String]>,
        timeout: Duration,
    ) -> Result<NodeCut, DatabaseError> {
        let cut = self.take_node_cut(timeout)?;
        let interest = coordinode_modality::change_interest::resolve_change_interest_at(
            &self.engine,
            cut.bound(),
            labels,
            Some(&[]),
        )
        .map_err(coordinode_query::executor::ExecutionError::from)?
        .map_err(|refusal| DatabaseError::Semantic(refusal.to_string()))?;
        self.node_cut_of(cut, interest)
    }

    /// Take a fixed cut for a reader that follows nodes by label with
    /// `cursor`: the same as [`Self::node_cut`], with the labels the cursor
    /// is bound to, whatever their names name now. A label dropped since
    /// selects no node of the view, and one created again under its name is
    /// not followed.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Semantic`] for a cursor following relationships or
    /// blob objects, which a cut of nodes does not rebuild; a cursor of
    /// another history or namespace; as [`Self::node_cut`].
    pub fn node_cut_for(
        &self,
        cursor: &ChangeCursor,
        timeout: Duration,
    ) -> Result<NodeCut, DatabaseError> {
        let interest = cursor.interest();
        if interest.edge_types() != Some(&[][..]) || interest.blobs() {
            return Err(DatabaseError::Semantic(
                "a cut of nodes rebuilds a reader that follows nodes only".into(),
            ));
        }
        if interest.namespace() != Some(self.engine.namespace()) {
            return Err(DatabaseError::Semantic(
                "the cursor follows another namespace".into(),
            ));
        }
        let cut = self.take_node_cut(timeout)?;
        if cut.history() != cursor.history() {
            return Err(DatabaseError::Semantic(
                "the cursor reads another history".into(),
            ));
        }
        self.node_cut_of(cut, interest.clone())
    }

    /// A fixed cut of this database's node records.
    fn take_node_cut(&self, timeout: Duration) -> Result<FixedCut, DatabaseError> {
        self.found_deployment()?;
        let scope = CutScope::rows(Partition::Node, NODE_KEYS.to_vec())?;
        Ok(self.engine.fixed_cut(scope, timeout)?)
    }

    /// The node cut of `cut` for `interest`: what carries each of its labels
    /// is read once from the cut's catalog, never per node.
    fn node_cut_of(&self, cut: FixedCut, interest: Interest) -> Result<NodeCut, DatabaseError> {
        let carriers = match interest.labels() {
            None => None,
            Some(labels) => {
                let mut catalog = CatalogMembership::at(&self.engine, cut.bound());
                let mut carriers = Vec::with_capacity(labels.len());
                for &label in labels {
                    carriers.push(
                        label_carriers_of(&self.engine, label, &mut catalog)
                            .map_err(coordinode_query::executor::ExecutionError::from)?,
                    );
                }
                Some(carriers)
            }
        };
        let cursor = self.engine.change_cursor_at(cut.first(), interest)?;
        Ok(NodeCut {
            cut,
            carriers,
            cursor,
        })
    }

    /// One page of the cut's view: at most `max_rows` node records read in
    /// key order, after `after` when given, within `budget`, of which the
    /// page holds those the cut's interest selects.
    ///
    /// # Errors
    ///
    /// `CutLost` for a cut of a database closed since, or whose node records
    /// were repaired; `Budget` for a first record that does not fit the
    /// budget on its own; a record whose labels cannot be read.
    pub fn node_cut_scan<'b>(
        &self,
        cut: &NodeCut,
        after: Option<&[u8]>,
        max_rows: usize,
        budget: &'b QueryBudget,
    ) -> Result<NodeCutPage<'b>, DatabaseError> {
        let (rows, after, charge) =
            self.engine
                .cut_page_where(&cut.cut, after, max_rows, budget, |_, record| {
                    if cut.carriers.is_none() {
                        return Ok(true);
                    }
                    let labels =
                        coordinode_core::graph::node::NodeRecord::labels_from_msgpack(record)
                            .map_err(|e| {
                                coordinode_storage::error::StorageError::Serialization(format!(
                                    "a node record's labels: {e}"
                                ))
                            })?;
                    Ok(cut.selects(&labels))
                })?;
        Ok(NodeCutPage {
            rows,
            after,
            _charge: charge,
        })
    }

    /// Read up to `max_entries` log entries of the cut's continuation, within
    /// `budget`, from where the last read ended, as [`Self::read_changes`]
    /// reads them under the cut's interest: the units the view already holds
    /// are left out, every other one is handed over with its events.
    ///
    /// # Errors
    ///
    /// As [`Self::read_changes`]: `RetentionLost` when the log no longer
    /// holds the next position, and the copy starts again from a new cut. On
    /// each of them the cut stays where it was.
    pub fn node_cut_tail<'b>(
        &self,
        cut: &mut NodeCut,
        max_entries: usize,
        budget: &'b QueryBudget,
        follower: Option<&ChangeFollower>,
    ) -> Result<NodeCutBatch<'b>, DatabaseError> {
        self.engine.held_cut(&cut.cut)?;
        let (batch, charge) = self
            .read_changes(&cut.cursor, max_entries, budget, follower)?
            .into_parts();
        let mut units = batch.units;
        units.retain(|unit| !cut.cut.in_view(unit.position, unit.commit_ts));
        cut.cursor = batch.cursor;
        Ok(NodeCutBatch {
            units,
            joined: cut.joined(),
            _charge: charge,
        })
    }
}
