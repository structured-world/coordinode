//! What a transfer of the index partition carries about this member's
//! index copies, and how a receiving member takes it.
//!
//! The index copies a snapshot or a partition copy moves are not proved to
//! agree with their records by moving: a checksum proves the bytes, a log
//! position the entries applied, neither the index's logical content. A
//! transfer therefore carries an inventory of the copies it holds, each
//! qualified (with the evidence of a check or build that proved it),
//! excluded (found disagreeing) or unknown. The receiver keeps the evidence
//! of the qualified copies and checks the others before it uses them.
//!
//! The engine only carries the inventory and keeps one node-local record:
//! that an index transfer is being installed. What a copy is, how it is
//! proved and how a receiver marks one belong to the index layer, which
//! registers an [`IndexCopyQualifier`] with the engine.

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// Set while a transfer replaces the index partition, until the receiver
/// has taken its inventory. Node-local (under `meta:`): a crash in between
/// leaves it, and the next open takes every copy as unknown.
pub const INDEX_TRANSFER_KEY: &[u8] = b"meta:idxtransfer";

/// The index layer's side of a transfer of index copies. The inventory is
/// opaque to the engine, which stores, carries and hands it back.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot qualify transferred index copies",
    label = "not an index copy qualifier",
    note = "the index layer registers one with `StorageEngine::set_index_copy_qualifier`"
)]
pub trait IndexCopyQualifier: Send + Sync {
    /// At a capture of `engine`, taken while no entry applies: the inventory
    /// of its index copies as they stand, with what this member found
    /// against them that is not stored (a finding whose mark could not be
    /// written) and the revision of every finding.
    ///
    /// # Errors
    ///
    /// The catalog or the records of the copies could not be read.
    fn capture(&self, engine: &StorageEngine) -> StorageResult<Vec<u8>>;

    /// The inventory `captured` (from [`Self::capture`]) as a transfer
    /// sends it, sealed against what `engine` found since the capture: a
    /// copy found disagreeing after it is excluded.
    ///
    /// # Errors
    ///
    /// `captured` does not decode.
    fn seal(&self, engine: &StorageEngine, captured: &[u8]) -> StorageResult<Vec<u8>>;

    /// A transfer replaced `engine`'s index partition with copies whose
    /// inventory is `inventory` (`None`: the transfer carries none): keep the
    /// evidence of the copies it qualifies, rebound to this member's
    /// installations, take every other copy as unfit until checked, and
    /// clear [`INDEX_TRANSFER_KEY`], durably, in that order.
    ///
    /// # Errors
    ///
    /// The records could not be written: the key stays, and the next open
    /// takes every copy as unknown.
    fn installed(&self, engine: &StorageEngine, inventory: Option<&[u8]>) -> StorageResult<()>;

    /// The proof this member keeps of its copy of `generation`, in the form
    /// every member that proved the same generation encodes it alike, with
    /// an epoch that moves on every finding against the copy and every new
    /// proof of it; `None` when the copy is not proved or was found
    /// disagreeing. A history export reads it before and after the export
    /// and carries it only when the two agree, so a finding made and lifted
    /// while it read is not missed.
    ///
    /// # Errors
    ///
    /// The catalog or the records of the copy could not be read.
    fn copy_proof(
        &self,
        engine: &StorageEngine,
        generation: u64,
    ) -> StorageResult<Option<(Vec<u8>, u64)>>;
}

impl StorageEngine {
    /// Register the index layer's handling of transferred index copies.
    pub fn set_index_copy_qualifier(&self, qualifier: std::sync::Arc<dyn IndexCopyQualifier>) {
        *self.index_copy_qualifier.write() = Some(qualifier);
    }

    /// The registered handling of transferred index copies, if any.
    pub fn index_copy_qualifier(&self) -> Option<std::sync::Arc<dyn IndexCopyQualifier>> {
        self.index_copy_qualifier.read().clone()
    }

    /// Record, durably, that a transfer is about to replace the index
    /// partition: until [`Self::index_transfer_installed`] takes its
    /// inventory, no copy in it counts as qualified.
    ///
    /// # Errors
    ///
    /// The write or its flush failed.
    pub fn begin_index_transfer(&self) -> StorageResult<()> {
        // Durable before the transfer replaces anything: a crash midway must
        // find the record and treat every copy as unqualified.
        self.commit_local(
            vec![crate::engine::core::LocalWrite::Put {
                partition: Partition::Schema,
                key: INDEX_TRANSFER_KEY.to_vec(),
                value: Vec::new(),
            }],
            crate::engine::core::LocalSync::BeforeUse,
        )
    }

    /// Whether an index transfer was begun and its inventory not taken.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn index_transfer_pending(&self) -> StorageResult<bool> {
        Ok(self.get(Partition::Schema, INDEX_TRANSFER_KEY)?.is_some())
    }

    /// The transfer begun with [`Self::begin_index_transfer`] installed its
    /// copies, with `inventory` as it carried it: the registered qualifier
    /// takes it. Without one the record stays, and whoever opens the index
    /// layer next takes every copy as unknown.
    ///
    /// # Errors
    ///
    /// The qualifier's errors.
    pub fn index_transfer_installed(&self, inventory: Option<&[u8]>) -> StorageResult<()> {
        match self.index_copy_qualifier() {
            Some(qualifier) => qualifier.installed(self, inventory),
            None => Ok(()),
        }
    }

    /// The inventory of this engine's index copies at a capture taken now,
    /// or `None` without a registered qualifier.
    ///
    /// # Errors
    ///
    /// The qualifier's errors.
    pub fn capture_index_inventory(&self) -> StorageResult<Option<Vec<u8>>> {
        self.index_copy_qualifier()
            .map(|qualifier| qualifier.capture(self))
            .transpose()
    }

    /// The proof of this member's copy of `generation`, with the revision of
    /// its latest finding; `None` without one or without a registered
    /// qualifier.
    ///
    /// # Errors
    ///
    /// The qualifier's errors.
    pub fn index_copy_proof(&self, generation: u64) -> StorageResult<Option<(Vec<u8>, u64)>> {
        match self.index_copy_qualifier() {
            Some(qualifier) => qualifier.copy_proof(self, generation),
            None => Ok(None),
        }
    }

    /// `captured`, an inventory taken at a capture of this engine, sealed
    /// against what this member found since; `None` without a registered
    /// qualifier, which leaves every copy unknown to the receiver.
    ///
    /// # Errors
    ///
    /// The qualifier's errors.
    pub fn seal_index_inventory(&self, captured: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        self.index_copy_qualifier()
            .map(|qualifier| qualifier.seal(self, captured))
            .transpose()
    }
}
