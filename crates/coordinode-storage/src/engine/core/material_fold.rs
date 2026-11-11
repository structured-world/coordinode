//! Rounds of the material fold ([`crate::engine::material::fold`]).
//!
//! A round folds every accounted tree's units below one exclusive target
//! (the complete cut plus one) and then publishes the round. Each tree keeps
//! its own cursor (`0x00 "ac"`), written in the same batch as the
//! descriptors and summary that batch folded, so a crash anywhere resumes
//! where that tree's last batch left it and folds no unit twice. The round
//! (`0x00 "ag"` in the schema tree) names the target and the seqno of the
//! last write of the round: an observation reads every tree's summary at a
//! snapshot just above it, so writes of a round still running are not seen
//! and the trees are read at one cut. The observation pins that snapshot
//! for its read; one no longer readable gives no observation, never another
//! round's buckets.

use coordinode_core::budget::QueryBudget;
use lsm_tree::{AbstractTree, SeqNo};

use super::StorageEngine;
use crate::engine::footprint::Admit;
use crate::engine::material::fold::{Fold, RangePage, Tomb};
use crate::engine::material::{self, UnitKind};
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// Each tree's cursor.
pub(super) const CURSOR_KEY: [u8; 3] = [0x00, b'a', b'c'];
/// The published round, in the schema tree.
const ROUND_KEY: [u8; 3] = [0x00, b'a', b'g'];
/// Units read from the index, and keys of a range deletion folded, per
/// batch.
const PAGE: usize = 256;
/// Commits after which the commit path runs a round.
pub(super) const FOLD_STRIDE: u32 = 32;
/// Units (and range-deleted keys) the commit path folds per round at most; a
/// round left short is resumed by the next.
const COMMIT_BUDGET: usize = 4 * PAGE;
/// Units a catch-up before an observation, or at open, folds at most.
const CATCH_UP_BUDGET: usize = 64 * PAGE;
/// Memory a round holds at most: the units it reads and what it folds before
/// a batch is written, each reserved before it is taken.
const FOLD_MEMORY: u64 = 48 << 20;

/// The runs a round goes on through: as many as a first level holds before
/// writes to it are slowed down (the default limit), and one per deeper
/// level. Past them compaction is already pressed to merge.
pub(super) const FOLD_RUNS: usize = 40 + 6;

// The fold's share of a round's memory takes the longest unit and a range
// page over the longest keys, read through a tree of `FOLD_RUNS` runs, so a
// round goes on.
const _: () = assert!(
    FOLD_MEMORY - FOLD_MEMORY / 4 >= crate::engine::material::fold::least_memory(FOLD_RUNS)
);

#[cfg(test)]
std::thread_local! {
    /// Where a test refuses the fold's batches room, as the footprint does.
    pub(super) static REFUSE_ROOM: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    /// Where a test refuses the retirement of folded units room.
    pub(super) static REFUSE_RETIRE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    /// Where a test stops a fold write after its snapshot records, before
    /// its batch: what a crash between the two leaves.
    pub(super) static STOP_AFTER_MARKS: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// What a round may still spend: units (and range-deleted keys), the memory
/// of what it folds, and the memory of each page of units it reads (a
/// quarter of the round's, so a page read never takes the room its units
/// are folded in).
struct Work<'b> {
    left: usize,
    memory: &'b QueryBudget,
    page_memory: u64,
}

/// Held by the one fold running: rounds and clears of a tree's fold state
/// take turns on it.
#[derive(Default)]
pub(super) struct Folder {
    /// The trees whose folded unit records are still to retire, a bit per
    /// partition in [`Partition::all`] order: a retirement refused for room
    /// is retried by the next round, whatever it folds, and an open retries
    /// every tree's.
    retire_due: u64,
}

impl Folder {
    fn bit(part: Partition) -> u64 {
        Partition::all()
            .iter()
            .position(|p| *p == part)
            .map_or(0, |at| 1 << at)
    }
}

/// How far a fold got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Progress {
    /// Everything below the target is folded (and, where asked, the round
    /// published).
    Reached,
    /// The budget ran out first: the next round resumes.
    Short,
    /// A tree is under repair: no round is published until it is done.
    Repairing,
}

/// A published round: units below `through` are folded, and the round's
/// last write is at `at`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Round {
    pub(crate) through: SeqNo,
    pub(crate) at: SeqNo,
}

/// Where a tree's fold stands: the next unit to fold, inside a range
/// deletion the descriptor key its next page starts from, and the bytes a
/// fixed snapshot of the tree needs after everything folded so far.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Cursor {
    seqno: SeqNo,
    ordinal: u32,
    resume: Option<Vec<u8>>,
    visible: u64,
}

/// The record of the bytes a fixed snapshot of a tree needs, written at each
/// commit's seqno: a read at snapshot `s` finds the bytes after every commit
/// below `s`.
pub(super) const VISIBLE_KEY: [u8; 3] = [0x00, b'a', b'V'];

impl Cursor {
    /// This cursor moved to `(seqno, ordinal)`, its bytes kept.
    fn at(&self, seqno: SeqNo, ordinal: u32, resume: Option<Vec<u8>>) -> Self {
        Self {
            seqno,
            ordinal,
            resume,
            visible: self.visible,
        }
    }

    /// The commit whose units are folded in part: a later unit completes it.
    fn open(&self) -> Option<SeqNo> {
        (self.ordinal > 0 || self.resume.is_some()).then_some(self.seqno)
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(21 + self.resume.as_ref().map_or(0, Vec::len));
        out.extend_from_slice(&self.seqno.to_le_bytes());
        out.extend_from_slice(&self.ordinal.to_le_bytes());
        out.extend_from_slice(&self.visible.to_le_bytes());
        match &self.resume {
            Some(k) => {
                out.push(1);
                out.extend_from_slice(k);
            }
            None => out.push(0),
        }
        out
    }

    fn decode(value: &[u8]) -> StorageResult<Self> {
        let bad = || {
            StorageError::Serialization(format!("material fold cursor holds {} bytes", value.len()))
        };
        let seqno = value
            .get(..8)
            .and_then(|s| <[u8; 8]>::try_from(s).ok())
            .map(u64::from_le_bytes)
            .ok_or_else(bad)?;
        let ordinal = value
            .get(8..12)
            .and_then(|s| <[u8; 4]>::try_from(s).ok())
            .map(u32::from_le_bytes)
            .ok_or_else(bad)?;
        let visible = value
            .get(12..20)
            .and_then(|s| <[u8; 8]>::try_from(s).ok())
            .map(u64::from_le_bytes)
            .ok_or_else(bad)?;
        let resume = match value.get(20) {
            Some(0) if value.len() == 21 => None,
            Some(1) => Some(value[21..].to_vec()),
            _ => return Err(bad()),
        };
        Ok(Self {
            seqno,
            ordinal,
            resume,
            visible,
        })
    }
}

impl StorageEngine {
    /// The published fold round, if any.
    ///
    /// # Errors
    ///
    /// The record could not be read or is malformed.
    pub(crate) fn material_round(&self) -> StorageResult<Option<Round>> {
        let tree = self.tree(Partition::Schema)?;
        let Some(value) = tree.get(ROUND_KEY.as_slice(), SeqNo::MAX)? else {
            return Ok(None);
        };
        let v = <[u8; 16]>::try_from(value.as_ref()).map_err(|_| {
            StorageError::Serialization(format!("material fold round holds {} bytes", value.len()))
        })?;
        let mut through = [0u8; 8];
        let mut at = [0u8; 8];
        through.copy_from_slice(&v[..8]);
        at.copy_from_slice(&v[8..]);
        Ok(Some(Round {
            through: u64::from_le_bytes(through),
            at: u64::from_le_bytes(at),
        }))
    }

    /// Withdraw the published round: a tree's fold state was cleared, and no
    /// round covers it until the fold has gone through what is put back.
    ///
    /// # Errors
    ///
    /// The schema tree is unavailable.
    pub(super) fn forget_material_round(&self) -> StorageResult<()> {
        let schema = self.tree(Partition::Schema)?;
        schema.remove(ROUND_KEY.as_slice(), self.next_seqno());
        Ok(())
    }

    /// The exclusive bound every unit below which is complete: the complete
    /// cut is the last commit included, so the bound is one past it.
    fn fold_target(&self) -> Option<SeqNo> {
        self.complete_cut().and_then(|cut| cut.checked_add(1))
    }

    /// After a commit: count it, and every [`FOLD_STRIDE`] commits run a
    /// bounded round unless one is running.
    pub(super) fn fold_after_commit(&self) {
        use core::sync::atomic::Ordering;
        if self.material_fold_due.fetch_add(1, Ordering::AcqRel) + 1 < FOLD_STRIDE {
            return;
        }
        let Some(mut folder) = self.material_fold.try_lock() else {
            return;
        };
        self.material_fold_due.store(0, Ordering::Release);
        // Folded, not published: a round is published where an observation
        // needs it, so commits write only the trees they reached.
        if let Some(target) = self.fold_target()
            && let Err(e) = self.fold_round(&mut folder, target, COMMIT_BUDGET, false)
        {
            tracing::warn!(error = %e, "material fold round failed; the next one resumes it");
        }
    }

    /// Fold every unit below the complete cut, without publishing a round:
    /// the backlog an open left, folded before the store is handed out.
    ///
    /// # Errors
    ///
    /// A record could not be read or written, or the fold has no room.
    pub(super) fn fold_material_at_open(&self) -> StorageResult<()> {
        let mut folder = self.material_fold.lock();
        // A retirement refused before the store closed is not remembered:
        // every tree's is retried, each found due by its records.
        folder.retire_due = u64::MAX;
        if let Some(target) = self.fold_target() {
            // Bounded like any round: a longer backlog is resumed by the
            // commits and observations that follow.
            self.fold_round(&mut folder, target, CATCH_UP_BUDGET, false)?;
        }
        Ok(())
    }

    /// Fold every unit below the complete cut and publish the round: what a
    /// decision on the material a consumer requires reads first, so it is
    /// taken on the accounting at the cut it is taken at.
    ///
    /// # Errors
    ///
    /// A record could not be read or written, or the fold has no room.
    /// Returns whether the round reached the cut. A backlog past one bounded
    /// round, or a tree under repair, leaves it short: what was folded is
    /// kept and the next call goes on from there.
    pub fn fold_material_to_cut(&self) -> StorageResult<bool> {
        let mut folder = self.material_fold.lock();
        let Some(target) = self.fold_target() else {
            return Ok(true);
        };
        Ok(self.fold_round(&mut folder, target, CATCH_UP_BUDGET, true)? == Progress::Reached)
    }

    /// Fold up to `budget` units toward `target` and, once every tree
    /// reached it, publish the round when `publish` says so.
    pub(super) fn fold_round(
        &self,
        folder: &mut Folder,
        target: SeqNo,
        budget: usize,
        publish: bool,
    ) -> StorageResult<Progress> {
        self.fold_round_within(folder, target, budget, FOLD_MEMORY, publish)
    }

    /// [`Self::fold_round`] holding at most `memory` bytes at a time.
    pub(super) fn fold_round_within(
        &self,
        folder: &mut Folder,
        target: SeqNo,
        budget: usize,
        memory: u64,
        publish: bool,
    ) -> StorageResult<Progress> {
        if self.material_round()?.is_some_and(|r| r.through >= target) {
            // Nothing to fold; retirements still due are retried.
            self.retire_due(folder)?;
            return Ok(Progress::Reached);
        }
        let page_memory = memory / 4;
        let budget_memory = QueryBudget::new(memory - page_memory);
        let mut work = Work {
            left: budget,
            memory: &budget_memory,
            page_memory,
        };
        let mut repairing = false;
        for &part in Partition::all() {
            if !material::accounted(part) {
                continue;
            }
            match self.fold_tree(folder, part, target, &mut work)? {
                Progress::Reached => {}
                Progress::Short => return Ok(Progress::Short),
                Progress::Repairing => repairing = true,
            }
        }
        if repairing {
            return Ok(Progress::Repairing);
        }
        if !publish {
            return Ok(Progress::Reached);
        }
        let at = self.next_seqno();
        let mut value = [0u8; 16];
        value[..8].copy_from_slice(&target.to_le_bytes());
        value[8..].copy_from_slice(&at.to_le_bytes());
        let schema = self.tree(Partition::Schema)?;
        let mut batch = lsm_tree::WriteBatch::with_capacity(1);
        batch.insert(ROUND_KEY.as_slice(), value.as_slice());
        // With the flush owed for a memtable it starts.
        let _admitted = self.admit_footprint(
            crate::engine::footprint::footprint_of((ROUND_KEY.len() + 16) as u64, 1)
                + self.memtable_starts([Partition::Schema]),
            Admit::Recovery,
        )?;
        let (added, memtable) = schema.apply_batch(batch, at)?;
        self.coordinator.flush_trigger().wrote(added, memtable);
        Ok(Progress::Reached)
    }

    /// Retry the retirements `folder` holds due, each tree's below the
    /// window its cursor stands in.
    fn retire_due(&self, folder: &mut Folder) -> StorageResult<()> {
        if folder.retire_due == 0 {
            return Ok(());
        }
        for &part in Partition::all() {
            if !material::accounted(part) || folder.retire_due & Folder::bit(part) == 0 {
                continue;
            }
            let tree = self.tree(part)?;
            let cursor = match tree.get(CURSOR_KEY.as_slice(), SeqNo::MAX)? {
                Some(v) => Cursor::decode(&v)?,
                None => Cursor::default(),
            };
            self.retire_folded_units(folder, part, cursor.seqno)?;
        }
        Ok(())
    }

    /// Drop `part`'s unit records below the window `to` stands in: their
    /// effects are in the descriptors, and a rebuild reads the versions, not
    /// these records. The material records stay: they date the work
    /// consumers still owe. Without the room for the deletion, the tree is
    /// left due in `folder` and the next round retries it.
    fn retire_folded_units(
        &self,
        folder: &mut Folder,
        part: Partition,
        to: SeqNo,
    ) -> StorageResult<()> {
        let below = to - to % material::RETIRE_WINDOW;
        let tree = self.tree(part)?;
        if below == 0
            || material::units_from(tree, (0, 0), below, SeqNo::MAX, 1, None)?
                .0
                .is_empty()
        {
            folder.retire_due &= !Folder::bit(part);
            return Ok(());
        }
        // One range deletion, and the flush owed for a memtable it starts.
        #[cfg(test)]
        let admitted = if REFUSE_RETIRE.with(core::cell::Cell::get) {
            Err(StorageError::CapacityExhausted {
                endpoint_id: "test".to_string(),
                used_bytes: 0,
                hard_limit_bytes: 0,
            })
        } else {
            self.admit_footprint(
                crate::engine::footprint::footprint_of(material::RETIRE_RECORD_BYTES, 1)
                    + self.memtable_starts([part]),
                Admit::Recovery,
            )
        };
        #[cfg(not(test))]
        let admitted = self.admit_footprint(
            crate::engine::footprint::footprint_of(material::RETIRE_RECORD_BYTES, 1)
                + self.memtable_starts([part]),
            Admit::Recovery,
        );
        if let Err(StorageError::CapacityExhausted { .. }) = admitted {
            folder.retire_due |= Folder::bit(part);
            return Ok(());
        }
        let _admitted = admitted?;
        material::retire_units_below(tree, below, self.next_seqno());
        self.coordinator.flush_trigger().wrote_unmeasured();
        folder.retire_due &= !Folder::bit(part);
        Ok(())
    }

    /// Fold `part`'s units below `target`, spending `left`.
    fn fold_tree(
        &self,
        folder: &mut Folder,
        part: Partition,
        target: SeqNo,
        work: &mut Work<'_>,
    ) -> StorageResult<Progress> {
        // A tree under repair is still receiving units at their own commit
        // timestamps, below any cursor the fold would reach now.
        if self
            .under_repair
            .load(core::sync::atomic::Ordering::Acquire)
            & super::repair_bit(part)
            != 0
        {
            return Ok(Progress::Repairing);
        }
        let tree = self.tree(part)?;
        let mut cursor = match tree.get(CURSOR_KEY.as_slice(), SeqNo::MAX)? {
            Some(v) => Cursor::decode(&v)?,
            None => Cursor::default(),
        };
        let from = cursor.seqno;
        let progress = self.fold_tree_from(part, &mut cursor, target, work)?;
        // A cursor that stayed in its window has nothing new to retire,
        // unless a retirement is still due.
        let window = material::RETIRE_WINDOW;
        if cursor.seqno - cursor.seqno % window > from - from % window
            || folder.retire_due & Folder::bit(part) != 0
        {
            self.retire_folded_units(folder, part, cursor.seqno)?;
        }
        Ok(progress)
    }

    /// [`Self::fold_tree`] from `cursor`, left where it got to.
    fn fold_tree_from(
        &self,
        part: Partition,
        cursor: &mut Cursor,
        target: SeqNo,
        work: &mut Work<'_>,
    ) -> StorageResult<Progress> {
        let tree = self.tree(part)?;
        while cursor.seqno < target {
            if work.left == 0 {
                return Ok(Progress::Short);
            }
            let page = PAGE.min(work.left);
            let page_budget = QueryBudget::new(work.page_memory);
            let mut read = page_budget.empty_charge();
            // `squeezed`: fewer than asked for, without the memory to read
            // more; more may follow.
            let (units, squeezed) = material::units_from(
                tree,
                (cursor.seqno, cursor.ordinal),
                target,
                SeqNo::MAX,
                page,
                Some(&mut read),
            )?;
            if units.is_empty() {
                if squeezed {
                    return Ok(Progress::Short);
                }
                // Nothing below the target is left to fold. A commit folded
                // in part is now complete and its bytes are written; a
                // cursor left where it is otherwise reads the same nothing
                // next time, so it takes no write.
                if cursor.open().is_some() {
                    let mut fold = Fold::metered(tree, work.memory).after(cursor.open());
                    let done = cursor.at(target, 0, None);
                    // Without the memory for its mark and its write the
                    // commit stays open, completed by a later round.
                    if !fold.close() || !self.write_fold(part, fold, cursor, done)? {
                        return Ok(Progress::Short);
                    }
                }
                break;
            }
            let full = units.len() == page || squeezed;
            let mut fold = Fold::metered(tree, work.memory).after(cursor.open());
            // A range deletion spends the budget on its keys: what follows it
            // in this page is read again from the cursor, against what is
            // left. A unit without the memory to fold it ends the round.
            let mut reread = false;
            let mut out_of_room = false;
            for unit in units {
                if unit.kind == UnitKind::Range {
                    // At most `left` units were read.
                    work.left -= 1;
                    // What is folded so far goes first: the range reads the
                    // descriptors from the tree. A commit before the range's
                    // is whole by now.
                    if !fold.close_before(unit.seqno) {
                        return Ok(Progress::Short);
                    }
                    let folded = core::mem::replace(&mut fold, Fold::new(tree));
                    // A fold that took in nothing leaves the cursor where it
                    // is, at the range, with the key a page resumes from.
                    if !folded.is_empty() {
                        // Something was folded before the range in this page,
                        // so no page of it was folded yet: nothing to resume.
                        debug_assert!(cursor.resume.is_none());
                        let before = cursor.at(unit.seqno, unit.ordinal, None);
                        if !self.write_fold(part, folded, cursor, before)? {
                            return Ok(Progress::Short);
                        }
                    }
                    if self.fold_range(part, &unit, cursor, work)? == Progress::Short {
                        // The range's next page is in the cursor.
                        return Ok(Progress::Short);
                    }
                    reread = true;
                    break;
                }
                if !fold.unit(unit.seqno, &unit.key, unit.kind, unit.bytes)? {
                    out_of_room = true;
                    break;
                }
                work.left -= 1;
                cursor.seqno = unit.seqno;
                cursor.ordinal = unit.ordinal + 1;
                cursor.resume = None;
            }
            if reread {
                continue;
            }
            // Nothing left below the target: the last commit is whole, when
            // there is the memory for its mark.
            if !(full || out_of_room) && !fold.close() {
                out_of_room = true;
            }
            let next = if full || out_of_room {
                cursor.clone()
            } else {
                cursor.at(target, 0, None)
            };
            drop(read);
            if !self.write_fold(part, fold, cursor, next)? || out_of_room || squeezed {
                return Ok(Progress::Short);
            }
        }
        Ok(Progress::Reached)
    }

    /// Fold the range deletion `unit` page by page, each page with the
    /// cursor at it, from where `at` left it. Returns the cursor past it.
    /// Spends `left` a key at a time: a range over more keys than the budget
    /// stops with its next key in the cursor and returns
    /// [`Progress::Short`].
    fn fold_range(
        &self,
        part: Partition,
        unit: &material::Unit,
        cursor: &mut Cursor,
        work: &mut Work<'_>,
    ) -> StorageResult<Progress> {
        let tree = self.tree(part)?;
        let tomb = Tomb {
            seqno: unit.seqno,
            ordinal: unit.ordinal,
        };
        loop {
            if work.left == 0 {
                return Ok(Progress::Short);
            }
            let page = PAGE.min(work.left);
            let mut fold = Fold::metered(tree, work.memory).after(cursor.open());
            let ended = fold.range_page(
                tomb,
                &unit.key,
                &unit.end,
                unit.bytes,
                cursor.resume.as_deref(),
                page,
            )?;
            let (next, walked) = match ended {
                RangePage::Done(walked) => (cursor.at(unit.seqno, unit.ordinal + 1, None), walked),
                RangePage::From(k, walked) => (
                    cursor.at(unit.seqno, unit.ordinal, Some(k.to_vec())),
                    walked,
                ),
                // Nothing taken in, nothing to write: the round ends with
                // the range where it was.
                RangePage::NoRoom => return Ok(Progress::Short),
            };
            // At most `page <= left` keys were walked.
            work.left -= walked;
            let done = next.resume.is_none();
            if !self.write_fold(part, fold, cursor, next)? {
                return Ok(Progress::Short);
            }
            if done {
                return Ok(Progress::Reached);
            }
            // A page cut short of its keys by memory: the round ends there.
            if walked < page {
                return Ok(Progress::Short);
            }
        }
    }

    /// Write what `fold` folded from `cursor` and the cursor `next` after it
    /// as one batch of `part`'s tree, admitted on the footprint as the
    /// store's own work, and move `cursor` there.
    ///
    /// Before it, the bytes a fixed snapshot needs after each commit the
    /// fold completed are written at that commit's seqno: they are the same
    /// whenever this batch is written again from the same cursor, so a
    /// crash between them and the batch rewrites them unchanged.
    ///
    /// `false`, writing nothing and leaving `cursor`, when the fold's memory
    /// does not hold what writing it takes: a round goes no further then.
    fn write_fold(
        &self,
        part: Partition,
        mut fold: Fold<'_>,
        cursor: &mut Cursor,
        mut next: Cursor,
    ) -> StorageResult<bool> {
        if !fold.ready() {
            return Ok(false);
        }
        let tree = self.tree(part)?;
        // Built within the fold's memory, cursor record and marks included,
        // and held by it until applied.
        let written = fold.write(cursor.visible, CURSOR_KEY.as_slice(), |visible| {
            next.visible = visible;
            next.encode()
        })?;
        let (batch, marks) = (written.batch, written.marks);
        let bytes = written.bytes + marks_len(marks.len()) as u64;
        // Admitted before the first write, with the flush owed for a memtable
        // it starts: a refusal leaves the tree, the cursor and the snapshot
        // records as they were.
        #[cfg(test)]
        if REFUSE_ROOM.with(core::cell::Cell::get) {
            return Err(StorageError::CapacityExhausted {
                endpoint_id: "test".to_string(),
                used_bytes: 0,
                hard_limit_bytes: 0,
            });
        }
        let _admitted = self.admit_footprint(
            crate::engine::footprint::footprint_of(bytes, batch.len() + marks.len())
                + self.memtable_starts([part]),
            Admit::Recovery,
        )?;
        for &(commit, total) in &marks {
            // At the commit's own seqno: a snapshot sees what lies below it,
            // so one at `s` finds the bytes after the last commit below `s`.
            tree.insert(
                VISIBLE_KEY.as_slice(),
                total.to_le_bytes().as_slice(),
                commit,
            );
        }
        #[cfg(test)]
        if STOP_AFTER_MARKS.with(core::cell::Cell::get) {
            return Err(StorageError::Io(
                "stopped after the snapshot records".into(),
            ));
        }
        let (added, memtable) = tree.apply_batch(batch, self.next_seqno())?;
        drop(written.held);
        self.coordinator.flush_trigger().wrote(added, memtable);
        *cursor = next;
        Ok(true)
    }

    /// What a fixed snapshot at `snapshot` needs: the closures of the keys
    /// with a value there, over every accounted tree, as one observation of
    /// the published round. Its work is dated by its registration, not here
    /// (`first` is none). Unavailable while a tree is repaired, before the
    /// round has folded everything below the snapshot, or once the store no
    /// longer serves reads there.
    ///
    /// # Errors
    ///
    /// A record or the rebuild intents could not be read.
    pub fn observe_snapshot_material(
        &self,
        snapshot: SeqNo,
    ) -> StorageResult<crate::engine::material::Observation> {
        use crate::engine::material::{Material, Observation};
        use core::sync::atomic::Ordering;
        let generation = self.material_generation.load(Ordering::Acquire);
        if self.repairing_material()
            || self
                .pending_rebuilds()?
                .into_iter()
                .any(material::accounted)
        {
            return Ok(Observation::Rebuilding);
        }
        let Ok(_held) = self.pin_readable_snapshot_at(snapshot) else {
            return Ok(Observation::Lost);
        };
        let Some(round) = self.material_round()? else {
            return Ok(Observation::NoCut);
        };
        let Some(bytes) = self.snapshot_material(snapshot)? else {
            return Ok(Observation::NoCut);
        };
        if self.repairing_material()
            || self.material_generation.load(Ordering::Acquire) != generation
        {
            return Ok(Observation::Rebuilding);
        }
        Ok(Observation::Observed {
            through: round.through.saturating_sub(1),
            generation,
            material: Material { first: None, bytes },
        })
    }

    /// The bytes a fixed snapshot at `snapshot` needs of every accounted
    /// tree, when the published round has folded everything below it.
    ///
    /// # Errors
    ///
    /// A record could not be read or is malformed.
    pub(crate) fn snapshot_material(&self, snapshot: SeqNo) -> StorageResult<Option<u64>> {
        if !self
            .material_round()?
            .is_some_and(|round| round.through >= snapshot)
        {
            return Ok(None);
        }
        let mut total = 0u64;
        for &part in Partition::all() {
            if !material::accounted(part) {
                continue;
            }
            let tree = self.tree(part)?;
            let bytes = match tree.get(VISIBLE_KEY.as_slice(), snapshot)? {
                Some(v) => <[u8; 8]>::try_from(v.as_ref())
                    .map(u64::from_le_bytes)
                    .map_err(|_| {
                        StorageError::Serialization(format!(
                            "material snapshot record holds {} bytes",
                            v.len()
                        ))
                    })?,
                None => 0,
            };
            total = total.checked_add(bytes).ok_or_else(|| {
                StorageError::Serialization("material snapshot bytes overflow u64".to_string())
            })?;
        }
        Ok(Some(total))
    }
}

/// Bytes of `marks` snapshot records.
fn marks_len(marks: usize) -> usize {
    marks * (VISIBLE_KEY.len() + 8)
}
