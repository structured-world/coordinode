//! Room a consensus member holds for its leader's client entries before they
//! arrive.
//!
//! A follower that met its hard limit while appending a client entry would
//! leave that entry unacknowledged at the head of the log, and the group's
//! own work behind it could never commit. Instead the member holds a pool of
//! client capacity admitted ahead (a grant) and reports it to the leader,
//! which appends a client entry only while a quorum's grants cover it with
//! every entry still in flight. An entry the leader sent is then paid from
//! the pool, never refused for want of room.
//!
//! The pool is a real admission on the footprints: other writes cannot take
//! it. Payment follows the log, never the messages or their waiters: a
//! request's client entries are set aside from the pool when it arrives,
//! and each becomes paid when the log appends it, by its index and term. A
//! request's waiter going away proves nothing (the consensus core may still
//! append what it queued), so what was set aside goes back only on what the
//! log shows: another term's entry appended at that index, or the log
//! applied past it. A retransmission sets nothing aside twice; a truncation
//! gives back exactly what the cut suffix paid; an applied entry's room
//! goes back.
//!
//! What the member reported to its leader survives a restart. A report of
//! room `A` after log index `L` lets the leader send client entries past
//! `L` worth up to `A`. Counted in client bytes of the log (`cum(i)`: client
//! bytes through index `i`), that report reaches `cum(L) + A`: the member
//! owes room for every client byte its log may hold up to that point. The
//! coordinate is the group's own: the applied part comes from the state
//! machine, which records the total of the client bytes it applied with
//! the entries themselves, so it is the same on every member and a snapshot
//! carries it; the unapplied entries this member's log holds add theirs.
//! The stored horizon is the furthest any report reached, raised and
//! written synced before a report reaching further is sent, off the path
//! that answers the leader: until it is stored, reports reach no further
//! than the stored one. A restart owes the horizon less `cum` of the log it
//! opens with, whatever entries were set aside, appended, applied,
//! snapshotted, purged or truncated before.

use std::collections::BTreeMap;

use super::{Admit, Admitted, Footprints};
use crate::error::StorageError;

/// The room a member keeps granted unless configured otherwise.
pub const DEFAULT_CLIENT_GRANT_BYTES: u64 = 4 << 20;

/// The smallest growth of a grant asked for while room is short.
const GRANT_STEP: u64 = 64 << 10;

/// What a member has promised its leader, as stored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GrantRecord {
    /// The furthest any report reached, in client bytes of the group's log:
    /// those through the reported index plus the room reported.
    pub horizon: u64,
    /// The member's openings, this one included.
    pub epoch: u64,
}

/// The room one client entry holds, and whose it is: the grant's (it goes
/// back to the pool when the entry gives it up) or the recovery reserve's
/// (it goes back to the footprint).
#[derive(Debug)]
struct Held {
    room: Admitted,
    granted: bool,
}

/// A member's pool of granted room and what client entries hold of it.
#[derive(Debug)]
pub(super) struct Grants {
    /// How much room the pool keeps free when it can.
    target: u64,
    /// Granted and not set aside for any entry.
    free: Admitted,
    /// Set aside for entries a request carried, by index and term, until
    /// the log appends them or shows they never will be.
    staged: BTreeMap<(u64, u64), Held>,
    /// What each appended, unapplied client entry paid, by index, with the
    /// term that entry belongs to.
    paid: BTreeMap<u64, (u64, Held)>,
    /// The highest index applied: an entry at or below it is paid for by
    /// nothing, it already landed.
    applied: u64,
    /// The client entries of the log past `counted`, by index: their term
    /// and client bytes. As many as the log holds unapplied.
    logged: BTreeMap<u64, (u64, u64)>,
    /// The sum of `logged`.
    logged_bytes: u64,
    /// The last index the state machine's total counts, and that total.
    counted: (u64, u64),
    /// What is stored of the promise.
    record: GrantRecord,
    /// A horizon being stored: no report reaches past the stored one until
    /// it is.
    raising: bool,
    /// Further horizons stored by this process.
    #[cfg(any(test, feature = "test-faults"))]
    stores: u64,
    /// Reports made by this process: each takes the next, under the lock
    /// that reads the room it reports.
    seq: u64,
}

impl Default for Grants {
    fn default() -> Self {
        Self {
            target: DEFAULT_CLIENT_GRANT_BYTES,
            free: Admitted::default(),
            staged: BTreeMap::new(),
            paid: BTreeMap::new(),
            applied: 0,
            logged: BTreeMap::new(),
            logged_bytes: 0,
            counted: (0, 0),
            record: GrantRecord::default(),
            raising: false,
            #[cfg(any(test, feature = "test-faults"))]
            stores: 0,
            seq: 0,
        }
    }
}

impl Grants {
    /// The log is applied through `index`, and the state machine's total of
    /// applied client bytes is `total` through `through`: what the applied
    /// entries held goes back, what was set aside for entries at or below it
    /// that the log never appended returns to the pool, and the entries the
    /// total counts leave `logged`.
    pub(super) fn applied(&mut self, index: u64, through: u64, total: u64) {
        self.applied = self.applied.max(index);
        if through >= self.counted.0 {
            self.counted = (through, total);
            let kept = self.logged.split_off(&(through + 1));
            for (_, (_, bytes)) in std::mem::replace(&mut self.logged, kept) {
                self.logged_bytes -= bytes;
            }
        }
        while let Some(entry) = self.paid.first_entry() {
            if *entry.key() > index {
                break;
            }
            // Its drop gives the room back: the entry landed.
            drop(entry.remove());
        }
        while let Some(entry) = self.staged.first_entry() {
            if entry.key().0 > index {
                break;
            }
            let held = entry.remove();
            self.release(held);
        }
    }

    /// An entry gave up the room it held without landing: the grant's goes
    /// back to the pool, the recovery reserve's to the footprint.
    fn release(&mut self, held: Held) {
        if held.granted {
            self.free.absorb(held.room);
        }
    }

    /// Give what the appended entries from `index` on paid back, and take
    /// them out of the log's client bytes.
    fn refund_from(&mut self, index: u64) {
        for (_, (_, held)) in self.paid.split_off(&index) {
            self.release(held);
        }
        for (_, (_, bytes)) in self.logged.split_off(&index) {
            // Each was added to the sum when it was logged.
            self.logged_bytes -= bytes;
        }
    }

    /// The log holds `bytes` client bytes at `index` of `term` (none for an
    /// entry of the group's own work).
    fn log(&mut self, index: u64, term: u64, bytes: u64) {
        if index <= self.counted.0 {
            return;
        }
        if self
            .logged
            .get(&index)
            .is_some_and(|&(t, b)| t == term && b == bytes)
        {
            return;
        }
        if let Some((_, old)) = self.logged.remove(&index) {
            self.logged_bytes -= old;
        }
        if bytes > 0 {
            self.logged.insert(index, (term, bytes));
            // Bounded by the log's size.
            self.logged_bytes += bytes;
        }
    }

    /// Client bytes through the log's last entry.
    fn cum(&self) -> u64 {
        // Both bounded by the bytes ever logged.
        self.counted.1 + self.logged_bytes
    }
}

/// A client entry a member takes from its leader: its position in the log
/// and the room it is admitted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrantedEntry {
    /// Its index in the log.
    pub index: u64,
    /// The term of the leader that appended it.
    pub term: u64,
    /// The room it takes until it is applied.
    pub bytes: u64,
}

/// A report of a member's grant, as sent to its leader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrantReport {
    /// The room held for client entries after `last_index`.
    pub available: u64,
    /// The log index the room is held after.
    pub last_index: u64,
    /// The member's openings, this one included.
    pub epoch: u64,
    /// The report's place among this process's reports.
    pub seq: u64,
}

impl Footprints {
    /// Set aside room from the granted pool for `entries`, all or none: each
    /// once by its index and term, and nothing for an entry already applied,
    /// appended with the same term or set aside already.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] when the pool holds less than the
    /// new entries need: the leader sent more than this member granted.
    pub fn stage_granted(&self, entries: &[GrantedEntry]) -> Result<(), StorageError> {
        if self.is_empty() {
            return Ok(());
        }
        let mut grants = self.grants.lock();
        let new: Vec<&GrantedEntry> = entries
            .iter()
            .filter(|entry| {
                entry.index > grants.applied
                    && grants.paid.get(&entry.index).map(|(t, _)| *t) != Some(entry.term)
                    && !grants.staged.contains_key(&(entry.index, entry.term))
            })
            .collect();
        // Bounded by the size of the message the entries came in.
        let need: u64 = new.iter().map(|entry| entry.bytes).sum();
        if grants.free.bytes() < need {
            return Err(self.endpoints.first().map_or_else(
                || StorageError::Io("granted room exhausted".to_string()),
                |footprint| StorageError::CapacityExhausted {
                    endpoint_id: footprint.id().to_string(),
                    used_bytes: footprint.committed(),
                    hard_limit_bytes: footprint.limit(),
                },
            ));
        }
        for entry in new {
            // Checked against the pool above, under the same lock.
            let Some(room) = grants.free.split_off(entry.bytes) else {
                debug_assert!(false, "the pool held what the entries need");
                break;
            };
            grants.staged.insert(
                (entry.index, entry.term),
                Held {
                    room,
                    granted: true,
                },
            );
        }
        Ok(())
    }

    /// Set aside room for committed client `entries`, each once by its
    /// index and term like [`Self::stage_granted`]: from the grant while it
    /// holds what the entries not yet paid need, otherwise all of it from
    /// the recovery reserve, admitted here after `recovery_ok` passed (a
    /// committed entry is history the group holds; refusing it would only
    /// leave this member behind). Entries applied, appended or set aside
    /// already take nothing, so a repeat of the request takes nothing again.
    ///
    /// # Errors
    ///
    /// What `recovery_ok` returned, or [`StorageError::CapacityExhausted`]
    /// when the recovery reserve cannot hold the entries.
    pub fn stage_settled(
        &self,
        entries: &[GrantedEntry],
        recovery_ok: impl FnOnce() -> Result<(), StorageError>,
    ) -> Result<(), StorageError> {
        if self.is_empty() {
            return Ok(());
        }
        // The check may read the filesystem: it runs without the pool's
        // lock, and what to hold is decided again once it passed, from the
        // pool as it then is (entries may have landed, the grant refilled).
        let mut recovery_ok = Some(recovery_ok);
        loop {
            let mut grants = self.grants.lock();
            let new: Vec<&GrantedEntry> = entries
                .iter()
                .filter(|entry| {
                    entry.index > grants.applied
                        && grants.paid.get(&entry.index).map(|(t, _)| *t) != Some(entry.term)
                        && !grants.staged.contains_key(&(entry.index, entry.term))
                })
                .collect();
            // Bounded by the size of the message the entries came in.
            let need: u64 = new.iter().map(|entry| entry.bytes).sum();
            if need == 0 {
                return Ok(());
            }
            let granted = grants.free.bytes() >= need;
            let source = if granted {
                None
            } else if let Some(check) = recovery_ok.take() {
                drop(grants);
                check()?;
                continue;
            } else {
                Some(self.admit(need, Admit::Recovery)?)
            };
            Self::hold_settled(&mut grants, new, source, granted);
            return Ok(());
        }
    }

    /// Set aside room for `new` entries from `source` (the grant's free room
    /// when `None`), under the pool's lock.
    fn hold_settled(
        grants: &mut Grants,
        new: Vec<&GrantedEntry>,
        mut source: Option<Admitted>,
        granted: bool,
    ) {
        for entry in new {
            // Held for exactly what the entries need, under the same lock.
            let room = match &mut source {
                None => grants.free.split_off(entry.bytes),
                Some(reserve) => reserve.split_off(entry.bytes),
            };
            let Some(room) = room else {
                debug_assert!(false, "the room held what the entries need");
                break;
            };
            grants
                .staged
                .insert((entry.index, entry.term), Held { room, granted });
        }
    }

    /// The log appended the entry at `index` of `term`: what was set aside
    /// for it is paid. What was set aside for an entry the log can no longer
    /// take returns to the pool: another term's at that index, and any of an
    /// earlier term (terms never decrease along a Raft log). An entry of
    /// another term still paid there was cut from the log first and gives
    /// its room back. `bytes` are the entry's client bytes, none for the
    /// group's own work: every entry the log takes is counted, set aside
    /// before or not.
    pub fn granted_appended(&self, index: u64, term: u64, bytes: u64) {
        if self.is_empty() {
            return;
        }
        let mut grants = self.grants.lock();
        grants.log(index, term, bytes);
        // Bounded by the entries of the requests in flight.
        let dead: Vec<(u64, u64)> = grants
            .staged
            .keys()
            .filter(|key| key.1 < term || (key.0 == index && key.1 != term))
            .copied()
            .collect();
        for key in dead {
            if let Some(held) = grants.staged.remove(&key) {
                grants.release(held);
            }
        }
        let Some(held) = grants.staged.remove(&(index, term)) else {
            return;
        };
        if let Some((_, replaced)) = grants.paid.insert(index, (term, held)) {
            grants.release(replaced);
        }
    }

    /// The log was truncated: its entries from `index` on are gone, and what
    /// they paid goes back to the pool; the log holds their client bytes no
    /// more. What is set aside stays: the entries replacing them are
    /// appended next.
    pub fn granted_truncated(&self, index: u64) {
        if self.is_empty() {
            return;
        }
        self.grants.lock().refund_from(index);
    }

    /// How much room the pool keeps free when it can. A smaller target does
    /// not take back room already granted: the leader may count on it; the
    /// pool shrinks as entries pay from it.
    pub fn set_grant_target(&self, bytes: u64) {
        self.grants.lock().target = bytes;
    }

    /// The stored promise.
    #[must_use]
    pub fn grant_record(&self) -> GrantRecord {
        self.grants.lock().record
    }

    /// Whether a further horizon is being stored: until it is, nothing else
    /// can raise it. For tests.
    #[cfg(any(test, feature = "test-faults"))]
    #[must_use]
    pub fn grant_raising(&self) -> bool {
        self.grants.lock().raising
    }

    /// What the pool holds for client entries: granted and free, and set
    /// aside or paid for entries not yet applied. For tests.
    #[cfg(any(test, feature = "test-faults"))]
    #[must_use]
    pub fn granted_held(&self) -> (u64, u64) {
        let grants = self.grants.lock();
        let entries = grants
            .staged
            .values()
            .chain(grants.paid.values().map(|(_, held)| held))
            .map(|held| held.room.bytes())
            .sum();
        (grants.free.bytes(), entries)
    }

    /// Refill the pool in batches: nothing while it holds its target or
    /// more; once below it, one admission of client capacity up to twice the
    /// target, halving what it asks for while the footprints refuse, so a
    /// short endpoint grants what fits. Entries then spend the batch down to
    /// the target before the next one: the stored horizon, which only a
    /// refill moves, is raised once per batch rather than once per entry.
    /// The pool never takes the recovery reserve; a target of 0 holds none.
    pub fn top_up_grant(&self) {
        if self.is_empty() {
            return;
        }
        let mut grants = self.grants.lock();
        let free = grants.free.bytes();
        if free >= grants.target {
            return;
        }
        // Capped by the rule itself: a target past half the address space
        // asks for all of it, which the admission below cuts to what fits.
        let mut want = grants.target.saturating_mul(2) - free;
        while want > 0 {
            match self.admit(want, Admit::Client) {
                Ok(room) => {
                    grants.free.absorb(room);
                    return;
                }
                // No smaller ask than a step: room that short is no grant.
                Err(_) if want <= GRANT_STEP => return,
                Err(_) => want /= 2,
            }
        }
    }

    /// Start opening the pool with `record` (this opening's, stored), the
    /// log applied through `applied` and the state machine's total of
    /// applied client bytes `total` through `through`. The client entries the
    /// log holds past `applied` follow through [`Self::recover_granted`], a
    /// bounded batch at a time, and [`Self::recover_granted_owed`] ends the
    /// opening.
    pub fn recover_granted_start(
        &self,
        record: GrantRecord,
        applied: u64,
        through: u64,
        total: u64,
    ) {
        if self.is_empty() {
            return;
        }
        let mut grants = self.grants.lock();
        grants.record = record;
        grants.applied = grants.applied.max(applied);
        grants.counted = (through, total);
    }

    /// Client entries the log holds past its applied index, in log order:
    /// each holds again, as the store's own recovery, the room it takes
    /// until applied, before anything else is admitted.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] when the endpoint cannot hold them.
    pub fn recover_granted(&self, entries: &[GrantedEntry]) -> Result<(), StorageError> {
        if self.is_empty() {
            return Ok(());
        }
        let mut grants = self.grants.lock();
        for entry in entries {
            grants.log(entry.index, entry.term, entry.bytes);
            if entry.index <= grants.applied || grants.paid.contains_key(&entry.index) {
                continue;
            }
            let room = self.admit(entry.bytes, Admit::Recovery)?;
            grants.paid.insert(
                entry.index,
                (
                    entry.term,
                    Held {
                        room,
                        granted: false,
                    },
                ),
            );
        }
        Ok(())
    }

    /// End the opening: hold again, as the store's own recovery, what the
    /// stored horizon reaches past the log, which the leader may still send.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] when the endpoint cannot hold it.
    pub fn recover_granted_owed(&self) -> Result<(), StorageError> {
        if self.is_empty() {
            return Ok(());
        }
        let mut grants = self.grants.lock();
        // Floored at zero by the rule itself: a horizon the log already
        // passed owes nothing more.
        let owed = grants.record.horizon.saturating_sub(grants.cum());
        if owed > 0 {
            let room = self.admit(owed, Admit::Recovery)?;
            grants.free.absorb(room);
        }
        Ok(())
    }

    /// A report of the grant after `last_index` (read by the caller before
    /// this, so entries appended since were paid from the room reported):
    /// the room granted and not set aside, with this process's epoch and the
    /// report's place among its reports, taken together so a later place
    /// never reports an earlier state of the pool. The room reported
    /// reaches no further than the stored horizon; when the pool reaches
    /// further, the horizon to store is returned with it, once until it is
    /// stored ([`Self::grant_horizon_stored`]).
    pub fn grant_report(&self, last_index: u64) -> (GrantReport, Option<GrantRecord>) {
        let mut grants = self.grants.lock();
        let free = grants.free.bytes();
        let cum = grants.cum();
        let record = grants.record;
        // Bounded by the bytes this endpoint can hold.
        let reach = cum + free;
        // Floored at zero by the rule: a horizon the log passed leaves no
        // room to report.
        let available = free.min(record.horizon.saturating_sub(cum));
        // Only when the stored horizon holds back room the pool has: with
        // none free a further horizon promises nothing more.
        let raise = (available < free && !grants.raising).then(|| {
            grants.raising = true;
            GrantRecord {
                horizon: reach,
                ..record
            }
        });
        // One per reply sent: far below u64::MAX.
        grants.seq += 1;
        (
            GrantReport {
                available,
                last_index,
                epoch: record.epoch,
                seq: grants.seq,
            },
            raise,
        )
    }

    /// A raise of the horizon [`Self::grant_report`] returned is over:
    /// `stored` when it reached the disk, synced; reports reach it from now
    /// on.
    pub fn grant_horizon_stored(&self, raised: GrantRecord, stored: bool) {
        let mut grants = self.grants.lock();
        grants.raising = false;
        if stored {
            grants.record.horizon = grants.record.horizon.max(raised.horizon);
            // One per synced store: far below u64::MAX.
            #[cfg(any(test, feature = "test-faults"))]
            {
                grants.stores += 1;
            }
        }
    }

    /// Further horizons this process stored, each a synced write of the
    /// member's journal: what the grant costs in syncs. For tests.
    #[cfg(any(test, feature = "test-faults"))]
    #[must_use]
    pub fn grant_stores(&self) -> u64 {
        self.grants.lock().stores
    }

    /// The room granted and not set aside for any entry; `u64::MAX` with no
    /// limit.
    #[must_use]
    pub fn granted_room(&self) -> u64 {
        if self.is_empty() {
            return u64::MAX;
        }
        self.grants.lock().free.bytes()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
