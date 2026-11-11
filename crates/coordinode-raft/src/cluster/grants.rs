//! The leader's view of the room its followers granted for client entries.
//!
//! A follower holds room for its leader's client entries admitted ahead
//! against its own hard limit and reports it on every append response: how
//! much it holds for entries after its last log index. The leader appends a
//! client entry only while, in every voter set of the membership, a majority
//! (itself included) can pay for it together with every client entry it
//! already sent past that member's reported index and every hand-over it
//! admitted that has not reached its log yet. A follower then pays for each
//! entry it receives from that room and never refuses it, so a client entry
//! never sits unacknowledged at the head of the log for want of space, and
//! the group's own work behind it commits.
//!
//! The view belongs to one leadership, set by the leader's own term: a
//! report of an older term changes nothing, one of a newer term (this member
//! leads again) starts the view over. A member's reports are taken in the
//! order of its process and of each process's replies, so a reply that
//! arrives late gives back nothing. Each hand-over of a proposal is counted
//! on its own until the log takes it or the leadership ends: a hand-over
//! whose wait timed out may still be appended.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use coordinode_core::txn::proposal::ProposalId;

use crate::proto::replication::ClientGrant;

/// The room a log entry takes until it is applied, by whose room pays: the
/// group's own work (as its leader admitted it), and its client proposals.
/// The leader admits a client proposal by the same measure.
pub(crate) fn entry_room(entry: &crate::storage::Entry) -> (u64, u64) {
    use coordinode_storage::engine::footprint::{closure_footprint, write_footprint};
    let openraft::entry::EntryPayload::Normal(request) = &entry.payload else {
        return (0, 0);
    };
    let (mut own, mut client) = (0u64, 0u64);
    for proposal in &request.proposals {
        // Bounded by the size of one entry.
        let room = write_footprint(&proposal.mutations) + closure_footprint();
        if proposal.bypass_rate_limiter {
            own += room;
        } else {
            client += room;
        }
    }
    (own, client)
}

/// The room a client proposal is admitted for, by the measure members pay
/// its entry with.
pub(crate) fn proposal_room(proposal: &coordinode_core::txn::proposal::RaftProposal) -> u64 {
    coordinode_storage::engine::footprint::write_footprint(&proposal.mutations)
        + coordinode_storage::engine::footprint::closure_footprint()
}

/// What a member last reported of its grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Granted {
    /// The member has no hard limit: any entry fits.
    Unbounded,
    /// Room after a log index, from one process of the member.
    Room(ClientGrant),
}

#[derive(Debug, Default)]
struct View {
    /// The leadership the view belongs to.
    term: u64,
    reports: HashMap<u64, Granted>,
    /// The client bytes this leader appended, cumulative through each index
    /// that holds some: what is in flight to a member is `total` less the
    /// value through its reported index.
    cum: BTreeMap<u64, u64>,
    /// Client bytes appended in all.
    total: u64,
    /// Entries through `floor` are no longer kept: through it `base` bytes
    /// were appended. A member reported below `floor` has more in flight
    /// than any member grants (or reported past it before), and covers
    /// nothing until it catches up.
    floor: u64,
    base: u64,
    /// Hand-overs admitted and not yet in the log, by proposal, in order.
    pending: HashMap<ProposalId, VecDeque<u64>>,
    pending_total: u64,
    /// Reports of a newer term than the view's, kept aside: they become the
    /// view's only when this member admits under that term as its leader.
    ahead: Option<(u64, HashMap<u64, Granted>)>,
}

impl View {
    /// Take the leadership of `term`, as the leader's own metrics show it: a
    /// newer one starts the view over with the reports kept aside for it.
    /// Whether the view now belongs to `term`.
    fn lead(&mut self, term: u64) -> bool {
        if term > self.term {
            let reports = match self.ahead.take() {
                Some((ahead, reports)) if ahead == term => reports,
                _ => HashMap::new(),
            };
            *self = Self {
                term,
                reports,
                ..Self::default()
            };
        }
        term == self.term
    }

    /// The reports `term` is recorded in: the view's own, or those kept
    /// aside for a newer term; `None` for an older one.
    fn reports_of(&mut self, term: u64) -> Option<&mut HashMap<u64, Granted>> {
        if term == self.term {
            return Some(&mut self.reports);
        }
        if term < self.term {
            return None;
        }
        if self.ahead.as_ref().is_none_or(|(ahead, _)| *ahead < term) {
            self.ahead = Some((term, HashMap::new()));
        }
        match &mut self.ahead {
            Some((ahead, reports)) if *ahead == term => Some(reports),
            _ => None,
        }
    }

    /// The bytes appended through `index`, or `None` below what is kept.
    fn through(&self, index: u64) -> Option<u64> {
        if index < self.floor {
            return None;
        }
        Some(
            self.cum
                .range(..=index)
                .next_back()
                .map_or(self.base, |(_, bytes)| *bytes),
        )
    }

    /// Whether `member` can pay for `bytes` more past what is in flight to it.
    fn covers(&self, member: u64, bytes: u64) -> bool {
        match self.reports.get(&member) {
            None => false,
            Some(Granted::Unbounded) => true,
            Some(Granted::Room(grant)) => self.through(grant.last_index).is_some_and(|sent| {
                // Bounded by what this leader admitted.
                let in_flight = self.total - sent + self.pending_total;
                in_flight
                    .checked_add(bytes)
                    .is_some_and(|need| need <= grant.available)
            }),
        }
    }

    /// Keep only the reports of `voters`, and only the appended entries that
    /// can still decide whether a voter covers a write: those past the
    /// lowest index a voter reported, and of those no more than the largest
    /// grant can hold. A member that left holds nothing back, and a voter
    /// that lags or is silent costs no more than the largest grant.
    fn compact(&mut self, voters: &[BTreeSet<u64>]) {
        self.reports
            .retain(|member, _| voters.iter().any(|set| set.contains(member)));
        let (mut lowest, mut largest) = (None::<u64>, 0u64);
        for granted in self.reports.values() {
            if let Granted::Room(grant) = granted {
                lowest = Some(lowest.map_or(grant.last_index, |l| l.min(grant.last_index)));
                largest = largest.max(grant.available);
            }
        }
        while let Some(entry) = self.cum.first_entry() {
            let (index, bytes) = (*entry.key(), *entry.get());
            let reported_past = lowest.is_some_and(|l| index <= l);
            if !reported_past && self.total - bytes <= largest {
                break;
            }
            entry.remove();
            // What was appended through `index` is still known: `base`.
            self.floor = index;
            self.base = bytes;
        }
    }
}

/// Why a client proposal was refused before it reached the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GrantRefusal {
    /// The leader's first entry of its term is not applied yet: entries of
    /// earlier terms may still be in flight to members, unaccounted here.
    NotReady,
    /// No majority of some voter set holds room for it.
    NoRoom,
}

/// The leader's view of its followers' grants.
#[derive(Debug, Default)]
pub(crate) struct LeaderGrants {
    view: parking_lot::Mutex<View>,
}

impl LeaderGrants {
    /// Record what `member` reported in its reply to an append this member
    /// sent while leading in `term`; `None` from a member without a limit. A
    /// report never moves the view to another leadership: only an admission
    /// under the leader's own metrics does. One of an older term changes
    /// nothing; one of a newer term is kept aside for it.
    pub(crate) fn report(&self, term: u64, member: u64, grant: Option<ClientGrant>) {
        let mut view = self.view.lock();
        let Some(reports) = view.reports_of(term) else {
            return;
        };
        let granted = grant.map_or(Granted::Unbounded, Granted::Room);
        if let (Some(Granted::Room(old)), Granted::Room(new)) = (reports.get(&member), granted)
            && (new.epoch, new.seq) <= (old.epoch, old.seq)
        {
            return;
        }
        reports.insert(member, granted);
    }

    /// Admit a hand-over of client proposal `id` taking `bytes` under
    /// leadership `term` of `leader`, when `ready` (the term's first entry
    /// is applied) and each voter set in `voters` has a majority, the leader
    /// included, that can pay for it. Every hand-over is a log entry of its
    /// own that members pay for, so a retry is admitted again.
    pub(crate) fn admit(
        &self,
        term: u64,
        leader: u64,
        ready: bool,
        voters: &[BTreeSet<u64>],
        id: ProposalId,
        bytes: u64,
    ) -> Result<(), GrantRefusal> {
        if !ready {
            return Err(GrantRefusal::NotReady);
        }
        let mut view = self.view.lock();
        if !view.lead(term) {
            // A leadership this member no longer holds.
            return Err(GrantRefusal::NoRoom);
        }
        // A leader alone in every voter set pays for its entries itself.
        if voters.iter().flatten().all(|&member| member == leader) {
            return Ok(());
        }
        view.compact(voters);
        let quorate = voters.iter().all(|set| {
            let paying = set
                .iter()
                .filter(|&&member| member == leader || view.covers(member, bytes))
                .count();
            paying * 2 > set.len()
        });
        if !quorate {
            return Err(GrantRefusal::NoRoom);
        }
        view.pending.entry(id).or_default().push_back(bytes);
        // Bounded by what members grant.
        view.pending_total += bytes;
        Ok(())
    }

    /// The hand-over admitted last for proposal `id` under `term` was not
    /// made, as its caller proved before handing it over: its room is no
    /// longer in flight. Nothing once the leadership changed (the view was
    /// started over) or the log took it.
    pub(crate) fn withdraw(&self, term: u64, id: ProposalId) {
        let mut view = self.view.lock();
        if view.term != term {
            return;
        }
        let Some(bytes) = view.pending.get_mut(&id).and_then(VecDeque::pop_back) else {
            return;
        };
        if view.pending.get(&id).is_some_and(VecDeque::is_empty) {
            view.pending.remove(&id);
        }
        view.pending_total -= bytes;
    }

    /// A hand-over of proposal `id` is in the log at `index`: what it takes
    /// is in flight to every member that has not reported past it. Nothing
    /// for a proposal this leader did not admit (a follower's own log).
    pub(crate) fn appended(&self, id: ProposalId, index: u64) {
        let mut view = self.view.lock();
        let Some(bytes) = view.pending.get_mut(&id).and_then(VecDeque::pop_front) else {
            return;
        };
        if view.pending.get(&id).is_some_and(VecDeque::is_empty) {
            view.pending.remove(&id);
        }
        view.pending_total -= bytes;
        view.total += bytes;
        let total = view.total;
        view.cum.insert(index, total);
    }
}

/// One hand-over's claim on its members' grants: withdrawn when the caller
/// proves the hand-over was not made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Claim {
    term: u64,
    id: ProposalId,
}

impl Claim {
    /// The hand-over was not made (refused before it reached the core).
    pub(crate) fn withdraw(self, grants: &LeaderGrants) {
        grants.withdraw(self.term, self.id);
    }
}

/// Claim, for one hand-over of a client `proposal` by `raft`, room on a
/// majority of each voter set; the claim lasts until the log takes the
/// hand-over, the leadership ends or the claim is withdrawn. Made right
/// before the hand-over. The group's own work claims nothing, and neither
/// does a member that does not lead: its write is refused with where the
/// leader is.
///
/// # Errors
///
/// [`ProposalError::LeaderNotReady`] before the term's first entry is
/// applied; [`ProposalError::MembersFull`] when no majority holds room.
pub(crate) fn claim(
    grants: &LeaderGrants,
    raft: &openraft::Raft<crate::storage::TypeConfig, crate::storage::CoordinodeStateMachine>,
    proposal: &coordinode_core::txn::proposal::RaftProposal,
) -> Result<Option<Claim>, coordinode_core::txn::proposal::ProposalError> {
    use coordinode_core::txn::proposal::ProposalError;
    use openraft::rt::watch::WatchReceiver;
    if proposal.bypass_rate_limiter {
        return Ok(None);
    }
    let (term, leader, ready, voters) = {
        let m = raft.metrics().borrow_watched().clone();
        if m.current_leader != Some(m.id) {
            return Ok(None);
        }
        let ready = m
            .last_applied
            .is_some_and(|applied| applied.committed_leader_id().term == m.current_term);
        let voters: Vec<BTreeSet<u64>> =
            m.membership_config.membership().get_joint_config().clone();
        (m.current_term, m.id, ready, voters)
    };
    match grants.admit(
        term,
        leader,
        ready,
        &voters,
        proposal.id,
        proposal_room(proposal),
    ) {
        Ok(()) => Ok(Some(Claim {
            term,
            id: proposal.id,
        })),
        Err(GrantRefusal::NotReady) => Err(ProposalError::LeaderNotReady { waited_ms: 0 }),
        Err(GrantRefusal::NoRoom) => Err(ProposalError::MembersFull),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
