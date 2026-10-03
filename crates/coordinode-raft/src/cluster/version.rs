//! Version matching between the members of one group.
//!
//! Members whose version pairs differ exchange nothing but the frozen
//! [`Handshake`]: every inter-node call carries the caller's record in
//! [`HANDSHAKE_METADATA_KEY`], the receiver refuses one that does not match
//! before reading its payload, and answers with its own record either way. A
//! member whose own pair is not the one its group runs is read-only: it
//! refuses writes and serves reads as of the last entry it applied.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use coordinode_core::version::{Handshake, RecordedPair, VersionPair};

use crate::storage::GroupPairs;

/// The request and response metadata key carrying a member's encoded
/// [`Handshake`] on every inter-node call. Frozen with the record.
pub const HANDSHAKE_METADATA_KEY: &str = "cn-version-handshake-bin";

/// Whether this member runs the pair its group runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemberState {
    /// It does, or the group has recorded no pair yet: it serves everything.
    Matched,
    /// It does not: read-only, frozen at what it held when it stopped
    /// matching.
    Mismatched(Mismatch),
}

pub use coordinode_core::version::Mismatch;

/// Why a peer's call is refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    /// The call carries no version record, or one that does not decode: it
    /// comes from a release without the handshake, or is not a member's.
    #[error("the caller sent no version handshake: {0}")]
    NoHandshake(String),
    /// The caller speaks for another consensus group.
    #[error("the caller belongs to group {theirs}, this member to group {ours}")]
    OtherGroup {
        /// The caller's group.
        theirs: u64,
        /// This member's group.
        ours: u64,
    },
    /// The two members run different pairs.
    #[error("the caller runs {theirs}, this member runs {ours}")]
    Pair {
        /// The caller's pair.
        theirs: VersionPair,
        /// This member's pair.
        ours: VersionPair,
    },
    /// This member is behind its group and follows no one.
    #[error("{0}")]
    Behind(Mismatch),
}

/// One member's view of versions: its own pair, the pair its group runs,
/// and what each peer last reported.
pub struct VersionGate {
    node_id: u64,
    group_id: u64,
    pair: VersionPair,
    /// The group's pair records as this member applied them.
    applied: tokio::sync::watch::Receiver<GroupPairs>,
    applied_commit_ts: Arc<AtomicU64>,
    // no-std: spin::Mutex; the view is touched once per inter-node call.
    view: parking_lot::Mutex<View>,
}

#[derive(Default)]
struct View {
    /// The latest record a peer reported, when later than the applied one.
    reported: Option<RecordedPair>,
    leader: Option<(u64, String)>,
    peers: BTreeMap<u64, Handshake>,
    /// When this member first saw no pair able to write: none held by a
    /// majority of the voters, or the majority's pair not yet recorded.
    paused_since: Option<std::time::Instant>,
}

/// What a member reports about versions: its own, its group's, whether it
/// serves writes, and the pair each voter runs as far as it knows.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct VersionReport {
    /// This member.
    pub node_id: u64,
    /// The pair it runs.
    pub pair: VersionPair,
    /// The pair its group runs, with the record that set it.
    pub group_pair: Option<RecordedPair>,
    /// Why it is read-only; `None` when it serves writes.
    pub read_only: Option<ReadOnly>,
    /// Every voter and the pair it last reported (this member's own for
    /// itself); `None` for a voter not heard from.
    pub voters: Vec<VoterPair>,
    /// The pair a majority of the voters runs, if one does.
    pub majority_pair: Option<VersionPair>,
    /// How long no pair has been able to write, in milliseconds, while the
    /// group is paused; `None` while it writes.
    pub pause_ms: Option<u64>,
}

/// A read-only member's reason, as its refusals name it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ReadOnly {
    /// The group moved past this member; otherwise this member is ahead.
    pub behind: bool,
    /// The commit timestamp its reads are as of.
    pub as_of: u64,
    /// The leader to send writes to, when known.
    pub leader_id: Option<u64>,
    /// That leader's address, when known.
    pub leader_addr: Option<String>,
    /// The refusal, in words.
    pub reason: String,
}

/// One voter's pair.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct VoterPair {
    /// The voter.
    pub node_id: u64,
    /// The pair it last reported.
    pub pair: Option<VersionPair>,
}

impl VersionGate {
    /// The gate of member `node_id` of group `group_id`, running `pair`,
    /// over the applied record stream of its state machine.
    pub fn new(
        node_id: u64,
        group_id: u64,
        pair: VersionPair,
        applied: tokio::sync::watch::Receiver<GroupPairs>,
        applied_commit_ts: Arc<AtomicU64>,
    ) -> Self {
        Self {
            node_id,
            group_id,
            pair,
            applied,
            applied_commit_ts,
            view: parking_lot::Mutex::new(View::default()),
        }
    }

    /// The pair this member runs.
    pub fn pair(&self) -> VersionPair {
        self.pair
    }

    /// The pair the group runs as far as this member knows: the later of
    /// what it applied and what a peer reported.
    pub fn group_pair(&self) -> Option<RecordedPair> {
        let applied = self.applied.borrow().last().copied();
        let reported = self.view.lock().reported;
        later(applied, reported)
    }

    /// Whether this member serves writes, and if not, why.
    pub fn state(&self) -> MemberState {
        match self.group_pair() {
            Some(group) if group.pair != self.pair => MemberState::Mismatched(Mismatch {
                own: self.pair,
                group: group.pair,
                behind: self.behind_at(group.seq),
                leader: self.view.lock().leader.clone(),
                as_of: self.applied_commit_ts.load(Ordering::Acquire),
            }),
            _ => MemberState::Matched,
        }
    }

    /// The report of this member among `voters`, the group's voting members.
    /// Also marks the start or end of a write pause, so the pause it reports
    /// is measured from the first report or leadership change that saw it.
    pub fn report(&self, voters: &[u64]) -> VersionReport {
        let group_pair = self.group_pair();
        let read_only = match self.state() {
            MemberState::Matched => None,
            MemberState::Mismatched(m) => Some(ReadOnly {
                behind: m.behind,
                as_of: m.as_of,
                leader_id: m.leader.as_ref().map(|(id, _)| *id),
                leader_addr: m.leader.as_ref().map(|(_, addr)| addr.clone()),
                reason: m.to_string(),
            }),
        };
        let mut view = self.view.lock();
        let voters: Vec<VoterPair> = voters
            .iter()
            .map(|&node_id| VoterPair {
                node_id,
                pair: if node_id == self.node_id {
                    Some(self.pair)
                } else {
                    view.peers.get(&node_id).map(|h| h.pair)
                },
            })
            .collect();
        let majority_pair = majority(&voters);
        // The group writes when a majority runs one pair and that pair is the
        // recorded one (or nothing is recorded yet, so its leader records it
        // as its first entry).
        let writes = majority_pair.is_some_and(|m| group_pair.is_none_or(|g| g.pair == m));
        let pause_ms = if writes {
            view.paused_since = None;
            None
        } else {
            let since = *view
                .paused_since
                .get_or_insert_with(std::time::Instant::now);
            // A pause measured in u64 milliseconds outlasts any process.
            Some(u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX))
        };
        VersionReport {
            node_id: self.node_id,
            pair: self.pair,
            group_pair,
            read_only,
            voters,
            majority_pair,
            pause_ms,
        }
    }

    /// The record this member sends.
    pub fn local_handshake(&self) -> Handshake {
        Handshake {
            node_id: self.node_id,
            group_id: self.group_id,
            pair: self.pair,
            group_pair: self.group_pair(),
            leader: self.view.lock().leader.clone(),
        }
    }

    /// The group's leader as this member's consensus sees it. Ignored once a
    /// peer reported a later record of the group's pair than this member
    /// applied: this member's consensus is then a stale view of a group that
    /// moved on, and the leader the peer named is the one to retry at.
    pub fn set_leader(&self, leader: Option<(u64, String)>) {
        let applied = self.applied.borrow().last().copied();
        let mut view = self.view.lock();
        let moved_on = match (view.reported, applied) {
            (Some(reported), Some(applied)) => reported.is_later_than(&applied),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if !moved_on {
            view.leader = leader;
        }
    }

    /// Learn from a peer's record: its pair, and the group's pair and leader
    /// when it knows a later record than this member.
    pub fn observe(&self, peer: &Handshake) {
        if peer.group_id != self.group_id {
            return;
        }
        let mut view = self.view.lock();
        if let Some(reported) = peer.group_pair {
            let applied = self.applied.borrow().last().copied();
            let known = later(applied, view.reported);
            if known.is_none_or(|k| reported.is_later_than(&k)) {
                view.reported = Some(reported);
                // A member that knows a later record than ours knows the
                // leader of the group that made it better than we do.
                if peer.leader.is_some() {
                    view.leader = peer.leader.clone();
                }
            }
        }
        view.peers.insert(peer.node_id, peer.clone());
    }

    /// Whether `peer` last reported the pair this member runs; `None` before
    /// it reported anything.
    pub fn peer_matches(&self, peer: u64) -> Option<bool> {
        self.view
            .lock()
            .peers
            .get(&peer)
            .map(|h| h.pair == self.pair && h.group_id == self.group_id)
    }

    /// Admit a call from the peer whose record is `peer`, or say why not.
    /// The record is learnt from either way.
    pub fn admit(&self, peer: Result<Handshake, Refusal>) -> Result<(), Refusal> {
        let peer = peer?;
        self.observe(&peer);
        if peer.group_id != self.group_id {
            return Err(Refusal::OtherGroup {
                theirs: peer.group_id,
                ours: self.group_id,
            });
        }
        if peer.pair != self.pair {
            return Err(Refusal::Pair {
                theirs: peer.pair,
                ours: self.pair,
            });
        }
        match self.state() {
            MemberState::Mismatched(m) if m.behind => Err(Refusal::Behind(m)),
            _ => Ok(()),
        }
    }

    /// The pair the group should record now that this member leads: its own,
    /// unless the group runs it already or has moved past it.
    pub fn pair_to_record(&self) -> Option<VersionPair> {
        match self.group_pair() {
            Some(group) if group.pair == self.pair => None,
            Some(group) if self.behind_at(group.seq) => None,
            _ => Some(self.pair),
        }
    }

    /// Whether this member's pair was recorded before record `seq`: the
    /// group ran it and moved on.
    fn behind_at(&self, seq: u64) -> bool {
        self.applied
            .borrow()
            .iter()
            .any(|r| r.pair == self.pair && r.seq < seq)
    }
}

/// The pair more than half of `voters` run, if any.
fn majority(voters: &[VoterPair]) -> Option<VersionPair> {
    let mut counts: Vec<(VersionPair, usize)> = Vec::new();
    for pair in voters.iter().filter_map(|v| v.pair) {
        match counts.iter_mut().find(|(p, _)| *p == pair) {
            Some((_, n)) => *n += 1,
            None => counts.push((pair, 1)),
        }
    }
    counts
        .into_iter()
        .find(|(_, n)| *n > voters.len() / 2)
        .map(|(p, _)| p)
}

fn later(a: Option<RecordedPair>, b: Option<RecordedPair>) -> Option<RecordedPair> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if b.is_later_than(&a) { b } else { a }),
        (a, b) => a.or(b),
    }
}

/// The handshake a call's metadata carries.
pub fn read_handshake(metadata: &tonic::metadata::MetadataMap) -> Result<Handshake, Refusal> {
    let value = metadata
        .get_bin(HANDSHAKE_METADATA_KEY)
        .ok_or_else(|| Refusal::NoHandshake("no version record".into()))?;
    let bytes = value
        .to_bytes()
        .map_err(|e| Refusal::NoHandshake(e.to_string()))?;
    Handshake::decode(&bytes).map_err(|e| Refusal::NoHandshake(e.to_string()))
}

/// Put `handshake` into a call's metadata.
pub fn write_handshake(metadata: &mut tonic::metadata::MetadataMap, handshake: &Handshake) {
    metadata.insert_bin(
        HANDSHAKE_METADATA_KEY,
        tonic::metadata::MetadataValue::from_bytes(&handshake.encode()),
    );
}

/// The status a refused call is answered with: FAILED_PRECONDITION naming
/// both sides, carrying this member's record.
pub fn refusal_status(refusal: &Refusal, local: &Handshake) -> tonic::Status {
    let mut metadata = tonic::metadata::MetadataMap::new();
    write_handshake(&mut metadata, local);
    tonic::Status::with_metadata(
        tonic::Code::FailedPrecondition,
        format!("version mismatch: {refusal}"),
        metadata,
    )
}

/// Serves the frozen exchange: learns the caller's record and answers with
/// this member's, whether or not they match.
pub struct HandshakeService {
    gate: Arc<VersionGate>,
}

impl HandshakeService {
    /// The exchange of `gate`'s member.
    pub fn new(gate: Arc<VersionGate>) -> Self {
        Self { gate }
    }
}

#[tonic::async_trait]
impl crate::proto::internode::version_handshake_server::VersionHandshake for HandshakeService {
    async fn exchange(
        &self,
        request: tonic::Request<crate::proto::internode::HandshakeRecord>,
    ) -> Result<tonic::Response<crate::proto::internode::HandshakeRecord>, tonic::Status> {
        match Handshake::decode(&request.into_inner().record) {
            Ok(peer) => self.gate.observe(&peer),
            Err(e) => tracing::debug!(%e, "an undecodable version handshake"),
        }
        Ok(tonic::Response::new(
            crate::proto::internode::HandshakeRecord {
                record: self.gate.local_handshake().encode(),
            },
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
