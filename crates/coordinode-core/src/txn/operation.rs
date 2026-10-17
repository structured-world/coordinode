//! Client operation identity: which logical write an effect belongs to, so a
//! retry after a lost reply never adds a second effect.
//!
//! An issuer (a logical client, independent of connections and servers)
//! enrolls an incarnation and numbers its writes. Each write carries an
//! [`OperationBinding`]: its [`OperationId`], a digest of what it means
//! (computed by the database from the statement, never taken from the
//! client), and the attempt that carries it. The binding is decided where
//! the write is applied, in log order, in the same unit as its effects: the
//! first attempt of an operation that reaches the log applies and records its
//! outcome; any later attempt of the same operation applies nothing, and its
//! proposer reads the recorded outcome instead.
//!
//! The records live in the replicated Schema keyspace, so snapshots, backups
//! and a store's growth into a group carry them with the effects they
//! protect.

use alloc::vec::Vec;

use crate::group::DeploymentId;

/// Bytes of an [`IssuerId`].
pub const ISSUER_ID_LEN: usize = 16;

/// Bytes of an [`IntentDigest`].
pub const INTENT_DIGEST_LEN: usize = 32;

/// Operations an incarnation may hold above its retired prefix unless its
/// enrollment says otherwise: admission past it is refused until the issuer
/// retires completed operations.
pub const DEFAULT_ADMISSION_WINDOW: u64 = 1 << 16;

/// A logical client that numbers its writes. Chosen by the client (128 random
/// bits), bound to its owner at enrollment.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct IssuerId(pub [u8; ISSUER_ID_LEN]);

/// One write of one issuer: the issuer, the incarnation it enrolled and the
/// write's number within it. Numbers start at 1.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct OperationId {
    /// The logical client.
    pub issuer: IssuerId,
    /// The enrolled incarnation the number belongs to.
    pub incarnation: u64,
    /// The write's number, from 1.
    pub sequence: u64,
}

/// What an operation means: a digest the database computes over the
/// statement text, its parameters and the settings that change its effect.
/// Two attempts of one operation must carry the same digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct IntentDigest(pub [u8; INTENT_DIGEST_LEN]);

/// The binding a bound write carries in its proposal, in the same unit as
/// its effects: the two commit together, under every write concern, or not
/// at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct OperationBinding {
    /// The operation.
    pub op: OperationId,
    /// What it means.
    pub intent: IntentDigest,
    /// The attempt carrying it: the proposal's id.
    pub attempt: u64,
}

/// A change to the operation records, decided where it is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OperationCommand {
    /// Open `incarnation` of `issuer`, closing every lower one: a closed
    /// incarnation admits nothing new, and what it admitted stays. Refused
    /// unless `incarnation` is above every incarnation the issuer has.
    Enroll {
        /// The issuer.
        issuer: IssuerId,
        /// The incarnation to open.
        incarnation: u64,
        /// Operations it may hold above its retired prefix.
        window: u64,
    },
    /// Bind the proposal's effects to an operation.
    Bind(OperationBinding),
    /// Exclude an operation no attempt has committed, so none ever will: the
    /// proof that it does not take effect. Leaves a committed operation as it
    /// is.
    Exclude(OperationId),
    /// Release the outcome of one operation the caller no longer needs, above
    /// the retired prefix and out of number order: a committed or excluded
    /// operation keeps only the evidence that it is settled, and no attempt
    /// of it is admitted again. Leaves a number nothing is recorded of as it
    /// is.
    Release(OperationId),
    /// Retire every operation of `issuer`'s `incarnation` numbered below
    /// `below`: their records are released, and the retired prefix refuses
    /// them for good, including numbers never used.
    Retire {
        /// The issuer.
        issuer: IssuerId,
        /// The incarnation.
        incarnation: u64,
        /// The first number not to retire.
        below: u64,
    },
}

impl OperationCommand {
    /// Approximate encoded size, for rate limiting and buffer accounting:
    /// the issuer, the numbers and, for a binding, its intent.
    #[must_use]
    pub fn size_estimate(&self) -> usize {
        match self {
            Self::Bind(_) => ISSUER_ID_LEN + INTENT_DIGEST_LEN + 4 * 8 + 1,
            Self::Enroll { .. } | Self::Exclude(_) | Self::Release(_) | Self::Retire { .. } => {
                ISSUER_ID_LEN + 3 * 8
            }
        }
    }

    /// The issuer whose records the command reads and writes.
    #[must_use]
    pub fn issuer(&self) -> IssuerId {
        match self {
            Self::Enroll { issuer, .. } | Self::Retire { issuer, .. } => *issuer,
            Self::Bind(binding) => binding.op.issuer,
            Self::Exclude(op) | Self::Release(op) => op.issuer,
        }
    }
}

/// Where in a log a unit was applied: its entry index and its place in the
/// entry. [`LogPosition::LOCAL`] for a unit applied without a log, which is
/// never delivered twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LogPosition {
    /// The log entry.
    pub index: u64,
    /// The unit's place in the entry.
    pub sub: u32,
}

impl LogPosition {
    /// A unit applied without a log.
    pub const LOCAL: Self = Self { index: 0, sub: 0 };
}

/// What is recorded of one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationState {
    /// Its effects committed with `attempt`, at `commit_ts`, from the unit at
    /// `at`.
    Committed {
        /// The attempt that committed.
        attempt: u64,
        /// Its commit timestamp.
        commit_ts: u64,
        /// The unit that applied it.
        at: LogPosition,
    },
    /// No attempt will ever bind it.
    Excluded,
    /// It was settled, and its outcome was released at the caller's
    /// acknowledgement: no attempt binds it again.
    Released,
}

/// The record of one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationRecord {
    /// What it means; zero for an operation excluded before any attempt.
    pub intent: IntentDigest,
    /// Where it stands.
    pub state: OperationState,
}

impl OperationRecord {
    /// Whether an attempt meaning `intent` means something else than what is
    /// recorded; never for an operation excluded before any attempt, which
    /// records no intent.
    #[must_use]
    pub fn intent_differs(&self, intent: &IntentDigest) -> bool {
        self.intent != IntentDigest([0; INTENT_DIGEST_LEN]) && self.intent != *intent
    }
}

/// The record of one incarnation of an issuer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncarnationRecord {
    /// Whether it still admits operations.
    pub open: bool,
    /// Every number below this one is retired.
    pub retired_below: u64,
    /// Operations it may hold at or above `retired_below`.
    pub window: u64,
}

/// What a proposal's binding decides against the recorded state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindDecision {
    /// The effects apply, and this record is written with them.
    Apply(OperationRecord),
    /// The effects do not apply; the operation's record (if any) says why.
    Skip,
}

/// Decide `binding`, carried by the unit at `at` applying at `commit_ts`,
/// against the operation's current `record` and its incarnation's record.
///
/// The unit that applied an operation may be delivered again (a replay after
/// a crash left some partitions behind it): it applies again, idempotently.
/// Any other unit of an operation that has a record applies nothing.
#[must_use]
pub fn decide_binding(
    binding: &OperationBinding,
    incarnation: Option<&IncarnationRecord>,
    record: Option<&OperationRecord>,
    commit_ts: u64,
    at: LogPosition,
) -> BindDecision {
    let bound = OperationRecord {
        intent: binding.intent,
        state: OperationState::Committed {
            attempt: binding.attempt,
            commit_ts,
            at,
        },
    };
    let logged = at != LogPosition::LOCAL;
    match record {
        Some(existing) if existing.intent != binding.intent => BindDecision::Skip,
        // This very unit applied it: a re-delivery.
        Some(existing) if logged && *existing == bound => BindDecision::Apply(bound),
        Some(_) => BindDecision::Skip,
        None => {
            let admissible = incarnation.is_some_and(|inc| {
                inc.open
                    && binding.op.sequence >= inc.retired_below
                    && binding.op.sequence - inc.retired_below < inc.window
            });
            if admissible {
                BindDecision::Apply(bound)
            } else {
                BindDecision::Skip
            }
        }
    }
}

/// How an operation stands, as a lookup reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationOutcome {
    /// It committed with `attempt` at `commit_ts`.
    Committed {
        /// The attempt that committed.
        attempt: u64,
        /// Its commit timestamp.
        commit_ts: u64,
    },
    /// It is proven not to take effect: excluded before any attempt
    /// committed it.
    Aborted,
    /// Nothing is recorded: an attempt may still be on its way. Not a proof
    /// that it was never submitted; excluding it gives one.
    Unknown,
    /// Its number is retired: its record is released, so its outcome is no
    /// longer known, and no attempt of it is admitted again.
    Retired,
}

/// Classify the operation `op` from its incarnation's record and its own.
#[must_use]
pub fn outcome_of(
    op: &OperationId,
    incarnation: Option<&IncarnationRecord>,
    record: Option<&OperationRecord>,
) -> OperationOutcome {
    match record.map(|r| r.state) {
        Some(OperationState::Committed {
            attempt, commit_ts, ..
        }) => OperationOutcome::Committed { attempt, commit_ts },
        Some(OperationState::Excluded) => OperationOutcome::Aborted,
        Some(OperationState::Released) => OperationOutcome::Retired,
        None => match incarnation {
            Some(inc) if op.sequence < inc.retired_below => OperationOutcome::Retired,
            // A closed incarnation binds nothing new, so an attempt still on
            // its way is refused where it is applied.
            Some(inc) if !inc.open => OperationOutcome::Aborted,
            _ => OperationOutcome::Unknown,
        },
    }
}

/// Prefix of every operation record key.
pub const OPERATION_KEY_PREFIX: &[u8] = b"op:b:";

/// Prefix of every incarnation record key.
pub const INCARNATION_KEY_PREFIX: &[u8] = b"op:i:";

/// The key of `issuer`'s incarnation records, followed by the incarnation.
#[must_use]
pub fn issuer_incarnations_prefix(issuer: &IssuerId) -> Vec<u8> {
    let mut key = Vec::with_capacity(INCARNATION_KEY_PREFIX.len() + ISSUER_ID_LEN);
    key.extend_from_slice(INCARNATION_KEY_PREFIX);
    key.extend_from_slice(&issuer.0);
    key
}

/// The key of one incarnation's record.
#[must_use]
pub fn incarnation_key(issuer: &IssuerId, incarnation: u64) -> Vec<u8> {
    let mut key = issuer_incarnations_prefix(issuer);
    key.extend_from_slice(&incarnation.to_be_bytes());
    key
}

/// The incarnation a key from [`issuer_incarnations_prefix`] names.
#[must_use]
pub fn decode_incarnation_key(key: &[u8]) -> Option<u64> {
    let rest = key.strip_prefix(INCARNATION_KEY_PREFIX)?;
    let (_, incarnation) = rest.split_at_checked(ISSUER_ID_LEN)?;
    Some(u64::from_be_bytes(incarnation.try_into().ok()?))
}

/// The key prefix of one incarnation's operation records, in number order.
#[must_use]
pub fn incarnation_operations_prefix(issuer: &IssuerId, incarnation: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(OPERATION_KEY_PREFIX.len() + ISSUER_ID_LEN + 16);
    key.extend_from_slice(OPERATION_KEY_PREFIX);
    key.extend_from_slice(&issuer.0);
    key.extend_from_slice(&incarnation.to_be_bytes());
    key
}

/// The key of one operation's record.
#[must_use]
pub fn operation_key(op: &OperationId) -> Vec<u8> {
    let mut key = incarnation_operations_prefix(&op.issuer, op.incarnation);
    key.extend_from_slice(&op.sequence.to_be_bytes());
    key
}

/// The operation a key from [`operation_key`] names.
#[must_use]
pub fn decode_operation_key(key: &[u8]) -> Option<OperationId> {
    let rest = key.strip_prefix(OPERATION_KEY_PREFIX)?;
    let (issuer, rest) = rest.split_first_chunk::<ISSUER_ID_LEN>()?;
    let (incarnation, rest) = rest.split_first_chunk::<8>()?;
    let sequence: [u8; 8] = rest.try_into().ok()?;
    Some(OperationId {
        issuer: IssuerId(*issuer),
        incarnation: u64::from_be_bytes(*incarnation),
        sequence: u64::from_be_bytes(sequence),
    })
}

const STATE_COMMITTED: u8 = 1;
const STATE_EXCLUDED: u8 = 3;
const STATE_RELEASED: u8 = 4;

fn put_position(at: LogPosition, out: &mut Vec<u8>) {
    out.extend_from_slice(&at.index.to_be_bytes());
    out.extend_from_slice(&at.sub.to_be_bytes());
}

fn take_position(bytes: &[u8]) -> Option<(LogPosition, &[u8])> {
    let (index, rest) = bytes.split_first_chunk::<8>()?;
    let (sub, rest) = rest.split_first_chunk::<4>()?;
    Some((
        LogPosition {
            index: u64::from_be_bytes(*index),
            sub: u32::from_be_bytes(*sub),
        },
        rest,
    ))
}

impl OperationRecord {
    /// The stored form: the intent, a state tag and the state's numbers.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(INTENT_DIGEST_LEN + 29);
        out.extend_from_slice(&self.intent.0);
        match self.state {
            OperationState::Committed {
                attempt,
                commit_ts,
                at,
            } => {
                out.push(STATE_COMMITTED);
                out.extend_from_slice(&attempt.to_be_bytes());
                out.extend_from_slice(&commit_ts.to_be_bytes());
                put_position(at, &mut out);
            }
            OperationState::Excluded => out.push(STATE_EXCLUDED),
            OperationState::Released => out.push(STATE_RELEASED),
        }
        out
    }

    /// Read the form [`Self::encode`] writes.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (intent, rest) = bytes.split_first_chunk::<INTENT_DIGEST_LEN>()?;
        let (&tag, rest) = rest.split_first()?;
        let state = match tag {
            STATE_COMMITTED => {
                let (attempt, rest) = rest.split_first_chunk::<8>()?;
                let (commit_ts, rest) = rest.split_first_chunk::<8>()?;
                let (at, rest) = take_position(rest)?;
                if !rest.is_empty() {
                    return None;
                }
                OperationState::Committed {
                    attempt: u64::from_be_bytes(*attempt),
                    commit_ts: u64::from_be_bytes(*commit_ts),
                    at,
                }
            }
            STATE_EXCLUDED if rest.is_empty() => OperationState::Excluded,
            STATE_RELEASED if rest.is_empty() => OperationState::Released,
            _ => return None,
        };
        Some(Self {
            intent: IntentDigest(*intent),
            state,
        })
    }
}

impl IncarnationRecord {
    /// The stored form: open flag, retired prefix, window.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(17);
        out.push(u8::from(self.open));
        out.extend_from_slice(&self.retired_below.to_be_bytes());
        out.extend_from_slice(&self.window.to_be_bytes());
        out
    }

    /// Read the form [`Self::encode`] writes.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (&open, rest) = bytes.split_first()?;
        let (retired_below, rest) = rest.split_first_chunk::<8>()?;
        let window: [u8; 8] = rest.try_into().ok()?;
        Some(Self {
            open: match open {
                0 => false,
                1 => true,
                _ => return None,
            },
            retired_below: u64::from_be_bytes(*retired_below),
            window: u64::from_be_bytes(window),
        })
    }
}

/// What a client keeps to find its operation again after a lost reply, a
/// restart or a leader change: the operation and the deployment that holds
/// its record. Opaque to the client: it stores and hands back the bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationDescriptor {
    /// The operation.
    pub op: OperationId,
    /// The deployment its record lives in: the domain that issued it, which
    /// every other database refuses.
    pub deployment: DeploymentId,
}

/// Leading bytes of an encoded [`OperationDescriptor`].
const DESCRIPTOR_MAGIC: &[u8; 4] = b"cnop";

/// Format of the encoded descriptor.
const DESCRIPTOR_VERSION: u8 = 2;

impl OperationDescriptor {
    /// The opaque bytes a client persists.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 1 + ISSUER_ID_LEN + 16 + 16);
        out.extend_from_slice(DESCRIPTOR_MAGIC);
        out.push(DESCRIPTOR_VERSION);
        out.extend_from_slice(&self.op.issuer.0);
        out.extend_from_slice(&self.op.incarnation.to_be_bytes());
        out.extend_from_slice(&self.op.sequence.to_be_bytes());
        out.extend_from_slice(self.deployment.as_bytes());
        out
    }

    /// Read bytes [`Self::to_bytes`] wrote; `None` for anything else.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let rest = bytes.strip_prefix(DESCRIPTOR_MAGIC)?;
        let (&version, rest) = rest.split_first()?;
        if version != DESCRIPTOR_VERSION {
            return None;
        }
        let (issuer, rest) = rest.split_first_chunk::<ISSUER_ID_LEN>()?;
        let (incarnation, rest) = rest.split_first_chunk::<8>()?;
        let (sequence, rest) = rest.split_first_chunk::<8>()?;
        let deployment = DeploymentId::from_slice(rest)?;
        Some(Self {
            op: OperationId {
                issuer: IssuerId(*issuer),
                incarnation: u64::from_be_bytes(*incarnation),
                sequence: u64::from_be_bytes(*sequence),
            },
            deployment,
        })
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests;
