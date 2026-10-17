//! Client operation records the ordered application decides.
//!
//! A unit carrying an [`OperationCommand`] is decided against the records the
//! engine holds when it applies, under the decision lock of its issuer, and
//! turned into plain Schema writes before anything else sees it. A `Bind`
//! decides the whole unit: the first attempt of an operation applies its
//! effects with the operation's record, any later attempt applies nothing.
//! The decision reads nothing but the records earlier decisions wrote, so it
//! is the same on every member and on every replay.

use coordinode_core::txn::operation::{
    BindDecision, INCARNATION_KEY_PREFIX, INTENT_DIGEST_LEN, ISSUER_ID_LEN, IncarnationRecord,
    IntentDigest, IssuerId, LogPosition, OperationCommand, OperationId, OperationOutcome,
    OperationRecord, OperationState, decide_binding, decode_incarnation_key, decode_operation_key,
    incarnation_key, incarnation_operations_prefix, issuer_incarnations_prefix, operation_key,
    outcome_of,
};
use coordinode_core::txn::proposal::{Mutation, PartitionId};

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// Stripes of the operation decision lock: issuers are random, so a stripe
/// is contended only by decisions that would serialize anyway or by chance.
const DECISION_STRIPES: usize = 64;

/// The operation decision locks, one per issuer stripe.
pub(crate) struct DecisionLocks(Box<[parking_lot::Mutex<()>]>);

impl Default for DecisionLocks {
    fn default() -> Self {
        Self(
            (0..DECISION_STRIPES)
                .map(|_| parking_lot::Mutex::new(()))
                .collect(),
        )
    }
}

impl DecisionLocks {
    /// The lock of `issuer`'s stripe.
    pub(crate) fn lock(&self, issuer: &IssuerId) -> parking_lot::MutexGuard<'_, ()> {
        let mut head = [0u8; 8];
        head.copy_from_slice(&issuer.0[..8]);
        // The remainder is below DECISION_STRIPES, so it indexes the slice.
        let stripe = (u64::from_le_bytes(head) % DECISION_STRIPES as u64) as usize;
        self.0[stripe].lock()
    }
}

/// A decided unit.
pub(crate) struct Decided {
    /// What applies.
    pub(crate) effects: Vec<Mutation>,
    /// The unit's binding refused it: none of its effects apply. A reader of
    /// the raw log must leave it out too.
    pub(crate) skipped: bool,
}

/// The operation command `mutations` carries, if any. It is the unit's last
/// operation, so the positions DERIVED work names in the rest stay as they
/// are once it is decided away.
///
/// # Errors
///
/// The unit carries one elsewhere, more than one, or one beside a metadata
/// command: either of the last two would be decided against state lacking
/// the other's effects.
pub(crate) fn unit_command(
    mutations: &[Mutation],
) -> StorageResult<Option<(&OperationCommand, &[Mutation])>> {
    let Some((Mutation::Operation(command), rest)) = mutations.split_last() else {
        if mutations
            .iter()
            .any(|m| matches!(m, Mutation::Operation(_)))
        {
            return Err(StorageError::InvalidConfig(
                "an operation command is not the last operation of its unit".into(),
            ));
        }
        return Ok(None);
    };
    if rest
        .iter()
        .any(|m| matches!(m, Mutation::Operation(_) | Mutation::Command(_)))
    {
        return Err(StorageError::InvalidConfig(
            "a unit carries an operation command beside another command".into(),
        ));
    }
    Ok(Some((command, rest)))
}

/// Decide `command`, the operation command of the unit at `at` whose other
/// operations are `rest`, against the records `engine` holds, for the unit
/// applying at `commit_ts`.
pub(crate) fn decide(
    engine: &StorageEngine,
    command: &OperationCommand,
    rest: &[Mutation],
    commit_ts: u64,
    at: LogPosition,
) -> StorageResult<Decided> {
    let with = |records: Vec<Mutation>| {
        let mut effects = rest.to_vec();
        effects.extend(records);
        Decided {
            effects,
            skipped: false,
        }
    };
    Ok(match command {
        OperationCommand::Bind(binding) => {
            let incarnation =
                incarnation_record(engine, &binding.op.issuer, binding.op.incarnation)?;
            let record = operation_record(engine, &binding.op)?;
            match decide_binding(
                binding,
                incarnation.as_ref(),
                record.as_ref(),
                commit_ts,
                at,
            ) {
                BindDecision::Apply(record) => with(vec![schema_put(
                    operation_key(&binding.op),
                    record.encode(),
                )]),
                BindDecision::Skip => Decided {
                    effects: Vec::new(),
                    skipped: true,
                },
            }
        }
        OperationCommand::Enroll {
            issuer,
            incarnation,
            window,
        } => with(decide_enrollment(engine, issuer, *incarnation, *window)?),
        OperationCommand::Exclude(op) => with(decide_exclusion(engine, op)?),
        OperationCommand::Release(op) => with(decide_release(engine, op)?),
        OperationCommand::Retire {
            issuer,
            incarnation,
            below,
        } => with(decide_retirement(engine, issuer, *incarnation, *below)?),
    })
}

fn schema_put(key: Vec<u8>, value: Vec<u8>) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Schema,
        key,
        value,
    }
}

/// Open `incarnation`, closing every lower open one; nothing when the issuer
/// already has it or a higher one, or when it admits nothing.
fn decide_enrollment(
    engine: &StorageEngine,
    issuer: &IssuerId,
    incarnation: u64,
    window: u64,
) -> StorageResult<Vec<Mutation>> {
    if incarnation == 0 || window == 0 {
        tracing::warn!(incarnation, window, "operation enrollment refused");
        return Ok(Vec::new());
    }
    let existing = incarnations(engine, issuer)?;
    if existing.iter().any(|(number, _)| *number >= incarnation) {
        return Ok(Vec::new());
    }
    let mut effects = Vec::with_capacity(existing.len() + 1);
    for (number, record) in existing.into_iter().filter(|(_, r)| r.open) {
        let closed = IncarnationRecord {
            open: false,
            ..record
        };
        effects.push(schema_put(incarnation_key(issuer, number), closed.encode()));
    }
    let opened = IncarnationRecord {
        open: true,
        // Numbers start at 1.
        retired_below: 1,
        window,
    };
    effects.push(schema_put(
        incarnation_key(issuer, incarnation),
        opened.encode(),
    ));
    Ok(effects)
}

/// Exclude `op` when no attempt committed it: one nothing is recorded of and
/// that could still be admitted, or one admitted and not completed (a
/// provisional write whose drain is delayed or whose payload was lost), so
/// the admitted attempt never completes either. Nothing otherwise.
fn decide_exclusion(engine: &StorageEngine, op: &OperationId) -> StorageResult<Vec<Mutation>> {
    match operation_record(engine, op)? {
        Some(OperationRecord {
            intent,
            state: OperationState::Admitted { .. },
        }) => {
            let excluded = OperationRecord {
                intent,
                state: OperationState::Excluded,
            };
            return Ok(vec![schema_put(operation_key(op), excluded.encode())]);
        }
        Some(_) => return Ok(Vec::new()),
        None => {}
    }
    let Some(incarnation) = incarnation_record(engine, &op.issuer, op.incarnation)? else {
        return Ok(Vec::new());
    };
    let admissible = incarnation.open
        && op.sequence >= incarnation.retired_below
        && op.sequence - incarnation.retired_below < incarnation.window;
    if !admissible {
        // Outside the window nothing can bind it; a lookup reports that.
        return Ok(Vec::new());
    }
    let excluded = OperationRecord {
        intent: IntentDigest([0; INTENT_DIGEST_LEN]),
        state: OperationState::Excluded,
    };
    Ok(vec![schema_put(operation_key(op), excluded.encode())])
}

/// Release the outcome of `op` when it is settled (committed or excluded),
/// keeping its intent and the evidence that it is settled. An admitted
/// operation is still owed its outcome, and a number nothing is recorded of
/// may still be on its way: neither is released.
fn decide_release(engine: &StorageEngine, op: &OperationId) -> StorageResult<Vec<Mutation>> {
    match operation_record(engine, op)? {
        Some(OperationRecord {
            intent,
            state: OperationState::Committed { .. } | OperationState::Excluded,
        }) => {
            let released = OperationRecord {
                intent,
                state: OperationState::Released,
            };
            Ok(vec![schema_put(operation_key(op), released.encode())])
        }
        _ => Ok(Vec::new()),
    }
}

/// Retire the incarnation's numbers below `below`, up to its first admitted
/// operation: release their records and move the retired prefix.
fn decide_retirement(
    engine: &StorageEngine,
    issuer: &IssuerId,
    incarnation: u64,
    below: u64,
) -> StorageResult<Vec<Mutation>> {
    let Some(record) = incarnation_record(engine, issuer, incarnation)? else {
        return Ok(Vec::new());
    };
    if below <= record.retired_below {
        return Ok(Vec::new());
    }
    let mut effects = Vec::new();
    let mut retired_below = below;
    let prefix = incarnation_operations_prefix(issuer, incarnation);
    for guard in engine.prefix_scan(Partition::Schema, &prefix)? {
        let (key, value) = guard.into_inner()?;
        let op = decode_operation_key(&key).ok_or_else(|| malformed("operation key"))?;
        if op.sequence >= below {
            break;
        }
        let state = OperationRecord::decode(&value)
            .ok_or_else(|| malformed("operation record"))?
            .state;
        if matches!(state, OperationState::Admitted { .. }) {
            // Still owed its commit: the prefix stops in front of it.
            retired_below = op.sequence;
            break;
        }
        effects.push(Mutation::Delete {
            partition: PartitionId::Schema,
            key: key.to_vec(),
        });
    }
    if retired_below <= record.retired_below {
        return Ok(Vec::new());
    }
    effects.push(schema_put(
        incarnation_key(issuer, incarnation),
        IncarnationRecord {
            retired_below,
            ..record
        }
        .encode(),
    ));
    Ok(effects)
}

fn malformed(what: &str) -> StorageError {
    StorageError::Serialization(format!("an {what} does not decode"))
}

/// The record of `op`, if one is held.
///
/// # Errors
///
/// The record cannot be read or does not decode.
pub fn operation_record(
    engine: &StorageEngine,
    op: &OperationId,
) -> StorageResult<Option<OperationRecord>> {
    engine
        .get(Partition::Schema, &operation_key(op))?
        .map(|bytes| OperationRecord::decode(&bytes).ok_or_else(|| malformed("operation record")))
        .transpose()
}

/// The record of `issuer`'s `incarnation`, if it was enrolled.
///
/// # Errors
///
/// The record cannot be read or does not decode.
pub fn incarnation_record(
    engine: &StorageEngine,
    issuer: &IssuerId,
    incarnation: u64,
) -> StorageResult<Option<IncarnationRecord>> {
    engine
        .get(Partition::Schema, &incarnation_key(issuer, incarnation))?
        .map(|bytes| {
            IncarnationRecord::decode(&bytes).ok_or_else(|| malformed("incarnation record"))
        })
        .transpose()
}

/// Every incarnation `issuer` enrolled, lowest first.
///
/// # Errors
///
/// A record cannot be read or does not decode.
pub fn incarnations(
    engine: &StorageEngine,
    issuer: &IssuerId,
) -> StorageResult<Vec<(u64, IncarnationRecord)>> {
    let mut out = Vec::new();
    for guard in engine.prefix_scan(Partition::Schema, &issuer_incarnations_prefix(issuer))? {
        let (key, value) = guard.into_inner()?;
        let number = decode_incarnation_key(&key).ok_or_else(|| malformed("incarnation key"))?;
        let record =
            IncarnationRecord::decode(&value).ok_or_else(|| malformed("incarnation record"))?;
        out.push((number, record));
    }
    Ok(out)
}

/// How `op` stands in `engine`'s records.
///
/// # Errors
///
/// A record cannot be read or does not decode.
pub fn operation_outcome(
    engine: &StorageEngine,
    op: &OperationId,
) -> StorageResult<OperationOutcome> {
    let incarnation = incarnation_record(engine, &op.issuer, op.incarnation)?;
    let record = operation_record(engine, op)?;
    Ok(outcome_of(op, incarnation.as_ref(), record.as_ref()))
}

/// Every incarnation record `engine` holds, with its issuer and number: the
/// admission fences of its history.
///
/// # Errors
///
/// A record cannot be read or does not decode.
pub fn admission_fences(
    engine: &StorageEngine,
) -> StorageResult<Vec<(IssuerId, u64, IncarnationRecord)>> {
    let mut out = Vec::new();
    for guard in engine.prefix_scan(Partition::Schema, INCARNATION_KEY_PREFIX)? {
        let (key, value) = guard.into_inner()?;
        let issuer = key
            .get(INCARNATION_KEY_PREFIX.len()..)
            .and_then(|rest| rest.first_chunk::<ISSUER_ID_LEN>())
            .ok_or_else(|| malformed("incarnation key"))?;
        let number = decode_incarnation_key(&key).ok_or_else(|| malformed("incarnation key"))?;
        let record =
            IncarnationRecord::decode(&value).ok_or_else(|| malformed("incarnation record"))?;
        out.push((IssuerId(*issuer), number, record));
    }
    Ok(out)
}

/// Fence the operation records of an image installed by a restore, before
/// anything is served from it. The restored store is a new domain: every
/// incarnation the image holds is closed, so a descriptor issued before the
/// restore resolves to its recorded outcome and admits nothing new. The
/// fences of the history the image replaced, `surviving`, are kept too: the
/// image cannot reopen an incarnation or a retired number that history
/// closed. Each incarnation keeps the higher retired prefix of the two.
///
/// Writes the store directly, as the install does; the caller persists both.
///
/// # Errors
///
/// A record cannot be read, decoded or written.
pub fn fence_restored(
    engine: &StorageEngine,
    surviving: &[(IssuerId, u64, IncarnationRecord)],
) -> StorageResult<()> {
    let mut fences: std::collections::BTreeMap<([u8; ISSUER_ID_LEN], u64), IncarnationRecord> =
        admission_fences(engine)?
            .into_iter()
            .map(|(issuer, number, record)| ((issuer.0, number), record))
            .collect();
    for (issuer, number, kept) in surviving {
        fences
            .entry((issuer.0, *number))
            .and_modify(|record| {
                record.retired_below = record.retired_below.max(kept.retired_below);
            })
            .or_insert(*kept);
    }
    for ((issuer, number), record) in fences {
        let closed = IncarnationRecord {
            open: false,
            ..record
        };
        engine.put(
            Partition::Schema,
            &incarnation_key(&IssuerId(issuer), number),
            &closed.encode(),
        )?;
    }
    Ok(())
}

/// Prefix of the marks of log units whose binding refused them. Node-local
/// (`raft:`): each member decides the same, and the marks describe this
/// member's own log, read by what reads that log raw.
const SKIPPED_UNIT_PREFIX: &[u8] = b"raft:skip:";

/// The mark of the unit at `at`.
#[must_use]
pub(crate) fn skipped_unit_key(at: LogPosition) -> Vec<u8> {
    let mut key = Vec::with_capacity(SKIPPED_UNIT_PREFIX.len() + 12);
    key.extend_from_slice(SKIPPED_UNIT_PREFIX);
    key.extend_from_slice(&at.index.to_be_bytes());
    key.extend_from_slice(&at.sub.to_be_bytes());
    key
}

/// Whether the log unit at `at` was refused by its binding: none of its
/// operations took effect, whatever its frame carries.
///
/// # Errors
///
/// The mark cannot be read.
pub fn unit_skipped(engine: &StorageEngine, at: LogPosition) -> StorageResult<bool> {
    Ok(engine
        .get(Partition::Schema, &skipped_unit_key(at))?
        .is_some())
}

/// [`unit_skipped`] of `engine`, for a reader of the log `engine` applies.
#[must_use]
pub fn unit_decisions(
    engine: std::sync::Arc<StorageEngine>,
) -> crate::oplog::tailer::UnitDecisionSource {
    std::sync::Arc::new(move |at| unit_skipped(&engine, at))
}

/// Whether `key` of the Schema partition is an operation or incarnation
/// record.
#[must_use]
pub fn is_operation_record(key: &[u8]) -> bool {
    key.starts_with(coordinode_core::txn::operation::OPERATION_KEY_PREFIX)
        || key.starts_with(INCARNATION_KEY_PREFIX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
