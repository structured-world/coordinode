//! AFTER COMMIT trigger dispatcher (the trigger architecture, ADR-026).
//!
//! AFTER COMMIT triggers are not run inline. When a committed mutation matches
//! one, the executor enqueues a durable [`PendingTriggerEvent`] under
//! `trigger_pending:<name>:<seq>` in the same transaction as the user write
//! (see `coordinode_query`'s `enqueue_after_commit_trigger`). This module is
//! the consumer: it scans the queue, runs each trigger body in a fresh
//! auto-commit transaction, and removes the entry on success. Failures follow
//! the trigger's `ON ERROR` policy — retried in place with exponential backoff,
//! or dead-lettered into `trigger_failures:<name>:<seq>` (retries exhausted,
//! `DEAD_LETTER`/`PROPAGATE` on an already-committed transaction, or an async
//! cascade-depth overflow). Every failed event lands somewhere durable and
//! inspectable; silent loss is impossible.
//!
//! ## Driving the dispatcher
//!
//! - **Embedded** (`cluster_mode == false`): the queue is drained inline at the
//!   end of each committed write — deterministic, no background thread. Retries
//!   scheduled into the future fire on a later write's drain.
//! - **Cluster** (`cluster_mode == true`): a leader-gated background worker
//!   calls [`Database::dispatch_after_commit_triggers`] in a loop (wired in
//!   `coordinode-server`), so bodies and their replicated writes only run on
//!   the node that owns the Raft lease.
//!
//! At-least-once: a crash between "run body" and "delete pending entry"
//! re-runs the body on recovery, so bodies must be idempotent (the same
//! contract `RETRY` / `REPLAY` already imposes).

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use coordinode_core::graph::types::Value;
use coordinode_core::schema::triggers::{
    decode_trigger_event_seq, encode_trigger_failure_key, encode_trigger_key,
    encode_trigger_pending_key, trigger_pending_scan_prefix, FailedTriggerEvent,
    OnErrorPolicySchema, PendingTriggerEvent, TriggerSchema,
};
use coordinode_core::txn::proposal::{Mutation, PartitionId, RaftProposal};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use super::{Database, DatabaseError, QuerySession, TxnMode};

/// Operator-tunable knobs for the AFTER COMMIT trigger dispatcher (R192). The
/// server wires these from `coordinode.conf` / CLI flags; embedded callers can
/// set them via [`Database::set_trigger_dispatch_config`]. Defaults match the
/// trigger architecture (ADR-026 / ADR-026A): cascade depth 10, `RETRY 3 WITH
/// BACKOFF 1000`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriggerDispatchConfig {
    /// Cluster cap on async AFTER COMMIT cascade generations (the trigger
    /// architecture L1). An event whose generation exceeds this is
    /// dead-lettered (cascade overflow) instead of executed.
    pub max_cascade_depth: u32,
    /// Total execution attempts for an AFTER COMMIT trigger that declares no
    /// `ON ERROR` policy, before the event is dead-lettered.
    pub default_retry_attempts: u32,
    /// Base backoff (ms) for the default retry policy; the per-attempt wait is
    /// `backoff * 2^attempt`.
    pub default_backoff_ms: u64,
}

impl Default for TriggerDispatchConfig {
    fn default() -> Self {
        Self {
            max_cascade_depth: 10,
            default_retry_attempts: 3,
            default_backoff_ms: 1000,
        }
    }
}

/// Hard cap on body executions per `dispatch_after_commit_triggers` pass — a
/// backstop against a pathological self-enqueueing cascade that the generation
/// bound somehow fails to stop. Far above any real per-drive trigger volume.
const MAX_EXECUTIONS_PER_PASS: usize = 100_000;

thread_local! {
    /// Reentrancy guard: a trigger body executes through `execute_cypher_impl`,
    /// which would otherwise inline-drain again. The outer dispatch pass already
    /// re-scans for newly enqueued events, so a nested drain is both redundant
    /// and a recursion hazard — this flag makes the inner call a no-op.
    static IN_DISPATCH: Cell<bool> = const { Cell::new(false) };
}

/// Summary of one [`Database::dispatch_after_commit_triggers`] pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AfterCommitDispatchReport {
    /// Bodies that executed successfully (event removed from the queue).
    pub fired: usize,
    /// Events rescheduled for a later retry (left in the queue with backoff).
    pub retried: usize,
    /// Events moved to `trigger_failures` (retries exhausted / dead-letter /
    /// cascade overflow).
    pub dead_lettered: usize,
    /// Events discarded because their trigger no longer exists.
    pub discarded: usize,
    /// Non-fatal dispatcher errors (queue bookkeeping failures). Body errors are
    /// not here — they drive the retry / dead-letter path instead.
    pub errors: Vec<String>,
}

impl AfterCommitDispatchReport {
    /// `true` when the pass changed any queue state.
    pub fn did_work(&self) -> bool {
        self.fired + self.retried + self.dead_lettered + self.discarded > 0
    }
}

/// Outcome of processing a single queued event.
enum Outcome {
    Fired,
    Retried,
    DeadLettered,
    Discarded,
    /// Trigger is disabled — left in the queue untouched, skipped this pass.
    Skipped,
    /// Queue bookkeeping failed; carries the message for the report.
    Err(String),
}

impl Database {
    /// Drain all currently-due AFTER COMMIT trigger events, executing each
    /// trigger body once and applying its `ON ERROR` policy on failure.
    ///
    /// Re-scans after each round so events enqueued by a trigger body (the next
    /// async cascade generation) are drained in the same call, bounded by the
    /// per-trigger `generation` cap. Events scheduled for a future retry are
    /// left in place for a later call. Reentrant calls (a trigger body running
    /// through the executor) are no-ops — the outer pass owns the drain.
    ///
    /// Returns a [`AfterCommitDispatchReport`] of what changed. Body errors are
    /// not surfaced as dispatcher errors; they land in the retry / dead-letter
    /// queues per policy.
    pub fn dispatch_after_commit_triggers(&self) -> AfterCommitDispatchReport {
        let mut report = AfterCommitDispatchReport::default();
        if IN_DISPATCH.with(|f| f.replace(true)) {
            // Already dispatching on this thread (we are inside a trigger body):
            // the outer pass re-scans and will pick up whatever we enqueue.
            return report;
        }
        let _guard = DispatchGuard;

        // Events handled this pass — a disabled or rescheduled-due event must
        // not be re-collected and spun on within the same drive.
        let mut seen: HashSet<u64> = HashSet::new();
        let mut budget = MAX_EXECUTIONS_PER_PASS;

        loop {
            let due = match self.collect_due_pending(&seen) {
                Ok(d) => d,
                Err(e) => {
                    report.errors.push(format!("scan pending: {e}"));
                    break;
                }
            };
            if due.is_empty() || budget == 0 {
                break;
            }
            for (seq, event) in due {
                if budget == 0 {
                    break;
                }
                budget -= 1;
                seen.insert(seq);
                match self.process_pending_event(seq, event) {
                    Outcome::Fired => report.fired += 1,
                    Outcome::Retried => report.retried += 1,
                    Outcome::DeadLettered => report.dead_lettered += 1,
                    Outcome::Discarded => report.discarded += 1,
                    Outcome::Skipped => {}
                    Outcome::Err(e) => report.errors.push(e),
                }
            }
        }
        report
    }

    /// Number of AFTER COMMIT events currently queued (pending execution or
    /// awaiting retry). `0` means the queue is drained. Cheap prefix count —
    /// used by the cluster worker's idle check and by tests.
    pub fn after_commit_pending_count(&self) -> usize {
        self.count_prefix(coordinode_core::schema::triggers::trigger_pending_scan_prefix())
    }

    /// Number of dead-lettered AFTER COMMIT events (`trigger_failures`). The full
    /// per-event surface is `SHOW TRIGGER FAILURES`; this is the bare count.
    pub fn after_commit_failure_count(&self) -> usize {
        self.count_prefix(coordinode_core::schema::triggers::trigger_failures_scan_prefix())
    }

    /// Count committed keys under a schema-partition prefix.
    fn count_prefix(&self, prefix: &[u8]) -> usize {
        match self.engine.prefix_scan(Partition::Schema, prefix) {
            Ok(iter) => iter.filter_map(|g| g.key().ok()).count(),
            Err(_) => 0,
        }
    }

    /// Scan the pending queue and return events that are due now (`next_attempt_us
    /// <= now`) and not already handled this pass, oldest-key first.
    fn collect_due_pending(
        &self,
        seen: &HashSet<u64>,
    ) -> Result<Vec<(u64, PendingTriggerEvent)>, String> {
        let now = now_us();
        let iter = self
            .engine
            .prefix_scan(Partition::Schema, trigger_pending_scan_prefix())
            .map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for guard in iter {
            let Ok((key, value)) = guard.into_inner() else {
                continue;
            };
            let Some(seq) = decode_trigger_event_seq(&key) else {
                continue;
            };
            if seen.contains(&seq) {
                continue;
            }
            let event: PendingTriggerEvent = match rmp_serde::from_slice(&value) {
                Ok(ev) => ev,
                // A corrupt queue entry cannot be executed; skip it (it stays
                // in the queue for an operator to inspect rather than vanishing).
                Err(_) => continue,
            };
            if event.next_attempt_us <= now {
                out.push((seq, event));
            }
        }
        Ok(out)
    }

    /// Execute one queued event and apply its outcome to the queue.
    fn process_pending_event(&self, seq: u64, event: PendingTriggerEvent) -> Outcome {
        let name = event.trigger_name.clone();

        // Live definition lookup: ALTER/DROP/DISABLE take effect on queued events.
        let def = match self.load_trigger_def(&name) {
            Ok(Some(d)) => d,
            Ok(None) => {
                // Trigger was dropped — discard its queued events.
                return match self.delete_pending(&name, seq) {
                    Ok(()) => Outcome::Discarded,
                    Err(e) => Outcome::Err(format!("discard `{name}`#{seq}: {e}")),
                };
            }
            Err(e) => return Outcome::Err(format!("load trigger `{name}`: {e}")),
        };
        if !def.enabled {
            // Disabled: leave queued, resume when re-enabled. `seen` keeps this
            // pass from re-collecting it.
            return Outcome::Skipped;
        }

        // L1 async cascade depth bound.
        let max_depth = self.trigger_dispatch_config.max_cascade_depth;
        if event.generation > max_depth {
            let chain = vec![format!(
                "async cascade depth {} exceeds limit {max_depth}",
                event.generation
            )];
            return self.dead_letter(seq, &event, chain, true);
        }

        let params: HashMap<String, Value> = event
            .params
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        match self.execute_after_commit_body(&def.body_source, params, event.generation) {
            Ok(()) => match self.delete_pending(&name, seq) {
                Ok(()) => Outcome::Fired,
                Err(e) => Outcome::Err(format!("ack `{name}`#{seq}: {e}")),
            },
            Err(exec_err) => self.apply_error_policy(seq, &event, &def, exec_err.to_string()),
        }
    }

    /// Apply the trigger's resolved `ON ERROR` policy after a body failure.
    fn apply_error_policy(
        &self,
        seq: u64,
        event: &PendingTriggerEvent,
        def: &TriggerSchema,
        error: String,
    ) -> Outcome {
        match self.resolve_policy(def) {
            // PROPAGATE on a committed transaction can't abort anything — the
            // arch routes it straight to the dead-letter surface. DEAD_LETTER is
            // the explicit form of the same.
            OnErrorPolicySchema::Propagate | OnErrorPolicySchema::DeadLetter => {
                self.dead_letter(seq, event, vec![error], false)
            }
            OnErrorPolicySchema::Retry { n, backoff_ms } => {
                let attempts_done = event.attempt.saturating_add(1);
                if attempts_done >= n {
                    self.dead_letter(seq, event, vec![error], false)
                } else {
                    // Exponential backoff: base * 2^attempt, capped to avoid overflow.
                    let shift = event.attempt.min(20);
                    let backoff_ms = (backoff_ms as u64).saturating_mul(1u64 << shift);
                    let updated = PendingTriggerEvent {
                        attempt: attempts_done,
                        next_attempt_us: now_us().saturating_add(backoff_ms.saturating_mul(1000)),
                        ..event.clone()
                    };
                    match self.overwrite_pending(seq, &updated) {
                        Ok(()) => Outcome::Retried,
                        Err(e) => {
                            Outcome::Err(format!("reschedule `{}`#{seq}: {e}", event.trigger_name))
                        }
                    }
                }
            }
        }
    }

    /// Resolve a trigger's effective `ON ERROR` policy, applying the configured
    /// AFTER COMMIT default (`RETRY default_retry_attempts WITH BACKOFF
    /// default_backoff_ms`) when the trigger declares none.
    fn resolve_policy(&self, def: &TriggerSchema) -> OnErrorPolicySchema {
        def.on_error.clone().unwrap_or(OnErrorPolicySchema::Retry {
            n: self.trigger_dispatch_config.default_retry_attempts,
            backoff_ms: self
                .trigger_dispatch_config
                .default_backoff_ms
                .min(u32::MAX as u64) as u32,
        })
    }

    /// Run a trigger body in a fresh auto-commit transaction at the given async
    /// cascade generation. Writes replicate through the proposal pipeline like
    /// any user statement.
    fn execute_after_commit_body(
        &self,
        body: &str,
        params: HashMap<String, Value>,
        generation: u32,
    ) -> Result<(), DatabaseError> {
        let session = QuerySession {
            read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::default(),
            snapshot_read_ts: None,
            write_concern: coordinode_core::txn::write_concern::WriteConcern::default(),
            vector_consistency: coordinode_core::graph::types::VectorConsistencyMode::default(),
            after_commit_generation: generation,
        };
        let params = if params.is_empty() {
            None
        } else {
            Some(params)
        };
        self.execute_cypher_impl(body, None, params, &session, TxnMode::AutoCommit, &mut None)
            .map(|_| ())
    }

    // ── Queue bookkeeping (raw schema-partition mutations through the pipeline) ──

    /// Read and decode a trigger definition straight from the engine (committed
    /// state). `None` when the trigger has been dropped.
    fn load_trigger_def(&self, name: &str) -> Result<Option<TriggerSchema>, String> {
        let bytes = self
            .engine
            .get(Partition::Schema, &encode_trigger_key(name))
            .map_err(|e| e.to_string())?;
        match bytes {
            Some(b) => rmp_serde::from_slice(&b)
                .map(Some)
                .map_err(|e| format!("decode trigger `{name}`: {e}")),
            None => Ok(None),
        }
    }

    /// Remove a queued event (successful ack or discard).
    fn delete_pending(&self, name: &str, seq: u64) -> Result<(), DatabaseError> {
        self.submit_schema_mutations(vec![Mutation::Delete {
            partition: PartitionId::Schema,
            key: encode_trigger_pending_key(name, seq),
        }])
    }

    /// Overwrite a queued event in place (retry reschedule).
    fn overwrite_pending(
        &self,
        seq: u64,
        event: &PendingTriggerEvent,
    ) -> Result<(), DatabaseError> {
        let value = rmp_serde::to_vec(event)
            .map_err(|e| DatabaseError::Other(format!("encode pending event: {e}")))?;
        self.submit_schema_mutations(vec![Mutation::Put {
            partition: PartitionId::Schema,
            key: encode_trigger_pending_key(&event.trigger_name, seq),
            value,
        }])
    }

    /// Atomically move a queued event to the dead-letter surface: write the
    /// failure record and delete the pending entry in one proposal.
    fn dead_letter(
        &self,
        seq: u64,
        event: &PendingTriggerEvent,
        error_chain: Vec<String>,
        cascade_overflow: bool,
    ) -> Outcome {
        let now = now_us();
        let failure = FailedTriggerEvent {
            trigger_name: event.trigger_name.clone(),
            params: event.params.clone(),
            error_chain,
            attempts: event.attempt.saturating_add(1),
            first_fail_us: if event.attempt == 0 {
                now
            } else {
                event.first_seen_us
            },
            last_fail_us: now,
            cascade_overflow,
        };
        let value = match rmp_serde::to_vec(&failure) {
            Ok(v) => v,
            Err(e) => return Outcome::Err(format!("encode failure record: {e}")),
        };
        let muts = vec![
            Mutation::Put {
                partition: PartitionId::Schema,
                key: encode_trigger_failure_key(&event.trigger_name, seq),
                value,
            },
            Mutation::Delete {
                partition: PartitionId::Schema,
                key: encode_trigger_pending_key(&event.trigger_name, seq),
            },
        ];
        match self.submit_schema_mutations(muts) {
            Ok(()) => Outcome::DeadLettered,
            Err(e) => Outcome::Err(format!("dead-letter `{}`#{seq}: {e}", event.trigger_name)),
        }
    }

    /// Submit raw schema-partition mutations through the proposal pipeline so
    /// queue bookkeeping replicates with the rest of the metadata.
    fn submit_schema_mutations(&self, mutations: Vec<Mutation>) -> Result<(), DatabaseError> {
        if mutations.is_empty() {
            return Ok(());
        }
        let proposal = RaftProposal {
            id: self.proposal_id_gen.next(),
            mutations,
            commit_ts: Timestamp::from_raw(0),
            start_ts: Timestamp::from_raw(0),
            bypass_rate_limiter: true,
        };
        self.pipeline
            .propose_and_wait(&proposal)
            .map(|_| ())
            .map_err(|e| DatabaseError::Other(format!("trigger queue write: {e}")))
    }

    /// Drive the AFTER COMMIT queue inline after a committed embedded write.
    /// No-op in cluster mode (the leader-gated worker owns the drain) and when
    /// already dispatching on this thread.
    pub(super) fn drive_after_commit_inline(&self) {
        if self.cluster_mode {
            return;
        }
        let report = self.dispatch_after_commit_triggers();
        for err in &report.errors {
            tracing::warn!("after-commit trigger dispatch: {err}");
        }
    }
}

/// Wall-clock microseconds since the Unix epoch — the queue's scheduling clock.
fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// RAII reset for the reentrancy flag.
struct DispatchGuard;
impl Drop for DispatchGuard {
    fn drop(&mut self) {
        IN_DISPATCH.with(|f| f.set(false));
    }
}
