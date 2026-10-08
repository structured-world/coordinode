//! NodeId leases granted through the database's proposal pipeline.
//!
//! A lease is a [`MetadataCommand::GrantNodeLease`] the ordered application
//! decides: it takes `(base, target]` only while the last granted ceiling is
//! still `base`, and records the grant as a write-once record keyed by its
//! ceiling. The reserver reads that record back after its proposal applied
//! and owns the range only when it finds its own token there. The records are
//! not `meta:` keys, because those are left out of Raft snapshots and every
//! member must see every grant.

use std::sync::Arc;
use std::thread::JoinHandle;

// no-std: spin::Mutex; the condvar has no no-std counterpart, the worker is std-only.
use parking_lot::{Condvar, Mutex};

use coordinode_core::graph::node::{
    IdLease, IdLeaseError, IdLeaseReserver, NODE_ID_MAX_SEQUENCE, NODE_LEASE_TOKEN_LEN,
};
use coordinode_core::txn::proposal::{
    MetadataCommand, Mutation, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::metadata::{node_lease_ceiling, node_lease_holder};

/// Sequences per lease: one log round trip per this many created nodes.
pub(crate) const NODE_LEASE_SIZE: u64 = 10_000;

/// Grants lost to concurrent reservers before a reservation gives up. Each
/// loss means another member granted a range in the meantime, so this bounds
/// a contention that only a change of leader produces.
const MAX_GRANT_ATTEMPTS: u32 = 16;

/// Takes NodeId leases through a proposal pipeline.
pub(crate) struct LogLeaseReserver {
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    proposal_ids: Arc<ProposalIdGenerator>,
    oracle: Arc<TimestampOracle>,
    /// One grant in flight per process: two from here would read the same
    /// base and one of them would lose a round trip for nothing.
    granting: Mutex<()>,
}

impl LogLeaseReserver {
    pub(crate) fn new(
        engine: Arc<StorageEngine>,
        pipeline: Arc<dyn ProposalPipeline>,
        proposal_ids: Arc<ProposalIdGenerator>,
        oracle: Arc<TimestampOracle>,
    ) -> Self {
        Self {
            engine,
            pipeline,
            proposal_ids,
            oracle,
            granting: Mutex::new(()),
        }
    }

    /// The highest sequence the log has granted so far.
    pub(crate) fn ceiling(&self) -> Result<u64, IdLeaseError> {
        node_lease_ceiling(&self.engine)
            .map_err(|e| IdLeaseError::NotGranted(format!("read the lease records: {e}")))
    }

    /// Propose the grant `(base, target]` under `token` and report whether
    /// this proposal is the one the log applied.
    fn grant(
        &self,
        base: u64,
        target: u64,
        token: Option<[u8; NODE_LEASE_TOKEN_LEN]>,
    ) -> Result<bool, IdLeaseError> {
        let id = self.proposal_ids.next();
        // Held as not yet logged until the entry is handed over, so no
        // closed bound passes over it meanwhile.
        let (commit_ts, _held) = self
            .engine
            .pending_commits()
            .obligate(|| self.oracle.next().as_raw());
        let commit_ts = Timestamp::from_raw(commit_ts);
        let token = token.unwrap_or_else(|| {
            let mut token = [0u8; NODE_LEASE_TOKEN_LEN];
            token[..8].copy_from_slice(&id.as_raw().to_be_bytes());
            token[8..].copy_from_slice(&commit_ts.as_raw().to_be_bytes());
            token
        });

        self.pipeline
            .propose_and_wait(&RaftProposal {
                id,
                mutations: vec![Mutation::Command(MetadataCommand::GrantNodeLease {
                    base,
                    ceiling: target,
                    token,
                })],
                commit_ts,
                start_ts: Timestamp::from_raw(0),
                bypass_rate_limiter: false,
            })
            .map_err(|e| IdLeaseError::NotGranted(e.to_string()))?;

        let holder = node_lease_holder(&self.engine, target)
            .map_err(|e| IdLeaseError::NotGranted(format!("read the lease record: {e}")))?;
        Ok(holder == Some(token))
    }

    /// Take `(base, target]` under `token` while the granted ceiling is
    /// still `base`, so no allocator of the group ever hands out a sequence
    /// of it: a restore takes the sequences of the identifiers it writes this
    /// way. `false` when another grant moved the ceiling first, since a range
    /// taken in between may hold sequences the caller checked as free.
    pub(crate) fn raise_from(
        &self,
        base: u64,
        target: u64,
        token: [u8; NODE_LEASE_TOKEN_LEN],
    ) -> Result<bool, IdLeaseError> {
        if target > NODE_ID_MAX_SEQUENCE {
            return Err(IdLeaseError::Exhausted { shard_hint: 0 });
        }
        let _granting = self.granting.lock();
        if self.ceiling()? != base {
            return Ok(false);
        }
        self.grant(base, target, Some(token))
    }
}

impl IdLeaseReserver for LogLeaseReserver {
    fn reserve(&self) -> Result<IdLease, IdLeaseError> {
        let _granting = self.granting.lock();
        for _ in 0..MAX_GRANT_ATTEMPTS {
            let base = self.ceiling()?;
            if base >= NODE_ID_MAX_SEQUENCE {
                return Err(IdLeaseError::Exhausted { shard_hint: 0 });
            }
            // base < 2^44, so the sum cannot overflow a u64.
            let target = (base + NODE_LEASE_SIZE).min(NODE_ID_MAX_SEQUENCE);
            if self.grant(base, target, None)? {
                return Ok(IdLease {
                    base,
                    ceiling: target,
                });
            }
            // Another grant applied first from the same base: retry above it.
        }
        Err(IdLeaseError::NotGranted(format!(
            "lost {MAX_GRANT_ATTEMPTS} grants in a row to concurrent reservers"
        )))
    }
}

/// Keeps one spare lease so that switching to the next lease costs no round
/// trip: when the allocator takes the spare, a background worker asks the log
/// for the next one straight away. A draw waits on the log only when no spare
/// is in hand (the first CREATE after open, or after the member stopped and
/// started leading again).
pub(crate) struct PrefetchingReserver {
    shared: Arc<Prefetch>,
    worker: Option<JoinHandle<()>>,
}

struct Prefetch {
    source: LogLeaseReserver,
    state: Mutex<SpareState>,
    wake: Condvar,
}

#[derive(Default)]
struct SpareState {
    spare: Option<IdLease>,
    /// The spare was taken and the worker should fetch the next one.
    wanted: bool,
    stop: bool,
}

impl PrefetchingReserver {
    pub(crate) fn start(source: LogLeaseReserver) -> std::io::Result<Self> {
        let shared = Arc::new(Prefetch {
            source,
            state: Mutex::new(SpareState::default()),
            wake: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("coordinode-id-lease".into())
            .spawn(move || worker_shared.run())?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }
}

impl Prefetch {
    fn run(&self) {
        loop {
            {
                let mut state = self.state.lock();
                while !state.stop && !state.wanted {
                    self.wake.wait(&mut state);
                }
                if state.stop {
                    return;
                }
                state.wanted = false;
            }
            match self.source.reserve() {
                Ok(lease) => self.state.lock().spare = Some(lease),
                // A member that does not lead gets no lease; the next draw
                // asks again, in the foreground.
                Err(e) => tracing::debug!(error = %e, "no spare NodeId lease"),
            }
        }
    }

    fn want_next(&self, state: &mut SpareState) {
        state.wanted = true;
        self.wake.notify_one();
    }
}

impl IdLeaseReserver for PrefetchingReserver {
    fn reserve(&self) -> Result<IdLease, IdLeaseError> {
        {
            let mut state = self.shared.state.lock();
            if let Some(lease) = state.spare.take() {
                self.shared.want_next(&mut state);
                return Ok(lease);
            }
        }
        let lease = self.shared.source.reserve()?;
        let mut state = self.shared.state.lock();
        if state.spare.is_none() {
            self.shared.want_next(&mut state);
        }
        Ok(lease)
    }
}

impl Drop for PrefetchingReserver {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock();
            state.stop = true;
            self.shared.wake.notify_one();
        }
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                tracing::error!("the NodeId lease worker panicked");
            }
        }
    }
}
