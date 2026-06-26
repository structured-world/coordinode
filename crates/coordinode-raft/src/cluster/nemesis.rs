//! Test-only network nemesis: a process-global directed-partition matrix the
//! Raft network layer consults before each outbound RPC, so integration tests
//! (R147 Jepsen suite) can inject network partitions into an in-process cluster.
//!
//! **Production-safe by construction.** An `AtomicBool` gate (`ENABLED`) is the
//! only thing the hot path touches when no test has armed the nemesis: a single
//! relaxed load returning `false`, after which [`is_blocked`] short-circuits
//! without locking. The matrix itself is never allocated until a test calls
//! [`block`] / [`isolate`]. Production code never arms it, so it stays a no-op.
//!
//! Not behind `#[cfg(test)]` because integration tests live in a separate crate
//! (`tests/r147_jepsen.rs`) and cannot see the library's test-cfg items; the
//! atomic gate makes the always-compiled form free in production.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

/// Armed flag. `false` (default) → [`is_blocked`] is a single relaxed load.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Directed blocked links `(from_node, to_node)`. Allocated on first `block`.
static BLOCKED: OnceLock<RwLock<HashSet<(u64, u64)>>> = OnceLock::new();

fn matrix() -> &'static RwLock<HashSet<(u64, u64)>> {
    BLOCKED.get_or_init(|| RwLock::new(HashSet::new()))
}

/// Block the given directed links and arm the nemesis. An RPC from `from` to
/// `to` is then reported unreachable by the network layer.
pub fn block(pairs: &[(u64, u64)]) {
    {
        // Poison-tolerant: a test that panicked mid-update must not wedge the
        // nemesis for the rest of the process — recover the guard and continue.
        let mut g = matrix().write().unwrap_or_else(|e| e.into_inner());
        for &p in pairs {
            g.insert(p);
        }
    }
    ENABLED.store(true, Ordering::Relaxed);
}

/// Symmetrically isolate `node` from every peer in `peers` (both directions) —
/// the common "partition this node off" nemesis.
pub fn isolate(node: u64, peers: &[u64]) {
    let mut pairs = Vec::with_capacity(peers.len() * 2);
    for &p in peers {
        pairs.push((node, p));
        pairs.push((p, node));
    }
    block(&pairs);
}

/// Heal all partitions and disarm the nemesis.
pub fn heal() {
    if let Some(m) = BLOCKED.get() {
        m.write().unwrap_or_else(|e| e.into_inner()).clear();
    }
    ENABLED.store(false, Ordering::Relaxed);
}

/// Whether an outbound RPC from `from` to `to` is currently partitioned.
/// Hot-path: a single relaxed load when disarmed.
#[inline]
pub fn is_blocked(from: u64, to: u64) -> bool {
    if !ENABLED.load(Ordering::Relaxed) {
        return false;
    }
    matrix()
        .read()
        .map(|m| m.contains(&(from, to)))
        .unwrap_or(false)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn disarmed_by_default_then_block_and_heal() {
        // Note: process-global; this test owns the keys it uses and heals after.
        assert!(!is_blocked(91, 92));
        block(&[(91, 92)]);
        assert!(is_blocked(91, 92));
        assert!(!is_blocked(92, 91), "block is directed");
        isolate(93, &[94]);
        assert!(
            is_blocked(93, 94) && is_blocked(94, 93),
            "isolate is symmetric"
        );
        heal();
        assert!(!is_blocked(91, 92) && !is_blocked(93, 94));
    }
}
