//! Loom model-check for [`EntryPoint`] CAS-loop promotion.
//!
//! Exhaustively enumerates every legal interleaving of two concurrent
//! `try_promote` callers and asserts the post-condition spelled out in
//! `arch/search/vector-parallel-insert.md` §"Layer-promotion
//! linearisability":
//!
//! > Concurrent inserts at the same novel max-level produce one
//! > entry-point; the loser's level becomes the highest-level
//! > non-entry. CAS-loop on `entry_point` with packed (level, id)
//! > payload.
//!
//! Build with:
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test --test loom_entry_point --release
//! ```
//!
//! Loom complexity grows factorially in the number of atomic
//! operations per thread, so each scenario sticks to ≤ 2 threads and
//! one CAS-loop iteration of `try_promote`.

#![cfg(loom)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_vector::hnsw::{LoomEntryPoint as EntryPoint, LoomPromoteOutcome as PromoteOutcome};
use loom::sync::Arc;
use loom::thread;

/// First-insert race: two threads attempt to seed an empty EntryPoint
/// simultaneously. Exactly ONE must report `Installed` and that
/// thread's `(level, idx)` is the final state. The loser reports
/// `NotNeeded { current_level }` where `current_level` matches the
/// winner's level. No outcome can leave the EntryPoint empty.
#[test]
fn first_insert_seed_race() {
    loom::model(|| {
        let ep: Arc<EntryPoint> = Arc::new(EntryPoint::new());

        let a = {
            let ep = ep.clone();
            thread::spawn(move || ep.try_promote(3, 10))
        };
        let b = {
            let ep = ep.clone();
            thread::spawn(move || ep.try_promote(3, 20))
        };

        let r_a = a.join().unwrap();
        let r_b = b.join().unwrap();

        // Exactly one Installed. Either order is valid; loom will
        // explore both.
        let installed_count = [r_a, r_b]
            .iter()
            .filter(|r| matches!(r, PromoteOutcome::Installed))
            .count();
        assert_eq!(
            installed_count, 1,
            "exactly one promote must Install on the seed race (got {r_a:?}, {r_b:?})"
        );

        // Final state matches whichever thread won.
        let (level, idx) = ep.load().expect("EntryPoint must be set after the race");
        assert_eq!(level, 3, "winner level must be 3");
        assert!(
            idx == 10 || idx == 20,
            "winner idx must be one of {{10, 20}}"
        );

        // The loser must have observed the winner's level (3).
        for r in [r_a, r_b] {
            if let PromoteOutcome::NotNeeded { current_level } = r {
                assert_eq!(
                    current_level, 3,
                    "loser must observe winner's level on the seed race"
                );
            }
        }
    });
}

/// Promotion race: EntryPoint is pre-seeded; two threads race to
/// promote past it with different levels. The thread with the higher
/// level must win; the thread with the lower level must observe the
/// new top and NOT install.
#[test]
fn higher_level_wins_concurrent_promote() {
    loom::model(|| {
        let ep: Arc<EntryPoint> = Arc::new(EntryPoint::new());
        // Pre-seed serially — loom does not need to enumerate this
        // since it happens before any spawn.
        ep.try_promote(2, 0);

        let a = {
            let ep = ep.clone();
            thread::spawn(move || ep.try_promote(5, 10)) // higher
        };
        let b = {
            let ep = ep.clone();
            thread::spawn(move || ep.try_promote(4, 20)) // lower
        };

        let r_a = a.join().unwrap();
        let r_b = b.join().unwrap();

        // Final state must have level 5, idx 10 — the higher-level
        // thread MUST eventually install regardless of CAS ordering.
        let (final_level, final_idx) = ep.load().unwrap();
        assert_eq!(final_level, 5, "highest layer (5) must own the entry-point");
        assert_eq!(
            final_idx, 10,
            "winning thread's idx must be the final entry"
        );

        // r_a (level 5) must always Install (it's the highest).
        assert!(
            matches!(r_a, PromoteOutcome::Installed),
            "level-5 thread must Install (got {r_a:?})"
        );
        // r_b (level 4) may have raced with r_a:
        //   - If r_b ran first, it installs, then r_a installs over it → r_b reports Installed.
        //   - If r_a ran first, r_b sees level 5 and reports NotNeeded.
        // Both are valid; the post-condition (final = level 5) holds in either case.
        match r_b {
            PromoteOutcome::Installed => { /* r_b ran before r_a */ }
            PromoteOutcome::NotNeeded { current_level } => {
                assert!(
                    current_level >= 4,
                    "lower-level loser must observe a level ≥ its own (got {current_level})"
                );
            }
        }
    });
}

/// Tie race: two threads attempt to promote with the same level the
/// EntryPoint already holds. NEITHER should install — both report
/// NotNeeded with the pre-existing level.
#[test]
fn equal_level_promote_is_idempotent_under_race() {
    loom::model(|| {
        let ep: Arc<EntryPoint> = Arc::new(EntryPoint::new());
        ep.try_promote(7, 100);

        let a = {
            let ep = ep.clone();
            thread::spawn(move || ep.try_promote(7, 200))
        };
        let b = {
            let ep = ep.clone();
            thread::spawn(move || ep.try_promote(7, 300))
        };

        let r_a = a.join().unwrap();
        let r_b = b.join().unwrap();

        // Both report NotNeeded; the pre-existing (7, 100) stays.
        for r in [r_a, r_b] {
            assert!(
                matches!(r, PromoteOutcome::NotNeeded { current_level: 7 }),
                "equal-level promote must be NotNeeded (got {r:?})"
            );
        }
        assert_eq!(ep.load(), Some((7, 100)), "pre-existing entry must survive");
    });
}
