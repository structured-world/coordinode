use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::Journal;

use super::{Grounds, Round};
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;

/// The first index of each journal segment under `root`, oldest first.
fn journal_segments(root: &std::path::Path) -> Vec<u64> {
    let mut starts: Vec<u64> = std::fs::read_dir(root.join("oplog").join("0"))
        .expect("journal dir")
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            name.strip_prefix("oplog-")?
                .strip_suffix(".bin")?
                .parse()
                .ok()
        })
        .collect();
    starts.sort_unstable();
    starts
}

/// A round that failed after its flushes, with no memtable, sealed or
/// active, left to flush: room given back on another thread wakes the
/// reclaimer itself, with no write and no new table, and the round that
/// follows removes the covered journal. The test runs the reclaimer's
/// rounds by hand on its own wake, so no other round runs meanwhile.
#[test]
fn a_reclamation_failed_with_nothing_to_flush_runs_again_when_room_comes_back() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )
        .with_hard_limit_bytes(crate::engine::config::MIN_HARD_LIMIT_BYTES),
    ]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open"));
    let footprints = engine.footprints().clone();
    let scratch = dir.path().join("scratch");
    let held = footprints.file(&scratch).expect("a charge");
    held.grow_to(64 << 10).expect("room");

    let landed = engine.table_landed();
    landed.bind();
    footprints.set_reclaim_wake(Arc::clone(&landed));
    // Written until a segment is sealed: a write admitted only after a
    // reclamation may have removed the earlier ones.
    let mut index = 0u64;
    while journal_segments(dir.path()).len() < 2 {
        assert!(index < 200, "no segment sealed");
        index += 1;
        let mut state = index.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let value = (0..64 << 10)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state.to_le_bytes()[0]
            })
            .collect();
        let write = [Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:1:w{index}").into_bytes(),
            value,
        }];
        engine
            .commit_logged(&write, oracle.next().as_raw(), Journal::Journal)
            .expect("commit");
    }

    // The round flushes every memtable, then fails before any removal. Room
    // background work gives back while it runs or after wakes it again at
    // once (no wait to test then): it is run until its failure stays
    // registered with no flush left running or to run. A persist waits out
    // any flush in flight and finds every memtable empty; a table such a
    // flush landed is spent with the wake, so only room coming back can
    // wake the reclaimer afterwards.
    let first = journal_segments(dir.path()).first().copied();
    engine.fail_journal_purge.store(true, Ordering::SeqCst);
    let mut last = None;
    let mut registered = false;
    for _ in 0..20 {
        landed.wait(Some(Duration::ZERO));
        engine.journal_reclaim_round(&mut last);
        assert!(last.is_some_and(|round| round.failed), "the round failed");
        engine.persist().expect("persist");
        landed.wait(Some(Duration::ZERO));
        if footprints.flush_stalled() {
            registered = true;
            break;
        }
    }
    assert!(registered, "the failed round waits for room");
    assert_eq!(
        journal_segments(dir.path()).first().copied(),
        first,
        "nothing removed"
    );
    engine.fail_journal_purge.store(false, Ordering::SeqCst);

    let footprints = footprints.clone();
    std::thread::spawn(move || {
        footprints.removed(&scratch);
        drop(held);
    })
    .join()
    .expect("room given back");
    let start = Instant::now();
    landed.wait(Some(Duration::from_secs(5)));
    assert!(
        start.elapsed() < Duration::from_secs(4),
        "room given back woke the reclaimer"
    );
    engine.journal_reclaim_round(&mut last);
    assert!(
        last.is_some_and(|round| !round.failed),
        "the round succeeded"
    );
    assert!(
        journal_segments(dir.path()).first().copied() != first,
        "the covered journal went: {:?}",
        journal_segments(dir.path())
    );
}

const BEFORE: Grounds = Grounds {
    newest_sealed: Some(40),
    floor: u64::MAX,
};

/// A round that succeeded runs again only on a new ground: a segment sealed
/// since it began (also one sealed while it ran, since it records what it
/// read before), or a floor raised. Its own removals, or a journal emptied
/// by it, are no ground.
#[test]
fn a_succeeded_round_runs_again_only_on_a_new_ground() {
    let round = Round {
        checked: BEFORE,
        failed: false,
    };
    assert!(!round.due(BEFORE), "nothing new");
    let emptied = Grounds {
        newest_sealed: None,
        ..BEFORE
    };
    assert!(
        !round.due(emptied),
        "its own removals emptied the sealed list"
    );
    let sealed = Grounds {
        newest_sealed: Some(56),
        ..BEFORE
    };
    assert!(round.due(sealed), "a segment sealed while it ran or since");
    let held = Round {
        checked: Grounds { floor: 0, ..BEFORE },
        failed: false,
    };
    let raised = Grounds { floor: 1, ..BEFORE };
    assert!(held.due(raised), "a checkpoint or consumer floor raised");
}

/// A failed round runs on the next wake: only an event wakes the reclaimer
/// (a table landing, which after a failure follows room coming back or a
/// raised limit), so the failure does not repeat by itself.
#[test]
fn a_failed_round_runs_on_the_next_wake() {
    let round = Round {
        checked: BEFORE,
        failed: true,
    };
    assert!(round.due(BEFORE), "the wake is the new ground");
}
