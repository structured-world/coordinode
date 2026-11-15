use std::sync::Arc;

use super::*;
use crate::engine::config::{Durability, Media, Tier};

const UNIT: u64 = 4096;

fn endpoint(dir: &Path, limit: u64) -> EndpointConfig {
    EndpointConfig::new("ep", dir, Media::Ssd, Durability::Durable, Tier::Warm)
        .with_hard_limit_bytes(limit)
        .with_recovery_reserve_bytes(0)
}

fn opened(dir: &Path, limit: u64) -> Arc<Footprint> {
    Footprint::open(&endpoint(dir, limit)).unwrap()
}

/// A removal that stopped partway gives back only the files that are gone,
/// reads the disk without the file table locked, and leaves a name charged
/// again meanwhile with its new charge.
#[test]
fn a_partial_removal_reads_the_disk_unlocked_and_keeps_a_recharged_name() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 64 * UNIT);
    let under = dir.path().join("ckpt");
    std::fs::create_dir_all(&under).unwrap();
    let charge = |name: &str| {
        let path = under.join(name);
        std::fs::write(&path, b"x").unwrap();
        footprint.file(&path).grow_to(UNIT).unwrap();
        path
    };
    let gone = charge("gone");
    let kept = charge("kept");
    let reused = charge("reused");
    std::fs::remove_file(&gone).unwrap();
    std::fs::remove_file(&reused).unwrap();
    assert_eq!(footprint.charged(), 3 * UNIT);

    let during = Arc::clone(&footprint);
    let other = dir.path().join("other");
    let renamed = reused.clone();
    footprint.on_removed_partly_checking(move || {
        // The table is free while the disk is read: another thread charges
        // a file, and the reused name is charged anew.
        let (done, waited) = std::sync::mpsc::channel();
        let elsewhere = Arc::clone(&during);
        let path = other.clone();
        std::thread::spawn(move || {
            elsewhere.file(&path).grow_to(UNIT).unwrap();
            done.send(()).unwrap();
        });
        waited
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the file table was free");
        // Charged anew under the same name before its file is created: the
        // disk still shows no file, but the charge is not the one read.
        during.removed(&renamed);
        during.file(&renamed).grow_to(2 * UNIT).unwrap();
    });
    footprint.removed_partly(&under);
    assert!(kept.exists(), "the kept file is still on disk");

    // gone: given back; kept: still charged; reused: its new 2 units; other:
    // the unit charged meanwhile.
    assert_eq!(footprint.charged(), UNIT + 2 * UNIT + UNIT);
}

/// A file cannot grow past the limit: the growth that would pass it is
/// refused with StorageFull and charges nothing.
#[test]
fn growth_past_the_limit_is_refused_and_charges_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 4 * UNIT);
    let file = footprint.file(&dir.path().join("a"));
    file.grow_to(3 * UNIT).unwrap();
    let refused = file.grow_to(5 * UNIT).unwrap_err();
    assert_eq!(refused.kind(), io::ErrorKind::StorageFull);
    assert_eq!(footprint.charged(), 3 * UNIT);
    assert_eq!(file.bytes(), 3 * UNIT);
}

/// Growth is charged in whole allocation units: one byte takes a unit.
#[test]
fn growth_is_charged_in_whole_allocation_units() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 64 * UNIT);
    let file = footprint.file(&dir.path().join("a"));
    file.grow_to(1).unwrap();
    assert_eq!(footprint.charged(), footprint.allocation(1));
    assert!(footprint.charged() >= 1);
    // Growing within the same unit charges nothing more.
    let before = footprint.charged();
    file.grow_to(2).unwrap();
    assert_eq!(footprint.charged(), before);
}

/// Many writers growing at once never pass the limit together: each growth
/// is checked and charged in one step.
#[test]
fn concurrent_growth_never_passes_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let limit = 64 * UNIT;
    let footprint = opened(dir.path(), limit);
    let granted = std::sync::atomic::AtomicU64::new(0);
    std::thread::scope(|scope| {
        for writer in 0..16 {
            let footprint = &footprint;
            let granted = &granted;
            let root = dir.path();
            scope.spawn(move || {
                for n in 0..16 {
                    let file = footprint.file(&root.join(format!("{writer}-{n}")));
                    if file.grow_to(UNIT).is_ok() {
                        granted.fetch_add(1, Ordering::Relaxed);
                    }
                    // Kept charged: the names stay in the account.
                    std::mem::forget(file);
                }
            });
        }
    });
    let unit = footprint.allocation(1);
    assert!(footprint.charged() <= limit);
    assert_eq!(footprint.charged(), granted.into_inner() * unit);
}

/// A client write leaves the recovery reserve; a recovery record may use it.
/// An admission holds its bytes until it drops.
#[test]
fn client_admission_leaves_the_recovery_reserve() {
    let dir = tempfile::tempdir().unwrap();
    let footprint =
        Footprint::open(&endpoint(dir.path(), 100 * UNIT).with_recovery_reserve_bytes(20 * UNIT))
            .unwrap();
    let held = footprint.admit(80 * UNIT, Admit::Client).unwrap();
    assert_eq!(footprint.admitted(), 80 * UNIT);
    assert!(matches!(
        footprint.admit(1, Admit::Client),
        Err(StorageError::CapacityExhausted { .. })
    ));
    let recovery = footprint.admit(20 * UNIT, Admit::Recovery).unwrap();
    assert!(footprint.admit(1, Admit::Recovery).is_err());
    drop(recovery);
    drop(held);
    assert_eq!(footprint.admitted(), 0);
    assert!(footprint.admit(80 * UNIT, Admit::Client).is_ok());
}

/// The room maintenance sees is the limit less what is charged and what
/// admitted writes hold, and nothing more: the memtables' future flushes
/// are not held against it, so a store with unflushed data but free space
/// still compacts.
#[test]
fn the_room_is_what_charged_and_admitted_leave() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 100 * UNIT);
    footprint
        .file(&dir.path().join("a"))
        .grow_to(30 * UNIT)
        .unwrap();
    let held = footprint.admit(20 * UNIT, Admit::Client).unwrap();
    assert_eq!(footprint.committed(), 50 * UNIT);
    assert_eq!(footprint.room(), 50 * UNIT);
    drop(held);
    assert_eq!(footprint.room(), 70 * UNIT);
}

/// A failed flush's own output, given back on its thread while it ran, does
/// not count as room coming back, so a flush that cannot fit is not tried
/// again in a loop; an admission given back is no room for a flush either.
/// A file another thread gives back afterwards wakes it.
#[test]
fn a_failed_flush_waits_for_room_others_give_back() {
    let dir = tempfile::tempdir().unwrap();
    let footprints = Footprints::open(&[endpoint(dir.path(), 100 * UNIT)]).unwrap();
    let other = footprints.file(&dir.path().join("other")).unwrap();
    other.grow_to(UNIT).unwrap();
    let held = footprints.admit(10 * UNIT, Admit::Client).unwrap();

    let attempt = footprints.flush_attempt();
    let own = footprints.file(&dir.path().join("own")).unwrap();
    own.grow_to(UNIT).unwrap();
    drop(own);
    footprints.removed(&dir.path().join("own"));
    attempt.failed();
    assert!(footprints.flush_stalled(), "its own cleanup woke nothing");
    drop(held);
    assert!(footprints.flush_stalled(), "an admission woke nothing");

    std::thread::scope(|scope| {
        scope.spawn(|| {
            drop(other);
            footprints.removed(&dir.path().join("other"));
        });
    });
    assert!(!footprints.flush_stalled(), "room given back woke it");
}

/// Racing admissions never admit more than fits.
#[test]
fn concurrent_admissions_never_overcommit() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 100 * UNIT);
    let admitted: Vec<Admission> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..32)
            .map(|_| {
                let footprint = &footprint;
                scope.spawn(move || footprint.admit(7 * UNIT, Admit::Client).ok())
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| handle.join().unwrap())
            .collect()
    });
    assert_eq!(
        admitted.len(),
        14,
        "100 units hold fourteen admissions of 7"
    );
    assert!(footprint.admitted() <= 100 * UNIT);
}

/// A second name of a file is the same bytes: charged once, given back only
/// when the last name goes.
#[test]
fn a_hard_link_is_charged_once() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 64 * UNIT);
    let a = dir.path().join("a");
    let b = dir.path().join("b");
    footprint.file(&a).grow_to(4 * UNIT).unwrap();
    footprint.linked(&a, &b);
    assert_eq!(footprint.charged(), 4 * UNIT);
    footprint.removed(&a);
    assert_eq!(footprint.charged(), 4 * UNIT, "the file lives on under b");
    footprint.removed(&b);
    assert_eq!(footprint.charged(), 0);
}

/// An open handle keeps its file's bytes charged after the name is removed,
/// as the file system keeps the blocks of an open, unlinked file.
#[test]
fn an_open_handle_keeps_a_removed_file_charged() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 64 * UNIT);
    let a = dir.path().join("a");
    let open = footprint.file(&a);
    open.grow_to(2 * UNIT).unwrap();
    footprint.removed(&a);
    assert_eq!(footprint.charged(), 2 * UNIT);
    drop(open);
    assert_eq!(footprint.charged(), 0);
}

/// A rename moves the charge: nothing is given back or charged again, and a
/// replaced file's bytes are given back.
#[test]
fn a_rename_moves_the_charge_and_releases_the_replaced_file() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 64 * UNIT);
    let (a, b) = (dir.path().join("a"), dir.path().join("b"));
    footprint.file(&a).grow_to(3 * UNIT).unwrap();
    footprint.file(&b).grow_to(5 * UNIT).unwrap();
    footprint.renamed(&a, &b);
    assert_eq!(footprint.charged(), 3 * UNIT);
    assert_eq!(footprint.file(&b).bytes(), 3 * UNIT);
}

/// Opening an existing store charges each file with what the file system
/// allocated to it, a hard-linked file once.
#[cfg(unix)]
#[test]
fn open_charges_existing_files_by_allocation_and_links_once() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub/a"), vec![7u8; 10_000]).unwrap();
    std::fs::hard_link(dir.path().join("sub/a"), dir.path().join("b")).unwrap();
    std::fs::write(dir.path().join("c"), b"x").unwrap();
    let expected: u64 = ["sub/a", "c"]
        .iter()
        .map(|name| {
            use std::os::unix::fs::MetadataExt;
            std::fs::metadata(dir.path().join(name)).unwrap().blocks() * 512
        })
        .sum();
    let footprint = opened(dir.path(), 1 << 30);
    assert_eq!(footprint.charged(), expected);
    footprint.removed(&dir.path().join("sub/a"));
    footprint.removed(&dir.path().join("b"));
    footprint.removed(&dir.path().join("c"));
    assert_eq!(footprint.charged(), 0);
}

/// A limit lowered below what is held deletes nothing: the footprint keeps
/// its bytes, client writes are refused, the store's own work grows only by
/// the bounded over-limit room, and once the footprint fits the limit holds
/// again.
#[test]
fn lowering_the_limit_below_the_footprint_refuses_growth_and_deletes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 64 * UNIT);
    let file = footprint.file(&dir.path().join("a"));
    file.grow_to(10 * UNIT).unwrap();
    footprint.set_limit(4 * UNIT, 0);
    assert_eq!(footprint.charged(), 10 * UNIT, "nothing is deleted");
    let room = footprint.over_limit_room();
    assert_eq!(footprint.room(), room);
    assert!(matches!(
        footprint.admit(1, Admit::Client),
        Err(StorageError::CapacityExhausted { .. })
    ));
    assert!(
        file.grow_to(10 * UNIT + room + UNIT).is_err(),
        "the store's own work is bounded"
    );
    file.truncate_to(2 * UNIT);
    assert_eq!(footprint.charged(), 2 * UNIT);
    assert_eq!(footprint.pending_limit(), None, "the footprint fits again");
    assert!(file.grow_to(4 * UNIT).is_ok());
    assert!(file.grow_to(5 * UNIT).is_err(), "the limit holds again");
}

/// A store opened holding more than its limit opens: the limit is pending.
/// Client writes are refused, the store's own work grows only by the
/// over-limit room past what it holds, and the pending ceiling follows the
/// space freed down to the limit, where it ends.
#[cfg(unix)]
#[test]
fn a_store_opened_past_its_limit_holds_its_own_growth_to_a_bounded_room() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("old"), vec![1u8; 40 * UNIT as usize]).unwrap();
    let held = {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(dir.path().join("old")).unwrap().blocks() * 512
    };
    let footprint =
        Footprint::open(&endpoint(dir.path(), 10 * UNIT).with_recovery_reserve_bytes(2 * UNIT))
            .unwrap();
    let room = footprint.over_limit_room();
    assert_eq!(room, OVER_LIMIT_ALLOWANCE, "a small reserve is raised");
    assert_eq!(footprint.pending_limit(), Some(held + room));
    assert!(matches!(
        footprint.admit(1, Admit::Client),
        Err(StorageError::CapacityExhausted { .. })
    ));

    let work = footprint.file(&dir.path().join("manifest"));
    work.grow_to(room)
        .expect("the store's own work grows into the room");
    assert!(work.grow_to(room + UNIT).is_err(), "and no further");
    drop(work);
    footprint.removed(&dir.path().join("manifest"));

    // The old file goes: the ceiling comes down with it, to the limit.
    footprint.removed(&dir.path().join("old"));
    assert_eq!(footprint.pending_limit(), None, "the footprint fits again");
    assert_eq!(footprint.enforced_limit(), 10 * UNIT);
    assert!(footprint.admit(UNIT, Admit::Client).is_ok());
}

/// While a store opens, its own work grows past the limit by a bounded
/// allowance; when the open phase ends, a store holding more than its limit
/// is pending, one that fits is held to the limit again.
#[test]
fn an_open_phase_bounds_growth_past_the_limit_and_ends_in_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let footprints = Footprints::open(&[endpoint(dir.path(), 4 * UNIT)]).unwrap();
    let footprint = Arc::clone(footprints.get("ep").unwrap());
    let skeleton = footprint.file(&dir.path().join("skeleton"));
    {
        let _opening = footprints.open_phase();
        skeleton
            .grow_to(OVER_LIMIT_ALLOWANCE)
            .expect("an empty store's files fit the open allowance");
        assert!(
            skeleton.grow_to(OVER_LIMIT_ALLOWANCE + UNIT).is_err(),
            "the open allowance is a bound"
        );
    }
    assert_eq!(
        footprint.pending_limit(),
        Some(OVER_LIMIT_ALLOWANCE + footprint.over_limit_room()),
        "a store that does not fit its limit is pending once opened"
    );
    assert!(matches!(
        footprint.admit(1, Admit::Client),
        Err(StorageError::CapacityExhausted { .. })
    ));
}

/// A limit lowered below what the endpoint holds is pending at once, and the
/// ceiling never rises again on growth: it only follows space being freed.
#[test]
fn the_pending_ceiling_only_comes_down() {
    let dir = tempfile::tempdir().unwrap();
    let footprint = opened(dir.path(), 64 * UNIT);
    let file = footprint.file(&dir.path().join("a"));
    file.grow_to(20 * UNIT).unwrap();
    footprint.set_limit(8 * UNIT, UNIT);
    let room = footprint.over_limit_room();
    assert_eq!(footprint.pending_limit(), Some(20 * UNIT + room));
    file.truncate_to(15 * UNIT);
    assert_eq!(footprint.pending_limit(), Some(15 * UNIT + room));
    assert!(
        file.grow_to(15 * UNIT + room + UNIT).is_err(),
        "no growth past the lowered ceiling"
    );
    file.truncate_to(5 * UNIT);
    assert_eq!(footprint.pending_limit(), None, "the footprint fits again");
    assert_eq!(footprint.enforced_limit(), 8 * UNIT);
}

/// A path finds the footprint of the deepest endpoint root holding it, and
/// an endpoint without a hard limit has none.
#[test]
fn a_path_finds_the_footprint_of_its_endpoint() {
    let dir = tempfile::tempdir().unwrap();
    let outer = dir.path().join("outer");
    let inner = outer.join("inner");
    let free = dir.path().join("free");
    let footprints = Footprints::open(
        &[
            endpoint(&outer, 1 << 20),
            endpoint(&inner, 1 << 20).with_recovery_reserve_bytes(0),
            EndpointConfig::new("free", &free, Media::Ssd, Durability::Durable, Tier::Warm),
        ]
        .map(|mut e| {
            let id = e.path.file_name().unwrap().to_string_lossy().into_owned();
            e.id = id;
            e
        }),
    )
    .unwrap();
    assert_eq!(footprints.for_path(&inner.join("x")).unwrap().id(), "inner");
    assert_eq!(footprints.for_path(&outer.join("x")).unwrap().id(), "outer");
    assert!(footprints.for_path(&free.join("x")).is_none());
}
