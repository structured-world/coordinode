use super::*;
use crate::engine::footprint::Footprints;

/// The marker written outside any engine: nothing to charge.
fn mark(dir: &Path, format: u32) -> StorageResult<()> {
    write_marker(dir, format, &Footprints::default())
}

fn touch(dir: &Path, name: &str) {
    std::fs::create_dir_all(dir).expect("dir");
    std::fs::write(dir.join(name), b"data").expect("write");
}

/// Marks what the step saw, so a test can tell it ran.
fn mark_step(dir: &Path) -> StorageResult<()> {
    std::fs::write(dir.join("migrated"), b"").map_err(|e| StorageError::Io(e.to_string()))
}

fn failing_step(_: &Path) -> StorageResult<()> {
    Err(StorageError::Io("step failed".into()))
}

fn unchanged(_: &Path) -> StorageResult<()> {
    Ok(())
}

static STEPS: &[MigrationStep] = &[
    MigrationStep {
        from: 0,
        migrate: unchanged,
    },
    MigrationStep {
        from: 4,
        migrate: mark_step,
    },
];

/// A config of durable endpoints at `dirs`, the first one primary.
fn endpoints(dirs: &[&Path]) -> StorageConfig {
    use crate::engine::config::{EndpointConfig, Media, Tier};
    StorageConfig::with_endpoints(
        dirs.iter()
            .enumerate()
            .map(|(i, dir)| {
                EndpointConfig::new(
                    format!("e{i}"),
                    dir,
                    Media::Hdd,
                    Durability::Durable,
                    Tier::Warm,
                )
            })
            .collect(),
    )
}

/// A restore claims every endpoint or none: one occupied directory, even a
/// secondary one, refuses the claim, and the primary is not created.
#[test]
fn a_restore_claim_needs_every_endpoint_empty() {
    let root = tempfile::tempdir().expect("tempdir");
    let (primary, secondary) = (root.path().join("p"), root.path().join("s"));
    touch(&secondary, "table");
    let config = endpoints(&[&primary, &secondary]);
    let refused = begin_restore(&config, 7).expect_err("an occupied endpoint");
    assert!(
        matches!(refused, StorageError::RestoreTarget { .. }),
        "{refused}"
    );
    assert!(!primary.exists(), "nothing was claimed");
}

/// Between the claim and its end only the open of that same restore is let
/// in; afterwards every open is.
#[test]
fn a_claimed_directory_opens_only_for_its_restore() {
    let root = tempfile::tempdir().expect("tempdir");
    let dir = root.path().join("data");
    let config = endpoints(&[&dir]);
    let claim = begin_restore(&config, 7).expect("claim");
    assert!(
        matches!(prepare(&config), Err(StorageError::StoreInUse { .. })),
        "the claim holds the directory"
    );
    drop(claim);
    assert!(matches!(
        prepare(&config),
        Err(StorageError::RestoreIncomplete { image: 7, .. })
    ));
    assert!(matches!(
        begin_restore(&config, 8),
        Err(StorageError::RestoreTarget { .. })
    ));
    let claim = begin_restore(&config, 7).expect("the same image claims it again");
    let restoring = prepare_restoring(&config, claim).expect("the restore itself");
    finish_restore(&config).expect("finish");
    drop(restoring);
    prepare(&config).expect("an ordinary open");
}

/// A restore owns its target from the check that found it empty until the
/// claim is written: neither another restore, even of the same image, nor an
/// ordinary open gets in between.
#[test]
fn nothing_gets_between_the_emptiness_check_and_the_claim() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = endpoints(&[&root.path().join("p"), &root.path().join("s")]);
    let mut first = true;
    let mut rival = None;
    let mut open = None;
    begin_restore_with(&config, 7, &mut || {
        if std::mem::take(&mut first) {
            rival = Some(begin_restore(&config, 7).is_ok());
            open = Some(prepare(&config).is_ok());
        }
        Ok(())
    })
    .expect("the first restore claims");
    assert_eq!(rival, Some(false), "a second restore got in");
    assert_eq!(open, Some(false), "an ordinary open got in");
}

/// A claim cut short at any of its writes leaves a target the same restore
/// takes up again, and that no ordinary open takes for a store once any of
/// the claim is on disk.
#[test]
fn a_claim_cut_short_at_any_write_is_taken_up_by_the_same_restore() {
    let root = tempfile::tempdir().expect("tempdir");
    for cut in 0.. {
        let base = root.path().join(format!("cut-{cut}"));
        let (primary, secondary) = (base.join("p"), base.join("s"));
        let config = endpoints(&[&primary, &secondary]);
        let mut calls = 0;
        let claimed = begin_restore_with(&config, 7, &mut || {
            calls += 1;
            if calls > cut {
                Err(StorageError::Io("cut".into()))
            } else {
                Ok(())
            }
        });
        if claimed.is_ok() {
            assert!(cut > 0, "the claim has steps");
            break;
        }
        let written = [&primary, &secondary]
            .iter()
            .any(|dir| !is_empty(dir).expect("list"));
        if written {
            assert!(
                prepare(&config).is_err(),
                "cut {cut}: an ordinary open took a partly claimed target"
            );
        }
        begin_restore(&config, 7)
            .unwrap_or_else(|e| panic!("cut {cut}: the same restore was refused: {e}"));
        assert!(
            prepare(&config).is_err(),
            "cut {cut}: claimed target opened"
        );
    }
}

/// Taking up an unfinished restore does not take in a directory the claim
/// never covered: with another endpoint in place of one it claimed, holding
/// a store, the restore is refused and that store left as it was.
#[test]
fn a_resumed_claim_does_not_take_in_another_endpoint() {
    let root = tempfile::tempdir().expect("tempdir");
    let (a, b, c) = (
        root.path().join("a"),
        root.path().join("b"),
        root.path().join("c"),
    );
    begin_restore(&endpoints(&[&a, &b]), 7).expect("claim a and b");
    touch(&c, "foreign-table");
    let refused = begin_restore(&endpoints(&[&a, &c]), 7);
    assert!(refused.is_err(), "a foreign endpoint was taken in");
    assert_eq!(
        std::fs::read_dir(&c).expect("list").count(),
        1,
        "the foreign endpoint was changed"
    );
}

/// Two endpoints naming one directory are refused by the restore and by an
/// ordinary open alike, rather than taken for a store holding itself.
#[test]
fn endpoints_naming_one_directory_are_refused() {
    let root = tempfile::tempdir().expect("tempdir");
    let dir = root.path().join("d");
    let alias = root.path().join("d").join("..").join("d");
    let config = endpoints(&[&dir, &alias]);
    assert!(matches!(
        begin_restore(&config, 7),
        Err(StorageError::InvalidConfig(_))
    ));
    assert!(matches!(
        prepare(&config),
        Err(StorageError::InvalidConfig(_))
    ));
}

/// A stop part way through ending a restore leaves the primary's claim,
/// which covers its directories: the same restore takes them up again,
/// whatever a secondary holds by then.
#[test]
fn a_restore_stopped_while_ending_is_taken_up_again() {
    let root = tempfile::tempdir().expect("tempdir");
    let (primary, secondary) = (root.path().join("p"), root.path().join("s"));
    let config = endpoints(&[&primary, &secondary]);
    drop(begin_restore(&config, 7).expect("claim"));
    touch(&secondary, "table");
    std::fs::remove_file(secondary.join(RESTORE_FILE)).expect("secondary claim ended");
    assert!(
        prepare(&config).is_err(),
        "the primary's claim still stands"
    );
    drop(begin_restore(&config, 7).expect("taken up again"));
}

/// A restore that stops while it ends never leaves one of its directories
/// free for another store: whatever claim went first, an open with another
/// primary over its secondary is refused, and the store itself still opens,
/// or its restore is taken up again, under its own configuration.
#[test]
fn a_restore_ending_part_way_keeps_its_secondary_bound() {
    let root = tempfile::tempdir().expect("tempdir");
    let (a, b, c) = (
        root.path().join("a"),
        root.path().join("b"),
        root.path().join("c"),
    );
    let config = endpoints(&[&a, &b]);
    let restoring =
        prepare_restoring(&config, begin_restore(&config, 7).expect("claim")).expect("open");
    touch(&b, "table");
    let stopped = finish_restore_with(&config, &mut || Err(StorageError::Io("cut".into())));
    assert!(
        stopped.is_err(),
        "the ending was cut after its first removal"
    );
    drop(restoring);

    assert!(
        prepare(&endpoints(&[&c, &b])).is_err(),
        "the secondary opened under another primary"
    );
    // The primary's claim went first: the store is published. It opens under
    // its own configuration, which clears the secondary's leftover claim,
    // and a restore over it is refused as over any store.
    let opened = prepare(&config).expect("the published store opens");
    assert!(
        read_claim(&b).expect("read").is_none(),
        "the leftover claim is cleared"
    );
    drop(opened);
    assert!(
        matches!(
            begin_restore(&config, 7),
            Err(StorageError::RestoreTarget { .. })
        ),
        "a restore over the published store was let in"
    );
}

/// A claim the ordinary open cannot see is no claim: with a volatile first
/// endpoint, either the restore is refused or the open still refuses the
/// claimed directory.
#[test]
fn a_claim_is_seen_by_every_open_of_its_configuration() {
    use crate::engine::config::{EndpointConfig, Media, Tier};
    let root = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "mem",
            root.path().join("mem"),
            Media::Hdd,
            Durability::Volatile,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "disk",
            root.path().join("disk"),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        ),
    ]);
    let claimed = begin_restore(&config, 7).is_ok();
    assert!(
        !claimed || prepare(&config).is_err(),
        "an ordinary open did not see the claim"
    );
}

#[test]
fn a_new_directory_is_marked_with_the_current_format() {
    let root = tempfile::tempdir().expect("tempdir");
    let dir = root.path().join("data");
    assert_eq!(
        prepare_dir(&dir, 5, STEPS).expect("prepare"),
        FormatOpen::Created
    );
    assert_eq!(read_marker(&dir).expect("read"), Some(5));
    assert_eq!(
        prepare_dir(&dir, 5, STEPS).expect("reopen"),
        FormatOpen::Current
    );
}

/// A directory with data and no marker predates the marker: format 0,
/// migrated by a release of format 1 and refused by any later one.
#[test]
fn an_unmarked_directory_with_data_is_format_zero() {
    let root = tempfile::tempdir().expect("tempdir");
    touch(root.path(), "schema");
    let opened = prepare_dir(root.path(), 1, STEPS).expect("migrate");
    assert!(matches!(
        opened,
        FormatOpen::Migrated { from: 0, to: 1, .. }
    ));
    settle(&opened, &Footprints::default()).expect("settle");
    assert_eq!(read_marker(root.path()).expect("read"), Some(1));

    let older = tempfile::tempdir().expect("tempdir");
    touch(older.path(), "schema");
    let err = prepare_dir(older.path(), 2, STEPS).expect_err("two behind");
    assert!(matches!(
        err,
        StorageError::UnsupportedFormat {
            found: 0,
            runs: 2,
            ..
        }
    ));
    assert_eq!(
        read_marker(older.path()).expect("read"),
        None,
        "left untouched"
    );
}

#[test]
fn the_previous_format_migrates_through_its_step() {
    let root = tempfile::tempdir().expect("tempdir");
    touch(root.path(), "schema");
    mark(root.path(), 4).expect("mark");
    let opened = prepare_dir(root.path(), 5, STEPS).expect("migrate");
    assert!(matches!(
        opened,
        FormatOpen::Migrated { from: 4, to: 5, .. }
    ));
    assert!(root.path().join("migrated").exists());
    assert_eq!(
        read_marker(root.path()).expect("read"),
        Some(4),
        "the marker waits for the open to succeed"
    );
    settle(&opened, &Footprints::default()).expect("settle");
    assert_eq!(read_marker(root.path()).expect("read"), Some(5));
}

/// The engine opens no directory of an earlier format: one from before the
/// marker (format 0) and one of the format before this release's (whose
/// index entries sit under generations, not installations) are both refused
/// by name and left as their writer left them.
#[test]
fn the_engine_refuses_a_directory_of_an_earlier_format() {
    use crate::engine::config::{EndpointConfig, Media, Tier};
    use crate::engine::core::StorageEngine;
    let runs = coordinode_core::version::engine_format_version();
    for earlier in [None, Some(runs - 1)] {
        let root = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            root.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        drop(StorageEngine::open(&config).expect("open"));
        match earlier {
            None => std::fs::remove_file(root.path().join(MARKER_FILE)).expect("unmark"),
            Some(format) => mark(root.path(), format).expect("mark earlier"),
        }
        let err = StorageEngine::open(&config).err().expect("refused");
        assert!(
            matches!(err, StorageError::UnsupportedFormat { runs: r, .. } if r == runs),
            "{err}"
        );
        assert_eq!(
            read_marker(root.path()).expect("read"),
            earlier,
            "left untouched"
        );
    }
}

/// A directory two formats behind, or written by a newer release, is
/// refused by name and not changed.
#[test]
fn other_formats_are_refused_untouched() {
    for found in [3, 6] {
        let root = tempfile::tempdir().expect("tempdir");
        touch(root.path(), "schema");
        mark(root.path(), found).expect("mark");
        let err = prepare_dir(root.path(), 5, STEPS).expect_err("refused");
        let text = err.to_string();
        assert!(
            matches!(err, StorageError::UnsupportedFormat { runs: 5, .. }),
            "{text}"
        );
        assert!(text.contains(&format!("format {found}")), "{text}");
        assert_eq!(read_marker(root.path()).expect("read"), Some(found));
    }
}

/// A step that fails leaves the marker where it was, so the next open runs
/// it again.
#[test]
fn a_failed_step_leaves_the_old_marker() {
    static FAILING: &[MigrationStep] = &[MigrationStep {
        from: 4,
        migrate: failing_step,
    }];
    let root = tempfile::tempdir().expect("tempdir");
    touch(root.path(), "schema");
    mark(root.path(), 4).expect("mark");
    prepare_dir(root.path(), 5, FAILING).expect_err("step fails");
    assert_eq!(read_marker(root.path()).expect("read"), Some(4));
}

#[test]
fn a_damaged_marker_is_an_error() {
    let root = tempfile::tempdir().expect("tempdir");
    std::fs::write(root.path().join(MARKER_FILE), b"five").expect("write");
    assert!(read_marker(root.path()).is_err());
}

/// Every engine open goes through the marker: a directory written by a
/// newer format fails the open without being touched.
#[test]
fn the_engine_refuses_a_directory_of_a_newer_format() {
    use crate::engine::config::{EndpointConfig, Media, Tier};
    use crate::engine::core::StorageEngine;
    let root = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        root.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    drop(StorageEngine::open(&config).expect("open"));
    let runs = coordinode_core::version::engine_format_version();
    assert_eq!(read_marker(root.path()).expect("read"), Some(runs));
    mark(root.path(), runs + 1).expect("mark newer");
    let err = StorageEngine::open(&config).err().expect("refused");
    assert!(
        matches!(err, StorageError::UnsupportedFormat { .. }),
        "{err}"
    );
}
