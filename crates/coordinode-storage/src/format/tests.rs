use super::*;

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

static STEPS: &[MigrationStep] = &[
    MigrationStep {
        from: 0,
        migrate: no_change,
    },
    MigrationStep {
        from: 4,
        migrate: mark_step,
    },
];

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
    assert_eq!(
        prepare_dir(root.path(), 1, STEPS).expect("migrate"),
        FormatOpen::Migrated { from: 0 }
    );
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
    write_marker(root.path(), 4).expect("mark");
    assert_eq!(
        prepare_dir(root.path(), 5, STEPS).expect("migrate"),
        FormatOpen::Migrated { from: 4 }
    );
    assert!(root.path().join("migrated").exists());
    assert_eq!(read_marker(root.path()).expect("read"), Some(5));
}

/// A directory two formats behind, or written by a newer release, is
/// refused by name and not changed.
#[test]
fn other_formats_are_refused_untouched() {
    for found in [3, 6] {
        let root = tempfile::tempdir().expect("tempdir");
        touch(root.path(), "schema");
        write_marker(root.path(), found).expect("mark");
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
    write_marker(root.path(), 4).expect("mark");
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
    write_marker(root.path(), runs + 1).expect("mark newer");
    let err = StorageEngine::open(&config).err().expect("refused");
    assert!(
        matches!(err, StorageError::UnsupportedFormat { .. }),
        "{err}"
    );
}
