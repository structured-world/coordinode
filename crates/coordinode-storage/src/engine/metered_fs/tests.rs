use std::io::Write;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};

const UNIT: u64 = 4096;

fn metered(dir: &Path, limit: u64) -> MeteredFs {
    let endpoint = EndpointConfig::new("ep", dir, Media::Ssd, Durability::Durable, Tier::Warm)
        .with_hard_limit_bytes(limit)
        .with_recovery_reserve_bytes(0);
    MeteredFs::new(
        Arc::new(lsm_tree::fs::StdFs),
        Footprints::open(&[endpoint]).unwrap(),
    )
}

fn charged(fs: &MeteredFs) -> u64 {
    fs.footprints().iter().map(|f| f.charged()).sum()
}

fn create(fs: &MeteredFs, path: &Path) -> Box<dyn FsFile> {
    fs.open(path, &FsOpenOptions::new().write(true).create_new(true))
        .unwrap()
}

/// A write that would pass the limit, with the space the file system may
/// allocate ahead of it, is refused before it reaches the disk: the file
/// keeps what was written before it, and the limit holds. Once the handle
/// closes the charge is what the file system allocated.
#[test]
fn a_write_past_the_limit_never_reaches_the_disk() {
    let dir = tempfile::tempdir().unwrap();
    let ahead = crate::engine::footprint::allocation_ahead(25 * UNIT);
    let fs = metered(dir.path(), 25 * UNIT + ahead);
    let path = dir.path().join("table");
    let mut file = create(&fs, &path);
    // 25 units and the allowance ahead of the writer: exactly the limit.
    file.write_all(&vec![1u8; 25 * UNIT as usize]).unwrap();
    assert!(charged(&fs) <= 25 * UNIT + ahead);
    let refused = file.write_all(&vec![2u8; 2 * UNIT as usize]).unwrap_err();
    assert_eq!(refused.kind(), std::io::ErrorKind::StorageFull);
    drop(file);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 25 * UNIT);
    assert_eq!(
        charged(&fs),
        crate::engine::footprint::allocated_at(&path).unwrap()
    );
}

/// While a file is written it is charged its length and the allowance a
/// file system may allocate ahead of the writer; a sync settles the charge
/// to what the file system actually allocated.
#[test]
fn a_file_being_written_is_charged_ahead_and_settled_at_sync() {
    let dir = tempfile::tempdir().unwrap();
    let fs = metered(dir.path(), 1 << 30);
    let path = dir.path().join("table");
    let mut file = create(&fs, &path);
    file.write_all(&vec![1u8; 10 * UNIT as usize]).unwrap();
    assert!(
        charged(&fs) >= 10 * UNIT + crate::engine::footprint::allocation_ahead(10 * UNIT),
        "the allowance ahead of the writer is held while it writes"
    );
    file.sync_data().unwrap();
    assert_eq!(
        charged(&fs),
        crate::engine::footprint::allocated_at(&path).unwrap(),
        "a sync brings the charge to the allocation"
    );
}

/// Removing, truncating and renaming a file through the wrapper keep the
/// account equal to what the files hold.
#[test]
fn removal_truncation_and_rename_keep_the_account_exact() {
    let dir = tempfile::tempdir().unwrap();
    let fs = metered(dir.path(), 64 * UNIT);
    let (a, b) = (dir.path().join("a"), dir.path().join("b"));
    let mut file = create(&fs, &a);
    file.write_all(&vec![1u8; 5 * UNIT as usize]).unwrap();
    drop(file);
    assert_eq!(charged(&fs), 5 * UNIT);
    fs.rename(&a, &b).unwrap();
    assert_eq!(charged(&fs), 5 * UNIT);
    fs.truncate_file(&b).unwrap();
    assert_eq!(charged(&fs), 0);
    let mut file = fs
        .open(&b, &FsOpenOptions::new().write(true).append(true))
        .unwrap();
    file.write_all(&vec![3u8; 2 * UNIT as usize]).unwrap();
    drop(file);
    assert_eq!(charged(&fs), 2 * UNIT);
    fs.remove_file(&b).unwrap();
    assert_eq!(charged(&fs), 0);
}

/// A hard link through the wrapper costs nothing more; the bytes come back
/// with the last name.
#[cfg(unix)]
#[test]
fn a_hard_link_through_the_wrapper_is_charged_once() {
    let dir = tempfile::tempdir().unwrap();
    let fs = metered(dir.path(), 64 * UNIT);
    let (a, b) = (dir.path().join("a"), dir.path().join("b"));
    let mut file = create(&fs, &a);
    file.write_all(&vec![1u8; 2 * UNIT as usize]).unwrap();
    drop(file);
    fs.hard_link(&a, &b).unwrap();
    assert_eq!(charged(&fs), 2 * UNIT);
    fs.remove_file(&a).unwrap();
    assert_eq!(charged(&fs), 2 * UNIT);
    fs.remove_file(&b).unwrap();
    assert_eq!(charged(&fs), 0);
}

/// Free space as compactions see it is the room the footprint leaves, never
/// more than the volume has.
#[test]
fn available_space_is_the_room_the_footprint_leaves() {
    let dir = tempfile::tempdir().unwrap();
    let fs = metered(dir.path(), 64 * UNIT);
    let path = dir.path().join("a");
    let mut file = create(&fs, &path);
    file.write_all(&vec![1u8; 10 * UNIT as usize]).unwrap();
    drop(file);
    let held = crate::engine::footprint::allocated_at(&path).unwrap();
    assert_eq!(fs.available_space(dir.path()).unwrap(), 64 * UNIT - held);
}

/// Extending a file with set_len is charged first and refused past the limit,
/// then the charge is what the file system allocated (nothing, for the hole
/// an extension leaves).
#[test]
fn set_len_growth_is_charged_and_refused_past_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let fs = metered(dir.path(), 8 * UNIT);
    let path = dir.path().join("a");
    let file = create(&fs, &path);
    file.set_len(6 * UNIT).unwrap();
    let after = charged(&fs);
    assert!(after <= 6 * UNIT);
    assert_eq!(
        after,
        crate::engine::footprint::allocated_at(&path).unwrap()
    );
    assert!(file.set_len(9 * UNIT).is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 6 * UNIT);
    file.set_len(UNIT).unwrap();
    assert!(charged(&fs) <= UNIT);
}
