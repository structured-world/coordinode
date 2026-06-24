use super::*;

/// Make a fake `ckpt-<tag>` directory under `root`.
fn mk_ckpt(root: &Path, tag: u64) {
    std::fs::create_dir_all(root.join(format!("{CHECKPOINT_PREFIX}{tag:020}"))).expect("mkdir");
}

#[test]
fn prune_keeps_newest_n() {
    let dir = tempfile::tempdir().expect("tempdir");
    for tag in [10u64, 20, 30, 40, 50] {
        mk_ckpt(dir.path(), tag);
    }
    prune_checkpoints(dir.path(), 2).expect("prune");

    let mut remaining = list_checkpoints(dir.path()).expect("list");
    remaining.sort();
    let names: Vec<String> = remaining
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec![
            format!("{CHECKPOINT_PREFIX}{:020}", 40u64),
            format!("{CHECKPOINT_PREFIX}{:020}", 50u64),
        ],
        "only the two newest checkpoints survive"
    );
}

#[test]
fn prune_noop_when_under_keep() {
    let dir = tempfile::tempdir().expect("tempdir");
    mk_ckpt(dir.path(), 1);
    mk_ckpt(dir.path(), 2);
    prune_checkpoints(dir.path(), 5).expect("prune");
    assert_eq!(list_checkpoints(dir.path()).expect("list").len(), 2);
}

#[test]
fn prune_ignores_non_checkpoint_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    mk_ckpt(dir.path(), 1);
    std::fs::create_dir_all(dir.path().join("oplog")).expect("mkdir");
    std::fs::write(dir.path().join("notes.txt"), b"x").expect("write");
    // keep 0 → the one checkpoint is removed, but the non-checkpoint entries stay.
    prune_checkpoints(dir.path(), 0).expect("prune");
    assert_eq!(list_checkpoints(dir.path()).expect("list").len(), 0);
    assert!(dir.path().join("oplog").exists(), "non-checkpoint dir kept");
    assert!(dir.path().join("notes.txt").exists(), "stray file kept");
}
