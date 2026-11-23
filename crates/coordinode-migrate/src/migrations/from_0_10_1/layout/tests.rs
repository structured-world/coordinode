use super::*;

/// A store holding a tree both at its top and in its set is refused: which
/// copy the store's records describe cannot be told, so neither moves.
#[test]
fn a_tree_both_at_the_top_and_in_its_set_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path();
    std::fs::create_dir_all(store.join("node")).expect("top");
    let in_set = store.join(TreeId::namespace(NAMESPACE, Partition::Node).dir());
    std::fs::create_dir_all(&in_set).expect("in its set");
    let finding = Finding {
        path: store.to_path_buf(),
        detail: String::new(),
    };
    let err = apply(&finding, &Backup::new(store, "run1")).expect_err("refused");
    assert!(format!("{err:#}").contains("both hold the tree"), "{err:#}");
    assert!(store.join("node").is_dir() && in_set.is_dir());
}

/// The deployment's partitions go to the system set, every other one and
/// the tables to the namespace's set.
#[test]
fn each_tree_goes_to_its_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path();
    for name in ["raft", "registry", "node", "idx", "tables"] {
        std::fs::create_dir_all(store.join(name)).expect("tree");
    }
    let mut got: Vec<(String, String)> = moves(store)
        .into_iter()
        .map(|(from, to)| {
            let rel = |p: &Path| p.strip_prefix(store).expect("inside").display().to_string();
            (rel(&from), rel(&to))
        })
        .collect();
    got.sort();
    let mut expected: Vec<(String, String)> = [
        ("idx", "ns/0000000000000001/idx"),
        ("node", "ns/0000000000000001/node"),
        ("raft", "system/raft"),
        ("registry", "system/registry"),
        ("tables", "ns/0000000000000001/tables"),
    ]
    .into_iter()
    .map(|(a, b)| (a.to_owned(), b.to_owned()))
    .collect();
    expected.sort();
    assert_eq!(got, expected);
}
