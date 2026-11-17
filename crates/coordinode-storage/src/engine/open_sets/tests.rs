use std::sync::Arc;

use super::*;

fn tree(dir: &tempfile::TempDir, name: &str) -> AnyTree {
    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    lsm_tree::Config::new_with_generators(dir.path().join(name), Arc::clone(&seqno), seqno)
        .open()
        .expect("open tree")
}

fn namespace_set(dir: &tempfile::TempDir, ns: u64) -> SetTrees {
    let mut set = SetTrees::new(Scope::Namespace(NamespaceId::from_raw(ns)));
    set.insert(Partition::Node, tree(dir, &format!("{ns}-node")));
    set.insert(Partition::Schema, tree(dir, &format!("{ns}-schema")));
    set
}

/// A set finds its trees by partition and reports the rest absent; the
/// listing names each tree with its set.
#[test]
fn a_set_finds_its_trees_by_partition() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let set = namespace_set(&dir, 7);
    let ns = NamespaceId::from_raw(7);
    assert!(set.get(Partition::Node).is_some());
    assert!(set.get(Partition::Adj).is_none());
    assert_eq!(set.len(), 2);
    let ids: Vec<TreeId> = set.iter().map(|(id, _)| id).collect();
    assert_eq!(
        ids,
        vec![
            TreeId::namespace(ns, Partition::Node),
            TreeId::namespace(ns, Partition::Schema)
        ]
    );
}

/// The open sets find a tree by its address, system or namespace, and list
/// every open tree with its address, the system set's first.
#[test]
fn trees_are_found_and_listed_by_address() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let mut system = SetTrees::new(Scope::System);
    system.insert(Partition::Raft, tree(&dir, "raft"));
    let sets = OpenSets::new(
        system,
        [
            Arc::new(namespace_set(&dir, 2)),
            Arc::new(namespace_set(&dir, 1)),
        ],
    );
    let one = NamespaceId::from_raw(1);
    let two = NamespaceId::from_raw(2);

    assert_eq!(sets.namespace_ids(), vec![one, two]);
    assert!(sets.tree(TreeId::namespace(two, Partition::Node)).is_some());
    assert!(sets.tree(TreeId::system(Partition::Raft)).is_some());
    assert!(sets.tree(TreeId::system(Partition::Registry)).is_none());
    assert!(
        sets.tree(TreeId::namespace(NamespaceId::from_raw(3), Partition::Node))
            .is_none(),
        "no set of a namespace not open"
    );
    let listed: Vec<TreeId> = sets.list().into_iter().map(|(id, _)| id).collect();
    assert_eq!(
        listed,
        vec![
            TreeId::system(Partition::Raft),
            TreeId::namespace(one, Partition::Node),
            TreeId::namespace(one, Partition::Schema),
            TreeId::namespace(two, Partition::Node),
            TreeId::namespace(two, Partition::Schema),
        ]
    );
}
