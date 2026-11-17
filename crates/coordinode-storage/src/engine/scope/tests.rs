use super::*;

/// The system set holds the consensus log, the consumer registry and a
/// Schema tree, and nothing of a namespace's data; a namespace set holds
/// every data partition and a Schema tree, and neither of the deployment's.
#[test]
fn each_set_holds_its_own_partitions() {
    let system: Vec<_> = Scope::System.partitions().collect();
    assert_eq!(
        system,
        vec![Partition::Schema, Partition::Raft, Partition::Registry]
    );

    let ns = Scope::Namespace(NamespaceId::from_raw(7));
    let data: Vec<_> = ns.partitions().collect();
    assert!(data.contains(&Partition::Schema));
    assert!(data.contains(&Partition::Node));
    assert!(data.contains(&Partition::NodeHistory));
    assert!(!data.contains(&Partition::Raft));
    assert!(!data.contains(&Partition::Registry));
    assert_eq!(data.len() + system.len(), Partition::all().len() + 1);
}

/// A namespace's trees live under its own directory, the system set's under
/// its own; two namespaces never share one.
#[test]
fn trees_live_under_their_sets_directory() {
    assert_eq!(
        TreeId::system(Partition::Raft).dir(),
        PathBuf::from("system/raft")
    );
    let a = TreeId::namespace(NamespaceId::from_raw(1), Partition::Node).dir();
    let b = TreeId::namespace(NamespaceId::from_raw(2), Partition::Node).dir();
    assert_eq!(a, PathBuf::from("ns/0000000000000001/node"));
    assert_ne!(a, b);
}

/// A tree's name gives the tree back. A name without its set, a bare
/// partition name as a copy of the store carried before trees had sets
/// included, and a partition the named set holds no tree of, give none.
#[test]
fn tree_names_round_trip_and_refuse_others() {
    let ns = NamespaceId::from_raw(0xab);
    for tree in [
        TreeId::system(Partition::Raft),
        TreeId::system(Partition::Schema),
        TreeId::namespace(ns, Partition::Node),
        TreeId::namespace(ns, Partition::Schema),
    ] {
        assert_eq!(TreeId::parse(&tree.name()), Some(tree), "{}", tree.name());
    }
    for name in [
        "node",
        "raft",
        "system/node",
        "ns/00000000000000ab/raft",
        "ns/ab/node",
        "ns/00000000000000ab",
        "ns/00000000000000ab/node/extra",
        "tables/t",
        "",
    ] {
        assert_eq!(TreeId::parse(name), None, "{name:?}");
    }
}

/// A namespace directory name gives its id back; any other name gives none,
/// so a stray directory is never taken for a namespace.
#[test]
fn namespace_dir_names_round_trip_and_refuse_others() {
    for raw in [0, 1, 0xdead_beef, u64::MAX] {
        let id = NamespaceId::from_raw(raw);
        assert_eq!(NamespaceId::from_dir_name(&id.dir_name()), Some(id));
    }
    for name in [
        "",
        "1",
        "000000000000000g",
        "00000000000000001",
        "+000000000000001",
    ] {
        assert_eq!(NamespaceId::from_dir_name(name), None, "{name:?}");
    }
}
