//! Namespaces of one deployment: each its own set of trees, created and
//! dropped through the catalog, never seeing another's data.

use std::sync::Arc;

use coordinode_core::namespace::DEFAULT_NAMESPACE;
use coordinode_core::txn::proposal::{Mutation, PartitionId, in_namespace};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};

fn durable_cfg(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// A store whose table writes can be refused, for a crash: what was only in
/// the memtables and the journal is all a reopen finds.
struct Crashable {
    dir: TempDir,
    faults: Arc<lsm_tree::fs::FaultInjector>,
    config: StorageConfig,
}

impl Crashable {
    fn new() -> Self {
        let dir = TempDir::new().expect("tmp");
        let faults = Arc::new(lsm_tree::fs::FaultInjector::new());
        let fs: Arc<dyn lsm_tree::fs::Fs> = Arc::new(lsm_tree::fs::FaultFs::with_injector(
            lsm_tree::fs::StdFs,
            Arc::clone(&faults),
        ));
        let config = durable_cfg(&dir).with_fs(fs);
        Self {
            dir,
            faults,
            config,
        }
    }

    /// Take the store away with every table write refused, so the flush
    /// its close attempts never lands: the journal keeps what it took.
    fn crash(&self, deployment: Arc<Deployment>) {
        use lsm_tree::fs::{Fault, FaultOp, FaultRule};
        let refuse = Fault::Error(lsm_tree::io::ErrorKind::Other);
        for op in [FaultOp::Open, FaultOp::Write, FaultOp::Rename] {
            self.faults.arm(FaultRule::new(op, refuse));
        }
        drop(deployment);
        self.faults.clear();
    }
}

/// A journaled deployment with no namespace yet: what a host opens before it
/// chooses its own store or a group to join.
fn open_deployment(config: &StorageConfig) -> (Arc<Deployment>, Arc<TimestampOracle>) {
    let oracle = Arc::new(TimestampOracle::new());
    let deployment = Deployment::open_embedded(config, Arc::clone(&oracle)).expect("open");
    (deployment, oracle)
}

/// Commit `mutations` of `engine`'s namespace through its journal.
fn commit(engine: &StorageEngine, oracle: &TimestampOracle, mutations: Vec<Mutation>) {
    engine
        .commit_journaled(
            &in_namespace(engine.namespace(), mutations),
            oracle.next().as_raw(),
        )
        .expect("commit");
}

fn put(key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Node,
        key: key.to_vec(),
        value: value.to_vec(),
    }
}

/// Every file under `dir`, for a failure to name what stayed.
fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

fn node(engine: &StorageEngine, key: &[u8]) -> Option<Vec<u8>> {
    engine
        .get(Partition::Node, key)
        .expect("get")
        .map(|v| v.to_vec())
}

/// Opening a deployment creates no namespace: a fresh store holds its system
/// trees alone until the host chooses it as its own.
#[test]
fn a_fresh_deployment_holds_no_namespace() {
    let dir = TempDir::new().expect("tmp");
    let (deployment, _) = open_deployment(&durable_cfg(&dir));
    assert!(deployment.live_namespaces().expect("catalog").is_empty());
    assert!(
        !dir.path()
            .join(crate::engine::scope::NAMESPACES_DIR)
            .exists(),
        "a namespace's trees exist without a namespace"
    );
}

/// Two namespaces with the same keys never see each other's rows, through a
/// point read, a scan or a snapshot read, before and after a reopen.
#[test]
fn two_namespaces_with_the_same_keys_stay_apart() {
    let dir = TempDir::new().expect("tmp");
    let config = durable_cfg(&dir);
    {
        let (deployment, oracle) = open_deployment(&config);
        let a = deployment
            .engine(deployment.create_namespace("a").expect("create a"))
            .expect("bind a");
        let b = deployment
            .engine(deployment.create_namespace("b").expect("create b"))
            .expect("bind b");
        assert_ne!(a.namespace(), b.namespace());
        commit(&a, &oracle, vec![put(b"k", b"of-a"), put(b"only-a", b"1")]);
        commit(&b, &oracle, vec![put(b"k", b"of-b")]);

        assert_eq!(node(&a, b"k").as_deref(), Some(&b"of-a"[..]));
        assert_eq!(node(&b, b"k").as_deref(), Some(&b"of-b"[..]));
        assert_eq!(node(&b, b"only-a"), None, "b reads a's row");
        let scanned: Vec<Vec<u8>> = b
            .prefix_scan(Partition::Node, b"")
            .expect("scan")
            .map(|guard| guard.into_owned().expect("row").0)
            .collect();
        assert_eq!(scanned, vec![b"k".to_vec()]);
        let snapshot = b.snapshot();
        assert_eq!(
            b.snapshot_get(&snapshot, Partition::Node, b"only-a")
                .expect("snapshot get"),
            None
        );
    }
    let (deployment, _) = open_deployment(&config);
    let a = deployment.engine_named("a").expect("a again");
    let b = deployment.engine_named("b").expect("b again");
    assert_eq!(node(&a, b"k").as_deref(), Some(&b"of-a"[..]));
    assert_eq!(node(&b, b"k").as_deref(), Some(&b"of-b"[..]));
    assert_eq!(node(&b, b"only-a"), None);
}

/// One unit writing two namespaces lands in both trees at one timestamp, and
/// a crash before any flush brings both back from the journal.
#[test]
fn a_unit_spanning_two_namespaces_replays_into_both() {
    let store = Crashable::new();
    let (a_id, b_id) = {
        let (deployment, oracle) = open_deployment(&store.config);
        let a_id = deployment.create_namespace("a").expect("a");
        let b_id = deployment.create_namespace("b").expect("b");
        let mut unit = in_namespace(a_id, [put(b"k", b"a-side")]);
        unit.extend(in_namespace(b_id, [put(b"k", b"b-side")]));
        deployment
            .commit_journaled(&unit, oracle.next().as_raw())
            .expect("commit");
        {
            let a = deployment.engine(a_id).expect("a");
            let b = deployment.engine(b_id).expect("b");
            assert_eq!(node(&a, b"k").as_deref(), Some(&b"a-side"[..]));
            assert_eq!(node(&b, b"k").as_deref(), Some(&b"b-side"[..]));
        }
        // Nothing flushed: the journal alone holds the unit.
        store.crash(deployment);
        (a_id, b_id)
    };
    let (deployment, _) = open_deployment(&store.config);
    let a = deployment.engine(a_id).expect("a after the crash");
    let b = deployment.engine(b_id).expect("b after the crash");
    assert_eq!(node(&a, b"k").as_deref(), Some(&b"a-side"[..]));
    assert_eq!(node(&b, b"k").as_deref(), Some(&b"b-side"[..]));
}

/// Every namespace's trees take the journal's base before it is purged: a
/// store whose journal no longer holds the commits reopens with both
/// namespaces whole.
#[test]
fn a_purged_journal_leaves_every_namespace_whole() {
    let dir = TempDir::new().expect("tmp");
    let mut config = durable_cfg(&dir);
    config.oplog_segment_max_entries = 8;
    config.oplog_retention_secs = 1;
    let commits = crate::engine::coverage::FOLD_EVERY + 50;
    {
        let (deployment, oracle) = open_deployment(&config);
        let a = deployment
            .engine(deployment.create_namespace("a").expect("a"))
            .expect("bind a");
        let b = deployment
            .engine(deployment.create_namespace("b").expect("b"))
            .expect("bind b");
        for i in 0..commits {
            let key = format!("k{i}");
            // Left to the persist below: no crash is under test here.
            for (engine, value) in [(&a, &b"a"[..]), (&b, &b"b"[..])] {
                engine
                    .commit_logged(
                        &in_namespace(engine.namespace(), [put(key.as_bytes(), value)]),
                        oracle.next().as_raw(),
                        coordinode_core::txn::write_concern::Journal::Cache,
                    )
                    .expect("commit");
            }
        }
        deployment.persist().expect("persist");
        let purged = deployment
            .oplog_purge_expired(4_000_000_000, u64::MAX)
            .expect("purge");
        assert!(purged > 0, "the journal kept every segment");
    }
    let (deployment, _) = open_deployment(&config);
    for (name, value) in [("a", &b"a"[..]), ("b", &b"b"[..])] {
        let engine = deployment.engine_named(name).expect("reopen");
        for i in [0, commits - 1] {
            assert_eq!(
                node(&engine, format!("k{i}").as_bytes()).as_deref(),
                Some(value),
                "{name} lost commit {i}"
            );
        }
    }
}

/// A dropped namespace is gone: its name is free, its trees leave every
/// endpoint, a bind to its id is refused, and another namespace keeps its
/// rows. A namespace created again under the name is a new one, with a new
/// id and none of the old rows.
#[test]
fn a_drop_removes_the_namespace_and_nothing_else() {
    let dir = TempDir::new().expect("tmp");
    let config = durable_cfg(&dir);
    let (deployment, oracle) = open_deployment(&config);
    let a_id = deployment.create_namespace("a").expect("a");
    let b_id = deployment.create_namespace("b").expect("b");
    {
        let a = deployment.engine(a_id).expect("a");
        let b = deployment.engine(b_id).expect("b");
        commit(&a, &oracle, vec![put(b"k", b"a")]);
        commit(&b, &oracle, vec![put(b"k", b"b")]);
        a.persist().expect("persist");
    }
    let a_dir = dir.path().join(Scope::Namespace(a_id).dir());
    assert!(a_dir.exists());

    assert_eq!(deployment.drop_namespace("a").expect("drop"), a_id);
    assert!(
        !a_dir.exists(),
        "the dropped namespace's trees stay: {:?}",
        walk(&a_dir)
    );
    assert!(matches!(
        deployment.engine(a_id),
        Err(StorageError::UnknownNamespace { .. })
    ));
    assert!(matches!(
        deployment.namespace_id("a"),
        Err(StorageError::NoSuchNamespace { .. })
    ));
    let b = deployment.engine(b_id).expect("b");
    assert_eq!(node(&b, b"k").as_deref(), Some(&b"b"[..]));

    let again = deployment.create_namespace("a").expect("a again");
    assert_ne!(again, a_id, "an id was handed out twice");
    let a = deployment.engine(again).expect("new a");
    assert_eq!(node(&a, b"k"), None, "the new namespace holds the old rows");
    drop((a, b));
    drop(deployment);

    let (deployment, _) = open_deployment(&config);
    assert_eq!(
        deployment.live_namespaces().expect("catalog"),
        vec![(b_id, "b".to_owned()), (again, "a".to_owned())]
    );
    assert!(!a_dir.exists(), "the dropped namespace came back");
}

/// Two namespaces each with a `STORAGE COLUMNAR` table of the same name keep
/// their own rows, and a crash before any flush replays each journalled row
/// into its own namespace's table, never the other's.
#[cfg(feature = "columnar")]
#[test]
fn columnar_tables_of_one_name_stay_apart_through_a_crash() {
    let store = Crashable::new();
    let rows = |engine: &StorageEngine| -> Vec<(Vec<u8>, Vec<u8>)> {
        engine.columnar_scan("t", engine.snapshot()).expect("scan")
    };
    let (a_id, b_id) = {
        let (deployment, oracle) = open_deployment(&store.config);
        let a = deployment
            .engine(deployment.create_namespace("a").expect("a"))
            .expect("bind a");
        let b = deployment
            .engine(deployment.create_namespace("b").expect("b"))
            .expect("bind b");
        a.create_columnar_table("t").expect("table of a");
        b.create_columnar_table("t").expect("table of b");
        a.columnar_insert("t", b"k".to_vec(), b"of-a".to_vec(), oracle.next().as_raw())
            .expect("row of a");
        b.columnar_insert("t", b"k".to_vec(), b"of-b".to_vec(), oracle.next().as_raw())
            .expect("row of b");
        b.columnar_insert(
            "t",
            b"only-b".to_vec(),
            b"1".to_vec(),
            oracle.next().as_raw(),
        )
        .expect("second row of b");
        assert_eq!(rows(&a), vec![(b"k".to_vec(), b"of-a".to_vec())]);
        assert_eq!(rows(&b).len(), 2);
        let ids = (a.namespace(), b.namespace());
        drop((a, b));
        // The table trees took their rows in memtables only.
        store.crash(deployment);
        ids
    };
    let (deployment, _) = open_deployment(&store.config);
    let a = deployment.engine(a_id).expect("a after the crash");
    let b = deployment.engine(b_id).expect("b after the crash");
    assert_eq!(
        rows(&a),
        vec![(b"k".to_vec(), b"of-a".to_vec())],
        "a's table lost its row or took b's"
    );
    assert_eq!(
        rows(&b),
        vec![
            (b"k".to_vec(), b"of-b".to_vec()),
            (b"only-b".to_vec(), b"1".to_vec())
        ],
        "b's table lost a row or took a's"
    );
}

/// A table rebuilt from a checkpoint takes the checkpoint's copy of its own
/// namespace's table and replays only its own namespace's journalled rows,
/// though a table of the same name in another namespace was written all
/// along.
#[cfg(feature = "columnar")]
#[test]
fn a_table_rebuild_takes_only_its_own_namespaces_rows() {
    let dir = TempDir::new().expect("tmp");
    let backups = TempDir::new().expect("tmp");
    let (deployment, oracle) = open_deployment(&durable_cfg(&dir));
    let a = deployment
        .engine(deployment.create_namespace("a").expect("a"))
        .expect("bind a");
    let b = deployment
        .engine(deployment.create_namespace("b").expect("b"))
        .expect("bind b");
    let row = |engine: &StorageEngine, key: &[u8], value: &[u8]| {
        engine
            .columnar_insert("t", key.to_vec(), value.to_vec(), oracle.next().as_raw())
            .expect("row");
    };
    row(&a, b"k", b"of-a");
    row(&b, b"k", b"of-b");
    let checkpoint = backups.path().join("checkpoint");
    deployment
        .create_checkpoint(&checkpoint)
        .expect("checkpoint");
    row(&a, b"later-a", b"1");
    row(&b, b"later-b", b"1");
    let from = deployment
        .checkpoint_replay_floor(&checkpoint)
        .expect("floor");
    let suffix = deployment
        .oplog_read_since(from)
        .expect("read journal")
        .expect("journal");

    let rebuilt = b
        .repair_table_from_checkpoint(&checkpoint, &suffix, "t", &[])
        .expect("rebuild");
    assert!(matches!(rebuilt, Rebuild::Rebuilt { .. }), "{rebuilt:?}");
    assert_eq!(
        b.columnar_scan("t", b.snapshot()).expect("scan b"),
        vec![
            (b"k".to_vec(), b"of-b".to_vec()),
            (b"later-b".to_vec(), b"1".to_vec())
        ],
        "the rebuilt table holds another namespace's rows"
    );
    assert_eq!(
        a.columnar_scan("t", a.snapshot()).expect("scan a"),
        vec![
            (b"k".to_vec(), b"of-a".to_vec()),
            (b"later-a".to_vec(), b"1".to_vec())
        ],
        "the rebuild touched another namespace's table"
    );
}

/// A partition rebuilt from a checkpoint takes the checkpoint's copy of its
/// own namespace's tree and replays only its own namespace's journalled
/// writes, though another namespace wrote the same keys all along.
#[test]
fn a_partition_rebuild_takes_only_its_own_namespaces_rows() {
    let dir = TempDir::new().expect("tmp");
    let backups = TempDir::new().expect("tmp");
    let (deployment, oracle) = open_deployment(&durable_cfg(&dir));
    let a = deployment
        .engine(deployment.create_namespace("a").expect("a"))
        .expect("bind a");
    let b = deployment
        .engine(deployment.create_namespace("b").expect("b"))
        .expect("bind b");
    commit(&a, &oracle, vec![put(b"k", b"of-a")]);
    commit(&b, &oracle, vec![put(b"k", b"of-b")]);
    let checkpoint = backups.path().join("checkpoint");
    deployment
        .create_checkpoint(&checkpoint)
        .expect("checkpoint");
    commit(&a, &oracle, vec![put(b"later", b"a")]);
    commit(&b, &oracle, vec![put(b"later", b"b")]);
    let from = deployment
        .checkpoint_replay_floor(&checkpoint)
        .expect("floor");
    let suffix = deployment
        .oplog_read_since(from)
        .expect("read journal")
        .expect("journal");

    let rebuilt = b
        .repair_partition_from_checkpoint(&checkpoint, &suffix, Partition::Node)
        .expect("rebuild");
    assert!(matches!(rebuilt, Rebuild::Rebuilt { .. }), "{rebuilt:?}");
    assert_eq!(node(&b, b"k").as_deref(), Some(&b"of-b"[..]));
    assert_eq!(node(&b, b"later").as_deref(), Some(&b"b"[..]));
    assert_eq!(node(&a, b"k").as_deref(), Some(&b"of-a"[..]));
    assert_eq!(node(&a, b"later").as_deref(), Some(&b"a"[..]));
}

/// Lost key ranges rebuilt from a checkpoint come from the checkpoint's copy
/// of the same namespace, and only that namespace's journalled writes are
/// replayed into them, though another namespace wrote the same keys.
#[test]
fn a_range_rebuild_takes_only_its_own_namespaces_rows() {
    let dir = TempDir::new().expect("tmp");
    let backups = TempDir::new().expect("tmp");
    let (deployment, oracle) = open_deployment(&durable_cfg(&dir));
    let a = deployment
        .engine(deployment.create_namespace("a").expect("a"))
        .expect("bind a");
    let b = deployment
        .engine(deployment.create_namespace("b").expect("b"))
        .expect("bind b");
    commit(&a, &oracle, vec![put(b"k", b"of-a")]);
    commit(&b, &oracle, vec![put(b"k", b"of-b")]);
    let checkpoint = backups.path().join("checkpoint");
    deployment
        .create_checkpoint(&checkpoint)
        .expect("checkpoint");
    commit(&a, &oracle, vec![put(b"later", b"a")]);
    commit(&b, &oracle, vec![put(b"later", b"b")]);
    let from = deployment
        .checkpoint_replay_floor(&checkpoint)
        .expect("floor");
    let suffix = deployment
        .oplog_read_since(from)
        .expect("read journal")
        .expect("journal");

    let lost = [(b"a".to_vec(), b"z".to_vec())];
    let rebuilt = b
        .repair_partition_ranges_from_checkpoint(&checkpoint, &suffix, Partition::Node, &lost)
        .expect("rebuild");
    assert!(matches!(rebuilt, Rebuild::Rebuilt { .. }), "{rebuilt:?}");
    assert_eq!(node(&b, b"k").as_deref(), Some(&b"of-b"[..]));
    assert_eq!(node(&b, b"later").as_deref(), Some(&b"b"[..]));
    assert_eq!(node(&a, b"k").as_deref(), Some(&b"of-a"[..]));
    assert_eq!(node(&a, b"later").as_deref(), Some(&b"a"[..]));
}

/// A repair holds its own namespace's tree only: the same partition of
/// another namespace can be repaired meanwhile, and neither shows the other
/// under repair.
#[test]
fn a_repair_holds_only_its_own_namespaces_tree() {
    let dir = TempDir::new().expect("tmp");
    let (deployment, _) = open_deployment(&durable_cfg(&dir));
    let a = deployment
        .engine(deployment.create_namespace("a").expect("a"))
        .expect("bind a");
    let b = deployment
        .engine(deployment.create_namespace("b").expect("b"))
        .expect("bind b");
    let a_node = a.tree_id(Partition::Node);
    let b_node = b.tree_id(Partition::Node);

    let held = deployment.enter_repair(b_node).expect("repair of b");
    assert!(deployment.is_under_repair(b_node));
    assert!(
        !deployment.is_under_repair(a_node),
        "a's tree shows b's repair"
    );
    let other = deployment
        .enter_repair(a_node)
        .expect("a's tree is refused while b's is repaired");
    assert!(
        deployment.enter_repair(b_node).is_err(),
        "a second repair of b's tree ran"
    );
    drop((held, other));
}

/// A rebuild cut short leaves its own namespace's tree owed across a reopen,
/// and only that one: the same partition of another namespace is not owed.
#[test]
fn a_rebuild_cut_short_stays_owed_by_its_own_namespaces_tree() {
    let dir = TempDir::new().expect("tmp");
    let config = durable_cfg(&dir);
    let b_node = {
        let (deployment, oracle) = open_deployment(&config);
        let a = deployment
            .engine(deployment.create_namespace("a").expect("a"))
            .expect("bind a");
        let b = deployment
            .engine(deployment.create_namespace("b").expect("b"))
            .expect("bind b");
        commit(&a, &oracle, vec![put(b"k", b"a")]);
        commit(&b, &oracle, vec![put(b"k", b"b")]);
        let b_node = b.tree_id(Partition::Node);
        assert!(
            b.clear_partition_with(
                Partition::Node,
                b.tree(Partition::Node).expect("tree"),
                || { Err(StorageError::Io("cut short".into())) }
            )
            .is_err()
        );
        b_node
    };
    let (deployment, _) = open_deployment(&config);
    assert_eq!(deployment.pending_rebuilds().expect("intents"), [b_node]);
    let a = deployment.engine_named("a").expect("a");
    assert_eq!(node(&a, b"k").as_deref(), Some(&b"a"[..]));
}

/// A name a live namespace holds, or one no namespace may have, is refused
/// and changes nothing.
#[test]
fn a_taken_or_invalid_name_is_refused() {
    let dir = TempDir::new().expect("tmp");
    let (deployment, _) = open_deployment(&durable_cfg(&dir));
    let a = deployment.create_namespace("a").expect("a");
    for name in ["a", "A", "", "with space"] {
        assert!(
            matches!(
                deployment.create_namespace(name),
                Err(StorageError::NamespaceRefused { .. })
            ),
            "{name:?} was not refused"
        );
    }
    assert_eq!(
        deployment.live_namespaces().expect("catalog"),
        vec![(a, "a".to_owned())]
    );
}

/// A store opened as a store of one namespace: a fresh one creates its
/// default once, a reopen binds the same one, one holding several asks which,
/// and one whose namespaces were dropped gets none back.
#[test]
fn a_store_of_one_namespace_creates_its_default_once() {
    let dir = TempDir::new().expect("tmp");
    let config = durable_cfg(&dir);
    let first = {
        let engine =
            StorageEngine::open_embedded(&config, Arc::new(TimestampOracle::new())).expect("fresh");
        engine.namespace()
    };
    {
        let (deployment, _) = open_deployment(&config);
        assert_eq!(
            deployment.live_namespaces().expect("catalog"),
            vec![(first, DEFAULT_NAMESPACE.to_owned())]
        );
        assert_eq!(deployment.single_store().expect("bind").namespace(), first);
        deployment.create_namespace("other").expect("other");
        assert!(matches!(
            deployment.single_store(),
            Err(StorageError::SeveralNamespaces { .. })
        ));
        deployment.drop_namespace("other").expect("drop other");
        deployment
            .drop_namespace(DEFAULT_NAMESPACE)
            .expect("drop default");
    }
    assert!(matches!(
        StorageEngine::open_embedded(&config, Arc::new(TimestampOracle::new())),
        Err(StorageError::NoNamespace { .. })
    ));
}

/// Rows a unit carries for a namespace dropped before it applies are left
/// out, the unit's other rows applied: nothing of the namespace is kept, and
/// a replay of the unit after a crash does the same.
#[test]
fn rows_of_a_dropped_namespace_are_left_out_on_apply_and_replay() {
    let store = Crashable::new();
    let (a_id, b_id) = {
        let (deployment, oracle) = open_deployment(&store.config);
        let a_id = deployment.create_namespace("a").expect("a");
        let b_id = deployment.create_namespace("b").expect("b");
        deployment.drop_namespace("a").expect("drop a");
        let mut unit = in_namespace(a_id, [put(b"k", b"late")]);
        unit.extend(in_namespace(b_id, [put(b"k", b"b")]));
        deployment
            .commit_journaled(&unit, oracle.next().as_raw())
            .expect("commit");
        {
            let b = deployment.engine(b_id).expect("b");
            assert_eq!(node(&b, b"k").as_deref(), Some(&b"b"[..]));
        }
        store.crash(deployment);
        (a_id, b_id)
    };
    let (deployment, _) = open_deployment(&store.config);
    assert!(matches!(
        deployment.engine(a_id),
        Err(StorageError::UnknownNamespace { .. })
    ));
    assert!(!store.dir.path().join(Scope::Namespace(a_id).dir()).exists());
    let b = deployment.engine(b_id).expect("b after the crash");
    assert_eq!(node(&b, b"k").as_deref(), Some(&b"b"[..]));
}
