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

/// Whether `dir`, a dropped namespace's directory, is gone within a bound:
/// it goes once the last of its trees closes, which the background work
/// still running in a tree after the drop (a flush's file pool, its deleter)
/// may delay a little past the drop itself.
fn removed_once_closed(dir: &std::path::Path) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while dir.exists() {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    true
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

/// A namespace nothing holds and nothing reached for a pass closes, its
/// trees flushed first: the next use opens it again from disk with every row
/// it had. A namespace a handle holds, or one reached since the last pass,
/// stays open.
#[test]
fn an_idle_namespace_closes_and_opens_again_with_its_rows() {
    let dir = TempDir::new().expect("tmp");
    let (deployment, oracle) = open_deployment(&durable_cfg(&dir));
    let held = deployment
        .engine(deployment.create_namespace("held").expect("create held"))
        .expect("bind held");
    let idle_id = deployment.create_namespace("idle").expect("create idle");
    {
        let idle = deployment.engine(idle_id).expect("bind idle");
        commit(&idle, &oracle, vec![put(b"k", b"kept")]);
    }
    let open = |id| deployment.coordinator.sets().namespace(id).is_some();
    // The first pass finds both reached since the open.
    assert_eq!(deployment.close_idle_namespaces().expect("pass"), 0);
    // A background pass (flush, compaction, capacity) holding the set for a
    // moment keeps it open that pass: the next one closes it.
    let close_idle = || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while open(idle_id) {
            assert!(std::time::Instant::now() < deadline, "never closed");
            deployment.close_idle_namespaces().expect("pass");
        }
    };
    close_idle();
    assert!(open(held.namespace()), "a held namespace stays open");

    // The reopen waits for the closed trees to let go of their files, which
    // no background holder delays past its work.
    let started = std::time::Instant::now();
    let again = deployment.engine(idle_id).expect("open again");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "the reopen waited {:?}",
        started.elapsed()
    );
    assert_eq!(
        node(&again, b"k").as_deref(),
        Some(&b"kept"[..]),
        "a closed namespace keeps its rows"
    );
    drop(again);
    // Reached by the read: the next pass leaves it open.
    assert_eq!(deployment.close_idle_namespaces().expect("pass"), 0);
    assert!(open(idle_id));
    close_idle();

    // Dropped while closed: its directories go with it.
    deployment.drop_namespace("idle").expect("drop");
    for gone in namespace_dirs(&deployment.endpoints, idle_id) {
        assert!(
            removed_once_closed(&gone),
            "a closed namespace's files stayed: {:?}",
            walk(&gone)
        );
    }
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
        removed_once_closed(&a_dir),
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

/// A dropped namespace's directories go only once its trees are closed:
/// while anything of a tree still works in its directory (here a handle a
/// background pass took before the drop), removing the directory under it
/// races that work and leaves files behind. Once the last handle goes, the
/// directories are removed without another call.
#[test]
fn a_drop_removes_the_directories_once_the_trees_are_closed() {
    let dir = TempDir::new().expect("tmp");
    let config = durable_cfg(&dir);
    let (deployment, oracle) = open_deployment(&config);
    let a_id = deployment.create_namespace("a").expect("a");
    {
        let a = deployment.engine(a_id).expect("a");
        commit(&a, &oracle, vec![put(b"k", b"a")]);
        a.persist().expect("persist");
    }
    let held = deployment
        .coordinator
        .sets()
        .namespace(a_id)
        .and_then(|set| set.get(Partition::Node).cloned())
        .expect("a's node tree is open");
    let a_dir = dir.path().join(Scope::Namespace(a_id).dir());

    deployment.drop_namespace("a").expect("drop");
    assert!(
        a_dir.exists(),
        "the directories went while a tree still worked in them"
    );
    drop(held);
    assert!(
        removed_once_closed(&a_dir),
        "the closed trees' directories stay: {:?}",
        walk(&a_dir)
    );
}

/// A node row's value: compressible, and its own for every row and
/// namespace, so a read proves the exact value came back.
fn row_value(namespace: &str, row: u32) -> Vec<u8> {
    format!("{namespace}:{row:08};").repeat(64).into_bytes()
}

/// Commit node rows `rows` of `namespace` into `engine` and flush them into
/// one new table.
fn write_rows(
    engine: &StorageEngine,
    oracle: &TimestampOracle,
    namespace: &str,
    rows: std::ops::Range<u32>,
) {
    let mutations = rows
        .map(|i| put(format!("row-{i:08}").as_bytes(), &row_value(namespace, i)))
        .collect();
    commit(engine, oracle, mutations);
    engine.persist().expect("persist");
}

/// Every row of `rows` of `namespace` reads back its exact value.
fn assert_rows(engine: &StorageEngine, namespace: &str, rows: std::ops::Range<u32>) {
    for i in rows {
        assert_eq!(
            node(engine, format!("row-{i:08}").as_bytes()),
            Some(row_value(namespace, i)),
            "row {i} of {namespace}"
        );
    }
}

/// One installed table of a node tree, as it lies on disk.
#[derive(Debug, PartialEq)]
struct InstalledTable {
    id: lsm_tree::TableId,
    level: usize,
    /// The data-block codec the table's own metadata records: the codec is
    /// not repeated per block, a block's header only says it is compressed.
    codec: lsm_tree::CompressionType,
    bytes: Vec<u8>,
}

/// The installed tables of `engine`'s node tree, in id order.
fn node_tables(engine: &StorageEngine) -> Vec<InstalledTable> {
    use lsm_tree::AbstractTree;
    let version = engine
        .tree(Partition::Node)
        .expect("tree")
        .current_version();
    let mut tables = Vec::new();
    for level in 0..version.level_count() {
        let Some(runs) = version.level(level) else {
            continue;
        };
        for table in runs.iter().flat_map(|run| run.iter()) {
            tables.push(InstalledTable {
                id: table.id(),
                level,
                codec: table.metadata.data_block_compression,
                bytes: std::fs::read(table.path.as_path()).expect("table bytes"),
            });
        }
    }
    tables.sort_by_key(|t| t.id);
    tables
}

/// The namespace compression policy on real tables, the hot band: two
/// namespaces on one endpoint flush with their own codecs; changing one
/// namespace's codec while its tree is open rewrites nothing, leaves its
/// installed table under the same identity and bytes, and only its next
/// flush writes the new codec while the other namespace keeps its own; both
/// generations read back exactly after a cold reopen.
#[test]
fn a_namespace_codec_change_applies_to_its_next_tables_only() {
    use coordinode_core::namespace::{BlockCodec, NamespaceStorage};
    use lsm_tree::CompressionType;
    let dir = TempDir::new().expect("tmp");
    let config = durable_cfg(&dir);
    let (deployment, oracle) = open_deployment(&config);
    let hot = |codec| NamespaceStorage {
        compression: Some(codec),
        ..NamespaceStorage::default()
    };
    let first_id = deployment
        .create_namespace_with("first", hot(BlockCodec::Lz4))
        .expect("first");
    let second_id = deployment
        .create_namespace_with("second", hot(BlockCodec::None))
        .expect("second");
    let first = deployment.engine(first_id).expect("first");
    let second = deployment.engine(second_id).expect("second");
    write_rows(&first, &oracle, "first", 0..500);
    write_rows(&second, &oracle, "second", 0..500);
    let first_before = node_tables(&first);
    let second_before = node_tables(&second);
    // A flush writes with the hot codec, wherever the engine then installs
    // the table (into an empty tree it goes straight to the bottom level).
    assert_eq!(
        first_before.iter().map(|t| t.codec).collect::<Vec<_>>(),
        vec![CompressionType::Lz4]
    );
    assert_eq!(
        second_before.iter().map(|t| t.codec).collect::<Vec<_>>(),
        vec![CompressionType::None]
    );

    deployment
        .set_namespace_storage("first", hot(BlockCodec::Zstd(3)))
        .expect("alter");
    assert_eq!(
        node_tables(&first),
        first_before,
        "the change rewrote a table"
    );
    assert_eq!(node_tables(&second), second_before);

    write_rows(&first, &oracle, "first", 500..1000);
    write_rows(&second, &oracle, "second", 500..1000);
    // Each namespace's tables after the next flush: one written before the
    // change and still installed is the very one (same bytes); every table
    // written since, by the flush or by the engine's own compaction of the
    // two, carries that namespace's current codec, never the other's. Whether
    // a compaction ran by now is the machine's timing, not something the
    // codec change decides.
    let first_after = node_tables(&first);
    let second_after = node_tables(&second);
    for (name, before, after, codec) in [
        (
            "first",
            &first_before,
            &first_after,
            CompressionType::Zstd(3),
        ),
        (
            "second",
            &second_before,
            &second_after,
            CompressionType::None,
        ),
    ] {
        for table in after {
            match before.iter().find(|t| t.id == table.id) {
                Some(earlier) => assert_eq!(
                    table, earlier,
                    "{name}'s table {} written before the change stays the same bytes",
                    table.id
                ),
                None => assert_eq!(
                    table.codec, codec,
                    "{name}'s table {} written since carries its own codec",
                    table.id
                ),
            }
        }
        assert!(
            after.iter().any(|t| before.iter().all(|b| b.id != t.id)),
            "{name}'s next flush wrote a table: {after:?}"
        );
    }
    drop((first, second));
    drop(deployment);

    // After a cold reopen each namespace serves what it had installed: a
    // table still installed is the very one (same bytes), and one the
    // engine's own compaction wrote since carries the namespace's current
    // codec, never the other's; every row of both generations reads back.
    // Whether a compaction ran by the time of the check is the machine's
    // timing, not something the codec change decides.
    let (deployment, _) = open_deployment(&config);
    for (name, installed, codec) in [
        ("first", &first_after, CompressionType::Zstd(3)),
        ("second", &second_after, CompressionType::None),
    ] {
        let engine = deployment.engine_named(name).expect("reopened");
        for table in node_tables(&engine) {
            match installed.iter().find(|t| t.id == table.id) {
                Some(before) => assert_eq!(
                    &table, before,
                    "{name}'s table {} changed across the reopen",
                    table.id
                ),
                None => assert_eq!(
                    table.codec, codec,
                    "{name}'s compacted table {} carries its own codec",
                    table.id
                ),
            }
        }
        assert_rows(&engine, name, 0..1000);
    }
}

/// The namespace compression policy on real tables, the cold band: a
/// compaction onto a cold level writes the namespace's own cold codec,
/// whatever its hot codec; a namespace that sets no cold codec takes the
/// deployment's; and a change naming only the cold codec keeps the hot one
/// across a reopen, which replays the catalog.
#[cfg(feature = "zstd")]
#[test]
fn a_cold_level_takes_the_namespaces_own_cold_codec() {
    use coordinode_core::namespace::{BlockCodec, NamespaceStorage};
    use lsm_tree::CompressionType;
    let dir = TempDir::new().expect("tmp");
    let config = durable_cfg(&dir);
    let (deployment, oracle) = open_deployment(&config);
    let first_id = deployment
        .create_namespace_with(
            "first",
            NamespaceStorage {
                compression: Some(BlockCodec::Zstd(3)),
                ..NamespaceStorage::default()
            },
        )
        .expect("first");
    let second_id = deployment
        .create_namespace_with(
            "second",
            NamespaceStorage {
                compression: Some(BlockCodec::None),
                ..NamespaceStorage::default()
            },
        )
        .expect("second");
    deployment
        .set_namespace_storage(
            "first",
            NamespaceStorage {
                cold_compression: Some(BlockCodec::Lz4),
                ..NamespaceStorage::default()
            },
        )
        .expect("alter cold");
    let first = deployment.engine(first_id).expect("first");
    let second = deployment.engine(second_id).expect("second");
    for (engine, name) in [(&first, "first"), (&second, "second")] {
        write_rows(engine, &oracle, name, 0..500);
        write_rows(engine, &oracle, name, 500..1000);
        engine.major_compact(Partition::Node).expect("compact");
    }
    let cold_from = usize::from(config.compression.cold_level_threshold);
    let codecs = |engine: &StorageEngine| {
        let tables = node_tables(engine);
        assert!(
            !tables.is_empty() && tables.iter().all(|t| t.level >= cold_from),
            "the compaction wrote onto a cold level: {tables:?}"
        );
        tables.iter().map(|t| t.codec).collect::<Vec<_>>()
    };
    assert!(codecs(&first).iter().all(|c| *c == CompressionType::Lz4));
    // The deployment's cold codec is zstd at its default level.
    assert!(
        codecs(&second)
            .iter()
            .all(|c| *c == CompressionType::Zstd(3))
    );
    drop((first, second));
    drop(deployment);

    let (deployment, _) = open_deployment(&config);
    assert_eq!(
        deployment.namespace_storage("first").expect("settings"),
        NamespaceStorage {
            compression: Some(BlockCodec::Zstd(3)),
            cold_compression: Some(BlockCodec::Lz4),
            dictionaries: None,
        },
        "the cold change kept the hot codec it did not name"
    );
    for name in ["first", "second"] {
        assert_rows(
            &deployment.engine_named(name).expect("reopened"),
            name,
            0..1000,
        );
    }
}

/// A drop removes the namespace's directory on every endpoint its trees can
/// be placed on, hot and cold alike, and nothing of another namespace on
/// either.
#[test]
fn a_drop_removes_the_namespace_on_every_tier() {
    let hot = TempDir::new().expect("hot");
    let cold = TempDir::new().expect("cold");
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let (deployment, oracle) = open_deployment(&config);
    let a_id = deployment.create_namespace("a").expect("a");
    let b_id = deployment.create_namespace("b").expect("b");
    for id in [a_id, b_id] {
        let engine = deployment.engine(id).expect("bind");
        commit(&engine, &oracle, vec![put(b"k", b"v")]);
        engine.persist().expect("persist");
    }
    // Whatever placement put on the cold tier: a file the namespace holds
    // there, which the drop must take with the rest.
    for id in [a_id, b_id] {
        let on_cold = cold.path().join(Scope::Namespace(id).dir()).join("placed");
        std::fs::create_dir_all(&on_cold).expect("cold dir");
        std::fs::write(on_cold.join("segment"), b"x").expect("cold file");
    }
    deployment.drop_namespace("a").expect("drop");
    for root in [hot.path(), cold.path()] {
        let gone = root.join(Scope::Namespace(a_id).dir());
        assert!(
            removed_once_closed(&gone),
            "left on {root:?}: {:?}",
            walk(&gone)
        );
        assert!(
            root.join(Scope::Namespace(b_id).dir()).exists(),
            "another namespace's files went on {root:?}"
        );
    }
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

/// The tiered cache keeps each namespace's values apart: a value one
/// namespace read into the cache is never served to another reading the same
/// key, and a write in one namespace leaves the other's cached value served.
#[test]
fn the_tiered_cache_keeps_namespaces_apart() {
    let dir = TempDir::new().expect("tmp");
    let cache_dir = TempDir::new().expect("tmp");
    let mut config = durable_cfg(&dir);
    config.cache = crate::cache::config::TieredCacheConfig {
        layers: vec![crate::cache::config::CacheLayerConfig {
            path: cache_dir.path().join("layer0.cache"),
            max_bytes: 1 << 20,
            max_entries: 1000,
            compaction_threshold: 0.5,
        }],
        compaction_interval_secs: 0,
        ..Default::default()
    };
    let (deployment, oracle) = open_deployment(&config);
    let a = deployment
        .engine(deployment.create_namespace("a").expect("a"))
        .expect("bind a");
    let b = deployment
        .engine(deployment.create_namespace("b").expect("b"))
        .expect("bind b");
    commit(&a, &oracle, vec![put(b"k", b"of-a")]);
    commit(&b, &oracle, vec![put(b"k", b"of-b")]);

    // a's read fills the cache; b's read of the same key is b's own row.
    assert_eq!(node(&a, b"k").as_deref(), Some(&b"of-a"[..]));
    assert_eq!(node(&b, b"k").as_deref(), Some(&b"of-b"[..]));
    assert_eq!(node(&a, b"k").as_deref(), Some(&b"of-a"[..]));

    commit(&b, &oracle, vec![put(b"k", b"of-b-2")]);
    assert_eq!(node(&b, b"k").as_deref(), Some(&b"of-b-2"[..]));
    assert_eq!(node(&a, b"k").as_deref(), Some(&b"of-a"[..]));
    let values = a.multi_get(Partition::Node, &[b"k"]).expect("multi get");
    assert_eq!(values[0].as_deref(), Some(&b"of-a"[..]));
}

/// A tap on one namespace's partition sees that namespace's writes only:
/// another namespace's writes and range deletes to the same partition reach
/// it neither as keys nor as a replacement.
#[test]
fn a_write_tap_sees_its_own_namespace_only() {
    use crate::engine::tap::Tapped;
    let dir = TempDir::new().expect("tmp");
    let (deployment, oracle) = open_deployment(&durable_cfg(&dir));
    let a = deployment
        .engine(deployment.create_namespace("a").expect("a"))
        .expect("bind a");
    let b = deployment
        .engine(deployment.create_namespace("b").expect("b"))
        .expect("bind b");
    let (tap, _) = a.tap_writes(Partition::Node).expect("tap");

    commit(&b, &oracle, vec![put(b"of-b", b"1")]);
    b.put(Partition::Node, b"direct-b", b"1")
        .expect("direct put");
    assert_eq!(
        tap.take(),
        Tapped::Keys(Vec::new()),
        "a's tap took b's writes"
    );

    b.remove_range(Partition::Node, b"a", b"z")
        .expect("range delete");
    assert_eq!(
        tap.take(),
        Tapped::Keys(Vec::new()),
        "b's range delete replaced a's tap"
    );

    commit(&a, &oracle, vec![put(b"of-a", b"1")]);
    assert_eq!(tap.take(), Tapped::Keys(vec![b"of-a".to_vec()]));
}

/// A follower of one namespace's applied commits hears of that namespace's
/// keys only: another namespace's commit to the same partition queues it no
/// event, and its own commit queues one with its own key.
#[test]
fn an_applied_feed_reports_its_own_namespace_only() {
    use crate::engine::applied::AppliedEvent;
    let dir = TempDir::new().expect("tmp");
    let (deployment, oracle) = open_deployment(&durable_cfg(&dir));
    let a = deployment
        .engine(deployment.create_namespace("a").expect("a"))
        .expect("bind a");
    let b = deployment
        .engine(deployment.create_namespace("b").expect("b"))
        .expect("bind b");
    let feed = a.subscribe_applied(Partition::Node, 16);

    commit(&b, &oracle, vec![put(b"of-b", b"1")]);
    assert_eq!(feed.try_next(), None, "a's feed heard of b's commit");

    commit(&a, &oracle, vec![put(b"of-a", b"1")]);
    match feed.try_next() {
        Some(AppliedEvent::Keys { keys, .. }) => assert_eq!(&*keys, &[b"of-a".to_vec()]),
        other => panic!("a's own commit was not reported: {other:?}"),
    }
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
    // The namespace tests build units and paths for before holding the
    // store's engine is the one a fresh store creates.
    assert_eq!(first, crate::engine::scope::STORE_NAMESPACE);
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

/// A store that created a namespace, even an empty one, or dropped it again,
/// has a history of its own a group it joins would have to lose: only a
/// store that never created one is free to join. Without the check an empty
/// `default` passed as nothing, and the joiner's catalog was overwritten.
#[test]
fn a_created_namespace_is_local_state_even_empty() {
    let dir = TempDir::new().expect("tmp");
    let (deployment, _) = open_deployment(&durable_cfg(&dir));
    assert!(
        !deployment
            .holds_local_state()
            .expect("read the fresh store"),
        "a store that never created a namespace holds nothing of its own"
    );

    deployment.create_namespace("scratch").expect("create");
    assert!(
        !deployment.holds_user_data().expect("read"),
        "an empty namespace holds no data"
    );
    assert!(
        deployment.holds_local_state().expect("read"),
        "an empty namespace is still the store's own"
    );

    deployment.drop_namespace("scratch").expect("drop");
    assert!(
        deployment.holds_local_state().expect("read"),
        "a namespace dropped here was still created here"
    );
}

/// A member that joins a group opens with no namespace: its engine reaches
/// the deployment's records at once and binds to the namespace its group's
/// catalog brings on its first use after that.
#[test]
fn a_member_engine_binds_to_the_namespace_its_catalog_brings() {
    let dir = TempDir::new().expect("tmp");
    let (deployment, _) = open_deployment(&durable_cfg(&dir));
    let member = deployment.member_engine();
    assert!(
        deployment.live_namespaces().expect("catalog").is_empty(),
        "a member's engine creates no namespace"
    );
    member
        .put(Partition::Deployment, b"raft:sm:applied", b"1")
        .expect("the deployment's records need no namespace");
    assert!(matches!(
        member.get(Partition::Node, b"k"),
        Err(StorageError::NoNamespace { .. })
    ));

    // The catalog arrives: the next use binds to its one namespace.
    let id = deployment
        .create_namespace(DEFAULT_NAMESPACE)
        .expect("create");
    member.put(Partition::Node, b"k", b"v").expect("put");
    assert_eq!(member.namespace(), id);
    let named = deployment.engine(id).expect("engine");
    assert_eq!(node(&named, b"k").as_deref(), Some(&b"v"[..]));
}
