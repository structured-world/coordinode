//! The open settles the Schema tree before it reads the routes, the closure
//! ledger or anything else the rest of the open depends on: a tree whose
//! rebuild was cut short, or that an open salvaged with loss, is rebuilt or
//! its own records proved, and refused otherwise. An absent key in such a
//! tree is never read as a fresh store.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use crate::engine::routing::PartitionRouting;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;

fn endpoint(id: &str, dir: &Path, tier: Tier) -> EndpointConfig {
    EndpointConfig::new(id, dir, Media::Hdd, Durability::Durable, tier)
}

/// A store over `endpoints`, with no background rewrite of its tables: the
/// faults below address tables on disk.
fn config_of(endpoints: Vec<EndpointConfig>) -> StorageConfig {
    let mut config = StorageConfig::with_endpoints(endpoints);
    config.compaction_workers = 0;
    config
}

fn open_journaled(config: &StorageConfig) -> StorageResult<StorageEngine> {
    StorageEngine::open_embedded(config, Arc::new(TimestampOracle::new()))
}

fn commit(engine: &StorageEngine, ts: u64, key: &[u8], value: &[u8]) {
    commit_to(engine, PartitionId::Node, ts, key, value);
}

fn commit_to(engine: &StorageEngine, partition: PartitionId, ts: u64, key: &[u8], value: &[u8]) {
    engine
        .commit_journaled(
            &[Mutation::Put {
                partition,
                key: key.to_vec(),
                value: value.to_vec(),
            }],
            ts,
        )
        .expect("commit");
}

fn routing_key(part: Partition) -> Vec<u8> {
    format!("meta:routing:{}", part.name()).into_bytes()
}

fn read(engine: &StorageEngine, part: Partition, key: &[u8]) -> Option<Vec<u8>> {
    engine
        .get(part, key)
        .expect("read")
        .map(|value| value.to_vec())
}

/// The ids of the tables the open Schema tree's version holds. A table's
/// file is named by its id; a file name alone is reused across opens.
fn schema_table_ids(engine: &StorageEngine) -> HashSet<u64> {
    engine
        .tree(Partition::Schema)
        .expect("schema")
        .current_version()
        .iter_tables()
        .map(lsm_tree::Table::id)
        .collect()
}

/// Flip a byte of the data section of each Schema table under `endpoint`
/// named in `victims`, writing a new file in its place, so a checkpoint
/// sharing the old one by a hard link keeps it whole.
fn damage_schema_tables(endpoint: &Path, victims: &HashSet<u64>) -> usize {
    let dir = endpoint.join(Partition::Schema.name()).join("tables");
    let mut damaged = 0;
    for id in victims {
        let path = dir.join(id.to_string());
        let mut bytes = std::fs::read(&path).expect("read table");
        let archive = lsm_tree::sfa::Reader::from_reader(&mut std::io::Cursor::new(&bytes))
            .expect("read the table layout");
        let data = archive.toc().section(b"data").expect("data section");
        assert!(data.len() > 0, "the victim holds a data block");
        let offset = usize::try_from(data.pos()).expect("data position");
        bytes[offset] ^= 0xFF;
        std::fs::remove_file(&path).expect("unlink the table");
        std::fs::write(&path, &bytes).expect("write the damaged table");
        damaged += 1;
    }
    damaged
}

/// Remove the Schema tree's manifest snapshots and edit logs under
/// `endpoint`, its tables left in place: the next open repairs the tree.
fn lose_schema_manifest(endpoint: &Path) {
    let dir = endpoint.join(Partition::Schema.name());
    let numbered = |rest: Option<&str>| {
        rest.is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()))
    };
    let mut lost = 0;
    for entry in std::fs::read_dir(&dir)
        .expect("read the Schema dir")
        .flatten()
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        if numbered(name.strip_prefix('v')) || numbered(name.strip_prefix("edits-")) {
            std::fs::remove_file(entry.path()).expect("remove a manifest file");
            lost += 1;
        }
    }
    assert!(lost > 0, "no manifest file in {}", dir.display());
}

/// A Schema rebuild cut short after its clear leaves the routes of the other
/// trees nowhere in the tree. The next open takes them from what the rebuild
/// kept before the clear, so every tree opens where its tables are and the
/// store's own records written last are there: a store whose routes differ
/// from the default for its endpoints (an endpoint added after the store was
/// created) finds its node row again. Read from the cleared tree, the routes
/// were absent and the open wrote the default in their place.
#[test]
fn an_interrupted_schema_rebuild_keeps_the_routes_of_the_other_trees() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let (a, b) = (root.path().join("a"), root.path().join("b"));
    std::fs::create_dir_all(&a).expect("dir a");
    std::fs::create_dir_all(&b).expect("dir b");
    {
        let engine =
            open_journaled(&config_of(vec![endpoint("a", &a, Tier::Warm)])).expect("create");
        commit(&engine, 10, b"node:0:00000001", b"first");
        engine.persist().expect("persist");
    }
    let both = config_of(vec![
        endpoint("a", &a, Tier::Warm),
        endpoint("b", &b, Tier::Hot),
    ]);
    let routes = {
        let engine = open_journaled(&both).expect("open with an endpoint added");
        let routes = engine
            .get(Partition::Schema, &routing_key(Partition::Node))
            .expect("read the node routes")
            .expect("the node routes are stored");
        let default = rmp_serde::to_vec(&PartitionRouting::default_for_endpoints(&both.endpoints))
            .expect("encode the default");
        assert_ne!(
            routes.as_ref(),
            default.as_slice(),
            "the store's routes must differ from the default for this to test anything"
        );
        engine
            .put(Partition::Schema, b"meta:probe", b"written last")
            .expect("a record of the store's own");
        let schema = engine.tree(Partition::Schema).expect("schema").clone();
        assert!(
            engine
                .clear_partition_with(Partition::Schema, &schema, || {
                    Err(StorageError::Io("cut short".into()))
                })
                .is_err()
        );
        routes
    };

    let engine = open_journaled(&both).expect("reopen");
    assert_eq!(
        engine
            .get(Partition::Schema, &routing_key(Partition::Node))
            .expect("read the node routes")
            .as_deref(),
        Some(routes.as_ref()),
        "the open replaced the store's routes"
    );
    assert_eq!(
        engine
            .get(Partition::Schema, b"meta:probe")
            .expect("read the record")
            .as_deref(),
        Some(&b"written last"[..]),
        "the store's own record written last is gone"
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:00000001")
            .expect("read the node row")
            .as_deref(),
        Some(&b"first"[..]),
        "the node tree opened elsewhere than its tables"
    );
    assert_eq!(
        engine.pending_rebuilds().expect("intents"),
        [Partition::Schema],
        "with no checkpoint the rebuild stays owed"
    );
}

/// Apply Raft entry `index` as the apply path does: its commit, the closure
/// records it plans with its coverage, then the plan installed.
fn closure_entry(engine: &StorageEngine, index: u64, commits: &[u64], below: u64) {
    for &ts in commits {
        engine.note_raft_applied(index, ts);
    }
    if let Some((records, at)) = engine
        .closure_record(below, index, false)
        .expect("plan the closure records")
    {
        engine
            .apply_raft_proposal(&records, at, index, 0, |_| false)
            .expect("write the closure records");
    }
    engine
        .raise_closure_frontier(below, index)
        .expect("install the plan");
}

/// An open whose salvage of the Schema tree lost the closure ledger's
/// records, the store's routes intact in an older table, is refused: the
/// ledger is not read as absent, and nothing the open would initialise is
/// written. Read from the salvaged tree, the head was absent and the store
/// opened as if no prefix of its log had a commit ceiling.
#[test]
fn a_salvage_that_lost_the_closure_ledger_refuses_the_open() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let config = config_of(vec![endpoint("default", dir.path(), Tier::Warm)]);
    {
        let engine = StorageEngine::open(&config).expect("create");
        engine.persist().expect("persist the routes");
    }
    let ledger_tables = {
        let engine = StorageEngine::open(&config).expect("open");
        let with_routes = schema_table_ids(&engine);
        assert!(!with_routes.is_empty(), "the routes are in a table");
        for index in 1..=5 {
            closure_entry(&engine, index, &[index * 10], 0);
        }
        closure_entry(&engine, 6, &[], 51);
        assert_eq!(engine.closure_frontier(), 51, "the ledger closed the log");
        engine.persist().expect("persist the ledger");
        &schema_table_ids(&engine) - &with_routes
    };
    assert_eq!(
        damage_schema_tables(dir.path(), &ledger_tables),
        1,
        "the ledger is in a table of its own"
    );
    lose_schema_manifest(dir.path());

    match StorageEngine::open(&config) {
        Ok(engine) => panic!(
            "opened without its closure ledger: frontier {}",
            engine.closure_frontier()
        ),
        Err(StorageError::SchemaRecoveryRefused(RebuildRefusal::LocalStateLost)) => {}
        Err(e) => panic!("refused for another reason: {e}"),
    }
    // The salvage that ran before the refusal set the unreadable table
    // aside; the refusal itself writes nothing, so the next open finds the
    // tree's tables as this one left them (the tree rewrites only its own
    // manifest when it opens) and refuses again.
    let tables = dir.path().join(Partition::Schema.name()).join("tables");
    let refused = tree_files(&tables);
    assert!(matches!(
        StorageEngine::open(&config),
        Err(StorageError::SchemaRecoveryRefused(
            RebuildRefusal::LocalStateLost
        ))
    ));
    assert_eq!(
        tree_files(&tables),
        refused,
        "a refused open changed the Schema tables"
    );
}

/// Every file under `dir` but its lock, with its bytes.
fn tree_files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in std::fs::read_dir(&at).expect("read dir").flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if entry.file_name() != "LOCK" {
                let bytes = std::fs::read(&path).expect("read file");
                files.insert(path, bytes);
            }
        }
    }
    files
}

/// A journaled store whose routes differ from the default for its endpoints
/// (a hot endpoint `b` added to a store created on a warm `a`), with a
/// checkpoint between two Schema writes, journaled writes after it and, when
/// `late_record`, a record of its own written after it as well.
struct Checkpointed {
    root: tempfile::TempDir,
    config: StorageConfig,
    routes: Vec<u8>,
    checkpoint: PathBuf,
    /// Every row the store holds, by partition and key.
    rows: Vec<(Partition, &'static [u8], &'static [u8])>,
}

impl Checkpointed {
    fn new(late_record: bool) -> Self {
        let root = tempfile::TempDir::new().expect("tempdir");
        let (a, b) = (root.path().join("a"), root.path().join("b"));
        std::fs::create_dir_all(&a).expect("dir a");
        std::fs::create_dir_all(&b).expect("dir b");
        {
            let engine =
                open_journaled(&config_of(vec![endpoint("a", &a, Tier::Warm)])).expect("create");
            commit(&engine, 10, b"node:0:00000001", b"first");
            engine.persist().expect("persist");
        }
        let config = config_of(vec![
            endpoint("a", &a, Tier::Warm),
            endpoint("b", &b, Tier::Hot),
        ]);
        let checkpoint = crate::engine::rebuild::checkpoint_root(&a).join(format!(
            "{}{:020}",
            crate::engine::rebuild::CHECKPOINT_PREFIX,
            0
        ));
        let routes = {
            let engine = open_journaled(&config).expect("open with an endpoint added");
            let routes = read(&engine, Partition::Schema, &routing_key(Partition::Node))
                .expect("the node routes are stored");
            let default =
                rmp_serde::to_vec(&PartitionRouting::default_for_endpoints(&config.endpoints))
                    .expect("encode the default");
            assert_ne!(routes, default, "the routes must differ from the default");
            commit_to(&engine, PartitionId::Schema, 20, b"schema:x", b"before");
            engine.create_checkpoint(&checkpoint).expect("checkpoint");
            commit_to(&engine, PartitionId::Schema, 30, b"schema:x", b"after");
            commit_to(&engine, PartitionId::Schema, 40, b"schema:y", b"tail");
            commit(&engine, 50, b"node:0:00000002", b"second");
            if late_record {
                engine
                    .put(Partition::Schema, b"meta:probe", b"written last")
                    .expect("a record of the store's own");
            }
            engine.persist().expect("persist");
            routes
        };
        let mut rows = vec![
            (Partition::Schema, &b"schema:x"[..], &b"after"[..]),
            (Partition::Schema, b"schema:y", b"tail"),
            (Partition::Node, b"node:0:00000001", b"first"),
            (Partition::Node, b"node:0:00000002", b"second"),
        ];
        if late_record {
            rows.push((Partition::Schema, b"meta:probe", b"written last"));
        }
        Self {
            root,
            config,
            routes,
            checkpoint,
            rows,
        }
    }

    fn primary(&self) -> PathBuf {
        self.root.path().join("a")
    }

    /// Clear the Schema tree as a rebuild does, and stop right after.
    fn cut_a_schema_rebuild_short(&self) {
        let engine = open_journaled(&self.config).expect("open");
        let schema = engine.tree(Partition::Schema).expect("schema").clone();
        assert!(
            engine
                .clear_partition_with(Partition::Schema, &schema, || {
                    Err(StorageError::Io("cut short".into()))
                })
                .is_err()
        );
    }

    /// The store, opened, holds its routes and every row, owes no rebuild
    /// and no repair, and left no staged copy behind.
    fn assert_whole(&self, engine: &StorageEngine) {
        assert_eq!(
            read(engine, Partition::Schema, &routing_key(Partition::Node)).as_deref(),
            Some(self.routes.as_slice()),
            "the store's routes"
        );
        for &(part, key, value) in &self.rows {
            assert_eq!(
                read(engine, part, key).as_deref(),
                Some(value),
                "{} {}",
                part.name(),
                key.escape_ascii()
            );
        }
        assert!(engine.pending_rebuilds().expect("intents").is_empty());
        assert!(
            engine.open_repairs().is_empty(),
            "{:?}",
            engine.open_repairs()
        );
        assert!(!self.primary().join("schema-source").exists());
    }
}

/// A Schema rebuild cut short after its clear is finished by the next open
/// from the latest checkpoint and the journal, before anything reads the
/// tree: the routes (which differ from the default) and the record of its
/// own written after the checkpoint come from what the rebuild kept, the
/// rows from the checkpoint and the journal. The checkpoint is read, not
/// opened: its files are as they were. A second open finds a whole tree and
/// nothing owed.
#[test]
fn a_cut_short_schema_rebuild_is_finished_from_the_checkpoint() {
    let store = Checkpointed::new(true);
    let checkpoint = tree_files(&store.checkpoint);
    store.cut_a_schema_rebuild_short();

    {
        let engine = open_journaled(&store.config).expect("reopen");
        store.assert_whole(&engine);
    }
    assert_eq!(
        tree_files(&store.checkpoint),
        checkpoint,
        "the open changed the checkpoint"
    );
    let engine = open_journaled(&store.config).expect("open again");
    store.assert_whole(&engine);
}

/// A rebuild at the open cut short after its clear leaves the rebuild owed
/// with the records it keeps; the next open finishes it.
#[test]
fn a_rebuild_at_the_open_cut_short_is_finished_by_the_next_open() {
    let store = Checkpointed::new(true);
    store.cut_a_schema_rebuild_short();

    super::schema_bootstrap::CUT_AFTER_CLEAR.store(true, std::sync::atomic::Ordering::Release);
    assert!(
        open_journaled(&store.config).is_err(),
        "the cut did not stop the open"
    );
    assert_eq!(
        read_rebuild_intents(&store.primary())
            .expect("intents")
            .partitions,
        [Partition::Schema],
        "the rebuild is no longer owed"
    );
    let engine = open_journaled(&store.config).expect("open after the cut");
    store.assert_whole(&engine);
}

/// A Schema rebuild owed when the journal no longer holds entries the other
/// trees hold (and the checkpoint lacks) is refused: the tree is not
/// rebuilt without them, and the rebuild stays owed with the records it
/// keeps, the checkpoint as it was.
#[test]
fn a_rebuild_the_journal_cannot_cover_is_refused() {
    let store = Checkpointed::new(true);
    store.cut_a_schema_rebuild_short();
    let checkpoint = tree_files(&store.checkpoint);
    let journal = store.primary().join("oplog").join("0");
    for entry in std::fs::read_dir(&journal).expect("journal").flatten() {
        std::fs::remove_file(entry.path()).expect("lose a journal segment");
    }

    match open_journaled(&store.config) {
        Err(StorageError::SchemaRecoveryRefused(RebuildRefusal::UncoveredEntry { .. })) => {}
        Err(e) => panic!("refused for another reason: {e}"),
        Ok(_) => panic!("opened without the entries the Schema tree lacks"),
    }
    let kept = read_kept_rows(&store.primary())
        .expect("read the intent")
        .expect("the rebuild keeps its records");
    assert!(
        kept.iter().any(
            |(key, value)| key.as_slice() == routing_key(Partition::Node).as_slice()
                && value == &store.routes
        ),
        "the routes are no longer kept"
    );
    assert_eq!(tree_files(&store.checkpoint), checkpoint);
}

/// A Schema rebuild owed from a checkpoint whose own Schema tree opens only
/// by a salvage that drops data is refused, and the checkpoint is left as
/// it was.
#[test]
fn a_rebuild_from_a_damaged_checkpoint_is_refused() {
    let store = Checkpointed::new(true);
    store.cut_a_schema_rebuild_short();
    let schema = store.checkpoint.join(Partition::Schema.name());
    let tables: HashSet<u64> = std::fs::read_dir(schema.join("tables"))
        .expect("checkpoint tables")
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
        .collect();
    assert!(damage_schema_tables(&store.checkpoint, &tables) > 0);
    lose_schema_manifest(&store.checkpoint);
    let checkpoint = tree_files(&store.checkpoint);

    match open_journaled(&store.config) {
        Err(StorageError::SchemaRecoveryRefused(RebuildRefusal::CheckpointDamaged)) => {}
        Err(e) => panic!("refused for another reason: {e}"),
        Ok(_) => panic!("rebuilt from a damaged checkpoint"),
    }
    assert_eq!(tree_files(&store.checkpoint), checkpoint);
}

/// A salvage of the Schema tree that lost a table whose key range is not
/// known cannot show the store's own records survived: the open is refused
/// with a checkpoint at hand, since nothing but the tree holds the routes.
#[test]
fn a_salvage_with_an_unknown_loss_refuses_the_open() {
    let store = Checkpointed::new(false);
    let primary = store.primary();
    let newest = {
        let engine = open_journaled(&store.config).expect("open");
        schema_table_ids(&engine)
            .into_iter()
            .max()
            .expect("a Schema table")
    };
    let path = primary
        .join(Partition::Schema.name())
        .join("tables")
        .join(newest.to_string());
    let bytes = std::fs::read(&path).expect("read the table");
    std::fs::remove_file(&path).expect("unlink the table");
    std::fs::write(&path, &bytes[..bytes.len() / 2]).expect("write half of it");
    lose_schema_manifest(&primary);

    match open_journaled(&store.config) {
        Err(StorageError::SchemaRecoveryRefused(RebuildRefusal::LocalStateLost)) => {}
        Err(e) => panic!("refused for another reason: {e}"),
        Ok(engine) => panic!("opened after an unknown loss: {:?}", engine.open_repairs()),
    }
}

/// A Schema tree with a corrupt block, found by the open's verification
/// where a checkpoint backs the trees, is rebuilt before anything reads it.
/// A clean one is verified once: the repair pass after the open reuses it.
#[test]
fn a_corrupt_schema_tree_is_rebuilt_and_a_clean_one_verified_once() {
    let store = Checkpointed::new(false);
    {
        let engine = open_journaled(&store.config).expect("open");
        assert!(
            engine.verified_at_open(Partition::Schema),
            "a clean tree verified at the open"
        );
        let after_checkpoint = &schema_table_ids(&engine)
            - &std::fs::read_dir(
                store
                    .checkpoint
                    .join(Partition::Schema.name())
                    .join("tables"),
            )
            .expect("checkpoint tables")
            .flatten()
            .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
            .collect::<HashSet<u64>>();
        drop(engine);
        assert!(damage_schema_tables(&store.primary(), &after_checkpoint) > 0);
    }

    let engine = open_journaled(&store.config).expect("reopen");
    store.assert_whole(&engine);
    assert!(
        !engine.verified_at_open(Partition::Schema),
        "a rebuilt tree is verified again by the repair pass"
    );
}

/// Bytes the files under `dir` hold, each name counted.
fn dir_len(dir: &Path) -> u64 {
    if !dir.exists() {
        return 0;
    }
    tree_files(dir)
        .values()
        .map(|bytes| bytes.len() as u64)
        .sum()
}

/// The copy of a checkpoint's Schema tree a rebuild at the open stages is
/// charged to the endpoint's footprint before it is written: an endpoint
/// with less room than the copy refuses the open, grows by nothing past its
/// allowance, keeps the checkpoint as it was and the rebuild owed, and the
/// next open with room finishes it. Copied unmetered, the open passed the
/// endpoint's limit.
#[test]
fn a_schema_copy_that_does_not_fit_the_endpoint_is_refused() {
    let store = Checkpointed::new(true);
    let primary = store.primary();
    {
        // Rows that do not compress, kept in the checkpoint taken after them.
        let engine = open_journaled(&store.config).expect("open");
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for i in 0..6_000u32 {
            let value: Vec<u8> = (0..1024)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state.to_le_bytes()[0]
                })
                .collect();
            engine
                .put(
                    Partition::Schema,
                    format!("schema:big:{i:06}").as_bytes(),
                    &value,
                )
                .expect("a schema row");
        }
        engine.persist().expect("persist");
        engine
            .create_checkpoint(
                &crate::engine::rebuild::checkpoint_root(&primary).join(format!(
                    "{}{:020}",
                    crate::engine::rebuild::CHECKPOINT_PREFIX,
                    1
                )),
            )
            .expect("checkpoint");
    }
    store.cut_a_schema_rebuild_short();
    let latest = crate::engine::rebuild::latest_checkpoint(
        &crate::engine::rebuild::checkpoint_root(&primary),
    )
    .expect("list")
    .expect("a checkpoint");
    let schema_copy = dir_len(&latest.join(Partition::Schema.name()));
    assert!(
        schema_copy > 4 * crate::engine::footprint::OVER_LIMIT_ALLOWANCE,
        "the Schema copy ({schema_copy} bytes) must not fit the room left"
    );
    let checkpoint = tree_files(&latest);

    // Room for what the endpoint holds and a little more, not the copy.
    let held = dir_len(&primary);
    let mut tight = store.config.clone();
    tight.endpoints[0] = tight.endpoints[0].clone().with_hard_limit_bytes(
        crate::engine::config::MIN_HARD_LIMIT_BYTES
            .max(held + crate::engine::footprint::OVER_LIMIT_ALLOWANCE),
    );
    let opened = open_journaled(&tight);
    let staged = dir_len(&primary.join("schema-source"));
    assert!(
        staged <= crate::engine::footprint::OVER_LIMIT_ALLOWANCE,
        "the refused open wrote {staged} bytes of the copy, past its allowance"
    );
    assert!(
        !primary.join("schema-source").exists(),
        "the refused copy was left behind"
    );
    match opened {
        Err(StorageError::CapacityExhausted { .. }) => {}
        Err(e) => panic!("refused for another reason: {e}"),
        Ok(_) => panic!("the open copied the Schema tree past the endpoint's limit"),
    }
    assert_eq!(tree_files(&latest), checkpoint, "the checkpoint changed");
    assert_eq!(
        read_rebuild_intents(&primary).expect("intents").partitions,
        [Partition::Schema],
        "the rebuild is no longer owed"
    );

    let engine = open_journaled(&store.config).expect("open with room");
    store.assert_whole(&engine);
}
