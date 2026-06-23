use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use tempfile::TempDir;

fn disk_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

/// R159 Part C: a `page_ecc = ForceOn` endpoint makes
/// `to_tree_config_with_routing` request `Config::page_ecc(true)`.
/// Verify the engine opens (no `PageEccUnsupported`), and a value
/// survives a flush-to-SST + reopen — i.e. it round-trips through
/// the Reed-Solomon-trailered block codec. Only meaningful with the
/// `page_ecc` feature compiled, so the test is gated on it.
#[cfg(feature = "page_ecc")]
#[test]
fn page_ecc_force_on_round_trips_through_sst() {
    use crate::engine::config::PageEccPolicy;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ecc",
        dir.path(),
        Media::Hdd,
        Durability::Durable, // ForceOn overrides Durable's Auto-off
        Tier::Warm,
    )
    .with_page_ecc(PageEccPolicy::ForceOn)]);

    let engine = StorageEngine::open(&config).expect("open engine with page_ecc");
    engine
        .put(Partition::Node, b"k-ecc", b"v-ecc")
        .expect("put");
    // Flush memtable → SST so the page-ECC block codec runs on disk.
    engine.persist().expect("persist to SST");
    drop(engine);

    // Reopen and read back through the ECC-decoded SST block path.
    let reopened = StorageEngine::open(&config).expect("reopen engine with page_ecc");
    assert_eq!(
        reopened
            .get(Partition::Node, b"k-ecc")
            .expect("get")
            .as_deref(),
        Some(b"v-ecc".as_ref()),
        "value must round-trip through page-ECC SST blocks",
    );
}

#[test]
fn checkpoint_round_trips_all_partitions() {
    let src_dir = TempDir::new().expect("src tempdir");
    let engine = disk_engine(src_dir.path());

    // Seed a couple of partitions, including Schema (which carries the
    // interner) and Node (the main data partition).
    engine
        .put(Partition::Node, b"k-node", b"v-node")
        .expect("put node");
    engine
        .put(Partition::Schema, b"k-schema", b"v-schema")
        .expect("put schema");
    engine
        .put(Partition::Adj, b"adj:T:out:x", b"posting")
        .expect("put adj");

    // Checkpoint into a fresh sibling dir.
    let ckpt_parent = TempDir::new().expect("ckpt parent");
    let target = ckpt_parent.path().join("snap1");
    let summary = engine.create_checkpoint(&target).expect("checkpoint");
    assert_eq!(
        summary.partitions,
        Partition::all().len(),
        "every partition tree must be checkpointed"
    );
    // Partition directories are named by `Partition::name()`, which is
    // lower-case ("node", "schema"). Assert the exact names: a capitalised
    // path silently passes on case-insensitive filesystems (macOS) and
    // fails on case-sensitive ones (Linux CI).
    assert!(
        target.join(Partition::Node.name()).exists(),
        "Node partition dir in checkpoint"
    );
    assert!(
        target.join(Partition::Schema.name()).exists(),
        "Schema partition dir in checkpoint"
    );

    // Drop the source engine, then open a brand-new engine against the
    // checkpoint and confirm every written key is present and correct.
    drop(engine);
    let restored = disk_engine(&target);
    assert_eq!(
        restored
            .get(Partition::Node, b"k-node")
            .expect("get node")
            .as_deref(),
        Some(b"v-node".as_ref()),
        "node value must survive the checkpoint"
    );
    assert_eq!(
        restored
            .get(Partition::Schema, b"k-schema")
            .expect("get schema")
            .as_deref(),
        Some(b"v-schema".as_ref()),
        "schema value must survive the checkpoint"
    );
    assert_eq!(
        restored
            .get(Partition::Adj, b"adj:T:out:x")
            .expect("get adj")
            .as_deref(),
        Some(b"posting".as_ref()),
        "adjacency value must survive the checkpoint"
    );
}

#[test]
fn checkpoint_refuses_existing_target() {
    let src_dir = TempDir::new().expect("src tempdir");
    let engine = disk_engine(src_dir.path());
    let existing = TempDir::new().expect("existing target");
    match engine.create_checkpoint(existing.path()) {
        Err(e) => assert!(
            format!("{e}").contains("already exists"),
            "must refuse to overwrite an existing target, got: {e}"
        ),
        Ok(_) => panic!("checkpoint must refuse an existing target"),
    }
}
