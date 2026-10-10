use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;

fn config(path: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        path,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open() -> (StorageEngine, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    (StorageEngine::open(&config(dir.path())).expect("open"), dir)
}

/// The logical key of an entry of `generation`: `tag / generation / rest`.
fn entry(tag: u8, generation: u64, rest: &[u8]) -> Vec<u8> {
    key(tag, generation, rest)
}

/// Every key of the index tree as stored, reserved ones left out.
fn stored_keys(engine: &StorageEngine) -> Vec<Vec<u8>> {
    use lsm_tree::Guard as _;
    engine
        .tree(Partition::Idx)
        .expect("idx")
        .range(
            crate::engine::coverage::USER_KEYSPACE_START..,
            SeqNo::MAX,
            None,
        )
        .map(|guard| guard.key().expect("key").to_vec())
        .collect()
}

/// Every key of the index partition as the engine gives it back.
fn logical_keys(engine: &StorageEngine) -> Vec<Vec<u8>> {
    engine
        .prefix_scan(Partition::Idx, b"")
        .expect("scan")
        .map(|guard| guard.into_owned().expect("pair").0)
        .collect()
}

/// Bindings of `generation → installation` pairs, as a loaded catalog holds.
fn bindings(pairs: &[(u64, u64)]) -> Bindings {
    let mut out = Bindings::default();
    for &(generation, installation) in pairs {
        out.installed.insert(generation, installation);
        out.owner.insert(installation, generation);
        out.generations.push(generation);
    }
    out.generations.sort_unstable();
    out
}

fn key(tag: u8, id: u64, rest: &[u8]) -> Vec<u8> {
    let mut out = prefix(tag, id);
    out.extend_from_slice(rest);
    out
}

/// A generation key is stored under its installation, with its tag and the
/// rest of the key unchanged; a key of another family is stored as written;
/// a key of a generation this member holds no installation of has no
/// address. The physical form reads back as the generation.
#[test]
fn a_generation_key_is_stored_under_its_installation() {
    let b = bindings(&[(40, 7)]);
    let mut buf = Vec::new();
    let logical = key(0x01, 40, b"tuple:\x00\x00\x00\x00\x00\x00\x00\x09");
    let physical = key(0x01, 7, b"tuple:\x00\x00\x00\x00\x00\x00\x00\x09");
    assert_eq!(b.locate(&logical, &mut buf), Located::Bound(&physical));
    assert_eq!(b.logical(&physical).expect("logical"), Some(40));

    let unique = key(0x02, 40, b"v");
    assert_eq!(
        b.locate(&unique, &mut buf),
        Located::Bound(&key(0x02, 7, b"v"))
    );

    assert_eq!(
        b.locate(b"idx:spatial:1", &mut buf),
        Located::Raw(&b"idx:spatial:1"[..])
    );
    assert_eq!(b.logical(b"idx:spatial:1").expect("raw"), None);
    assert_eq!(b.locate(&key(0x01, 41, b"t"), &mut buf), Located::Unbound);
}

/// A generation key under an installation no generation owns is refused,
/// never read as some generation's entry.
#[test]
fn a_key_under_an_unowned_installation_is_refused() {
    let b = bindings(&[(40, 7)]);
    assert!(matches!(
        b.logical(&key(0x01, 8, b"t")),
        Err(StorageError::InstallationCatalog(_))
    ));
    assert!(matches!(
        b.logical(&[0x01, 0, 0]),
        Err(StorageError::InstallationCatalog(_))
    ));
}

/// The whole partition splits into the keys below the generation tags, one
/// piece per bound generation in generation order (whatever order their
/// installations sort in), and the keys above the tags.
#[test]
fn a_range_over_the_partition_comes_apart_in_logical_order() {
    // Generation 5's installation sorts after generation 9's.
    let b = bindings(&[(5, 30), (9, 2)]);
    let pieces = b.pieces(Bound::Unbounded, Bound::Unbounded);
    let expected = vec![
        Piece::Raw {
            lo: Bound::Unbounded,
            hi: Bound::Excluded(vec![0x01]),
        },
        Piece::Domain {
            generation: 5,
            lo: Bound::Included(prefix(0x01, 30)),
            hi: Bound::Excluded(prefix(0x01, 31)),
        },
        Piece::Domain {
            generation: 9,
            lo: Bound::Included(prefix(0x01, 2)),
            hi: Bound::Excluded(prefix(0x01, 3)),
        },
        Piece::Domain {
            generation: 5,
            lo: Bound::Included(prefix(0x02, 30)),
            hi: Bound::Excluded(prefix(0x02, 31)),
        },
        Piece::Domain {
            generation: 9,
            lo: Bound::Included(prefix(0x02, 2)),
            hi: Bound::Excluded(prefix(0x02, 3)),
        },
        Piece::Raw {
            lo: Bound::Included(vec![0x03]),
            hi: Bound::Unbounded,
        },
    ];
    assert_eq!(pieces, expected);
}

/// A range inside one generation keeps its bounds, translated; a range
/// ending where the generation's keys end ends at its installation's end; a
/// range of an unbound generation has no pieces.
#[test]
fn a_range_inside_one_generation_keeps_its_bounds() {
    let b = bindings(&[(5, 30)]);
    let lo = key(0x01, 5, b"a");
    let hi = key(0x01, 5, b"m");
    assert_eq!(
        b.pieces(Bound::Included(&lo), Bound::Included(&hi)),
        vec![Piece::Domain {
            generation: 5,
            lo: Bound::Included(key(0x01, 30, b"a")),
            hi: Bound::Included(key(0x01, 30, b"m")),
        }]
    );
    let [(start, end), _] = coordinode_core::index::encoding::generation_ranges(
        coordinode_core::index::identity::GenerationId::from_raw(5),
    );
    assert_eq!(
        b.pieces(Bound::Included(&start), Bound::Excluded(&end)),
        vec![Piece::Domain {
            generation: 5,
            lo: Bound::Included(prefix(0x01, 30)),
            hi: Bound::Excluded(prefix(0x01, 31)),
        }]
    );
    let other = key(0x01, 6, b"");
    assert!(
        b.pieces(Bound::Included(&other), Bound::Excluded(&prefix(0x01, 7)))
            .is_empty()
    );
}

/// The last possible installation and generation still have an end: past
/// them comes the next tag.
#[test]
fn the_last_installation_ends_at_the_next_tag() {
    let b = bindings(&[(u64::MAX, u64::MAX)]);
    assert_eq!(
        b.pieces(
            Bound::Included(&prefix(0x01, u64::MAX)),
            Bound::Excluded(&[0x02][..])
        ),
        vec![Piece::Domain {
            generation: u64::MAX,
            lo: Bound::Included(prefix(0x01, u64::MAX)),
            hi: Bound::Excluded(vec![0x02]),
        }]
    );
}

/// A fresh catalog binds a generation once, to a new installation that is
/// never handed out again, also across a reopen of the catalog.
#[test]
fn a_binding_is_made_once_and_never_reused() {
    let (engine, _dir) = open();
    let tree = engine.tree(Partition::Idx).expect("idx");
    let installations = Installations::load(tree).expect("load");
    let a = installations
        .bind(tree, 40, engine.next_seqno())
        .expect("bind");
    assert_eq!(
        installations
            .bind(tree, 40, engine.next_seqno())
            .expect("again"),
        a,
        "a bound generation keeps its installation"
    );
    let b = installations
        .bind(tree, 41, engine.next_seqno())
        .expect("bind");
    assert_ne!(a, b);

    let reloaded = Installations::load(tree).expect("reload");
    assert_eq!(reloaded.current().installation(40), Some(a));
    assert_eq!(reloaded.current().installation(41), Some(b));
    let c = reloaded.bind(tree, 42, engine.next_seqno()).expect("bind");
    assert!(c != a && c != b, "the allocator survives the reload");
}

/// A binding asked for at an old seqno (an entry replayed from a journal
/// carries its commit timestamp) still lands above the catalog's earlier
/// writes, so the allocator reads back as advanced and no installation is
/// handed out twice.
#[test]
fn a_binding_at_an_old_seqno_still_advances_the_allocator() {
    let (engine, _dir) = open();
    let tree = engine.tree(Partition::Idx).expect("idx");
    let installations = Installations::load(tree).expect("load");
    let high = engine.next_seqno() + 1_000;
    let a = installations.bind(tree, 40, high).expect("bind high");
    let b = installations
        .bind(tree, 41, 1)
        .expect("bind at an old seqno");

    let reloaded = Installations::load(tree).expect("reload");
    let c = reloaded.bind(tree, 42, engine.next_seqno()).expect("bind");
    assert!(c != a && c != b, "{a} {b} {c}");
    assert_eq!(reloaded.current().installation(41), Some(b));
}

/// Entries under an installation no binding names fail the load: their
/// keys could belong to any generation.
#[test]
fn entries_without_a_binding_fail_the_load() {
    let (engine, _dir) = open();
    let tree = engine.tree(Partition::Idx).expect("idx");
    let installations = Installations::load(tree).expect("load");
    let bound = installations
        .bind(tree, 40, engine.next_seqno())
        .expect("bind");
    tree.insert(key(0x01, bound, b"t"), b"", engine.next_seqno());
    Installations::load(tree).expect("a bound installation's entries load");

    tree.insert(key(0x02, bound + 5, b"t"), b"", engine.next_seqno());
    assert!(matches!(
        Installations::load(tree),
        Err(StorageError::InstallationCatalog(_))
    ));
}

/// Two generations bound to one installation, or a binding outside what was
/// allocated, fail the load.
#[test]
fn contradictory_bindings_fail_the_load() {
    let (engine, _dir) = open();
    let tree = engine.tree(Partition::Idx).expect("idx");
    tree.insert(NEXT_KEY, 10u64.to_be_bytes(), engine.next_seqno());
    tree.insert(binding_key(1), 3u64.to_be_bytes(), engine.next_seqno());
    tree.insert(binding_key(2), 3u64.to_be_bytes(), engine.next_seqno());
    assert!(matches!(
        Installations::load(tree),
        Err(StorageError::InstallationCatalog(_))
    ));

    tree.insert(binding_key(2), 12u64.to_be_bytes(), engine.next_seqno());
    assert!(matches!(
        Installations::load(tree),
        Err(StorageError::InstallationCatalog(_))
    ));

    tree.insert(binding_key(2), [1, 2, 3], engine.next_seqno());
    assert!(matches!(
        Installations::load(tree),
        Err(StorageError::InstallationCatalog(_))
    ));
}

/// An entry written through the engine sits under its generation's
/// installation and nowhere under the generation itself; every read path
/// (point, batch, scan, snapshot, version) gives it back by its logical key,
/// also after a reopen. Keys of other families are stored as written.
#[test]
fn an_entry_is_stored_under_its_installation_and_read_back_logically() {
    let dir = tempfile::tempdir().expect("tempdir");
    let generation = 900;
    let logical = entry(0x01, generation, b"tuple:k");
    {
        let engine = StorageEngine::open(&config(dir.path())).expect("open");
        engine
            .put(Partition::Idx, &logical, b"v")
            .expect("put entry");
        engine
            .put(Partition::Idx, b"idx:spatial:1", b"s")
            .expect("put spatial");
        let installation = engine
            .installations
            .current()
            .installation(generation)
            .expect("bound on its first write");
        let physical = entry(0x01, installation, b"tuple:k");
        let stored = stored_keys(&engine);
        assert!(stored.contains(&physical), "{stored:?}");
        assert!(stored.contains(&b"idx:spatial:1".to_vec()));
        assert_ne!(installation, generation, "the first installation is 1");
        assert!(!stored.contains(&logical), "nothing under the generation");
        engine.persist().expect("persist");
    }
    let engine = StorageEngine::open(&config(dir.path())).expect("reopen");
    let snapshot = engine.snapshot();
    assert_eq!(
        engine
            .get(Partition::Idx, &logical)
            .expect("get")
            .as_deref(),
        Some(&b"v"[..])
    );
    assert_eq!(
        engine
            .multi_get(Partition::Idx, &[logical.as_slice(), b"idx:spatial:1"])
            .expect("multi"),
        vec![
            Some(bytes::Bytes::from_static(b"v")),
            Some(bytes::Bytes::from_static(b"s"))
        ]
    );
    assert!(engine.contains_key(Partition::Idx, &logical).expect("has"));
    assert!(
        engine
            .snapshot_get(&snapshot, Partition::Idx, &logical)
            .expect("snapshot")
            .is_some()
    );
    assert!(
        engine
            .record_version(Partition::Idx, &logical)
            .expect("version")
            .is_some()
    );
    assert_eq!(
        logical_keys(&engine),
        vec![logical.clone(), b"idx:spatial:1".to_vec()]
    );
    let prefix = entry(0x01, generation, b"");
    assert_eq!(
        engine
            .snapshot_prefix_scan(&snapshot, Partition::Idx, &prefix)
            .expect("prefix")
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>(),
        vec![logical.clone()]
    );
    engine.delete(Partition::Idx, &logical).expect("delete");
    assert!(engine.get(Partition::Idx, &logical).expect("get").is_none());
}

/// A key naming as its generation the number of another generation's
/// installation reads as absent: a generation this member holds no
/// installation of holds nothing, and installations are never read as
/// generations.
#[test]
fn a_generation_without_an_installation_holds_nothing() {
    let (engine, _dir) = open();
    engine
        .put(Partition::Idx, &entry(0x02, 77, b"x"), b"holder")
        .expect("put");
    let installation = engine
        .installations
        .current()
        .installation(77)
        .expect("bound");
    assert_ne!(installation, 77);
    let alias = entry(0x02, installation, b"x");
    assert!(engine.get(Partition::Idx, &alias).expect("get").is_none());
    assert!(
        engine
            .prefix_scan(Partition::Idx, &entry(0x02, installation, b""))
            .expect("scan")
            .next()
            .is_none()
    );
}

/// A range delete spanning several generations removes each one's entries
/// from its installation and leaves keys outside the range alone.
#[test]
fn a_range_delete_across_generations_reaches_every_installation() {
    let (engine, _dir) = open();
    for generation in [10, 11, 12] {
        engine
            .put(Partition::Idx, &entry(0x01, generation, b"a"), b"")
            .expect("put");
    }
    engine
        .put(Partition::Idx, b"tkey:1", b"t")
        .expect("put table key");
    engine
        .remove_range(Partition::Idx, &entry(0x01, 10, b""), &entry(0x01, 12, b""))
        .expect("range delete");
    assert_eq!(
        logical_keys(&engine),
        vec![entry(0x01, 12, b"a"), b"tkey:1".to_vec()]
    );
}

/// Clearing the partition keeps its installations allocated: a generation
/// bound afterwards gets a fresh one, and the catalog survives a reopen with
/// the bindings still in use.
#[test]
fn a_cleared_partition_never_hands_an_installation_out_twice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first;
    let second;
    {
        let engine = StorageEngine::open(&config(dir.path())).expect("open");
        engine
            .put(Partition::Idx, &entry(0x01, 5, b"a"), b"")
            .expect("put");
        first = engine
            .installations
            .current()
            .installation(5)
            .expect("bound");
        engine.clear_partition(Partition::Idx).expect("clear");
        engine
            .put(Partition::Idx, &entry(0x01, 6, b"b"), b"")
            .expect("put after clear");
        second = engine
            .installations
            .current()
            .installation(6)
            .expect("bound");
        assert_ne!(first, second);
        engine.persist().expect("persist");
    }
    let engine = StorageEngine::open(&config(dir.path())).expect("reopen");
    assert_eq!(engine.installations.current().installation(5), Some(first));
    assert_eq!(engine.installations.current().installation(6), Some(second));
    assert_eq!(logical_keys(&engine), vec![entry(0x01, 6, b"b")]);
}

/// A seekable scan over several generations yields their entries in logical
/// order whatever order their installations sort in, seeks into a later
/// generation and back, peeks logical keys, and walks backwards.
#[test]
fn a_seekable_scan_moves_across_generations_in_logical_order() {
    let (engine, _dir) = open();
    // Bound in this order, generation 30 gets the smaller installation.
    for (generation, rest) in [(30, b"x"), (20, b"a"), (20, b"b"), (30, b"y")] {
        engine
            .put(Partition::Idx, &entry(0x01, generation, rest), b"")
            .expect("put");
    }
    let all = vec![
        entry(0x01, 20, b"a"),
        entry(0x01, 20, b"b"),
        entry(0x01, 30, b"x"),
        entry(0x01, 30, b"y"),
    ];
    let open = || {
        engine
            .range_seekable(Partition::Idx, &[0x01], &[0x02], engine.snapshot())
            .expect("seekable")
    };
    let owned = |guard: crate::engine::StorageGuard| guard.into_owned().expect("pair").0;
    assert_eq!(open().map(owned).collect::<Vec<_>>(), all);
    let mut reversed = all.clone();
    reversed.reverse();
    assert_eq!(open().rev().map(owned).collect::<Vec<_>>(), reversed);

    let mut scan = open();
    scan.seek_to(&entry(0x01, 30, b"y"));
    assert_eq!(
        scan.peek_key().expect("peek").expect("key").to_vec(),
        entry(0x01, 30, b"y")
    );
    scan.seek_to(&entry(0x01, 20, b"b"));
    assert_eq!(scan.map(owned).collect::<Vec<_>>(), all[1..].to_vec());

    let mut scan = open();
    scan.seek_to_for_prev(&entry(0x01, 20, b"z"));
    assert_eq!(scan.next_back().map(owned), Some(entry(0x01, 20, b"b")));
}

/// A range of the tree as stored maps back to the logical ranges of the
/// generations whose installations it covers; untranslated keys keep their
/// range.
#[test]
fn a_stored_range_maps_back_to_logical_ranges() {
    let b = bindings(&[(5, 30), (9, 2)]);
    // Installation 2's entries in full, and installation 30's up to `z`.
    let mut ranges = b.logical_ranges(&key(0x01, 2, b""), &key(0x01, 30, b"z"));
    ranges.sort();
    assert_eq!(
        ranges,
        vec![
            (prefix(0x01, 5), key(0x01, 5, b"z\x00")),
            (prefix(0x01, 9), prefix(0x01, 10)),
        ]
    );
    assert_eq!(
        b.logical_ranges(b"tkey:a", b"tkey:b"),
        vec![(b"tkey:a".to_vec(), b"tkey:b\x00".to_vec())]
    );
}
