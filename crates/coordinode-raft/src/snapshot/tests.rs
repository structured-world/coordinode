use super::*;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use tempfile::tempdir;

fn open_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

/// A target opened as an offline restore opens it: no batch is synced on its
/// own, the restore persists once at its end.
fn open_restore_target(dir: &std::path::Path) -> StorageEngine {
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.flush_policy = coordinode_storage::engine::config::FlushPolicy::Manual;
    StorageEngine::open(&config).expect("open engine")
}

/// `engine`'s full snapshot, in memory.
fn build_full_snapshot(engine: &StorageEngine) -> io::Result<Vec<u8>> {
    let mut buf = io::Cursor::new(Vec::new());
    write_full_snapshot(engine, None, &mut buf)?;
    Ok(buf.into_inner())
}

/// Bind `engine` to deployment `id` and give it one history record, as a
/// store that founded its domain with its first history holds them.
fn with_history(engine: &StorageEngine, id: u8) {
    engine
        .put(
            Partition::Schema,
            coordinode_core::group::DEPLOYMENT_KEY,
            &[id; 16],
        )
        .unwrap();
    engine
        .put(Partition::NodeHistory, b"nhist:history", b"state")
        .unwrap();
}

fn domain_of(engine: &StorageEngine) -> Option<coordinode_core::group::DeploymentId> {
    coordinode_storage::engine::metadata::bound_deployment(engine).unwrap()
}

/// Every section of a snapshot describes the one cut it was asked for:
/// writes landing while the partitions stream out (a row of a partition not
/// yet written, the domain binding read last) stay out of it, so an image is
/// never half before and half after a commit.
#[test]
fn a_snapshot_is_one_cut_whatever_lands_while_it_streams() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine.put(Partition::Node, b"node:early", b"v").unwrap();
    engine.put(Partition::EdgeProp, b"ep:early", b"v").unwrap();
    let (at, _pin) = engine.pin_latest_snapshot();

    let mut buf = io::Cursor::new(Vec::new());
    let mut landed = false;
    write_snapshot(&engine, at, None, &mut buf, |partition| {
        if partition == Partition::Node && !landed {
            landed = true;
            engine
                .put(Partition::EdgeProp, b"ep:late", b"v")
                .map_err(|e| io::Error::other(e.to_string()))?;
            engine
                .put(
                    Partition::Schema,
                    coordinode_core::group::DEPLOYMENT_KEY,
                    &[7; 16],
                )
                .map_err(|e| io::Error::other(e.to_string()))?;
        }
        Ok(())
    })
    .unwrap();
    assert!(landed, "the Node partition is written first");

    let parsed = parse_verified_slice(&buf.into_inner()).unwrap();
    let keys = |wanted: Partition| -> Vec<Vec<u8>> {
        parsed
            .partitions
            .iter()
            .filter(|(partition, _)| *partition == wanted)
            .flat_map(|(_, entries)| entries.iter().map(|(key, _)| key.clone()))
            .collect()
    };
    assert_eq!(keys(Partition::EdgeProp), vec![b"ep:early".to_vec()]);
    assert!(
        !keys(Partition::Schema)
            .iter()
            .any(|key| key.as_slice() == coordinode_core::group::DEPLOYMENT_KEY),
        "a binding written after the cut is not in the image"
    );
    assert_eq!(parsed.domain, None, "the store was unbound at the cut");
}

/// An install whose second pass reads other bytes than the checked image
/// fails, and what it wrote is staging the next attempt starts over from:
/// retrying the checked image leaves none of the other input's records,
/// also after a reopen.
#[test]
fn a_retried_install_keeps_nothing_of_a_changed_input() {
    let source_dir = tempdir().unwrap();
    let source = open_engine(source_dir.path());
    source.put(Partition::Node, b"node:a", b"v").unwrap();
    let image = build_full_snapshot(&source).unwrap();
    // No longer than the checked record: written before the change shows.
    source.put(Partition::Node, b"node:b", b"v").unwrap();
    let changed = build_full_snapshot(&source).unwrap();
    drop(source);

    let target_dir = tempdir().unwrap();
    let budget = QueryBudget::new(1 << 20);
    let check = check_image(&mut io::Cursor::new(&image), &budget).unwrap();
    {
        let target = open_engine(target_dir.path());
        let failed = install_image(&target, &mut io::Cursor::new(&changed), &check, &budget)
            .expect_err("a changed input");
        assert!(failed.to_string().contains("changed"), "{failed}");
        assert!(
            target.get(Partition::Node, b"node:b").unwrap().is_some(),
            "the changed pass wrote its records"
        );
        install_image(&target, &mut io::Cursor::new(&image), &check, &budget).unwrap();
        target.persist().unwrap();
    }
    let target = open_engine(target_dir.path());
    assert!(target.get(Partition::Node, b"node:a").unwrap().is_some());
    assert!(
        target.get(Partition::Node, b"node:b").unwrap().is_none(),
        "the retry left a record of the changed input"
    );
}

/// A record the second pass reads longer than any the check saw is not of
/// the checked image: refused before it is written, so what a later
/// attempt clears reads no record longer than its image's.
#[test]
fn a_changed_input_with_a_longer_record_writes_none_of_it() {
    let source_dir = tempdir().unwrap();
    let source = open_engine(source_dir.path());
    source.put(Partition::Node, b"node:a", b"v").unwrap();
    let image = build_full_snapshot(&source).unwrap();
    source
        .put(Partition::Node, b"node:long", &vec![b'l'; 64 << 10])
        .unwrap();
    let changed = build_full_snapshot(&source).unwrap();
    drop(source);

    let budget = QueryBudget::new(4 << 20);
    let check = check_image(&mut io::Cursor::new(&image), &budget).unwrap();
    let target_dir = tempdir().unwrap();
    let target = open_engine(target_dir.path());
    let failed = install_image(&target, &mut io::Cursor::new(&changed), &check, &budget)
        .expect_err("a changed input");
    assert!(failed.to_string().contains("longer"), "{failed}");
    assert!(
        target.get(Partition::Node, b"node:long").unwrap().is_none(),
        "the longer record was written"
    );
}

/// Checking an image whose field dictionary is many short names holds, by
/// the allocator's count, no more than the memory limit at once: what the
/// check builds to verify the dictionary is reserved like the records.
#[test]
fn checking_a_large_dictionary_holds_no_more_than_the_limit() {
    use coordinode_test_fixtures::alloc_peak::peak_of;
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    for chunk in 0..20 {
        let names: Vec<String> = (0..1_000)
            .map(|i| format!("f{:05}", chunk * 1_000 + i))
            .collect();
        engine
            .apply_proposal_at(
                &command(coordinode_core::txn::proposal::MetadataCommand::RegisterFields { names }),
                0,
            )
            .unwrap();
    }
    let data = build_full_snapshot(&engine).unwrap();
    let limit: u64 = 6 << 20;
    let budget = QueryBudget::new(limit);
    let (checked, peak) = peak_of(|| {
        check_image(&mut io::Cursor::new(&data), &budget)
            .map(drop)
            .map_err(|e| e.to_string())
    });
    if let Err(e) = checked {
        assert!(e.contains("memory limit"), "{e}");
    }
    assert!(
        peak as u64 <= limit,
        "held {peak} bytes under a {limit}-byte limit"
    );
    // With room for it, the check passes, and still holds within its limit.
    let roomy: u64 = 32 << 20;
    let budget = QueryBudget::new(roomy);
    let (checked, peak) = peak_of(|| {
        check_image(&mut io::Cursor::new(&data), &budget)
            .map(|check| check.field_bindings().len())
            .map_err(|e| e.to_string())
    });
    assert_eq!(checked, Ok(20_000));
    assert!(
        peak as u64 <= roomy,
        "held {peak} bytes under a {roomy}-byte limit"
    );
}

/// Installing columnar rows holds, by the allocator's count, no more than
/// the memory limit at once: the copies the table write makes of a batch
/// are reserved, and a written batch gives its memory back.
#[test]
fn installing_columnar_rows_holds_no_more_than_the_limit() {
    use coordinode_test_fixtures::alloc_peak::peak_of;
    let limit: u64 = 1 << 20;
    let row = vec![b'r'; (limit / 8) as usize];
    let dir = tempdir().unwrap();
    let sender = open_engine(dir.path());
    let keys: Vec<Vec<u8>> = (0..12u8).map(|i| vec![b'k', i]).collect();
    sender
        .replace_columnar_tables(vec![(
            "Wide".to_owned(),
            keys.iter().map(|k| (k.clone(), row.clone())).collect(),
        )])
        .unwrap();
    let image = build_full_snapshot(&sender).unwrap();
    drop(sender);

    let budget = QueryBudget::new(limit);
    let check = check_image(&mut io::Cursor::new(&image), &budget).unwrap();
    let target_dir = tempdir().unwrap();
    let target = open_engine(target_dir.path());
    let (installed, peak) = peak_of(|| {
        install_image(&target, &mut io::Cursor::new(&image), &check, &budget)
            .map_err(|e| e.to_string())
    });
    installed.unwrap();
    // The table's tree the engine opens and keeps is the engine's memory,
    // not the restore's: measured alone, and allowed for.
    let engine_tree = {
        let probe_dir = tempdir().unwrap();
        let probe = open_engine(probe_dir.path());
        peak_of(|| probe.reset_columnar_table("Probe").unwrap()).1 as u64
    };
    assert!(
        peak as u64 <= limit + engine_tree,
        "held {peak} bytes under a {limit}-byte limit (the engine's tree takes {engine_tree})"
    );
    assert_eq!(target.columnar_tables_at(u64::MAX).unwrap()[0].1.len(), 12);
}

/// A record whose install cannot fit the memory limit (the batch and the
/// tree's write each hold a copy of it) is refused by the check, before any
/// target is touched.
#[test]
fn a_record_whose_install_does_not_fit_is_refused_by_the_check() {
    let limit: u64 = 1 << 20;
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine
        .put(
            Partition::Node,
            b"node:wide",
            &vec![b'w'; (limit * 6 / 10) as usize],
        )
        .unwrap();
    let image = build_full_snapshot(&engine).unwrap();
    let budget = QueryBudget::new(limit);
    let refused = check_image(&mut io::Cursor::new(&image), &budget)
        .expect_err("a record past half the limit");
    assert!(refused.to_string().contains("memory limit"), "{refused}");
}

/// A record whose commit holds more than the memory limit at once (a long
/// key is copied into the tree batch and once more into its unit record,
/// beside the buffer encoding unit records) is refused by the check, though
/// twice its bytes would fit.
#[test]
fn a_long_key_whose_commit_does_not_fit_is_refused_by_the_check() {
    let limit: u64 = 160 << 10;
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    let mut key = b"node:".to_vec();
    key.resize(60 << 10, b'k');
    engine.put(Partition::Node, &key, b"v").unwrap();
    let image = build_full_snapshot(&engine).unwrap();
    let budget = QueryBudget::new(limit);
    let refused = check_image(&mut io::Cursor::new(&image), &budget)
        .expect_err("a record whose commit holds more than the limit");
    assert!(refused.to_string().contains("memory limit"), "{refused}");
}

/// Installing holds, by the allocator's count, at no instant more than it
/// has reserved at that instant, besides what the store keeps (its
/// memtables), which is left out allocation by allocation rather than
/// subtracted as an end total: for a long key, for many short records,
/// over an earlier attempt's records, which the install deletes first, and
/// for batches large enough to be written by the rayon pool, whose threads
/// are counted too.
#[test]
fn installing_holds_no_more_than_it_reserves_and_the_store_keeps() {
    use coordinode_test_fixtures::alloc_peak::held_against;
    let image_of = |fill: &dyn Fn(&StorageEngine)| {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());
        fill(&engine);
        build_full_snapshot(&engine).unwrap()
    };
    let long_key = image_of(&|engine| {
        let mut key = b"node:".to_vec();
        key.resize(60 << 10, b'k');
        engine.put(Partition::Node, &key, b"v").unwrap();
    });
    let short = image_of(&|engine| {
        for i in 0..20_000u32 {
            engine
                .put(
                    Partition::Node,
                    format!("node:{i:08}").as_bytes(),
                    b"0123456789abcdef",
                )
                .unwrap();
        }
    });
    // Two partitions in one stream: a batch at their boundary carries over a
    // thousand mutations of both, which the store writes on the rayon pool.
    let two_partitions = image_of(&|engine| {
        for i in 0..6_000u32 {
            for partition in [Partition::Node, Partition::Adj] {
                engine
                    .put(
                        partition,
                        format!("{}:{i:08}", partition.name()).as_bytes(),
                        b"0123456789abcdef",
                    )
                    .unwrap();
            }
        }
    });
    for (name, image, limit, again) in [
        ("long key", &long_key, 1u64 << 20, false),
        ("short records", &short, 2 << 20, false),
        ("over an earlier attempt", &short, 2 << 20, true),
        (
            "two partitions on the rayon pool",
            &two_partitions,
            16 << 20,
            false,
        ),
    ] {
        let dir = tempdir().unwrap();
        let target = open_restore_target(dir.path());
        if again {
            let budget = QueryBudget::new(limit);
            let check = check_image(&mut io::Cursor::new(image), &budget).unwrap();
            install_image(&target, &mut io::Cursor::new(image), &check, &budget).unwrap();
        }
        let check_budget = QueryBudget::new(limit);
        let check = check_image(&mut io::Cursor::new(image), &check_budget).unwrap();
        let budget = std::sync::Arc::new(QueryBudget::new(limit));
        let read = std::sync::Arc::clone(&budget);
        let (installed, held) =
            held_against(std::sync::Arc::new(move || read.memory_used()), || {
                install_image(&target, &mut io::Cursor::new(image), &check, &budget)
                    .map_err(|e| e.to_string())
            });
        installed.unwrap();
        let reserved = budget.memory_peak();
        assert!(
            reserved <= limit,
            "{name}: reserved {reserved} past {limit}"
        );
        assert_eq!(
            held.over, None,
            "{name}: held more than was reserved at that instant: {held:?}"
        );
    }
}

/// An image many times the memory limit installs whole, holding no more
/// than the limit at once.
#[test]
fn an_image_larger_than_the_budget_installs_within_it() {
    let source_dir = tempdir().unwrap();
    let source = open_engine(source_dir.path());
    let pad = vec![b'x'; 1000];
    for i in 0..4000u32 {
        source
            .put(Partition::Node, format!("node:{i:06}").as_bytes(), &pad)
            .unwrap();
    }
    let image = build_full_snapshot(&source).unwrap();
    let limit = 256 << 10;
    assert!(image.len() as u64 > 8 * limit);

    let budget = QueryBudget::new(limit);
    let check = check_image(&mut io::Cursor::new(&image), &budget).unwrap();
    let target_dir = tempdir().unwrap();
    let target = open_engine(target_dir.path());
    install_image(&target, &mut io::Cursor::new(&image), &check, &budget).unwrap();
    assert!(
        budget.memory_peak() <= limit,
        "peak {}",
        budget.memory_peak()
    );
    assert_eq!(
        target
            .prefix_scan(Partition::Node, b"node:")
            .unwrap()
            .count(),
        4000
    );
}

/// A snapshot names the domain of its history. A receiver bound to another
/// one refuses it and keeps what it held; one bound to the same installs it;
/// one bound to none adopts the domain with the history, in both install
/// paths (the serialized snapshot and the stream).
#[test]
fn a_snapshot_installs_history_only_into_its_domain() {
    use coordinode_core::group::DeploymentId;
    let dir = tempdir().unwrap();
    let source = open_engine(dir.path());
    with_history(&source, 1);
    let data = build_full_snapshot(&source).unwrap();

    let other_dir = tempdir().unwrap();
    let other = open_engine(other_dir.path());
    other
        .put(
            Partition::Schema,
            coordinode_core::group::DEPLOYMENT_KEY,
            &[2; 16],
        )
        .unwrap();
    other.put(Partition::Node, b"node:mine", b"kept").unwrap();
    let refused = install_full_snapshot(&other, &data).expect_err("another domain");
    assert!(
        refused.to_string().contains("belongs to deployment"),
        "{refused}"
    );
    assert!(other.get(Partition::Node, b"node:mine").unwrap().is_some());
    assert!(
        other
            .get(Partition::NodeHistory, b"nhist:history")
            .unwrap()
            .is_none()
    );
    assert_eq!(domain_of(&other), Some(DeploymentId([2; 16])));

    let same_dir = tempdir().unwrap();
    let same = open_engine(same_dir.path());
    same.put(
        Partition::Schema,
        coordinode_core::group::DEPLOYMENT_KEY,
        &[1; 16],
    )
    .unwrap();
    install_full_snapshot(&same, &data).expect("the same domain");
    assert!(
        same.get(Partition::NodeHistory, b"nhist:history")
            .unwrap()
            .is_some()
    );

    let fresh_dir = tempdir().unwrap();
    let fresh = open_engine(fresh_dir.path());
    install_full_snapshot_from_reader(&fresh, &mut io::Cursor::new(&data)).expect("unbound");
    assert_eq!(domain_of(&fresh), Some(DeploymentId([1; 16])));
    assert!(
        fresh
            .get(Partition::NodeHistory, b"nhist:history")
            .unwrap()
            .is_some()
    );
}

/// Replacing a whole database replaces its domain with the image's, bound
/// with the image's records; history the image names no domain for is still
/// refused, and the receiver left as it was.
#[test]
fn a_replaced_database_takes_the_images_domain() {
    use coordinode_core::group::DeploymentId;
    let dir = tempdir().unwrap();
    let source = open_engine(dir.path());
    with_history(&source, 1);
    let data = build_full_snapshot(&source).unwrap();
    let target_dir = tempdir().unwrap();
    let target = open_engine(target_dir.path());
    target
        .put(
            Partition::Schema,
            coordinode_core::group::DEPLOYMENT_KEY,
            &[2; 16],
        )
        .unwrap();
    let budget = QueryBudget::new(1 << 20);
    let check = check_image(&mut io::Cursor::new(&data), &budget).expect("a valid image");
    assert_eq!(check.domain(), Some(DeploymentId([1; 16])));
    install_image(&target, &mut io::Cursor::new(&data), &check, &budget).expect("replace");
    assert_eq!(domain_of(&target), Some(DeploymentId([1; 16])));
    assert!(
        target
            .get(Partition::NodeHistory, b"nhist:history")
            .unwrap()
            .is_some()
    );

    let unbound_dir = tempdir().unwrap();
    let unbound_source = open_engine(unbound_dir.path());
    unbound_source
        .put(Partition::NodeHistory, b"nhist:history", b"state")
        .unwrap();
    let data = build_full_snapshot(&unbound_source).unwrap();
    let refused =
        check_image(&mut io::Cursor::new(&data), &budget).expect_err("history without a domain");
    assert!(refused.to_string().contains("names no domain"), "{refused}");
    assert_eq!(domain_of(&target), Some(DeploymentId([1; 16])));
}

/// History a snapshot names no domain for is refused: a receiver cannot
/// tell whose it is, and adopting none would leave it unbound.
#[test]
fn a_snapshot_of_history_without_a_domain_is_refused() {
    let dir = tempdir().unwrap();
    let source = open_engine(dir.path());
    source
        .put(Partition::NodeHistory, b"nhist:history", b"state")
        .unwrap();
    let data = build_full_snapshot(&source).unwrap();
    let fresh_dir = tempdir().unwrap();
    let fresh = open_engine(fresh_dir.path());
    let refused = install_full_snapshot(&fresh, &data).expect_err("no domain");
    assert!(refused.to_string().contains("names no domain"), "{refused}");
    assert!(
        fresh
            .get(Partition::NodeHistory, b"nhist:history")
            .unwrap()
            .is_none()
    );
    assert_eq!(domain_of(&fresh), None);
}

#[test]
fn test_snapshot_roundtrip_empty_db() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    let data = build_full_snapshot(&engine).unwrap();
    assert!(data.len() > 14); // magic(4) + version(1) + count(1) + checksum(8)

    // Install into fresh engine should succeed
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    install_full_snapshot(&engine2, &data).unwrap();
}

#[test]
fn test_snapshot_roundtrip_with_data() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    // Write some data across partitions
    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    engine.put(Partition::Node, b"node:0:2", b"bob").unwrap();
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x02")
        .unwrap();
    engine
        .put(Partition::EdgeProp, b"edgeprop:KNOWS:1:2", b"since=2020")
        .unwrap();
    engine
        .put(Partition::Schema, b"schema:label:User", b"{}")
        .unwrap();
    engine
        .put(Partition::Idx, b"idx:name:alice:1", b"")
        .unwrap();

    let data = build_full_snapshot(&engine).unwrap();

    // Install into fresh engine
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());

    install_full_snapshot(&engine2, &data).unwrap();

    // Verify data was restored
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"alice".to_vec())
    );
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:2")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"bob".to_vec())
    );
    assert_eq!(
        engine2
            .get(Partition::Adj, b"adj:KNOWS:out:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"\x02".to_vec())
    );
    assert_eq!(
        engine2
            .get(Partition::EdgeProp, b"edgeprop:KNOWS:1:2")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"since=2020".to_vec())
    );
    assert_eq!(
        engine2
            .get(Partition::Schema, b"schema:label:User")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"{}".to_vec())
    );
    assert_eq!(
        engine2
            .get(Partition::Idx, b"idx:name:alice:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"".to_vec())
    );
}

fn command(command: coordinode_core::txn::proposal::MetadataCommand) -> Vec<Mutation> {
    vec![Mutation::Command(command)]
}

fn grant(base: u64, ceiling: u64, token: u8) -> Vec<Mutation> {
    command(
        coordinode_core::txn::proposal::MetadataCommand::GrantNodeLease {
            base,
            ceiling,
            token: [token; 16],
        },
    )
}

use coordinode_core::txn::proposal::Mutation;
use coordinode_storage::engine::metadata::{
    load_field_dictionary, node_lease_ceiling, node_lease_holder,
};

#[test]
fn the_node_id_lease_record_travels_with_a_snapshot() {
    // A member caught up by a snapshot must see every lease granted before
    // it, or as leader it would grant the same ranges again.
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine.apply_proposal_at(&grant(0, 10_000, 1), 0).unwrap();
    engine
        .apply_proposal_at(&grant(10_000, 20_000, 2), 0)
        .unwrap();
    let data = build_full_snapshot(&engine).unwrap();

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    install_full_snapshot(&engine2, &data).unwrap();
    assert_eq!(node_lease_ceiling(&engine2).unwrap(), 20_000);
    assert_eq!(node_lease_holder(&engine2, 20_000).unwrap(), Some([2; 16]));

    // A grant from the base the snapshot carried applies on the new member;
    // one from a base below it does not.
    engine2.apply_proposal_at(&grant(0, 10_000, 3), 0).unwrap();
    engine2
        .apply_proposal_at(&grant(20_000, 30_000, 4), 0)
        .unwrap();
    assert_eq!(node_lease_ceiling(&engine2).unwrap(), 30_000);
    assert_eq!(node_lease_holder(&engine2, 30_000).unwrap(), Some([4; 16]));
    assert_eq!(node_lease_holder(&engine2, 10_000).unwrap(), Some([1; 16]));
}

/// A member caught up by a snapshot alone can read the data the snapshot
/// carries: the field dictionary travels with it, and registration continues
/// above the frontier it installed.
#[test]
fn the_field_dictionary_travels_with_a_snapshot() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine
        .apply_proposal_at(
            &command(
                coordinode_core::txn::proposal::MetadataCommand::RegisterFields {
                    names: vec!["name".into(), "age".into()],
                },
            ),
            0,
        )
        .unwrap();
    let data = build_full_snapshot(&engine).unwrap();

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    let before = engine2.field_dictionary_generation();
    install_full_snapshot(&engine2, &data).unwrap();
    assert!(
        engine2.field_dictionary_generation() > before,
        "an install tells readers to reread the dictionary"
    );
    let dictionary = load_field_dictionary(&engine2).unwrap();
    assert_eq!(dictionary.lookup("name"), Some(1));
    assert_eq!(dictionary.lookup("age"), Some(2));

    engine2
        .apply_proposal_at(
            &command(
                coordinode_core::txn::proposal::MetadataCommand::RegisterFields {
                    names: vec!["email".into()],
                },
            ),
            0,
        )
        .unwrap();
    assert_eq!(
        load_field_dictionary(&engine2).unwrap().lookup("email"),
        Some(3)
    );
}

/// A snapshot whose dictionary records disagree is refused before any of it
/// is installed: data it carries would be read under the wrong names.
#[test]
fn a_snapshot_with_a_contradictory_dictionary_is_refused() {
    use coordinode_core::graph::intern::{field_id_key, field_name_key};
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine
        .put(Partition::Schema, &field_name_key("a"), &1u32.to_be_bytes())
        .unwrap();
    engine
        .put(Partition::Schema, &field_id_key(1), b"b")
        .unwrap();
    engine.put(Partition::Node, b"node:0:1", b"x").unwrap();
    let data = build_full_snapshot(&engine).unwrap();

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    assert!(install_full_snapshot(&engine2, &data).is_err());
    assert!(
        engine2.get(Partition::Node, b"node:0:1").unwrap().is_none(),
        "nothing of a refused snapshot is installed"
    );
}

#[test]
fn test_snapshot_preserves_raft_keys_in_schema() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    // Source has raft keys and application keys
    engine
        .put(Partition::Schema, b"raft:vote", b"raft-data")
        .unwrap();
    engine
        .put(Partition::Schema, b"schema:label:User", b"{}")
        .unwrap();

    let data = build_full_snapshot(&engine).unwrap();

    // Target has different raft keys
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2
        .put(Partition::Schema, b"raft:vote", b"target-raft-data")
        .unwrap();
    engine2
        .put(Partition::Schema, b"schema:label:Old", b"old")
        .unwrap();

    install_full_snapshot(&engine2, &data).unwrap();

    // Raft keys preserved (target's own raft data kept)
    assert_eq!(
        engine2
            .get(Partition::Schema, b"raft:vote")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"target-raft-data".to_vec())
    );
    // Application data replaced from snapshot
    assert_eq!(
        engine2
            .get(Partition::Schema, b"schema:label:User")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"{}".to_vec())
    );
    // Old application data cleared
    assert!(
        engine2
            .get(Partition::Schema, b"schema:label:Old")
            .unwrap()
            .is_none()
    );
}

#[test]
fn test_snapshot_checksum_validation() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    let mut data = build_full_snapshot(&engine).unwrap();

    // Corrupt a byte in the payload
    if data.len() > 10 {
        data[5] ^= 0xFF;
    }

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    let result = install_full_snapshot(&engine2, &data);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch")
    );
}

#[test]
fn test_snapshot_invalid_magic() {
    let data = b"BADMxxxxxxxx";
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    let result = install_full_snapshot(&engine, data);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("invalid snapshot magic")
    );
}

#[test]
fn test_snapshot_replaces_existing_data() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine
        .put(Partition::Node, b"node:0:1", b"new-value")
        .unwrap();

    let data = build_full_snapshot(&engine).unwrap();

    // Target has different data
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2
        .put(Partition::Node, b"node:0:1", b"old-value")
        .unwrap();
    engine2
        .put(Partition::Node, b"node:0:99", b"stale")
        .unwrap();

    install_full_snapshot(&engine2, &data).unwrap();

    // Snapshot data overwrites
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"new-value".to_vec())
    );
    // Stale data removed
    assert!(
        engine2
            .get(Partition::Node, b"node:0:99")
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_snapshot_written_past_a_prefix_installs() {
    // A backup frames the snapshot after its own header; the counts patched
    // in place and the checksum read back must be the snapshot's own.
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x02")
        .unwrap();

    let mut buf = io::Cursor::new(b"frame".to_vec());
    buf.seek(SeekFrom::End(0)).unwrap();
    let written = write_full_snapshot(&engine, None, &mut buf).unwrap();
    let buf = buf.into_inner();
    assert_eq!(buf.len() as u64, 5 + written);
    assert_eq!(&buf[..5], b"frame", "the prefix is left as it was");
    assert_eq!(buf[5..], build_full_snapshot(&engine).unwrap()[..]);

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    install_full_snapshot(&engine2, &buf[5..]).unwrap();
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:1")
            .unwrap()
            .as_deref(),
        Some(b"alice".as_slice())
    );
    assert_eq!(
        engine2
            .get(Partition::Adj, b"adj:KNOWS:out:1")
            .unwrap()
            .as_deref(),
        Some(b"\x02".as_slice())
    );
}

#[test]
fn a_streamed_snapshot_with_bytes_past_its_checksum_is_refused() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    let mut data = build_full_snapshot(&engine).unwrap();
    data.extend_from_slice(b"x");

    let err = install_full_snapshot_from_reader(&engine, &mut io::Cursor::new(&data)).unwrap_err();
    assert!(
        err.to_string().contains("bytes past its checksum"),
        "got: {err}"
    );
}

// ── Chunked Transfer Protocol Tests ────────────────────────────

#[test]
fn test_snapshot_chunk_message_serde_roundtrip() {
    use crate::storage::Vote;

    // Header message roundtrip
    let header = SnapshotTransferHeader {
        vote: Vote::new(1, 1),
        meta: openraft::storage::SnapshotMeta {
            // A concrete log id, so the round-trip below proves the metadata
            // survives serialization rather than comparing two defaults.
            last_log_id: Some(openraft::LogId::new(
                crate::storage::CommittedLeaderId {
                    term: 7,
                    node_id: 0,
                },
                42,
            )),
            last_membership: openraft::StoredMembership::default(),
        },
        data_size: 12345,
    };
    let msg = SnapshotChunkMessage::Header(header);
    let bytes = rmp_serde::to_vec(&msg).expect("serialize header");
    let decoded: SnapshotChunkMessage = rmp_serde::from_slice(&bytes).expect("deserialize header");
    match decoded {
        SnapshotChunkMessage::Header(h) => {
            assert_eq!(h.data_size, 12345);
            let log_id = h.meta.last_log_id.expect("last_log_id survives round-trip");
            assert_eq!(log_id.index, 42);
            assert_eq!(log_id.committed_leader_id().term, 7);
        }
        _ => panic!("expected Header variant"),
    }

    // DataChunk message roundtrip
    let chunk_data = vec![1u8, 2, 3, 4, 5];
    let chunk_msg = SnapshotChunkMessage::DataChunk(chunk_data.clone());
    let bytes2 = rmp_serde::to_vec(&chunk_msg).expect("serialize chunk");
    let decoded2: SnapshotChunkMessage = rmp_serde::from_slice(&bytes2).expect("deserialize chunk");
    match decoded2 {
        SnapshotChunkMessage::DataChunk(d) => assert_eq!(d, chunk_data),
        _ => panic!("expected DataChunk variant"),
    }
}

/// A data chunk travels as one binary blob, not as an array of integers:
/// encoded byte by byte, a 2 MiB chunk took seconds to encode and decode and
/// every byte above 127 cost two on the wire.
#[test]
fn snapshot_chunk_data_is_one_binary_blob() {
    let chunk_data = vec![0xFFu8; 1 << 20];
    let bytes = rmp_serde::to_vec(&SnapshotChunkMessage::DataChunk(chunk_data.clone()))
        .expect("serialize chunk");
    assert!(
        bytes.len() <= chunk_data.len() + 16,
        "a {} byte chunk encoded to {} bytes",
        chunk_data.len(),
        bytes.len()
    );
    match rmp_serde::from_slice(&bytes).expect("deserialize chunk") {
        SnapshotChunkMessage::DataChunk(d) => assert_eq!(d, chunk_data),
        SnapshotChunkMessage::Header(_) => panic!("expected DataChunk variant"),
    }
}

#[test]
fn test_install_full_snapshot_from_reader() {
    // Build snapshot, install via reader, verify data matches
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine.put(Partition::Node, b"node:1", b"alice").unwrap();
    engine
        .put(Partition::EdgeProp, b"ep:1", b"prop_data")
        .unwrap();

    let snapshot_data = build_full_snapshot(&engine).unwrap();

    // Install to fresh engine via reader
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());

    let mut cursor = std::io::Cursor::new(&snapshot_data);
    install_full_snapshot_from_reader(&engine2, &mut cursor).unwrap();

    assert_eq!(
        engine2
            .get(Partition::Node, b"node:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"alice".to_vec())
    );
    assert_eq!(
        engine2
            .get(Partition::EdgeProp, b"ep:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"prop_data".to_vec())
    );
}

#[test]
fn test_install_full_snapshot_from_reader_cleans_stale() {
    // Pre-existing data not in snapshot gets cleaned up
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine.put(Partition::Node, b"node:1", b"alice").unwrap();

    let snapshot_data = build_full_snapshot(&engine).unwrap();

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2
        .put(Partition::Node, b"node:stale", b"old_data")
        .unwrap();

    let mut cursor = std::io::Cursor::new(&snapshot_data);
    install_full_snapshot_from_reader(&engine2, &mut cursor).unwrap();

    // Stale key removed
    assert!(
        engine2
            .get(Partition::Node, b"node:stale")
            .unwrap()
            .is_none()
    );
    // Snapshot key present
    assert!(engine2.get(Partition::Node, b"node:1").unwrap().is_some());
}

#[test]
fn test_install_full_from_reader_checksum_validation() {
    // Corrupt data should fail checksum
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine.put(Partition::Node, b"node:1", b"alice").unwrap();

    let mut snapshot_data = build_full_snapshot(&engine).unwrap();

    // Corrupt the checksum bytes (last 8 bytes) to trigger mismatch.
    // Corrupting data bytes could break CNSN parsing before reaching
    // the checksum, so we target the checksum directly.
    let len = snapshot_data.len();
    snapshot_data[len - 1] ^= 0xFF;

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());

    let mut cursor = std::io::Cursor::new(&snapshot_data);
    let result = install_full_snapshot_from_reader(&engine2, &mut cursor);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch")
    );
}

#[test]
fn test_chunked_full_snapshot_roundtrip() {
    // End-to-end: build → chunk → reassemble → install via reader
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine.put(Partition::Node, b"node:1", b"alice").unwrap();
    engine.put(Partition::Node, b"node:2", b"bob").unwrap();
    engine
        .put(Partition::EdgeProp, b"ep:1:2", b"friends")
        .unwrap();

    let snapshot_data = build_full_snapshot(&engine).unwrap();

    // Chunk with small size to test multi-chunk
    let small_chunk_size = 64;
    let chunks: Vec<&[u8]> = snapshot_data.chunks(small_chunk_size).collect();
    assert!(chunks.len() > 1, "should produce multiple chunks");

    // Reassemble
    let reassembled: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
    assert_eq!(reassembled, snapshot_data);

    // Install via reader from reassembled data
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());

    let mut cursor = std::io::Cursor::new(&reassembled);
    install_full_snapshot_from_reader(&engine2, &mut cursor).unwrap();

    assert_eq!(
        engine2
            .get(Partition::Node, b"node:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"alice".to_vec())
    );
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:2")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"bob".to_vec())
    );
    assert_eq!(
        engine2
            .get(Partition::EdgeProp, b"ep:1:2")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"friends".to_vec())
    );
}

/// A snapshot in the version 1 layout, as released builds write it: one Node
/// entry, no table section.
fn version_1_snapshot(key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.push(1);
    buf.push(1);
    buf.push(partition_tag(Partition::Node));
    put_entries(&mut buf, &[(key.to_vec(), value.to_vec())]).unwrap();
    let hash = fnv1a_64(&buf);
    buf.extend_from_slice(&hash.to_le_bytes());
    buf
}

fn table(name: &str, rows: &[(&[u8], &[u8])]) -> ColumnarTable {
    (
        name.to_owned(),
        rows.iter().map(|(k, v)| (k.to_vec(), v.to_vec())).collect(),
    )
}

#[test]
fn a_version_1_snapshot_installs_and_leaves_tables_alone() {
    // A dump taken with a released build is how a store reaches this one, so
    // the older layout must still install; it says nothing about columnar
    // tables, so the receiver's stay as they are.
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine
        .replace_columnar_tables(vec![table("Kept", &[(b"k1", b"v1")])])
        .unwrap();
    let data = version_1_snapshot(b"node:1", b"from-v1");

    install_full_snapshot(&engine, &data).unwrap();
    assert_eq!(
        engine.get(Partition::Node, b"node:1").unwrap().as_deref(),
        Some(b"from-v1".as_slice())
    );
    assert_eq!(
        engine.columnar_tables_at(u64::MAX).unwrap(),
        vec![table("Kept", &[(b"k1", b"v1")])]
    );

    let mut cursor = std::io::Cursor::new(&data);
    install_full_snapshot_from_reader(&engine, &mut cursor).unwrap();
}

#[test]
fn columnar_tables_travel_with_snapshots() {
    // The receiver ends with exactly the sender's tables: a table only the
    // receiver had is dropped, a shared name takes the sender's rows.
    let dir = tempdir().unwrap();
    let sender = open_engine(dir.path());
    sender
        .replace_columnar_tables(vec![
            table("Trade", &[(b"t1", b"AAPL"), (b"t2", b"MSFT")]),
            table("Quote", &[(b"q1", b"1.0")]),
        ])
        .unwrap();
    let expected = sender.columnar_tables_at(u64::MAX).unwrap();
    assert_eq!(expected.len(), 2, "sender holds both tables");

    let full = build_full_snapshot(&sender).unwrap();
    let dir2 = tempdir().unwrap();
    let receiver = open_engine(dir2.path());
    receiver
        .replace_columnar_tables(vec![
            table("Trade", &[(b"old", b"stale")]),
            table("Gone", &[(b"g", b"x")]),
        ])
        .unwrap();
    install_full_snapshot(&receiver, &full).unwrap();
    assert_eq!(receiver.columnar_tables_at(u64::MAX).unwrap(), expected);

    let dir3 = tempdir().unwrap();
    let streamed = open_engine(dir3.path());
    let mut cursor = std::io::Cursor::new(&full);
    install_full_snapshot_from_reader(&streamed, &mut cursor).unwrap();
    assert_eq!(streamed.columnar_tables_at(u64::MAX).unwrap(), expected);
}

#[test]
fn bytes_after_the_last_section_are_refused() {
    // A valid checksum over a body with trailing bytes means the producer
    // wrote something this parser does not know how to read.
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    let data = build_full_snapshot(&engine).unwrap();
    let mut body = data[..data.len() - 8].to_vec();
    body.extend_from_slice(b"extra");
    let hash = fnv1a_64(&body);
    body.extend_from_slice(&hash.to_le_bytes());

    let err = install_full_snapshot(&engine, &body).unwrap_err();
    assert!(
        err.to_string().contains("bytes past its last section"),
        "got: {err}"
    );
}

#[test]
fn a_corrupt_length_ends_in_eof_not_a_huge_allocation() {
    // The streaming parser cannot check the checksum first; a length field
    // claiming ~4 GiB must fail on the bytes that are not there instead of
    // allocating that much up front.
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.push(FORMAT_VERSION);
    buf.push(1);
    buf.push(partition_tag(Partition::Node));
    buf.extend_from_slice(&1u32.to_be_bytes());
    buf.extend_from_slice(&u32::MAX.to_be_bytes());
    buf.extend_from_slice(b"short");

    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    let mut cursor = std::io::Cursor::new(&buf);
    let err = install_full_snapshot_from_reader(&engine, &mut cursor).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
}

#[test]
fn a_version_newer_than_this_build_is_refused() {
    let mut data = version_1_snapshot(b"node:1", b"v");
    data[4] = FORMAT_VERSION + 1;
    let body_len = data.len() - 8;
    let hash = fnv1a_64(&data[..body_len]);
    data[body_len..].copy_from_slice(&hash.to_le_bytes());

    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    let err = install_full_snapshot(&engine, &data).unwrap_err();
    assert!(
        err.to_string().contains("unsupported snapshot version"),
        "got: {err}"
    );
}
