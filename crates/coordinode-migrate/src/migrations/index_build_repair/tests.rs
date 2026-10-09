use coordinode_modality::{BuildFailure, GenerationId, IndexBuildRecord, IndexId};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use super::{survey, upgrade_record};
use crate::Backup;

fn record() -> IndexBuildRecord {
    IndexBuildRecord::accepted(
        IndexId::from_raw(3),
        GenerationId::from_raw(9),
        BuildFailure::Withdraw,
    )
}

/// `record` as a build before duplicate repair wrote it: five fields.
fn old_bytes(record: &IndexBuildRecord) -> Vec<u8> {
    let current = rmp_serde::to_vec(record).expect("encode");
    let mut value = rmpv::decode::read_value(&mut &current[..]).expect("decode");
    if let rmpv::Value::Array(fields) = &mut value {
        fields.truncate(5);
    }
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &value).expect("encode");
    out
}

fn store(dir: &std::path::Path) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open")
}

/// An old record becomes a current one with no repair policy and no
/// repairs, the rest unchanged: the error the server retries forever on is
/// the one this rewrites.
#[test]
fn an_old_record_reads_as_current_without_repairs() {
    let old = old_bytes(&record());
    let refused = rmp_serde::from_slice::<IndexBuildRecord>(&old).expect_err("refused");
    assert!(
        refused.to_string().contains("invalid length 5"),
        "{refused}"
    );
    let new = upgrade_record(&old).expect("upgrade").expect("rewritten");
    assert_eq!(
        rmp_serde::from_slice::<IndexBuildRecord>(&new).expect("reads"),
        record()
    );
}

/// A current record is left alone; any other shape stops the run.
#[test]
fn current_records_stay_and_unknown_ones_stop() {
    let current = rmp_serde::to_vec(&record()).expect("encode");
    assert_eq!(upgrade_record(&current).expect("survey"), None);

    let mut six = rmpv::decode::read_value(&mut &current[..]).expect("decode");
    if let rmpv::Value::Array(fields) = &mut six {
        fields.truncate(6);
    }
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &six).expect("encode");
    let err = upgrade_record(&bytes).expect_err("six fields");
    assert!(format!("{err:#}").contains("6 fields"), "{err:#}");
}

/// The store and its checkpoints are rewritten in place; a directory that
/// is not a store is left without one.
#[test]
fn records_in_the_store_and_its_checkpoints_are_rewritten() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path().join("data");
    let key = IndexBuildRecord::key_of(GenerationId::from_raw(9));
    {
        let engine = store(&data);
        engine
            .put(Partition::Schema, &key, &old_bytes(&record()))
            .expect("put");
        engine
            .create_checkpoint(&data.join("checkpoints/ckpt-1"))
            .expect("checkpoint");
        engine.persist().expect("persist");
    }

    let found = survey(&data).expect("survey");
    assert_eq!(
        found.iter().map(|f| f.path.clone()).collect::<Vec<_>>(),
        vec![data.clone(), data.join("checkpoints/ckpt-1")]
    );
    let backup = Backup::new(&data, "run1");
    for finding in &found {
        super::apply(finding, &backup).expect("apply");
    }
    for at in [data.clone(), data.join("checkpoints/ckpt-1")] {
        let engine = StorageEngine::open_checkpoint(&at).expect("open");
        let value = engine
            .get(Partition::Schema, &key)
            .expect("get")
            .expect("present");
        assert_eq!(
            rmp_serde::from_slice::<IndexBuildRecord>(&value).expect("reads"),
            record(),
            "{}",
            at.display()
        );
    }
    assert!(survey(&data).expect("survey").is_empty());

    let empty = dir.path().join("not-a-store");
    std::fs::create_dir_all(&empty).expect("mkdir");
    assert!(survey(&empty).expect("survey").is_empty());
    assert_eq!(std::fs::read_dir(&empty).expect("list").count(), 0);
}
