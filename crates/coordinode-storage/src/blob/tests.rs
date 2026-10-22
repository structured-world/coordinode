use std::sync::Arc;

use coordinode_core::graph::blob::{create_blob, encode_blob_key};
use coordinode_core::txn::timestamp::TimestampOracle;
use lsm_tree::fs::{CrashFs, Fault, FaultFs, FaultInjector, FaultOp, FaultRule, StdFs};
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn config(dir: &TempDir, crash: &Arc<CrashFs>, faults: &Arc<FaultInjector>) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
    .with_fs(Arc::new(FaultFs::with_injector(
        CrashFs::clone(crash),
        Arc::clone(faults),
    )))
}

/// A blob whose chunks and metadata were written survives a power cut right
/// after the writes returned: an answer to an upload is a promise the store
/// keeps, and a reference committed after it never names missing bytes. A
/// deleted blob's metadata stays deleted across the cut.
#[test]
fn a_written_blob_survives_a_crash_right_after() {
    let dir = TempDir::new().expect("tempdir");
    let crash = Arc::new(CrashFs::new(StdFs));
    let faults = Arc::new(FaultInjector::new());
    let config = config(&dir, &crash, &faults);
    let (_, chunks) = create_blob(&vec![0x5a; 300 << 10]);
    {
        let engine =
            StorageEngine::open_embedded(&config, Arc::new(TimestampOracle::new())).expect("open");
        let held = put_chunks_durably(&engine, &chunks).expect("chunks");
        assert_eq!(held, 0);
        put_meta_durably(&engine, b"blobmeta:kept", b"meta").expect("meta");
        put_meta_durably(&engine, b"blobmeta:gone", b"meta").expect("meta");
        delete_meta_durably(&engine, b"blobmeta:gone").expect("delete");
        // A chunk already held is counted and not written again.
        assert_eq!(
            put_chunks_durably(&engine, &chunks).expect("again"),
            chunks.len()
        );
        // Killed, not shut down: nothing the drop would flush reaches the
        // disk, while its files and its directory locks are let go as a
        // dead process lets them go.
        let refuse = Fault::Error(lsm_tree::io::ErrorKind::Other);
        for op in [
            FaultOp::Open,
            FaultOp::Write,
            FaultOp::SyncAll,
            FaultOp::SyncData,
            FaultOp::Rename,
        ] {
            faults.arm(FaultRule::new(op, refuse));
        }
        drop(engine);
        faults.clear();
    }
    crash.crash();

    let engine =
        StorageEngine::open_embedded(&config, Arc::new(TimestampOracle::new())).expect("reopen");
    for (chunk_id, data) in &chunks {
        let held = engine
            .get(Partition::Blob, &encode_blob_key(chunk_id))
            .expect("get");
        // Compared, not printed: a chunk is hundreds of kilobytes.
        assert!(
            held.as_deref() == Some(data.as_slice()),
            "chunk {chunk_id} lost or changed across the power cut"
        );
    }
    assert_eq!(
        engine
            .get(Partition::BlobRef, b"blobmeta:kept")
            .expect("get")
            .as_deref(),
        Some(&b"meta"[..])
    );
    assert_eq!(
        engine
            .get(Partition::BlobRef, b"blobmeta:gone")
            .expect("get"),
        None,
        "a deleted blob came back"
    );
}
