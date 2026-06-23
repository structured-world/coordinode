use super::*;
use tempfile::TempDir;

fn test_entry(n: usize, ts: u64) -> DrainEntry {
    let mutations = (0..n)
        .map(|i| Mutation::Put {
            partition: PartitionId::Node,
            key: format!("key_{i}").into_bytes(),
            value: vec![i as u8, 42],
        })
        .collect();
    DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
}

fn delete_entry(ts: u64) -> DrainEntry {
    let mutations = vec![Mutation::Delete {
        partition: PartitionId::Adj,
        key: b"adj:key".to_vec(),
    }];
    DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
}

fn merge_entry(ts: u64) -> DrainEntry {
    let mutations = vec![Mutation::Merge {
        partition: PartitionId::Node,
        key: b"node:1".to_vec(),
        operand: vec![1, 2, 3, 4],
    }];
    DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
}

#[test]
fn roundtrip_put_entry() {
    let entry = test_entry(3, 100);
    let encoded = encode_entry(&entry);
    let decoded = decode_entry(&encoded[4..]).unwrap();

    assert_eq!(decoded.mutations.len(), 3);
    assert_eq!(decoded.commit_ts.as_raw(), 100);
    assert_eq!(decoded.start_ts.as_raw(), 1);

    let Mutation::Put {
        partition,
        key,
        value,
    } = &decoded.mutations[0]
    else {
        assert!(false, "expected Put variant");
        return;
    };
    assert_eq!(*partition, PartitionId::Node);
    assert_eq!(key, b"key_0");
    assert_eq!(value, &[0u8, 42]);
}

#[test]
fn roundtrip_delete_entry() {
    let entry = delete_entry(200);
    let encoded = encode_entry(&entry);
    let decoded = decode_entry(&encoded[4..]).unwrap();
    assert_eq!(decoded.commit_ts.as_raw(), 200);
    let Mutation::Delete { partition, key } = &decoded.mutations[0] else {
        assert!(false, "expected Delete variant");
        return;
    };
    assert_eq!(*partition, PartitionId::Adj);
    assert_eq!(key, b"adj:key");
}

#[test]
fn roundtrip_merge_entry() {
    let entry = merge_entry(300);
    let encoded = encode_entry(&entry);
    let decoded = decode_entry(&encoded[4..]).unwrap();
    let Mutation::Merge { operand, .. } = &decoded.mutations[0] else {
        assert!(false, "expected Merge variant");
        return;
    };
    assert_eq!(operand, &[1u8, 2, 3, 4]);
}

#[test]
fn open_and_append() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path()).unwrap();

    wb.append(&test_entry(2, 100)).unwrap();
    wb.append(&test_entry(3, 200)).unwrap();

    let recovered = NvmeWriteBuffer::recover(dir.path()).unwrap();
    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0].mutations.len(), 2);
    assert_eq!(recovered[0].commit_ts.as_raw(), 100);
    assert_eq!(recovered[1].mutations.len(), 3);
    assert_eq!(recovered[1].commit_ts.as_raw(), 200);
}

#[test]
fn recover_empty_dir() {
    let dir = TempDir::new().unwrap();
    let entries = NvmeWriteBuffer::recover(dir.path()).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn recover_clears_file() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path()).unwrap();
    wb.append(&test_entry(1, 100)).unwrap();
    drop(wb);

    // First recovery: returns entries.
    let first = NvmeWriteBuffer::recover(dir.path()).unwrap();
    assert_eq!(first.len(), 1);

    // Second recovery: file truncated, no entries.
    let second = NvmeWriteBuffer::recover(dir.path()).unwrap();
    assert!(second.is_empty());
}

#[test]
fn atomic_drain_checkpoint() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path()).unwrap();

    wb.append(&test_entry(2, 100)).unwrap();
    wb.append(&test_entry(1, 200)).unwrap();

    // begin_drain checkpoints current → draining
    let token = wb.begin_drain();

    // New entry goes to fresh current
    wb.append(&test_entry(1, 300)).unwrap();

    // complete_drain removes the draining file
    wb.complete_drain(token);

    // Only the post-checkpoint entry remains in current file
    let recovered = NvmeWriteBuffer::recover(dir.path()).unwrap();
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].commit_ts.as_raw(), 300);
}

#[test]
fn recover_draining_file_survives_crash() {
    // Simulate: begin_drain happened, crash before complete_drain.
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path()).unwrap();

    wb.append(&test_entry(2, 100)).unwrap();
    let token = wb.begin_drain();
    // Simulate crash: do NOT call complete_drain.
    drop(wb);

    // On restart: recover finds draining file.
    // 1 DrainEntry was written (with 2 mutations inside it).
    let recovered = NvmeWriteBuffer::recover(dir.path()).unwrap();
    assert_eq!(
        recovered.len(),
        1,
        "draining file entries must be recovered"
    );
    assert_eq!(recovered[0].commit_ts.as_raw(), 100);
    assert_eq!(
        recovered[0].mutations.len(),
        2,
        "entry should contain the 2 original mutations"
    );

    // Draining file was removed after recovery.
    let draining = find_draining_files(dir.path()).unwrap();
    assert!(
        draining.is_empty(),
        "draining file should be removed after recovery"
    );
    let _ = token; // suppress unused warning
}

#[test]
fn write_buffer_hook_begin_complete() {
    // Test WriteBufferHook trait interface.
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path()).unwrap();

    wb.append(&test_entry(1, 100)).unwrap();

    let hook: &dyn WriteBufferHook = &wb;
    let token = hook.begin_drain();
    hook.complete_drain(token);

    // After complete, no files remain.
    let draining = find_draining_files(dir.path()).unwrap();
    assert!(draining.is_empty());
}

#[test]
fn partition_roundtrip() {
    let partitions = [
        PartitionId::Node,
        PartitionId::Adj,
        PartitionId::EdgeProp,
        PartitionId::Blob,
        PartitionId::BlobRef,
        PartitionId::Schema,
        PartitionId::Idx,
        PartitionId::Counter,
    ];
    for p in partitions {
        let b = partition_to_u8(p);
        let back = u8_to_partition(b).unwrap();
        assert_eq!(p, back, "partition {p:?} roundtrip failed");
    }
}

#[test]
fn partial_entry_truncation_handled() {
    // Write a valid entry followed by a partial one (simulates crash mid-write).
    let dir = TempDir::new().unwrap();
    let current_path = dir.path().join(CURRENT_FILE);

    let entry = test_entry(1, 100);
    let encoded = encode_entry(&entry);

    // Write complete entry + partial garbage.
    std::fs::write(&current_path, {
        let mut data = encoded;
        data.extend_from_slice(&[0xFF, 0xFF, 0x00]); // partial/corrupt
        data
    })
    .unwrap();

    let recovered = NvmeWriteBuffer::recover(dir.path()).unwrap();
    // Only the complete entry should be recovered.
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].commit_ts.as_raw(), 100);
}
