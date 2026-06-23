use super::*;
use crate::oplog::entry::OplogOp;

fn make_entry(index: u64, ts: u64) -> OplogEntry {
    OplogEntry {
        ts,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: format!("key-{index}").into_bytes(),
            value: b"val".to_vec(),
        }],
        is_migration: false,
        pre_images: None,
    }
}

#[test]
fn varint_roundtrip_values() {
    for &v in &[
        0u64,
        1,
        127,
        128,
        255,
        16_383,
        16_384,
        u32::MAX as u64,
        u64::MAX / 2,
    ] {
        let mut buf = Vec::new();
        encode_varint(v, &mut buf);
        let mut cursor = Cursor::new(&buf);
        let decoded = decode_varint(&mut cursor).expect("decode");
        assert_eq!(v, decoded, "varint roundtrip failed for {v}");
    }
}

#[test]
fn varint_1_byte_for_small_values() {
    let mut buf = Vec::new();
    encode_varint(0, &mut buf);
    assert_eq!(buf.len(), 1);
    encode_varint(127, &mut buf);
    assert_eq!(buf.len(), 2);
}

#[test]
fn segment_write_read_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("seg-00000.bin");

    let entries: Vec<_> = (0..5u64).map(|i| make_entry(i, 1000 + i)).collect();

    // Write
    let mut writer = SegmentWriter::create(&path, 42, 0).expect("create");
    for e in &entries {
        writer.append(e).expect("append");
    }
    assert_eq!(writer.entry_count(), 5);
    writer.seal().expect("seal");

    // Read back
    let reader = SegmentReader::open(&path).expect("open");
    assert_eq!(reader.header.shard_id, 42);
    assert_eq!(reader.header.first_index, 0);
    assert_eq!(reader.header.version, FORMAT_VERSION);
    assert_eq!(reader.footer.entry_count, 5);
    assert_eq!(reader.footer.first_ts, 1000);
    assert_eq!(reader.footer.last_ts, 1004);
    assert_eq!(reader.entries(), &entries[..]);
}

#[test]
fn empty_segment_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty.bin");

    let writer = SegmentWriter::create(&path, 1, 100).expect("create");
    writer.seal().expect("seal");

    let reader = SegmentReader::open(&path).expect("open");
    assert_eq!(reader.footer.entry_count, 0);
    assert!(reader.entries().is_empty());
    assert_eq!(reader.header.first_index, 100);
}

#[test]
fn entry_checksum_mismatch_detected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("corrupt.bin");

    let mut writer = SegmentWriter::create(&path, 0, 0).expect("create");
    writer.append(&make_entry(0, 100)).expect("append");
    writer.seal().expect("seal");

    // Flip a bit in the middle of the entries section
    let mut data = std::fs::read(&path).expect("read");
    let mid = (HEADER_SIZE as usize) + 5; // middle of first entry payload
    data[mid] ^= 0xFF;
    std::fs::write(&path, &data).expect("write corrupt");

    let result = SegmentReader::open(&path);
    assert!(result.is_err(), "should detect entry corruption");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("checksum") || msg.contains("mismatch"),
        "unexpected error: {msg}"
    );
}

#[test]
fn footer_checksum_mismatch_detected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bad-footer.bin");

    let mut writer = SegmentWriter::create(&path, 0, 0).expect("create");
    writer.append(&make_entry(0, 100)).expect("append");
    writer.seal().expect("seal");

    // Corrupt the footer checksum bytes (last 4 bytes of file)
    let mut data = std::fs::read(&path).expect("read");
    let len = data.len();
    data[len - 1] ^= 0xFF;
    std::fs::write(&path, &data).expect("write");

    let result = SegmentReader::open(&path);
    assert!(result.is_err(), "should detect footer corruption");
}

#[test]
fn invalid_magic_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("invalid.bin");

    // Write garbage — valid length (>= 50B) but wrong magic
    let garbage = vec![0xFFu8; 64];
    std::fs::write(&path, &garbage).expect("write");

    let result = SegmentReader::open(&path);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("magic"), "unexpected error: {msg}");
}

#[test]
fn too_short_file_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("short.bin");
    std::fs::write(&path, b"OPLO").expect("write");

    let result = SegmentReader::open(&path);
    assert!(result.is_err());
}

#[test]
fn large_entry_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("large.bin");

    // 16 KB value
    let large_value = vec![0xABu8; 16 * 1024];
    let entry = OplogEntry {
        ts: 9999,
        term: 1,
        index: 0,
        shard: 0,
        ops: vec![crate::oplog::entry::OplogOp::Insert {
            partition: 0,
            key: b"bigkey".to_vec(),
            value: large_value.clone(),
        }],
        is_migration: false,
        pre_images: None,
    };

    let mut writer = SegmentWriter::create(&path, 0, 0).expect("create");
    writer.append(&entry).expect("append");
    writer.seal().expect("seal");

    let reader = SegmentReader::open(&path).expect("open");
    assert_eq!(reader.entries().len(), 1);
    if let crate::oplog::entry::OplogOp::Insert { ref value, .. } = reader.entries()[0].ops[0] {
        assert_eq!(value, &large_value);
    } else {
        panic!("expected Insert op");
    }
}

/// `flush_and_sync()` succeeds on a freshly created segment with no entries.
#[test]
fn flush_and_sync_empty_writer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty-sync.bin");
    let mut writer = SegmentWriter::create(&path, 0, 0).expect("create");
    // No entries written — flush_and_sync must still succeed.
    writer
        .flush_and_sync()
        .expect("flush_and_sync on empty writer");
}

/// `flush_and_sync()` makes appended entries durably readable after re-seal.
///
/// Verifies that calling flush_and_sync() after append() ensures the
/// BufWriter is flushed. After a subsequent seal() + SegmentReader::open(),
/// all entries must be present.
#[test]
fn flush_and_sync_then_seal_readable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("sync-then-seal.bin");

    let mut writer = SegmentWriter::create(&path, 7, 0).expect("create");
    for i in 0..4u64 {
        writer.append(&make_entry(i, 2000 + i)).expect("append");
    }
    // Fsync before sealing (the "ONE fsync per write batch" path).
    writer.flush_and_sync().expect("flush_and_sync");

    // After sync, seal and read back.
    writer.seal().expect("seal");
    let reader = SegmentReader::open(&path).expect("open");
    assert_eq!(
        reader.footer.entry_count, 4,
        "all 4 entries must survive flush_and_sync"
    );
    assert_eq!(reader.entries()[0].index, 0);
    assert_eq!(reader.entries()[3].index, 3);
}
