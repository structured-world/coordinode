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

/// `engine`'s full snapshot, in memory.
fn build_full_snapshot(engine: &StorageEngine) -> io::Result<Vec<u8>> {
    let mut buf = io::Cursor::new(Vec::new());
    write_full_snapshot(engine, &mut buf)?;
    Ok(buf.into_inner())
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
    let written = write_full_snapshot(&engine, &mut buf).unwrap();
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
