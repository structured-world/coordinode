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
    assert!(engine2
        .get(Partition::Schema, b"schema:label:Old")
        .unwrap()
        .is_none());
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
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("checksum mismatch"));
}

#[test]
fn test_snapshot_invalid_magic() {
    let data = b"BADMxxxxxxxx";
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    let result = install_full_snapshot(&engine, data);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("invalid snapshot magic"));
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
    assert!(engine2
        .get(Partition::Node, b"node:0:99")
        .unwrap()
        .is_none());
}

// -- Incremental snapshot tests (R135 / G057: native seqno MVCC) --

#[test]
fn test_incremental_no_changes_returns_none() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    // Snapshot boundary after write — incremental since this point finds nothing.
    let since = Timestamp::from_raw(engine.snapshot());

    let result = build_incremental_snapshot(&engine, since).unwrap();
    assert!(result.is_none(), "no changes after snapshot boundary");
}

#[test]
fn test_incremental_captures_recent_changes() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    // Phase 1: baseline
    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    engine.put(Partition::Node, b"node:0:2", b"bob").unwrap();
    let since = Timestamp::from_raw(engine.snapshot());

    // Phase 2: changes after baseline
    engine
        .put(Partition::Node, b"node:0:1", b"alice-updated")
        .unwrap();
    engine
        .put(Partition::Node, b"node:0:3", b"charlie")
        .unwrap();

    let data = build_incremental_snapshot(&engine, since)
        .unwrap()
        .expect("should have changes");

    // Install into target with baseline data
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    engine2.put(Partition::Node, b"node:0:2", b"bob").unwrap();

    install_incremental_snapshot(&engine2, &data).unwrap();

    // node:0:1 updated
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"alice-updated".to_vec())
    );

    // node:0:3 new
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:3")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"charlie".to_vec())
    );

    // node:0:2 unchanged (not in delta)
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:2")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"bob".to_vec())
    );
}

#[test]
fn test_incremental_schema_always_included() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine
        .put(Partition::Schema, b"schema:label:User", b"{name:string}")
        .unwrap();
    engine
        .put(Partition::Schema, b"raft:vote", b"raft-data")
        .unwrap();

    // Incremental with a future since_ts should still include schema keys
    let data = build_incremental_snapshot(&engine, Timestamp::from_raw(999999))
        .unwrap()
        .expect("schema keys should always be included");

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2
        .put(Partition::Schema, b"raft:vote", b"target-raft")
        .unwrap();

    install_incremental_snapshot(&engine2, &data).unwrap();

    // Schema key installed
    assert_eq!(
        engine2
            .get(Partition::Schema, b"schema:label:User")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"{name:string}".to_vec())
    );
    // Raft key preserved (not overwritten by incremental install)
    assert_eq!(
        engine2
            .get(Partition::Schema, b"raft:vote")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"target-raft".to_vec())
    );
}

#[test]
fn test_incremental_checksum_validation() {
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine.put(Partition::Node, b"node:0:1", b"data").unwrap();
    // since_ts=0 ensures the write is captured as a change
    let mut data = build_incremental_snapshot(&engine, Timestamp::from_raw(0))
        .unwrap()
        .expect("should have data");

    // Corrupt payload
    if data.len() > 10 {
        data[8] ^= 0xFF;
    }

    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    let result = install_incremental_snapshot(&engine2, &data);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("checksum mismatch"));
}

#[test]
fn test_incremental_detects_deleted_key() {
    // With native seqno MVCC, a key present in old snapshot but absent in
    // current state is detected as a deletion (empty value = tombstone).
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    let since = Timestamp::from_raw(engine.snapshot());

    // Delete the key after the snapshot boundary
    engine.delete(Partition::Node, b"node:0:1").unwrap();

    let data = build_incremental_snapshot(&engine, since)
        .unwrap()
        .expect("deletion should be a change");

    // Target has the key
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2
        .put(Partition::Node, b"node:0:1", b"to-be-deleted")
        .unwrap();

    install_incremental_snapshot(&engine2, &data).unwrap();

    // Key should be deleted
    assert!(
        engine2.get(Partition::Node, b"node:0:1").unwrap().is_none(),
        "tombstone should delete key on receiver"
    );
}

#[test]
fn test_incremental_multiple_partitions() {
    // Write data across Node, Adj, EdgeProp at different seqno phases,
    // build incremental, install on fresh engine, verify.
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    // Phase 1: baseline
    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x92\x02\x03")
        .unwrap();
    engine
        .put(Partition::EdgeProp, b"edgeprop:KNOWS:1:2", b"since=2020")
        .unwrap();
    let since = Timestamp::from_raw(engine.snapshot());

    // Phase 2: changes after baseline
    engine
        .put(Partition::Node, b"node:0:1", b"alice-v2")
        .unwrap();
    engine
        .put(Partition::Node, b"node:0:5", b"new-node")
        .unwrap();
    engine
        .put(Partition::Adj, b"adj:LIKES:out:5", b"\x92\x01")
        .unwrap();
    // EdgeProp unchanged

    // Schema always included
    engine
        .put(Partition::Schema, b"schema:label:User", b"{}")
        .unwrap();

    let data = build_incremental_snapshot(&engine, since)
        .unwrap()
        .expect("should have changes");

    // Target: has phase 1 data
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    engine2
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x92\x02\x03")
        .unwrap();
    engine2
        .put(Partition::EdgeProp, b"edgeprop:KNOWS:1:2", b"since=2020")
        .unwrap();

    install_incremental_snapshot(&engine2, &data).unwrap();

    // Node updated
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"alice-v2".to_vec())
    );
    // New node added
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:5")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"new-node".to_vec())
    );
    // New adj added
    assert!(engine2
        .get(Partition::Adj, b"adj:LIKES:out:5")
        .unwrap()
        .is_some());
    // Unchanged EdgeProp still exists
    assert!(engine2
        .get(Partition::EdgeProp, b"edgeprop:KNOWS:1:2")
        .unwrap()
        .is_some());
    // Schema installed
    assert_eq!(
        engine2
            .get(Partition::Schema, b"schema:label:User")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"{}".to_vec())
    );
}

#[test]
fn test_incremental_is_smaller_than_full() {
    // Verify incremental snapshot is smaller than full when most data is unchanged.
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());

    // Write 100 nodes
    for i in 0..100u64 {
        engine
            .put(
                Partition::Node,
                format!("node:0:{i:03}").as_bytes(),
                format!("value-{i}-with-some-payload-data").as_bytes(),
            )
            .unwrap();
    }
    let since = Timestamp::from_raw(engine.snapshot());

    // Update only 3
    engine
        .put(Partition::Node, b"node:0:007", b"updated-7")
        .unwrap();
    engine
        .put(Partition::Node, b"node:0:042", b"updated-42")
        .unwrap();
    engine
        .put(Partition::Node, b"node:0:099", b"updated-99")
        .unwrap();

    let full = build_full_snapshot(&engine).unwrap();
    let incr = build_incremental_snapshot(&engine, since)
        .unwrap()
        .expect("should have changes");

    assert!(
        incr.len() < full.len(),
        "incremental ({} bytes) should be smaller than full ({} bytes)",
        incr.len(),
        full.len()
    );
    // With 100 nodes and only 3 changed, incremental should be much smaller
    assert!(
        incr.len() < full.len() / 5,
        "incremental ({} bytes) should be <20% of full ({} bytes)",
        incr.len(),
        full.len()
    );
}

#[test]
fn test_incremental_with_oracle_engine() {
    // Production path: engine opened with TimestampOracle (HLC-like seqnos).
    // Verifies two-snapshot diff works when seqnos are large, non-contiguous
    // values (e.g., microsecond timestamps) instead of small sequential ints.
    use coordinode_core::txn::timestamp::TimestampOracle;
    use std::sync::Arc;

    let dir = tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::resume_from(
        coordinode_core::txn::timestamp::Timestamp::from_raw(1_000_000),
    ));
    let engine = StorageEngine::open_with_oracle(&config, oracle).unwrap();

    // Phase 1: baseline writes (oracle seqnos ~1_000_001+)
    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    engine.put(Partition::Node, b"node:0:2", b"bob").unwrap();
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x92\x02")
        .unwrap();
    let since = Timestamp::from_raw(engine.snapshot());

    // Phase 2: changes
    engine
        .put(Partition::Node, b"node:0:1", b"alice-v2")
        .unwrap();
    engine
        .put(Partition::Node, b"node:0:3", b"charlie")
        .unwrap();
    engine.delete(Partition::Node, b"node:0:2").unwrap();

    let data = build_incremental_snapshot(&engine, since)
        .unwrap()
        .expect("should have changes");

    // Install into fresh engine
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    engine2.put(Partition::Node, b"node:0:2", b"bob").unwrap();
    engine2
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x92\x02")
        .unwrap();

    install_incremental_snapshot(&engine2, &data).unwrap();

    // node:0:1 updated
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"alice-v2".to_vec())
    );
    // node:0:2 deleted
    assert!(
        engine2.get(Partition::Node, b"node:0:2").unwrap().is_none(),
        "node:0:2 should be deleted"
    );
    // node:0:3 new
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:0:3")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"charlie".to_vec())
    );
    // adj unchanged (not in delta)
    assert!(engine2
        .get(Partition::Adj, b"adj:KNOWS:out:1")
        .unwrap()
        .is_some());
}

#[test]
fn test_snapshot_transfer_serde_roundtrip() {
    // Verify SnapshotTransfer with since_ts serializes/deserializes correctly
    use crate::storage::Vote;

    let vote = Vote::new(1, 1);

    let transfer = SnapshotTransfer {
        vote,
        meta: openraft::storage::SnapshotMeta {
            last_log_id: None,
            last_membership: openraft::StoredMembership::default(),
            snapshot_id: "test-snap".to_string(),
        },
        data: vec![1, 2, 3],
        since_ts: Some(42000),
    };
    let bytes = rmp_serde::to_vec(&transfer).expect("serialize");
    let decoded: SnapshotTransfer = rmp_serde::from_slice(&bytes).expect("deserialize");
    assert_eq!(decoded.since_ts, Some(42000));
    assert_eq!(decoded.data, vec![1, 2, 3]);

    // Full snapshot (since_ts = None)
    let full_transfer = SnapshotTransfer {
        vote,
        meta: openraft::storage::SnapshotMeta {
            last_log_id: None,
            last_membership: openraft::StoredMembership::default(),
            snapshot_id: "full-snap".to_string(),
        },
        data: vec![4, 5],
        since_ts: None,
    };
    let bytes2 = rmp_serde::to_vec(&full_transfer).expect("serialize");
    let decoded2: SnapshotTransfer = rmp_serde::from_slice(&bytes2).expect("deserialize");
    assert_eq!(decoded2.since_ts, None);
}

// ── Chunked Transfer Protocol Tests ────────────────────────────

#[test]
fn test_chunk_snapshot_data_single_chunk() {
    // Data smaller than SNAPSHOT_CHUNK_SIZE → single chunk
    let data = vec![0u8; 100];
    let chunks: Vec<&[u8]> = chunk_snapshot_data(&data).collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].len(), 100);
}

#[test]
fn test_chunk_snapshot_data_multiple_chunks() {
    // Data larger than SNAPSHOT_CHUNK_SIZE → multiple chunks
    let size = SNAPSHOT_CHUNK_SIZE * 2 + 1000;
    let data = vec![0xABu8; size];
    let chunks: Vec<&[u8]> = chunk_snapshot_data(&data).collect();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].len(), SNAPSHOT_CHUNK_SIZE);
    assert_eq!(chunks[1].len(), SNAPSHOT_CHUNK_SIZE);
    assert_eq!(chunks[2].len(), 1000);

    // Reassembled data matches original
    let reassembled: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
    assert_eq!(reassembled, data);
}

#[test]
fn test_chunk_snapshot_data_exact_boundary() {
    // Data exactly SNAPSHOT_CHUNK_SIZE → single chunk, no remainder
    let data = vec![0u8; SNAPSHOT_CHUNK_SIZE];
    let chunks: Vec<&[u8]> = chunk_snapshot_data(&data).collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].len(), SNAPSHOT_CHUNK_SIZE);
}

#[test]
fn test_chunk_snapshot_data_empty() {
    let data: Vec<u8> = Vec::new();
    let chunks: Vec<&[u8]> = chunk_snapshot_data(&data).collect();
    assert_eq!(chunks.len(), 0);
}

#[test]
fn test_snapshot_chunk_message_serde_roundtrip() {
    use crate::storage::Vote;

    // Header message roundtrip
    let header = SnapshotTransferHeader {
        vote: Vote::new(1, 1),
        meta: openraft::storage::SnapshotMeta {
            last_log_id: None,
            last_membership: openraft::StoredMembership::default(),
            snapshot_id: "chunked-test".to_string(),
        },
        data_size: 12345,
        since_ts: Some(42000),
    };
    let msg = SnapshotChunkMessage::Header(header);
    let bytes = rmp_serde::to_vec(&msg).expect("serialize header");
    let decoded: SnapshotChunkMessage = rmp_serde::from_slice(&bytes).expect("deserialize header");
    match decoded {
        SnapshotChunkMessage::Header(h) => {
            assert_eq!(h.data_size, 12345);
            assert_eq!(h.since_ts, Some(42000));
            assert_eq!(h.meta.snapshot_id, "chunked-test");
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
    assert!(engine2
        .get(Partition::Node, b"node:stale")
        .unwrap()
        .is_none());
    // Snapshot key present
    assert!(engine2.get(Partition::Node, b"node:1").unwrap().is_some());
}

#[test]
fn test_install_incremental_snapshot_from_reader() {
    use coordinode_core::txn::timestamp::Timestamp;

    let dir = tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine =
        StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open oracle engine");

    // Write initial data
    engine.put(Partition::Node, b"node:1", b"v1").unwrap();
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"adj1")
        .unwrap();

    let since = Timestamp::from_raw(oracle.next().as_raw());

    // Write changes after snapshot point
    engine.put(Partition::Node, b"node:1", b"v2").unwrap();
    engine.put(Partition::Node, b"node:2", b"new").unwrap();

    let incr_data = build_incremental_snapshot(&engine, since)
        .unwrap()
        .expect("should have changes");

    // Install to fresh engine via reader
    let dir2 = tempdir().unwrap();
    let engine2 = open_engine(dir2.path());
    engine2.put(Partition::Node, b"node:1", b"v1").unwrap();
    engine2
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"adj1")
        .unwrap();

    let mut cursor = std::io::Cursor::new(&incr_data);
    install_incremental_snapshot_from_reader(&engine2, &mut cursor).unwrap();

    // node:1 updated to v2
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:1")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"v2".to_vec())
    );
    // node:2 added
    assert_eq!(
        engine2
            .get(Partition::Node, b"node:2")
            .unwrap()
            .map(|b| b.to_vec()),
        Some(b"new".to_vec())
    );
    // adj unchanged
    assert!(engine2
        .get(Partition::Adj, b"adj:KNOWS:out:1")
        .unwrap()
        .is_some());
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
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("checksum mismatch"));
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

#[test]
fn test_reconstruct_cnsn_payload_matches_original() {
    // Verify that reconstruct_cnsn_payload produces the same bytes as
    // build_full_snapshot (minus the 8-byte checksum at end)
    let dir = tempdir().unwrap();
    let engine = open_engine(dir.path());
    engine.put(Partition::Node, b"node:1", b"test").unwrap();
    engine.put(Partition::Adj, b"adj:X:out:1", b"adj").unwrap();

    let snapshot_data = build_full_snapshot(&engine).unwrap();

    // Parse the snapshot to get partitions
    let payload_without_checksum = &snapshot_data[..snapshot_data.len() - 8];

    // Parse manually for reconstruct
    let partitions: Vec<Partition> = snapshot_partitions().collect();
    let partition_count = partitions.len();

    let mut cursor = std::io::Cursor::new(&snapshot_data);
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic).unwrap();
    let mut version = [0u8; 1];
    cursor.read_exact(&mut version).unwrap();
    let mut pcount = [0u8; 1];
    cursor.read_exact(&mut pcount).unwrap();

    let mut parsed = Vec::new();
    for _ in 0..pcount[0] {
        let mut tag = [0u8; 1];
        cursor.read_exact(&mut tag).unwrap();
        let part = tag_to_partition(tag[0]).unwrap();
        let mut count_buf = [0u8; 4];
        cursor.read_exact(&mut count_buf).unwrap();
        let count = u32::from_be_bytes(count_buf) as usize;
        let mut entries = Vec::new();
        for _ in 0..count {
            let mut kl = [0u8; 4];
            cursor.read_exact(&mut kl).unwrap();
            let klen = u32::from_be_bytes(kl) as usize;
            let mut key = vec![0u8; klen];
            cursor.read_exact(&mut key).unwrap();
            let mut vl = [0u8; 4];
            cursor.read_exact(&mut vl).unwrap();
            let vlen = u32::from_be_bytes(vl) as usize;
            let mut val = vec![0u8; vlen];
            cursor.read_exact(&mut val).unwrap();
            entries.push((key, val));
        }
        parsed.push((part, entries));
    }

    let reconstructed = reconstruct_cnsn_payload(&parsed, partition_count);
    assert_eq!(reconstructed, payload_without_checksum);
}
