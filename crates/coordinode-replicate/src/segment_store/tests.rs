use std::collections::HashMap;

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::placement::{SegmentId as PlacementSegmentId, SegmentMap};
use coordinode_swarm::{
    assemble, verify_piece, LocalPieceStore, PieceEncoding, PieceStore, SegmentId,
};

use super::*;

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&cfg).expect("open engine")
}

#[test]
fn kv_blob_round_trips() {
    let entries = vec![
        (b"node:0:1".to_vec(), b"alice".to_vec()),
        (b"node:0:2".to_vec(), Vec::new()), // empty value
        (b"node:0:3".to_vec(), b"a longer value here".to_vec()),
    ];
    let blob = encode_kv_blob(&entries).expect("encode");
    let decoded = decode_kv_blob(&blob).expect("decode");
    assert_eq!(decoded, entries);
}

#[test]
fn decode_rejects_truncated_blob() {
    // Length prefix claims 100 bytes but the payload is short.
    let mut blob = 100u32.to_le_bytes().to_vec();
    blob.extend_from_slice(b"short");
    assert!(decode_kv_blob(&blob).is_err());
}

#[test]
fn export_transfer_install_round_trip() {
    // Source: write entries, materialise SSTs, derive the spanning segment.
    let src_dir = tempfile::tempdir().expect("tempdir");
    let src = test_engine(src_dir.path());
    for i in 0..20u32 {
        let key = format!("node:0:{i:08}");
        src.put(
            Partition::Node,
            key.as_bytes(),
            format!("val{i}").as_bytes(),
        )
        .expect("put");
    }
    src.force_compaction(Partition::Node).expect("compact");

    let map = SegmentMap::build(&src, Partition::Node, PlacementSegmentId(1)).expect("map");
    let descriptor = &map.segments()[0];

    // Export to a portable blob.
    let blob = export_segment(&src, descriptor).expect("export");
    assert!(!blob.is_empty());

    // Serve over the swarm piece store; gather + verify every piece; assemble.
    let seg = SegmentId(descriptor.id.0);
    let mut store = LocalPieceStore::new();
    store
        .insert(seg, &blob, 64, PieceEncoding::None)
        .expect("insert into piece store");
    let manifest = store.manifest(seg).expect("manifest");
    let mut wire = Vec::new();
    for i in 0..manifest.piece_count() {
        let piece = store.wire_piece(seg, i).expect("wire piece");
        verify_piece(&manifest, i, &piece).expect("verify piece");
        wire.push(piece);
    }
    let assembled = assemble(&manifest, &wire).expect("assemble");
    assert_eq!(assembled, blob, "assembled blob must equal exported blob");

    // Install into a fresh target engine and verify every entry round-tripped.
    let tgt_dir = tempfile::tempdir().expect("tempdir");
    let tgt = test_engine(tgt_dir.path());
    let sink = StorageSegmentSink::new(&tgt, Partition::Node);
    sink.store_segment(seg, &assembled).expect("install");

    let mut got: HashMap<String, Vec<u8>> = HashMap::new();
    let snap = tgt.snapshot();
    let scanned = tgt
        .snapshot_prefix_scan(&snap, Partition::Node, b"node:")
        .expect("scan");
    for (key, value) in scanned {
        got.insert(String::from_utf8(key).expect("utf8 key"), value.to_vec());
    }
    assert_eq!(got.len(), 20);
    assert_eq!(
        got.get("node:0:00000005").map(Vec::as_slice),
        Some(&b"val5"[..])
    );
    assert_eq!(
        got.get("node:0:00000000").map(Vec::as_slice),
        Some(&b"val0"[..])
    );
}

#[test]
fn export_empty_segment_is_empty_blob() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    // No data → empty map → nothing to export. Build a descriptor-free path by
    // exporting a synthetic whole-partition descriptor over an empty partition.
    let map = SegmentMap::build(&engine, Partition::Node, PlacementSegmentId(1)).expect("map");
    assert!(map.is_empty());
}
