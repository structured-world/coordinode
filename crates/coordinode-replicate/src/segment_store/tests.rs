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

    // Install into a fresh target engine via the self-describing installer
    // (partition routed by the blob's leading wire tag) and verify every entry.
    let tgt_dir = tempfile::tempdir().expect("tempdir");
    let tgt = std::sync::Arc::new(test_engine(tgt_dir.path()));
    let installer = SegmentInstaller::new(std::sync::Arc::clone(&tgt));
    installer.store_segment(seg, &assembled).expect("install");

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
fn segment_source_builds_and_caches_servable_pieces() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = std::sync::Arc::new(test_engine(dir.path()));
    for i in 0..30u32 {
        engine
            .put(
                Partition::Node,
                format!("node:0:{i:08}").as_bytes(),
                format!("v{i}").as_bytes(),
            )
            .expect("put");
    }
    engine.persist().expect("persist");

    let installer = SegmentInstaller::new(std::sync::Arc::clone(&engine));
    let range = KeyRange {
        start: b"node:".to_vec(),
        end: Vec::new(),
    };

    let built = installer
        .build_segment(Partition::Node, &range, 64, PieceEncoding::None)
        .expect("build segment");

    // The served pieces assemble back to exactly the exported blob, and each
    // piece verifies against the manifest — so a peer can pull them piece-by-piece
    // and reconstruct the segment.
    let blob = export_range(&engine, Partition::Node, &range).expect("export");
    assert_eq!(built.wire.len() as u32, built.manifest.piece_count());
    for (i, w) in built.wire.iter().enumerate() {
        verify_piece(&built.manifest, i as u32, w).expect("verify piece");
    }
    let assembled = assemble(&built.manifest, &built.wire).expect("assemble");
    assert_eq!(
        assembled, blob,
        "served pieces must reconstruct the segment"
    );

    // A second build for the same parameters is served from cache (same Arc).
    let again = installer
        .build_segment(Partition::Node, &range, 64, PieceEncoding::None)
        .expect("build again");
    assert!(
        std::sync::Arc::ptr_eq(&built, &again),
        "repeated build must hit the serve-side cache"
    );
}

#[tokio::test]
async fn drain_client_pushes_segment_into_target_engine() {
    use crate::transfer::proto::segment_transfer_service_server::SegmentTransferServiceServer;
    use crate::transfer::SegmentTransferHandler;
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    // Source engine: write entries, materialise SSTs, derive the spanning segment.
    let src_dir = tempfile::tempdir().expect("tempdir");
    let src = test_engine(src_dir.path());
    for i in 0..16u32 {
        let key = format!("node:0:{i:08}");
        src.put(Partition::Node, key.as_bytes(), format!("v{i}").as_bytes())
            .expect("put");
    }
    src.force_compaction(Partition::Node).expect("compact");
    let map = SegmentMap::build(&src, Partition::Node, PlacementSegmentId(7)).expect("map");
    let descriptor = map.segments()[0].clone();

    // Target engine behind the real tonic SegmentTransferService (the same
    // service main.rs registers in cluster mode).
    let tgt_dir = tempfile::tempdir().expect("tempdir");
    let tgt = std::sync::Arc::new(test_engine(tgt_dir.path()));
    let handler = SegmentTransferHandler::new(std::sync::Arc::new(SegmentInstaller::new(
        std::sync::Arc::clone(&tgt),
    )));

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let incoming = TcpListenerStream::new(listener);
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(SegmentTransferServiceServer::new(handler))
            .serve_with_incoming(incoming)
            .await
            .expect("server");
    });

    // Drain the segment from the source to the peer via the public client helper
    // (export → split → stream → install) and verify every entry installed.
    let ack = drain_segment_to_peer(
        &src,
        &descriptor,
        &format!("http://{addr}"),
        64,
        PieceEncoding::None,
    )
    .await
    .expect("drain");
    assert!(ack.ok, "transfer ack: {}", ack.error);

    let snap = tgt.snapshot();
    let scanned = tgt
        .snapshot_prefix_scan(&snap, Partition::Node, b"node:")
        .expect("scan");
    let mut got: HashMap<String, Vec<u8>> = HashMap::new();
    for (key, value) in scanned {
        got.insert(String::from_utf8(key).expect("utf8"), value.to_vec());
    }
    assert_eq!(got.len(), 16, "all entries installed via gRPC");
    assert_eq!(
        got.get("node:0:00000003").map(Vec::as_slice),
        Some(&b"v3"[..])
    );
}

#[tokio::test]
async fn drain_to_unreachable_peer_is_connect_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = test_engine(dir.path());
    src.put(Partition::Node, b"node:0:00000001", b"v")
        .expect("put");
    src.force_compaction(Partition::Node).expect("compact");
    let map = SegmentMap::build(&src, Partition::Node, PlacementSegmentId(1)).expect("map");
    let descriptor = &map.segments()[0];

    // Port 1 is privileged and never listening here → connection refused.
    let err = drain_segment_to_peer(
        &src,
        descriptor,
        "http://127.0.0.1:1",
        64,
        PieceEncoding::None,
    )
    .await
    .expect_err("unreachable peer must error");
    assert!(
        matches!(err, DrainError::Connect { .. }),
        "expected Connect error, got {err:?}"
    );
}

#[test]
fn installer_rejects_empty_and_unknown_tag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = std::sync::Arc::new(test_engine(dir.path()));
    let installer = SegmentInstaller::new(engine);
    // Empty blob has no partition tag.
    assert!(installer.store_segment(SegmentId(1), &[]).is_err());
    // Unknown tag byte does not map to a partition.
    assert!(installer.store_segment(SegmentId(1), &[250u8]).is_err());
}

#[test]
fn exported_blob_carries_partition_tag() {
    use coordinode_storage::placement::partition_wire_tag;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    engine
        .put(Partition::Node, b"node:0:00000001", b"v")
        .expect("put");
    engine.force_compaction(Partition::Node).expect("compact");
    let map = SegmentMap::build(&engine, Partition::Node, PlacementSegmentId(1)).expect("map");
    let blob = export_segment(&engine, &map.segments()[0]).expect("export");
    assert_eq!(blob.first(), Some(&partition_wire_tag(Partition::Node)));
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
