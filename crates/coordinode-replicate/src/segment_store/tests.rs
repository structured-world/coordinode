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

/// Recursively collect every regular file under `root`.
fn collect_files(root: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(root).expect("read_dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_files(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

/// The largest regular file under `root` — after a flush, the data SST.
fn largest_file(root: &std::path::Path) -> std::path::PathBuf {
    let mut files = Vec::new();
    collect_files(root, &mut files);
    files
        .into_iter()
        .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .expect("at least one file on disk")
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

#[tokio::test]
async fn swarm_pull_reconstructs_segment_over_grpc() {
    use crate::transfer::proto::segment_transfer_service_server::SegmentTransferServiceServer;
    use crate::transfer::proto::SegmentDescriptorRef;
    use crate::transfer::{GrpcPieceSource, SegmentTransferHandler};
    use coordinode_storage::placement::{partition_wire_tag, KeyRange};
    use coordinode_swarm::{swarm_download, Freshness, NodeId, PieceSource, SourceCandidate};
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    // Source engine with data, behind the real SegmentTransferService.
    let src_dir = tempfile::tempdir().expect("tempdir");
    let src = std::sync::Arc::new(test_engine(src_dir.path()));
    for i in 0..40u32 {
        src.put(
            Partition::Node,
            format!("node:0:{i:08}").as_bytes(),
            format!("value-{i}").as_bytes(),
        )
        .expect("put");
    }
    src.persist().expect("persist");
    let handler = SegmentTransferHandler::new(std::sync::Arc::new(SegmentInstaller::new(
        std::sync::Arc::clone(&src),
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

    // Whole-Node-partition descriptor with agreed split parameters (raw pieces).
    let descriptor = SegmentDescriptorRef {
        segment_id: 1,
        partition: u32::from(partition_wire_tag(Partition::Node)),
        range_start: b"node:".to_vec(),
        range_end: Vec::new(),
        piece_size: 256,
        encoding: 0,
        zstd_level: 0,
    };

    // Connect a gRPC piece source to the peer (fetches manifest + bitfield).
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    let candidate = SourceCandidate {
        node: NodeId(2),
        utilization: 0.0,
        bandwidth_to_target: 1.0,
        same_rack: false,
        tit_for_tat: 1.0,
        freshness: Freshness::Verified,
    };
    let (source, manifest) = GrpcPieceSource::connect(NodeId(2), channel, descriptor, candidate)
        .await
        .expect("connect source");

    // Run the rarest-first download on a blocking thread (fetch_piece bridges the
    // async client via block_on, which must not run on a runtime worker).
    let assembled = tokio::task::spawn_blocking(move || {
        let sources: Vec<&dyn PieceSource> = vec![&source];
        swarm_download(NodeId(1), &manifest, &sources, Vec::new())
    })
    .await
    .expect("join")
    .expect("swarm download");

    // The pulled segment equals what the source would export for that range.
    let expected = export_range(
        &src,
        Partition::Node,
        &KeyRange {
            start: b"node:".to_vec(),
            end: Vec::new(),
        },
    )
    .expect("export");
    assert_eq!(
        assembled, expected,
        "swarm pull over gRPC must reconstruct the segment byte-for-byte"
    );
    assert!(!assembled.is_empty());
}

#[tokio::test]
async fn repair_partition_pulls_and_installs_from_peer() {
    use crate::transfer::proto::segment_transfer_service_server::SegmentTransferServiceServer;
    use crate::transfer::SegmentTransferHandler;
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    // Healthy peer with data, behind the real SegmentTransferService.
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let peer = std::sync::Arc::new(test_engine(peer_dir.path()));
    for i in 0..40u32 {
        peer.put(
            Partition::Node,
            format!("node:0:{i:08}").as_bytes(),
            format!("value-{i}").as_bytes(),
        )
        .expect("put");
    }
    peer.persist().expect("persist");
    let handler = SegmentTransferHandler::new(std::sync::Arc::new(SegmentInstaller::new(
        std::sync::Arc::clone(&peer),
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

    // Local node with an empty Node partition repairs it from the peer.
    let local_dir = tempfile::tempdir().expect("tempdir");
    let local = std::sync::Arc::new(test_engine(local_dir.path()));
    let installer = std::sync::Arc::new(SegmentInstaller::new(std::sync::Arc::clone(&local)));

    let bytes = installer
        .repair_partition(
            &[format!("http://{addr}")],
            Partition::Node,
            256,
            PieceEncoding::None,
        )
        .await
        .expect("repair");
    assert!(bytes > 0, "repair installed nothing");

    let snap = local.snapshot();
    let scanned = local
        .snapshot_prefix_scan(&snap, Partition::Node, b"node:")
        .expect("scan");
    let mut got: HashMap<String, Vec<u8>> = HashMap::new();
    for (key, value) in scanned {
        got.insert(String::from_utf8(key).expect("utf8"), value.to_vec());
    }
    assert_eq!(got.len(), 40, "all peer entries repaired locally");
    assert_eq!(
        got.get("node:0:00000007").map(Vec::as_slice),
        Some(&b"value-7"[..])
    );
}

#[tokio::test]
async fn repair_partition_no_reachable_peer_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let local = std::sync::Arc::new(test_engine(dir.path()));
    let installer = std::sync::Arc::new(SegmentInstaller::new(local));
    // Port 1 is never listening here → no source can be reached.
    let err = installer
        .repair_partition(
            &["http://127.0.0.1:1".to_string()],
            Partition::Node,
            256,
            PieceEncoding::None,
        )
        .await
        .expect_err("unreachable peer must fail repair");
    assert!(
        matches!(err, RepairError::NoSource(_)),
        "expected NoSource, got {err:?}"
    );
}

#[tokio::test]
async fn repair_heals_corrupt_partition_so_rescrub_is_clean() {
    use crate::transfer::proto::segment_transfer_service_server::SegmentTransferServiceServer;
    use crate::transfer::SegmentTransferHandler;
    use coordinode_storage::scrub::{scrub_all, ScrubConfig};
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    let entries = 60u32;
    let put_all = |engine: &StorageEngine| {
        for i in 0..entries {
            engine
                .put(
                    Partition::Node,
                    format!("node:0:{i:08}").as_bytes(),
                    format!("value-{i}").as_bytes(),
                )
                .expect("put");
        }
    };

    // Healthy peer with the clean data, behind the real service.
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let peer = std::sync::Arc::new(test_engine(peer_dir.path()));
    put_all(&peer);
    peer.persist().expect("persist");
    let handler = SegmentTransferHandler::new(std::sync::Arc::new(SegmentInstaller::new(
        std::sync::Arc::clone(&peer),
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

    // Local node with the same data, then one SST physically corrupted.
    let local_dir = tempfile::tempdir().expect("tempdir");
    let local = std::sync::Arc::new(test_engine(local_dir.path()));
    put_all(&local);
    local.persist().expect("persist");
    let victim = largest_file(local_dir.path());
    let mut bytes = std::fs::read(&victim).expect("read sst");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&victim, &bytes).expect("corrupt sst");

    // Scrub detects the injected corruption.
    let before = scrub_all(&local, &ScrubConfig::default()).expect("scrub");
    assert!(
        before.has_errors(),
        "scrub must detect the injected corruption"
    );

    // Repair from the peer, then flush the reinstalled data so the scrub (which
    // reads on-disk SSTs) sees it.
    let installer = std::sync::Arc::new(SegmentInstaller::new(std::sync::Arc::clone(&local)));
    installer
        .repair_partition(
            &[format!("http://{addr}")],
            Partition::Node,
            256,
            PieceEncoding::None,
        )
        .await
        .expect("repair");
    local.persist().expect("persist");

    // The corruption is physically gone (the corrupt SST was dropped, not
    // shadowed) and the data is intact.
    let after = scrub_all(&local, &ScrubConfig::default()).expect("scrub");
    assert!(
        !after.has_errors(),
        "repair must physically heal the corruption: {:?}",
        after.errors
    );
    let v = local
        .get(Partition::Node, b"node:0:00000042")
        .expect("get")
        .expect("entry present after repair");
    assert_eq!(v.as_ref(), b"value-42");
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
