use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::placement::{SegmentId as PlacementSegmentId, SegmentMap};
use coordinode_swarm::{
    LocalPieceStore, PieceEncoding, PieceStore, SegmentId, assemble, verify_piece,
};

use coordinode_test_fixtures::TestPki;
use coordinode_wire::peer::{Accept, InboundAuth, PeerAdmission, PeerConnectInfo, PeerTimeouts};

use super::*;
use crate::transfer::SegmentTransferHandler;
use crate::transfer::proto::segment_transfer_service_server::SegmentTransferServiceServer;

/// Node credentials of the transfer tests: node 1 pulls or drains, every
/// served peer gets a node number of its own from 2 on.
static PKI: LazyLock<TestPki> = LazyLock::new(TestPki::new);

/// The next node number a served peer takes.
static NEXT_PEER: AtomicU64 = AtomicU64::new(2);

/// Admits any caller connected under a node identity, as a group whose
/// every node is a member would; the membership checks of a real group are
/// the consensus layer's.
struct AnyNode;

impl PeerAdmission for AnyNode {
    fn admit_peer(
        &self,
        info: Option<&PeerConnectInfo>,
        _metadata: &tonic::metadata::MetadataMap,
    ) -> Result<u64, tonic::Status> {
        info.and_then(|info| info.node.as_ref())
            .ok_or_else(|| tonic::Status::unauthenticated("not a node"))?
            .identity()
            .map(|_| 0)
            .map_err(|e| tonic::Status::unauthenticated(e.to_string()))
    }

    fn stamp(&self, _metadata: &mut tonic::metadata::MetadataMap) {}
}

/// How node 1 reaches the peers it is served by.
fn links() -> PeerLinks {
    PeerLinks {
        connections: PKI.connections(1),
        admission: Arc::new(AnyNode),
    }
}

/// Serve `engine`'s segments as a new peer node, accepting nodes only;
/// returns the target that reaches it.
async fn serve(engine: &Arc<StorageEngine>) -> PeerTarget {
    let node = NEXT_PEER.fetch_add(1, Ordering::Relaxed);
    let accept = Accept {
        inbound: InboundAuth::NodeOnly,
        plaintext_clients: false,
        handshake_timeout: PeerTimeouts::default().handshake,
    };
    let addr = serve_with(engine, &PKI.context(node), accept).await;
    PKI.target(node, addr)
}

/// Serve `engine`'s segments under `context` as `accept` says; returns the
/// address it listens on.
async fn serve_with(
    engine: &Arc<StorageEngine>,
    context: &Arc<coordinode_wire::peer::SecurityContext>,
    accept: Accept,
) -> String {
    let handler = SegmentTransferHandler::new(
        Arc::new(SegmentInstaller::new(Arc::clone(engine))),
        Arc::new(AnyNode),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    let incoming = coordinode_wire::peer::incoming(listener, context, accept).expect("incoming");
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(SegmentTransferServiceServer::new(handler))
            .serve_with_incoming(incoming)
            .await
            .expect("server");
    });
    addr
}

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
    let subject = crate::transfer::SegmentSubject::Range {
        partition: Partition::Node,
        range: range.clone(),
    };

    let built = installer
        .build_segment(&subject, 64, PieceEncoding::None, false)
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
        .build_segment(&subject, 64, PieceEncoding::None, false)
        .expect("build again");
    assert!(
        std::sync::Arc::ptr_eq(&built, &again),
        "repeated build must hit the serve-side cache"
    );

    // A manifest request builds fresh: a new pull sees the store as it
    // stands, not a build an earlier pull cached.
    let fresh = installer
        .build_segment(&subject, 64, PieceEncoding::None, true)
        .expect("build fresh");
    assert!(!std::sync::Arc::ptr_eq(&built, &fresh));
}

#[tokio::test]
async fn drain_client_pushes_segment_into_target_engine() {
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
    let target = serve(&tgt).await;

    // Drain the segment from the source to the peer via the public client helper
    // (export → split → stream → install) and verify every entry installed.
    let ack = drain_segment_to_peer(
        &src,
        &descriptor,
        &links(),
        &target,
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
        &links(),
        &PKI.target(2, "127.0.0.1:1"),
        64,
        PieceEncoding::None,
    )
    .await
    .expect_err("unreachable peer must error");
    assert!(
        matches!(&err, DrainError::Rpc(status) if status.code() == tonic::Code::Unavailable),
        "expected an unavailable peer, got {err:?}"
    );

    // An address without a port is refused before anything is dialed.
    let err = drain_segment_to_peer(
        &src,
        descriptor,
        &links(),
        &PKI.target(2, "no-port"),
        64,
        PieceEncoding::None,
    )
    .await
    .expect_err("an address without a port");
    assert!(
        matches!(err, DrainError::Connect { .. }),
        "expected Connect error, got {err:?}"
    );
}

/// The transfer service serves members of the group alone: a client with
/// no node identity, here in plain text on a port that takes clients, is
/// refused before anything is built or installed.
#[tokio::test]
async fn a_caller_without_a_node_identity_is_refused() {
    use crate::transfer::proto::segment_transfer_service_client::SegmentTransferServiceClient;
    use crate::transfer::proto::{SegmentDescriptorRef, SegmentManifestRequest};

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = Arc::new(test_engine(dir.path()));
    engine
        .put(Partition::Node, b"node:0:00000001", b"v")
        .expect("put");
    let accept = Accept {
        inbound: InboundAuth::Shared { clients: None },
        plaintext_clients: true,
        handshake_timeout: PeerTimeouts::default().handshake,
    };
    let addr = serve_with(&engine, &PKI.context(9), accept).await;
    let mut client = SegmentTransferServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("a client connects");
    let status = client
        .get_segment_manifest(SegmentManifestRequest {
            segment: Some(SegmentDescriptorRef {
                segment_id: 1,
                partition: u32::from(coordinode_storage::placement::partition_wire_tag(
                    Partition::Node,
                )),
                range_start: Vec::new(),
                range_end: Vec::new(),
                piece_size: 256,
                encoding: 0,
                zstd_level: 0,
                generation: None,
            }),
        })
        .await
        .expect_err("a client is no member");
    assert_eq!(status.code(), tonic::Code::Unauthenticated, "{status:?}");
}

#[tokio::test]
async fn swarm_pull_reconstructs_segment_over_grpc() {
    use crate::transfer::GrpcPieceSource;
    use crate::transfer::proto::SegmentDescriptorRef;
    use coordinode_storage::placement::{KeyRange, partition_wire_tag};
    use coordinode_swarm::{Freshness, NodeId, PieceSource, SourceCandidate, swarm_download};

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
    let target = serve(&src).await;

    // Whole-Node-partition descriptor with agreed split parameters (raw pieces).
    let descriptor = SegmentDescriptorRef {
        segment_id: 1,
        partition: u32::from(partition_wire_tag(Partition::Node)),
        range_start: b"node:".to_vec(),
        range_end: Vec::new(),
        piece_size: 256,
        encoding: 0,
        zstd_level: 0,
        generation: None,
    };

    // Connect a gRPC piece source to the peer (fetches manifest + bitfield).
    let candidate = SourceCandidate {
        node: NodeId(2),
        utilization: 0.0,
        bandwidth_to_target: 1.0,
        same_rack: false,
        tit_for_tat: 1.0,
        freshness: Freshness::Verified,
    };
    let (source, manifest) =
        GrpcPieceSource::connect(NodeId(2), &links(), &target, descriptor, candidate)
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
    let target = serve(&peer).await;

    // Local node with an empty Node partition repairs it from the peer.
    let local_dir = tempfile::tempdir().expect("tempdir");
    let local = std::sync::Arc::new(test_engine(local_dir.path()));
    let installer = std::sync::Arc::new(SegmentInstaller::new(std::sync::Arc::clone(&local)));

    let bytes = installer
        .repair_partition(
            &links(),
            &[target],
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
            &links(),
            &[PKI.target(2, "127.0.0.1:1")],
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
    use coordinode_storage::scrub::{ScrubConfig, scrub_all};

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
    let target = serve(&peer).await;

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
            &links(),
            &[target],
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

/// Stands in for the Raft state machine's fence: applies standing at a
/// chosen position, with the payload a coverage base would carry.
struct TestFence {
    state: parking_lot::Mutex<coordinode_storage::engine::core::RaftApplyState>,
    payload: Vec<u8>,
}

impl coordinode_storage::engine::core::RaftApplyFence for TestFence {
    fn with_applies_paused(
        &self,
        work: &mut dyn FnMut(
            &mut coordinode_storage::engine::core::RaftApplyState,
            &[u8],
        ) -> StorageResult<()>,
    ) -> StorageResult<()> {
        work(&mut self.state.lock(), &self.payload)
    }
}

/// An engine whose Raft applies stand at `next`.
fn raft_engine(dir: &std::path::Path, next: u64) -> (Arc<StorageEngine>, Arc<TestFence>) {
    let engine = Arc::new(test_engine(dir));
    let fence = Arc::new(TestFence {
        state: parking_lot::Mutex::new(coordinode_storage::engine::core::RaftApplyState::new(next)),
        payload: format!("id-{}", next.saturating_sub(1)).into_bytes(),
    });
    engine.register_raft_fence(Arc::clone(&fence) as _);
    (engine, fence)
}

/// A copy from a peer further along replaces the partition at the peer's
/// position: the tree records it, and the local applies leave the partition
/// alone until they pass it, so the entries in between are not applied twice.
#[tokio::test(flavor = "multi_thread")]
async fn a_copy_ahead_installs_at_its_position_and_raises_the_floor() {
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let (peer, _) = raft_engine(peer_dir.path(), 20);
    peer.put(Partition::Node, b"node:0:00000001", b"peer")
        .expect("put");
    let uri = serve(&peer).await;

    let local_dir = tempfile::tempdir().expect("tempdir");
    let (local, fence) = raft_engine(local_dir.path(), 10);
    local
        .put(Partition::Node, b"node:0:00000002", b"stale")
        .expect("put");
    let installer = Arc::new(SegmentInstaller::new(Arc::clone(&local)));
    installer
        .repair_partition(&links(), &[uri], Partition::Node, 256, PieceEncoding::None)
        .await
        .expect("repair");

    assert_eq!(
        local
            .get(Partition::Node, b"node:0:00000001")
            .expect("get")
            .as_deref(),
        Some(b"peer".as_slice())
    );
    assert!(
        local
            .get(Partition::Node, b"node:0:00000002")
            .expect("get")
            .is_none(),
        "the partition is replaced, not written over"
    );
    let coverage = local.raft_coverage().expect("coverage");
    assert!(
        coverage.holds(Partition::Node, 19, 0),
        "the peer's position"
    );
    assert!(!coverage.holds(Partition::Node, 20, 0));
    assert_eq!(fence.state.lock().floor(Partition::Node), 20);
    assert!(fence.state.lock().skips(Partition::Node, 15));
}

/// A copy behind this node's applies lacks entries this node applied and
/// will not apply again: it is refused, and the partition left as it was.
#[tokio::test(flavor = "multi_thread")]
async fn a_copy_behind_the_local_applies_is_refused() {
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let (peer, _) = raft_engine(peer_dir.path(), 5);
    peer.put(Partition::Node, b"node:0:00000001", b"peer")
        .expect("put");
    let uri = serve(&peer).await;

    let local_dir = tempfile::tempdir().expect("tempdir");
    let (local, _) = raft_engine(local_dir.path(), 10);
    local
        .put(Partition::Node, b"node:0:00000002", b"local")
        .expect("put");
    let installer = Arc::new(SegmentInstaller::new(Arc::clone(&local)));
    let err = installer
        .repair_partition(&links(), &[uri], Partition::Node, 256, PieceEncoding::None)
        .await
        .expect_err("a copy behind is refused");
    assert!(matches!(err, RepairError::Behind(_)), "{err:?}");
    assert_eq!(
        local
            .get(Partition::Node, b"node:0:00000002")
            .expect("get")
            .as_deref(),
        Some(b"local".as_slice())
    );
}

/// Two peers at one position whose partitions differ build segments with
/// different bytes: the installed partition is exactly one peer's copy,
/// never a splice of both.
#[tokio::test(flavor = "multi_thread")]
async fn a_repair_from_peers_with_different_copies_installs_one_of_them_whole() {
    let a_dir = tempfile::tempdir().expect("tempdir");
    let (a, _) = raft_engine(a_dir.path(), 20);
    let b_dir = tempfile::tempdir().expect("tempdir");
    let (b, _) = raft_engine(b_dir.path(), 20);
    for i in 0..200u32 {
        let key = format!("node:0:{i:08}");
        a.put(
            Partition::Node,
            key.as_bytes(),
            format!("a-{i:04}").as_bytes(),
        )
        .expect("put");
        b.put(
            Partition::Node,
            key.as_bytes(),
            format!("b-{i:04}").as_bytes(),
        )
        .expect("put");
    }
    let uri_a = serve(&a).await;
    let uri_b = serve(&b).await;

    let local_dir = tempfile::tempdir().expect("tempdir");
    let (local, _) = raft_engine(local_dir.path(), 10);
    let installer = Arc::new(SegmentInstaller::new(Arc::clone(&local)));
    installer
        .repair_partition(
            &links(),
            &[uri_a, uri_b],
            Partition::Node,
            256,
            PieceEncoding::None,
        )
        .await
        .expect("repair");

    let first = local
        .get(Partition::Node, b"node:0:00000000")
        .expect("get")
        .expect("present");
    let prefix = if first.as_ref().starts_with(b"a-") {
        "a"
    } else {
        "b"
    };
    for i in 0..200u32 {
        let key = format!("node:0:{i:08}");
        let v = local
            .get(Partition::Node, key.as_bytes())
            .expect("get")
            .expect("present");
        assert_eq!(
            v.as_ref(),
            format!("{prefix}-{i:04}").as_bytes(),
            "key {i} comes from the other peer"
        );
    }
}

/// Schema carries this node's own records (per-node routing, the state
/// machine's membership and snapshot metadata): they neither travel nor
/// get replaced by a peer's copy.
#[tokio::test(flavor = "multi_thread")]
async fn a_schema_copy_keeps_the_node_local_records() {
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let (peer, _) = raft_engine(peer_dir.path(), 20);
    peer.put(Partition::Schema, b"schema:label:A", b"peer")
        .expect("put");
    peer.put(Partition::Schema, b"raft:sm:membership", b"peer-membership")
        .expect("put");
    let uri = serve(&peer).await;

    let local_dir = tempfile::tempdir().expect("tempdir");
    let (local, _) = raft_engine(local_dir.path(), 10);
    local
        .put(
            Partition::Schema,
            b"raft:sm:membership",
            b"local-membership",
        )
        .expect("put");
    let installer = Arc::new(SegmentInstaller::new(Arc::clone(&local)));
    installer
        .repair_partition(
            &links(),
            &[uri],
            Partition::Schema,
            256,
            PieceEncoding::None,
        )
        .await
        .expect("repair");

    assert_eq!(
        local
            .get(Partition::Schema, b"schema:label:A")
            .expect("get")
            .as_deref(),
        Some(b"peer".as_slice())
    );
    assert_eq!(
        local
            .get(Partition::Schema, b"raft:sm:membership")
            .expect("get")
            .as_deref(),
        Some(b"local-membership".as_slice())
    );
}

/// The Raft partition is this node's vote and log metadata; another node's
/// copy of it would make this node vote and log as the other.
#[tokio::test(flavor = "multi_thread")]
async fn the_raft_partition_is_never_copied() {
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let (peer, _) = raft_engine(peer_dir.path(), 20);
    let uri = serve(&peer).await;

    let local_dir = tempfile::tempdir().expect("tempdir");
    let (local, _) = raft_engine(local_dir.path(), 10);
    let installer = Arc::new(SegmentInstaller::new(Arc::clone(&local)));
    installer
        .repair_partition(&links(), &[uri], Partition::Raft, 256, PieceEncoding::None)
        .await
        .expect_err("the peer refuses to export it");
}

/// A Raft store records one log position per partition; a key range from a
/// peer would leave the rest of the partition at another one.
#[test]
fn a_key_range_does_not_install_into_a_raft_store() {
    let src_dir = tempfile::tempdir().expect("tempdir");
    let (src, _) = raft_engine(src_dir.path(), 20);
    src.put(Partition::Node, b"node:0:00000001", b"v")
        .expect("put");
    let blob = export_range(
        &src,
        Partition::Node,
        &KeyRange {
            start: b"node:".to_vec(),
            end: Vec::new(),
        },
    )
    .expect("export");

    let tgt_dir = tempfile::tempdir().expect("tempdir");
    let (tgt, _) = raft_engine(tgt_dir.path(), 10);
    let installer = SegmentInstaller::new(Arc::clone(&tgt));
    installer
        .store_segment(SegmentId(1), &blob)
        .expect_err("a key range is refused");
    assert!(
        tgt.get(Partition::Node, b"node:0:00000001")
            .expect("get")
            .is_none()
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

/// The logical key `tag 0x01 / generation / rest` of an index entry.
fn index_entry(generation: u64, rest: &[u8]) -> Vec<u8> {
    let mut key = vec![coordinode_core::index::encoding::GENERATION_TAGS[0]];
    key.extend_from_slice(&generation.to_be_bytes());
    key.extend_from_slice(rest);
    key
}

/// A member whose copy of an index generation was damaged locally (an entry
/// written outside the log) replaces it with one filled from a peer's history
/// over the transfer service: the damage is gone, every logged write reads
/// back, a read at an earlier snapshot answers from the imported history, and
/// a write made while the replacement was prepared is in it.
#[tokio::test(flavor = "multi_thread")]
async fn a_generation_replaced_from_a_peer_drops_local_damage_and_keeps_history() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_core::txn::timestamp::TimestampOracle;

    const GEN: u64 = 40;
    let config = |dir: &std::path::Path| {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
    };
    // Two replicas applying the same log at the same commit timestamps.
    let oracle = std::sync::Arc::new(TimestampOracle::new());
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let local_dir = tempfile::tempdir().expect("tempdir");
    let peer = std::sync::Arc::new(
        StorageEngine::open_embedded(&config(peer_dir.path()), oracle.clone()).expect("open"),
    );
    let local = std::sync::Arc::new(
        StorageEngine::open_embedded(&config(local_dir.path()), oracle.clone()).expect("open"),
    );
    let commit = |key: &[u8], value: &[u8]| {
        let ts = oracle.next().as_raw();
        for engine in [&peer, &local] {
            let mutation = Mutation::Put {
                partition: PartitionId::Idx,
                key: key.to_vec(),
                value: value.to_vec(),
            };
            engine.commit_journaled(&[mutation], ts).expect("commit");
        }
        ts
    };
    let a = index_entry(GEN, b"a");
    let b = index_entry(GEN, b"b");
    let c = index_entry(GEN, b"c");
    let wrong = index_entry(GEN, b"wrong");
    let first = commit(&a, b"a1");
    commit(&a, b"a2");
    commit(&b, b"b1");
    local.put(Partition::Idx, &wrong, b"x").expect("damage");
    let target = serve(&peer).await;

    let installer = std::sync::Arc::new(SegmentInstaller::new(std::sync::Arc::clone(&local)));
    let generation = GenerationId::from_raw(GEN);
    let (bytes, proof) = installer
        .prepare_generation(&links(), &[target], generation, 128, PieceEncoding::None)
        .await
        .expect("prepare");
    assert!(bytes > 0, "nothing imported");
    assert_eq!(proof, None, "a peer that proved nothing carries no proof");
    // Logged after the replacement was registered: it reaches it too.
    commit(&c, b"c1");
    local.publish_generation(generation).expect("publish");

    let get = |key: &[u8]| {
        local
            .get(Partition::Idx, key)
            .expect("get")
            .map(|v| v.to_vec())
    };
    assert_eq!(get(&wrong), None, "the local damage is gone");
    assert_eq!(get(&a), Some(b"a2".to_vec()));
    assert_eq!(get(&b), Some(b"b1".to_vec()));
    assert_eq!(get(&c), Some(b"c1".to_vec()));
    let at_first = first + 1;
    assert_eq!(
        local
            .snapshot_get(&at_first, Partition::Idx, &a)
            .expect("historical read")
            .map(|v| v.to_vec()),
        Some(b"a1".to_vec()),
        "an earlier snapshot reads the imported history"
    );
}

/// A peer that holds no copy of the generation serves nothing, and the
/// replacement registered for the pull is dropped: the published copy stays
/// and a later replacement can be registered.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_generation_pull_drops_the_replacement() {
    const GEN: u64 = 40;
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let peer = std::sync::Arc::new(test_engine(peer_dir.path()));
    let target = serve(&peer).await;

    let local_dir = tempfile::tempdir().expect("tempdir");
    let local = std::sync::Arc::new(test_engine(local_dir.path()));
    let a = index_entry(GEN, b"a");
    local.put(Partition::Idx, &a, b"a1").expect("put");
    let installer = std::sync::Arc::new(SegmentInstaller::new(std::sync::Arc::clone(&local)));
    let generation = GenerationId::from_raw(GEN);
    let err = installer
        .prepare_generation(&links(), &[target], generation, 128, PieceEncoding::None)
        .await
        .expect_err("the peer holds no copy");
    assert!(matches!(err, RepairError::NoSource(_)), "got {err}");
    assert_eq!(
        local
            .get(Partition::Idx, &a)
            .expect("get")
            .map(|v| v.to_vec()),
        Some(b"a1".to_vec())
    );
    local
        .stage_generation(generation)
        .expect("a new replacement can be registered");
}

/// Stands in for a peer's index layer: proves its copy while `proved` holds.
struct PeerProof {
    proved: std::sync::atomic::AtomicBool,
}

impl coordinode_storage::engine::index_copies::IndexCopyQualifier for PeerProof {
    fn capture(&self, _: &StorageEngine) -> coordinode_storage::error::StorageResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn seal(
        &self,
        _: &StorageEngine,
        captured: &[u8],
    ) -> coordinode_storage::error::StorageResult<Vec<u8>> {
        Ok(captured.to_vec())
    }

    fn installed(
        &self,
        _: &StorageEngine,
        _: Option<&[u8]>,
    ) -> coordinode_storage::error::StorageResult<()> {
        Ok(())
    }

    fn copy_proof(
        &self,
        _: &StorageEngine,
        _: u64,
    ) -> coordinode_storage::error::StorageResult<Option<(Vec<u8>, u64)>> {
        Ok(self
            .proved
            .load(std::sync::atomic::Ordering::Relaxed)
            .then(|| (b"proof".to_vec(), 0)))
    }
}

/// A peer that proved its copy sends the proof with the history, so the
/// receiver need not check the copy it fills; once the peer's proof is gone
/// (a finding against its copy), a later pull carries none, even though the
/// segment it built with the proof is still cached.
#[tokio::test(flavor = "multi_thread")]
async fn a_generation_pull_carries_the_peers_standing_proof() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_core::txn::timestamp::TimestampOracle;

    const GEN: u64 = 40;
    let config = |dir: &std::path::Path| {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
    };
    let oracle = std::sync::Arc::new(TimestampOracle::new());
    let peer_dir = tempfile::tempdir().expect("tempdir");
    let local_dir = tempfile::tempdir().expect("tempdir");
    let peer = std::sync::Arc::new(
        StorageEngine::open_embedded(&config(peer_dir.path()), oracle.clone()).expect("open"),
    );
    let local = std::sync::Arc::new(
        StorageEngine::open_embedded(&config(local_dir.path()), oracle.clone()).expect("open"),
    );
    let ts = oracle.next().as_raw();
    for engine in [&peer, &local] {
        let mutation = Mutation::Put {
            partition: PartitionId::Idx,
            key: index_entry(GEN, b"a"),
            value: b"a1".to_vec(),
        };
        engine.commit_journaled(&[mutation], ts).expect("commit");
    }
    let layer = std::sync::Arc::new(PeerProof {
        proved: std::sync::atomic::AtomicBool::new(true),
    });
    peer.set_index_copy_qualifier(std::sync::Arc::clone(&layer) as _);
    let target = serve(&peer).await;

    let installer = std::sync::Arc::new(SegmentInstaller::new(std::sync::Arc::clone(&local)));
    let generation = GenerationId::from_raw(GEN);
    let peers = [target];
    let links = links();
    let pull =
        || installer.prepare_generation(&links, &peers, generation, 128, PieceEncoding::None);
    let (_, proof) = pull().await.expect("prepare");
    assert_eq!(proof.as_deref(), Some(&b"proof"[..]), "the peer's proof");
    local.publish_generation(generation).expect("publish");

    layer
        .proved
        .store(false, std::sync::atomic::Ordering::Relaxed);
    let (_, proof) = pull().await.expect("prepare again");
    assert_eq!(proof, None, "a proof gone since the segment was built");
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
