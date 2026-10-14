//! A cluster member's gRPC port is shared by clients and nodes. A client is
//! served the client services there and none of the inter-node ones: every
//! inter-node service refuses a caller whose connection proves no node
//! identity, whatever it sends.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use coordinode_integration::harness::{CoordinodeProcess, free_port};
use coordinode_integration::proto::internode::HandshakeRecord;
use coordinode_integration::proto::internode::version_handshake_client::VersionHandshakeClient;
use coordinode_integration::proto::query::ExecuteCypherRequest;
use coordinode_integration::proto::replication::raft_service_client::RaftServiceClient;
use coordinode_integration::proto::replication::segment_transfer_service_client::SegmentTransferServiceClient;
use coordinode_integration::proto::replication::{
    PieceData, PieceRequest, RaftPayload, SegmentManifestRequest,
};

/// The refusal a call without a node identity gets.
fn assert_refused<T: std::fmt::Debug>(service: &str, outcome: Result<T, tonic::Status>) {
    let status = outcome.expect_err(service);
    assert_eq!(
        status.code(),
        tonic::Code::Unauthenticated,
        "{service}: {status:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_client_is_served_its_services_and_refused_every_peer_service() {
    let member = CoordinodeProcess::start_cluster_member(1, free_port(), &[free_port()]).await;
    member.wait_for_leader(Duration::from_secs(20)).await;
    let channel = tonic::transport::Endpoint::from_shared(member.endpoint())
        .expect("endpoint")
        .connect()
        .await
        .expect("a client connects in plain text");

    let mut cypher = member.cypher_client().await;
    cypher
        .execute_cypher(ExecuteCypherRequest {
            query: "RETURN 1".to_string(),
            ..Default::default()
        })
        .await
        .expect("the client service answers");

    let mut raft = RaftServiceClient::new(channel.clone());
    assert_refused("Vote", raft.vote(RaftPayload::default()).await);
    assert_refused(
        "AppendEntries",
        raft.append_entries(RaftPayload::default()).await,
    );
    assert_refused(
        "StreamAppend",
        raft.stream_append(futures_stream(vec![RaftPayload::default()]))
            .await,
    );
    assert_refused(
        "Snapshot",
        raft.snapshot(futures_stream(vec![RaftPayload::default()]))
            .await,
    );
    assert_refused(
        "TransferLeader",
        raft.transfer_leader(RaftPayload::default()).await,
    );
    assert_refused("Members", raft.members(RaftPayload::default()).await);

    let mut segments = SegmentTransferServiceClient::new(channel.clone());
    assert_refused(
        "GetSegmentManifest",
        segments
            .get_segment_manifest(SegmentManifestRequest::default())
            .await,
    );
    assert_refused(
        "GetPiece",
        segments.get_piece(PieceRequest::default()).await,
    );
    assert_refused(
        "TransferPieces",
        segments
            .transfer_pieces(futures_stream(vec![PieceData::default()]))
            .await,
    );

    let mut handshake = VersionHandshakeClient::new(channel);
    assert_refused(
        "Exchange",
        handshake.exchange(HandshakeRecord::default()).await,
    );
}

/// `items` as a request stream.
fn futures_stream<T: Send + 'static>(items: Vec<T>) -> tokio_stream::Iter<std::vec::IntoIter<T>> {
    tokio_stream::iter(items)
}
