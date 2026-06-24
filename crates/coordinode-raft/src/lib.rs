pub mod cluster;
pub mod proposal;
pub mod read_fence;
pub mod snapshot;
pub mod storage;
pub mod wait_majority;

/// Generated protobuf types for Raft inter-node gRPC protocol.
pub mod proto {
    pub mod replication {
        tonic::include_proto!("coordinode.v1.replication");
    }
}
