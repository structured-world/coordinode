mod authority;
pub mod closure;
pub mod cluster;
pub mod engine_hold;
mod log_sync;
pub mod member;
pub mod proposal;
pub mod read_fence;
pub mod snapshot;
pub mod storage;
pub mod wait_majority;

/// The unit tests' allocator: the system one, with allocations attributed to
/// groups so a test counts what a piece of work held at once
/// (`coordinode_test_fixtures::alloc_peak`).
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

/// Generated protobuf types for Raft inter-node gRPC protocol.
pub mod proto {
    pub mod replication {
        tonic::include_proto!("coordinode.v1.replication");
    }
    /// The frozen version handshake between members.
    pub mod internode {
        tonic::include_proto!("coordinode.internode");

        /// Whole commands routed from the member a client reached to the
        /// member that executes them.
        pub mod routing {
            tonic::include_proto!("coordinode.internode.routing");
        }
    }
}
