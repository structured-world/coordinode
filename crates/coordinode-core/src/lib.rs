// The crate's target tier is `alloc`: its public surface is meant to compile
// without `std` once the remaining std-bound internals are migrated. Linking
// `alloc` explicitly lets new code name `alloc::` today, so it does not have
// to be rewritten when the `no_std` attribute goes on.
extern crate alloc;

/// The `google.rpc.ErrorInfo.domain` of every failure CoordiNode reports
/// over gRPC, to clients and between servers alike. Reasons are unique only
/// within a domain.
pub const ERROR_DOMAIN: &str = "coordinode.sw.foundation";

pub mod graph;
pub mod group;
pub mod index;
pub mod operations;
pub mod schema;
pub mod txn;
pub mod version;
