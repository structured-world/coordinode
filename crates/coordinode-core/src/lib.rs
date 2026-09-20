// The crate's target tier is `alloc`: its public surface is meant to compile
// without `std` once the remaining std-bound internals are migrated. Linking
// `alloc` explicitly lets new code name `alloc::` today, so it does not have
// to be rewritten when the `no_std` attribute goes on.
extern crate alloc;

pub mod graph;
pub mod index;
pub mod operations;
pub mod schema;
pub mod txn;
