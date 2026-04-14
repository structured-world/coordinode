pub mod blob;
pub mod cache;
pub mod compress;
pub mod engine;
pub mod error;
pub mod oplog;
pub mod scrub;
pub mod wal;

/// Re-export the `Guard` trait so downstream crates can call `into_inner()`
/// on `IterGuardImpl` without directly depending on `lsm_tree`.
pub use lsm_tree::Guard;

/// Re-export WAL config types for consumers that call `StorageEngine::open_with_wal`.
pub use wal::{WalConfig, WalSyncPolicy};
