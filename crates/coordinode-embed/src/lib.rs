//! CoordiNode embedded library API.
//!
//! Use CoordiNode as an in-process library — no server, no network.
//!
//! ```rust,no_run
//! use coordinode_embed::Database;
//! let mut db = Database::open("/tmp/coordinode-data").unwrap();
//! let results = db.execute_cypher("MATCH (n:User) RETURN n.name").unwrap();
//! ```
//!
//! ## Features
//!
//! - `replication`: a database that is a member of a replicated group, in
//!   [`replication`]. Without it no consensus, inter-node channel, listener or
//!   async runtime is compiled; with it the host supplies a multi-threaded
//!   tokio runtime.
//! - `io-uring`: the io_uring storage backend on Linux.

pub mod api;
pub mod backup;
pub mod db;
pub mod operation;
pub mod repair;
#[cfg(feature = "replication")]
pub mod replication;
pub mod text_worker;
pub mod vector_worker;

pub use coordinode_core::txn::transaction::CommitReceipt;
pub use db::{
    AfterCommitDispatchReport, ConstraintDeclaration, DEFAULT_VECTOR_BUILD_WAIT, Database,
    DatabaseError, LabelConstraint, ResumedIndexBuilds, TriggerDispatchConfig,
};
pub use repair::RepairReport;
