//! CoordiNode embedded library API.
//!
//! Use CoordiNode as an in-process library without a separate database server.
//!
//! ## Installation
//!
//! ```toml
//! [dependencies]
//! coordinode-embed = "0.8"
//! ```
//!
//! The registry package includes generated protocol bindings, so consumers do
//! not need a proto submodule or protobuf generation tools. Enable `replication`
//! when the host needs to participate in a replicated group (see below).
//!
//! ## Example
//!
//! ```rust,no_run
//! use coordinode_embed::{Database, DatabaseError};
//! # fn main() -> Result<(), DatabaseError> {
//! let mut db = Database::open("coordinode-data")?;
//! let results = db.execute_cypher("MATCH (n:User) RETURN n.name")?;
//! # Ok(())
//! # }
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
pub use coordinode_query::executor::runner::CatalogObject;
pub use db::{
    ActingContext, AfterCommitDispatchReport, ConstraintDeclaration, DEFAULT_VECTOR_BUILD_WAIT,
    Database, DatabaseError, Disposition, EdgeTypeDeclaration, Intent, LabelConstraint,
    LabelDeclaration, MAX_SCHEMA_PACKAGE_DECLARATIONS, PackageConstraint, PeerFailure, Principal,
    PublishedObject, ReadFenceFailure, Readiness, ResumedIndexBuilds, SchemaPackage,
    SchemaPublication, SchemaReadiness, StorageFailure, TransactionEnd, TransactionId,
    TransactionOptions, TriggerDispatchConfig,
};
pub use repair::RepairReport;
