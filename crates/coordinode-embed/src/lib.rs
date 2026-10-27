//! CoordiNode embedded library API.
//!
//! Use CoordiNode as an in-process library without a separate database server.
//!
//! ## Installation
//!
//! ```toml
//! [dependencies]
//! coordinode-embed = "0.9"
//! # Query parameters and result values are coordinode_core types.
//! coordinode-core = "0.9"
//! ```
//!
//! The registry package includes generated protocol bindings, so consumers do
//! not need a proto submodule or protobuf generation tools. Rust 1.95 or newer.
//! `Database::open_with_config` takes a `coordinode_storage` configuration:
//! add `coordinode-storage = "0.9"` to set endpoints, limits or retention.
//!
//! ## Example
//!
//! ```rust,no_run
//! use coordinode_embed::{Database, DatabaseError};
//! # fn main() -> Result<(), DatabaseError> {
//! let mut db = Database::open("coordinode-data")?;
//! let tx = db.begin_transaction()?;
//! db.execute_in_transaction(tx, "CREATE (:User {name: 'ada'})", None)?;
//! db.commit_transaction(tx)?;
//! let results = db.execute_cypher("MATCH (n:User) RETURN n.name")?;
//! // Dropping the database closes it and releases the directory.
//! drop(db);
//! # Ok(())
//! # }
//! ```
//!
//! A complete application (parameters, commit, rollback, a typed refusal,
//! reopen) is the repository's `examples/embedded-quickstart`; the guide is
//! *Embedded (Rust)* in the documentation. [`Database`] covers opening,
//! closing and durability; [`DatabaseError`] the typed refusals.
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

/// Every Rust snippet of the embedded guide, compiled as a doctest against
/// this crate so the guide cannot advertise code that does not build. The
/// replicated snippets need the feature; the documentation suite runs with
/// all features.
#[cfg(all(doctest, feature = "replication"))]
#[doc = include_str!("../../../docs/guide/embedded.md")]
pub struct EmbeddedGuide;
