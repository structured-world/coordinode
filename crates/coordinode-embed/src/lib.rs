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
//! not need a proto submodule or protobuf generation tools. Builds on current stable Rust.
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
//! - `tenancy`, on by default: namespaces, principals and the per-namespace
//!   access control. Without it a database is one store of one namespace,
//!   with no grant catalog or authorization step compiled; a directory
//!   holding several namespaces is refused by name, and a principal is
//!   refused with [`DatabaseError::WithoutTenancy`].
//! - `io-uring`: the io_uring storage backend on Linux.

pub mod api;
pub mod backup;
mod catalog_watch;
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
    AccessChange, AccessRefusal, ActingContext, AfterCommitDispatchReport, ConstraintDeclaration,
    DEFAULT_VECTOR_BUILD_WAIT, Database, DatabaseError, Disposition, EdgeTypeDeclaration,
    GranteeName, Intent, LabelConstraint, LabelDeclaration, MAX_SCHEMA_PACKAGE_DECLARATIONS,
    MissingRight, NamespaceRole, PackageConstraint, PeerFailure, Principal, PrincipalId,
    PublishedObject, ReadFenceFailure, Readiness, ResumedIndexBuilds, SchemaPackage,
    SchemaPublication, SchemaReadiness, StorageFailure, TransactionEnd, TransactionId,
    TransactionOptions, TriggerDispatchConfig,
};
#[cfg(feature = "tenancy")]
pub use db::{DeploymentAdmin, GrantView, RoleView};
pub use repair::RepairReport;

/// What a change-stream consumer registers with
/// ([`Database::register_consumer`]) and what the registry reports of it.
pub mod consumer {
    pub use crate::db::{CutBatch, CutHistory, CutPage, CutResume, CutRows, CutScope, StateCut};
    pub use coordinode_replicate::{
        ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, ConsumerSnapshot,
        InitialSeqno, PriorIncarnation, RegisteredHandle, RegistrationState, RegistryError,
        RetentionPressure, RetentionSpace, RetentionSummary, TerminalReason, TopologyScope,
    };
    pub use coordinode_storage::engine::fixed_cut::{CutEvents, CutUnit};
}

/// Committed changes as logical events ([`Database::read_changes`]): the
/// cursor a reader keeps, the units and events it receives.
pub mod changes {
    pub use crate::db::{
        ChangeFollower, ChangeInterest, ChangePage, ProjectionProgress, ProjectionTransaction,
    };
    pub use coordinode_storage::engine::change_feed::{
        Action, ChangeBatch, ChangeCursor, ChangeEvent, CursorError, Delivery, EventId,
        OwnedTarget, UnitChanges,
    };
}

/// Every Rust snippet of the embedded guide, compiled as a doctest against
/// this crate so the guide cannot advertise code that does not build. The
/// replicated and the access-control snippets need their features; the
/// documentation suite runs with all features.
#[cfg(all(doctest, feature = "replication", feature = "tenancy"))]
#[doc = include_str!("../../../docs/guide/embedded.md")]
pub struct EmbeddedGuide;
