//! CoordiNode embedded library API.
//!
//! Use CoordiNode as an in-process library — no server, no network.
//!
//! ```rust,no_run
//! use coordinode_embed::Database;
//! let mut db = Database::open("/tmp/coordinode-data").unwrap();
//! let results = db.execute_cypher("MATCH (n:User) RETURN n.name").unwrap();
//! ```

pub mod api;
pub mod backup;
pub mod db;
pub mod vector_worker;

pub use db::{Database, DatabaseError};
