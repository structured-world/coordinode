//! Integration test harness for CoordiNode standalone binary tests.
//!
//! # Architecture
//!
//! Each test spawns a real `coordinode` binary against a temporary data
//! directory, waits for the gRPC port to become ready, runs assertions,
//! and kills the process. Restart tests do this cycle twice.
//!
//! Schema management goes through the v2 `SchemaServiceClient`, Cypher
//! queries through `CypherServiceClient`.
//!
//! # Why not embedded (coordinode-embed)?
//!
//! The embedded API uses Rust structs directly, with no proto round-trip for
//! schema creation. Standalone tests exercise the full production path,
//! including proto serialisation and the server's schema service.

pub mod harness;
pub mod proto;
