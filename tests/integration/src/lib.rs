//! Integration test harness for CoordiNode standalone binary tests.
//!
//! # Architecture
//!
//! Each test spawns a real `coordinode` binary against a temporary data
//! directory, waits for the gRPC port to become ready, runs assertions,
//! and kills the process. Restart tests do this cycle twice.
//!
//! Schema management goes through `SchemaServiceClient` (generated from
//! schema.proto), Cypher queries through `CypherServiceClient`.
//!
//! # Why not embedded (coordinode-embed)?
//!
//! The embedded API uses Rust structs directly (no proto round-trip for
//! schema creation). The reported bugs manifest specifically via the gRPC
//! proto path where `proto_type_to_property_type(PROPERTY_TYPE_VECTOR=7)`
//! hardcodes `dimensions: 0`. Standalone tests exercise the full production
//! code path including proto serialisation and the server's schema service.

pub mod harness;
pub mod proto;
