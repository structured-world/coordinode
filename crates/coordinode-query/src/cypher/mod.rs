//! OpenCypher parser and AST.
//!
//! Parses OpenCypher read and write operations into a typed AST.
//!
//! Read: MATCH, WHERE, RETURN, WITH, UNWIND, ORDER BY, SKIP, LIMIT, OPTIONAL MATCH.
//! Write: CREATE, MERGE, DELETE, DETACH DELETE, SET, REMOVE.
//! Extensions: UPSERT MATCH, vector functions, AS OF TIMESTAMP, parameters ($name).

pub mod ast;
pub mod errors;
pub mod parser;
pub mod semantic;

pub use ast::*;
pub use errors::ParseError;
pub use parser::parse;
pub use semantic::{MapSchemaProvider, SchemaProvider, SemanticError, analyze};
