//! Language-neutral query IR kernel.
//!
//! The logical layer (planner optimizer passes, executor, advisor) operates on
//! this IR, which carries no dependency on any query dialect. A language
//! frontend (Cypher today, SQL later) parses its own surface syntax and lowers
//! it into this neutral IR; the same planner and executor then consume it,
//! whichever dialect produced it. Translating one dialect into another's AST is
//! never the path: both lower independently into the kernel here.
//!
//! This module is introduced incrementally. It first defines the neutral
//! expression surface; subsequent work migrates the operator tree and the
//! executor onto it and removes the dialect coupling from the layers below.

pub mod expr;

pub use expr::{BinOp, Expr, MapProjItem, Quantifier, StrOp, UnOp};
