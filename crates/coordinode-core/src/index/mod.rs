//! Index-related encoding shared across layers.
//!
//! Sortable binary encoding of [`crate::graph::types::Value`] for use
//! in B-tree-style secondary indexes (`Partition::Idx`). Lives in
//! `coordinode-core` so that both Layer 4 (`coordinode-stores`) and
//! Layer 5 (`coordinode-query`) can produce/consume the same key
//! format without a circular dependency.

pub mod encoding;
