//! Every migration this build ships, oldest first. A migration names the
//! shape it was written for in its own module, with the evidence that
//! identifies it.

mod raft_closed_bound;

use crate::Migration;

/// The migrations, in the order a run applies them.
pub static MIGRATIONS: &[Migration] = &[raft_closed_bound::MIGRATION];
