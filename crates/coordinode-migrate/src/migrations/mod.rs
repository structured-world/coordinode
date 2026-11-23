//! The migrations this build ships, one module per release they lead from:
//! a module brings a data directory that release wrote to the shape of the
//! release after it. A run applies the modules in order, so a directory
//! several releases behind goes through every step between, and each step
//! is a unit of its own for a rollout that moves a cluster one release at a
//! time. A build ships the step from the release before it only.

mod from_0_10_1;

use crate::Migration;

/// The migrations from one release to the next.
pub struct ReleaseStep {
    /// The release whose data directories the step reads.
    pub from: &'static str,
    /// What the step changes, in the order a run applies it.
    pub migrations: &'static [Migration],
}

/// The release steps, oldest first.
pub static RELEASE_STEPS: &[ReleaseStep] = &[from_0_10_1::STEP];

/// Every migration of every step, in the order a run applies them.
pub fn migrations() -> impl Iterator<Item = &'static Migration> {
    RELEASE_STEPS.iter().flat_map(|step| step.migrations)
}
