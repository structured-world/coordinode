//! Diagnostic builds only (`stage-timing`): how long each stage of a
//! statement's way through the server takes, recorded as
//! `coordinode_stage_duration_seconds{stage}` so a benchmark run can tell
//! where the time between a request and its answer goes. The crates on that
//! way call [`record`] under their own `stage-timing` feature, which turns
//! this one on; a shipped build compiles none of it.

// no-std: caller-provided clock trait.
use std::time::Instant;

/// Record that `stage` took the time since `since`.
pub fn record(stage: &'static str, since: Instant) {
    metrics::histogram!("coordinode_stage_duration_seconds", "stage" => stage)
        .record(since.elapsed().as_secs_f64());
}

/// Record that `stage`, called from `caller`, took the time since `since`:
/// the same stage told apart by where it is called from.
pub fn record_from(stage: &'static str, caller: &std::panic::Location<'_>, since: Instant) {
    let caller = format!("{}:{}", caller.file(), caller.line());
    metrics::histogram!(
        "coordinode_stage_duration_seconds",
        "stage" => stage,
        "caller" => caller
    )
    .record(since.elapsed().as_secs_f64());
}
