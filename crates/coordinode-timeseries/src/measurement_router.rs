//! Decides whether an incoming measurement fits in the bucket
//! currently open under its [`crate::BucketKey`], or whether the
//! bucket must roll over first. Pure function — no I/O — so the
//! catalog's stripe-locked critical section stays short.

use std::collections::BTreeMap;

use coordinode_modality::{BucketControl, FieldStats, Measurement};

use crate::config::CatalogConfig;

/// Decision for the catalog's per-measurement write path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Bucket has room — append the measurement to the in-memory
    /// buffer and update the control block in-place.
    Append,
    /// Bucket is full or schema-mismatched — flush this bucket to
    /// the underlying TimeSeriesStore, then open a fresh one for
    /// the same key and append the measurement there.
    Rollover(RolloverReason),
}

/// Specific trigger that caused a rollover. Surfaced for observability
/// (every rollover emits a tracing event tagged with the reason).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RolloverReason {
    /// Bucket reached [`crate::CatalogConfig::max_count`].
    Count,
    /// Bucket's serialised size estimate exceeded
    /// [`crate::CatalogConfig::max_size_bytes`]. The estimate is a
    /// running sum maintained as measurements arrive — accurate
    /// enough for rollover triggering, exact size is computed on
    /// flush.
    Size,
    /// Bucket's `time_max_us - time_min_us` exceeded
    /// [`crate::CatalogConfig::granularity_span`] (after considering
    /// the incoming measurement's timestamp).
    Time,
    /// Incoming measurement introduces a field the bucket's
    /// accumulated schema doesn't have.
    Schema,
}

/// Decide what to do with `measurement` against an open bucket
/// described by `control` + `size_estimate`.
///
/// `size_estimate` is the catalog's running byte count for the
/// bucket so far; the caller updates it after this returns `Append`.
pub fn route(
    control: &BucketControl,
    size_estimate: u32,
    measurement: &Measurement,
    config: &CatalogConfig,
) -> Decision {
    if control.count >= config.max_count {
        return Decision::Rollover(RolloverReason::Count);
    }

    let measurement_size = estimate_measurement_size(measurement);
    if size_estimate.saturating_add(measurement_size) > config.max_size_bytes {
        return Decision::Rollover(RolloverReason::Size);
    }

    // Schema check: incoming fields must be a SUBSET of the bucket's
    // known column set. New fields force a rollover so each bucket
    // has a stable schema (required for columnar compression on
    // flush). It is legal to omit fields the bucket has — they
    // become NULLs in the merged column on flush.
    if has_schema_drift(&control.fields_stats, &measurement.fields) {
        return Decision::Rollover(RolloverReason::Schema);
    }

    let new_min = control.time_min_us.min(measurement.timestamp_us);
    let new_max = control.time_max_us.max(measurement.timestamp_us);
    let span = new_max.saturating_sub(new_min);
    if span > config.granularity_span.as_micros() as i64 {
        return Decision::Rollover(RolloverReason::Time);
    }

    Decision::Append
}

/// Returns `true` if `incoming` introduces a field the bucket doesn't
/// yet know about. Identical-set incoming and subset-incoming both
/// return `false`; only "introduces a new key" triggers rollover.
fn has_schema_drift(
    bucket_columns: &BTreeMap<String, FieldStats>,
    incoming: &BTreeMap<String, f64>,
) -> bool {
    if bucket_columns.is_empty() {
        // Empty bucket has no schema yet; first measurement defines it.
        return false;
    }
    incoming.keys().any(|k| !bucket_columns.contains_key(k))
}

/// Conservative size estimate for one measurement. Used as a running
/// sum to detect [`RolloverReason::Size`] before the bucket actually
/// exceeds the limit. Slightly overestimates so we flush a bucket
/// one measurement early rather than crossing the BlobStore threshold.
fn estimate_measurement_size(m: &Measurement) -> u32 {
    // 8 B timestamp + per-field (name + 8 B f64 + BTreeMap entry overhead).
    let mut size: u32 = 8;
    for name in m.fields.keys() {
        size = size.saturating_add(name.len() as u32);
        size = size.saturating_add(16); // 8 B value + 8 B entry overhead
    }
    size
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn cfg_small_limits() -> CatalogConfig {
        CatalogConfig {
            max_count: 3,
            max_size_bytes: 4096,
            granularity_span: Duration::from_millis(100),
        }
    }

    fn empty_control() -> BucketControl {
        BucketControl {
            version: 1,
            count: 0,
            time_min_us: i64::MAX,
            time_max_us: i64::MIN,
            closed: false,
            fields_stats: BTreeMap::new(),
        }
    }

    fn make_measurement(ts_us: i64, fields: &[(&str, f64)]) -> Measurement {
        let mut m = BTreeMap::new();
        for (k, v) in fields {
            m.insert((*k).to_string(), *v);
        }
        Measurement {
            timestamp_us: ts_us,
            ingestion_ts_us: None,
            fields: m,
        }
    }

    #[test]
    fn empty_bucket_first_measurement_appends() {
        let control = empty_control();
        let m = make_measurement(0, &[("temp", 22.5)]);
        assert_eq!(
            route(&control, 0, &m, &cfg_small_limits()),
            Decision::Append,
        );
    }

    #[test]
    fn count_rollover_when_threshold_reached() {
        let mut control = empty_control();
        control.count = 3; // cfg max_count = 3
        control.time_min_us = 0;
        control.time_max_us = 10;
        let m = make_measurement(20, &[("temp", 22.5)]);
        assert_eq!(
            route(&control, 0, &m, &cfg_small_limits()),
            Decision::Rollover(RolloverReason::Count),
        );
    }

    #[test]
    fn size_rollover_when_running_sum_exceeds_max() {
        let control = empty_control();
        let m = make_measurement(0, &[("temp", 22.5)]);
        // size_estimate already at threshold; adding any measurement
        // forces Size rollover.
        let dec = route(&control, 4096, &m, &cfg_small_limits());
        assert_eq!(dec, Decision::Rollover(RolloverReason::Size));
    }

    #[test]
    fn time_rollover_when_span_exceeds_granularity() {
        let mut control = empty_control();
        control.count = 1;
        control.time_min_us = 0;
        control.time_max_us = 50_000; // 50ms
                                      // cfg granularity 100ms; measurement at 200ms extends span to 200ms > 100ms
        let m = make_measurement(200_000, &[("temp", 22.5)]);
        assert_eq!(
            route(&control, 0, &m, &cfg_small_limits()),
            Decision::Rollover(RolloverReason::Time),
        );
    }

    #[test]
    fn schema_rollover_when_new_field_introduced() {
        let mut control = empty_control();
        control.count = 1;
        control.time_min_us = 0;
        control.time_max_us = 10;
        control.fields_stats.insert(
            "temp".into(),
            FieldStats {
                min: 20.0,
                max: 25.0,
            },
        );
        // Incoming has a NEW field "humidity" — schema drift.
        let m = make_measurement(20, &[("temp", 22.5), ("humidity", 65.0)]);
        assert_eq!(
            route(&control, 0, &m, &cfg_small_limits()),
            Decision::Rollover(RolloverReason::Schema),
        );
    }

    #[test]
    fn schema_subset_does_not_trigger_rollover() {
        let mut control = empty_control();
        control.count = 1;
        control.time_min_us = 0;
        control.time_max_us = 10;
        control.fields_stats.insert(
            "temp".into(),
            FieldStats {
                min: 20.0,
                max: 25.0,
            },
        );
        control.fields_stats.insert(
            "humidity".into(),
            FieldStats {
                min: 60.0,
                max: 70.0,
            },
        );
        // Incoming is a SUBSET (only `temp`) — legal, becomes NULL
        // for `humidity` on flush.
        let m = make_measurement(20, &[("temp", 22.5)]);
        assert_eq!(
            route(&control, 0, &m, &cfg_small_limits()),
            Decision::Append,
        );
    }

    #[test]
    fn count_rollover_takes_precedence_over_size() {
        let mut control = empty_control();
        control.count = 3; // at limit
        control.time_min_us = 0;
        control.time_max_us = 10;
        let m = make_measurement(20, &[("temp", 22.5)]);
        // size_estimate ALSO over limit — Count must win since it
        // checks first (deterministic ordering for observability).
        let dec = route(&control, 4096, &m, &cfg_small_limits());
        assert_eq!(dec, Decision::Rollover(RolloverReason::Count));
    }

    #[test]
    fn out_of_order_measurement_within_window_appends() {
        let mut control = empty_control();
        control.count = 1;
        control.time_min_us = 50_000;
        control.time_max_us = 60_000;
        // Incoming is BEFORE current min — Tier 1 buffer absorption:
        // span 60 - 30 = 30ms ≤ 100ms cfg → Append, sort on flush.
        let m = make_measurement(30_000, &[("temp", 22.5)]);
        assert_eq!(
            route(&control, 0, &m, &cfg_small_limits()),
            Decision::Append,
        );
    }
}
