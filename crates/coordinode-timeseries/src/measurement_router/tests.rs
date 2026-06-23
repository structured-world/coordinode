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
