use super::*;

const NAME: u32 = 1;
const VALID_TO: u32 = 2;
const DELETED: u32 = 3;

const FIELDS: TimelineFields = TimelineFields {
    valid_to: Some(VALID_TO),
    deleted: Some(DELETED),
};

fn version(name: &str, valid_to: Option<i64>) -> NodeRecord {
    let mut record = NodeRecord::new("T");
    record.set(NAME, Value::String(name.into()));
    if let Some(to) = valid_to {
        record.set(VALID_TO, Value::Int(to));
    }
    record
}

fn tombstone() -> NodeRecord {
    let mut record = NodeRecord::new("T");
    record.set(DELETED, Value::Bool(true));
    record
}

fn name_at(versions: Vec<(i64, NodeRecord)>, at: i64) -> Option<String> {
    state_at(versions, at, FIELDS)
        .positive()
        .map(|(_, record)| match record.props.get(&NAME) {
            Some(Value::String(s)) => s.clone(),
            other => panic!("name: {other:?}"),
        })
}

/// The start belongs to a version, its end does not.
#[test]
fn intervals_are_half_open() {
    let timeline = || vec![(10, version("a", Some(20))), (20, version("b", None))];
    assert_eq!(name_at(timeline(), 9), None, "before the first start");
    assert_eq!(name_at(timeline(), 10).as_deref(), Some("a"));
    assert_eq!(name_at(timeline(), 19).as_deref(), Some("a"));
    assert_eq!(name_at(timeline(), 20).as_deref(), Some("b"));
    assert_eq!(name_at(timeline(), i64::MAX).as_deref(), Some("b"));
}

/// A finite-ended version is the state inside its interval even with an
/// open-ended version starting later.
#[test]
fn a_finite_current_version_beats_an_open_future_one() {
    let timeline = vec![
        (10, version("now", Some(30))),
        (30, version("future", None)),
    ];
    assert_eq!(name_at(timeline, 15).as_deref(), Some("now"));
}

/// A gap between two versions is absence, not the earlier version extended.
#[test]
fn a_gap_is_absent() {
    let timeline = vec![(10, version("a", Some(20))), (40, version("b", None))];
    assert_eq!(state_at(timeline, 30, FIELDS), StateAt::Absent);
}

/// After the last version ended there is no state.
#[test]
fn after_the_end_is_absent() {
    assert_eq!(
        state_at(vec![(10, version("a", Some(20)))], 20, FIELDS),
        StateAt::Absent
    );
}

/// A tombstone covering the instant is a deletion, told apart from absence.
#[test]
fn a_tombstone_is_deleted_not_absent() {
    let timeline = vec![(10, version("a", Some(20))), (20, tombstone())];
    assert_eq!(state_at(timeline.clone(), 25, FIELDS), StateAt::Deleted);
    assert_eq!(
        name_at(timeline, 15).as_deref(),
        Some("a"),
        "history before it"
    );
}

/// Without registered field names no version can carry an end or a
/// tombstone: every covering version is live.
#[test]
fn unregistered_fields_mean_open_live_versions() {
    let fields = TimelineFields {
        valid_to: None,
        deleted: None,
    };
    let state = state_at(vec![(10, version("a", Some(20)))], 50, fields);
    assert!(matches!(state, StateAt::Positive { valid_from: 10, .. }));
}

/// Valid time is signed: instants before the epoch order and bound like any
/// other, down to the smallest value.
#[test]
fn instants_before_the_epoch_bound_like_any_other() {
    let timeline = || {
        vec![
            (i64::MIN, version("first", Some(-20))),
            (-20, version("second", Some(0))),
        ]
    };
    assert_eq!(name_at(timeline(), i64::MIN).as_deref(), Some("first"));
    assert_eq!(name_at(timeline(), -21).as_deref(), Some("first"));
    assert_eq!(name_at(timeline(), -20).as_deref(), Some("second"));
    assert_eq!(name_at(timeline(), 0), None, "the end is outside");
}

/// An end stored as a timestamp value ends the version as an integer does.
#[test]
fn a_timestamp_end_ends_the_version() {
    let mut record = version("a", None);
    record.set(VALID_TO, Value::Timestamp(20));
    assert_eq!(state_at(vec![(10, record)], 20, FIELDS), StateAt::Absent);
}

/// Stored versions whose intervals overlap still give one state per
/// instant: the one starting later, never both.
#[test]
fn overlapping_versions_give_one_state() {
    let timeline = vec![(10, version("a", Some(40))), (20, version("b", None))];
    assert_eq!(name_at(timeline.clone(), 15).as_deref(), Some("a"));
    assert_eq!(name_at(timeline, 25).as_deref(), Some("b"));
}

/// An empty timeline has no state at any instant.
#[test]
fn an_empty_timeline_is_absent() {
    let empty = Vec::<(i64, NodeRecord)>::new;
    assert_eq!(state_at(empty(), 0, FIELDS), StateAt::Absent);
    let span = state_span(empty(), 0, FIELDS);
    assert_eq!((span.from, span.until), (None, None), "absent forever");
}

/// The span around an instant is where the state stays the same: every
/// instant inside it has that state, and on a well-formed timeline the
/// instants just outside it have another one.
#[test]
fn a_span_is_exactly_where_the_state_holds() {
    let timelines = [
        vec![(10, version("a", Some(20))), (20, version("b", None))],
        vec![(10, version("a", Some(20))), (40, version("b", None))],
        vec![(10, version("a", Some(40))), (20, version("b", Some(30)))],
        vec![(10, version("a", Some(20))), (20, tombstone())],
        vec![(10, version("a", None))],
        vec![(10, version("a", Some(20)))],
    ];
    for timeline in timelines {
        for at in 0..50 {
            let span = state_span(timeline.clone(), at, FIELDS);
            assert_eq!(span.state, state_at(timeline.clone(), at, FIELDS));
            let from = span.from.unwrap_or(0).max(0);
            let until = span.until.unwrap_or(50).min(50);
            assert!(from <= at && at < until, "{at} in [{from}, {until})");
            for inside in from..until {
                assert_eq!(
                    state_at(timeline.clone(), inside, FIELDS),
                    span.state,
                    "{inside} inside the span of {at}"
                );
            }
            if let Some(from) = span.from.filter(|f| *f > 0) {
                assert_ne!(state_at(timeline.clone(), from - 1, FIELDS), span.state);
            }
            if let Some(until) = span.until.filter(|u| *u < 50) {
                assert_ne!(state_at(timeline.clone(), until, FIELDS), span.state);
            }
        }
    }
}

/// A version that ended before the instant leaves the node absent from its
/// end until the next version, and a node before its first version is
/// absent since forever.
#[test]
fn absence_is_bounded_by_the_versions_around_it() {
    let timeline = || vec![(10, version("a", Some(20))), (40, version("b", None))];
    let gap = state_span(timeline(), 30, FIELDS);
    assert_eq!(
        (gap.state, gap.from, gap.until),
        (StateAt::Absent, Some(20), Some(40))
    );
    let before = state_span(timeline(), 5, FIELDS);
    assert_eq!(
        (before.state, before.from, before.until),
        (StateAt::Absent, None, Some(10))
    );
    let last = state_span(timeline(), 45, FIELDS);
    assert_eq!((last.from, last.until), (Some(40), None));
}
