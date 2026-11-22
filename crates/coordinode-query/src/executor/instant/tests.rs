use super::*;

/// The instants RFC 3339 §5.6 admits are read as microseconds since the
/// Unix epoch, whatever the separator of §5.6's NOTE and the case of `Z`.
#[test]
fn rfc3339_instants_are_read() {
    for (text, expected) in [
        ("2026-03-15T10:00:00Z", 1_773_568_800_000_000),
        ("2026-03-15t10:00:00z", 1_773_568_800_000_000),
        ("2026-03-15 10:00:00Z", 1_773_568_800_000_000),
        ("2026-03-15T05:00:00-05:00", 1_773_568_800_000_000),
        ("2026-03-15T10:00:00.25Z", 1_773_568_800_250_000),
        ("1970-01-01T00:00:00Z", 0),
        ("1969-12-31T23:59:59Z", -1_000_000),
    ] {
        assert_eq!(parse_instant_us(text), Ok(expected), "{text}");
    }
}

/// More than nine fractional digits are read, the digits past the
/// microsecond dropped; a part of a microsecond before the epoch rounds
/// down, so it stays before the epoch.
#[test]
fn long_fractions_are_read() {
    assert_eq!(
        parse_instant_us("2026-03-15T10:00:00.1234567891234Z"),
        Ok(1_773_568_800_123_456)
    );
    assert_eq!(parse_instant_us("1969-12-31T23:59:59.9999995Z"), Ok(-1));
}

/// A leap second (RFC 3339 §5.7) is read as the last instant of its
/// minute.
#[test]
fn a_leap_second_is_read() {
    let leap = parse_instant_us("2016-12-31T23:59:60Z").expect("leap second");
    let before = parse_instant_us("2016-12-31T23:59:59Z").expect("second before");
    let after = parse_instant_us("2017-01-01T00:00:00Z").expect("second after");
    assert!(before < leap && leap < after, "{before} {leap} {after}");
}

/// Forms RFC 3339 does not admit are refused, never guessed at: a time
/// without seconds, the ISO 8601 basic format, a decimal comma, an offset
/// without its colon or minutes or past 23 hours, a year of more than four
/// digits, a bracketed annotation (RFC 9557), and any other separator.
#[test]
fn other_forms_are_refused() {
    for text in [
        "2026-03-15T00Z",
        "20260315T100000Z",
        "2026-03-15T10:00:00,5Z",
        "2026-03-15T10:00:00+0500",
        "2026-03-15T10:00:00+05",
        "2026-03-15T10:00:00+25:00",
        "+202026-03-15T10:00:00Z",
        "202026-03-15T10:00:00Z",
        "2026-03-15T10:00:00Z[Europe/Paris]",
        "2026-03-15T10:00:00+01:00[Europe/Paris]",
        "2026-03-15_10:00:00Z",
        "2026-03-15",
        "",
    ] {
        assert!(parse_instant_us(text).is_err(), "{text:?} was read");
    }
}
