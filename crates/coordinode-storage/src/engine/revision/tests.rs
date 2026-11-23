use super::*;

/// A stamp round-trips: the payload comes back byte for byte, including an
/// empty one, and the revision with it.
#[test]
fn a_stamp_round_trips() {
    for payload in [&b""[..], b"x", b"a longer payload of a record"] {
        let value = stamp(payload, 0x0102_0304_0506_0708);
        assert_eq!(value.len(), payload.len() + STAMP_LEN);
        assert_eq!(split(&value), Some((payload, 0x0102_0304_0506_0708)));
        assert_eq!(payload_of(&value).expect("stamped"), payload);
    }
}

/// A value shorter than a stamp is not a stamped record: it is refused
/// rather than read as a short payload with an invented revision.
#[test]
fn a_value_shorter_than_a_stamp_is_refused() {
    assert_eq!(split(b"1234567"), None);
    assert!(payload_of(b"").is_err());
}

/// Only keys outside the reserved namespace of partitions that replace
/// their records carry a stamp.
#[test]
fn which_values_carry_a_stamp() {
    assert!(stamped(Partition::Node, b"node:1"));
    assert!(stamped(Partition::Idx, b"idx:1"));
    assert!(!stamped(Partition::Node, &[0x00, 1]), "reserved keys");
    assert!(!stamped(Partition::Adj, b"adj:1"), "folded records");
    assert!(!stamped(Partition::Counter, b"counter:1"), "folded records");
}
