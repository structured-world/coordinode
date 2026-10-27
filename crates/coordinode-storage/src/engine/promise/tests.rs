use super::*;

const INC: Incarnation = [7u8; 16];

fn written(payload: &[u8]) -> (tempfile::TempDir, Vec<u8>) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_copy(dir.path(), &INC, payload, &Footprints::default()).expect("write");
    let bytes = std::fs::read(dir.path().join(PROMISE_FILE)).expect("read back");
    (dir, bytes)
}

/// A written copy reads back as the same payload and incarnation, and leaves
/// no staged file behind.
#[test]
fn a_written_copy_reads_back_whole() {
    let (dir, _) = written(b"vote");
    assert_eq!(
        read_copy(dir.path()).expect("read"),
        PromiseCopy::Held {
            incarnation: INC,
            payload: b"vote".to_vec()
        }
    );
    assert!(!dir.path().join(format!("{PROMISE_FILE}.tmp")).exists());
}

/// No file is no copy, not an error.
#[test]
fn a_missing_copy_is_absent() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_eq!(read_copy(dir.path()).expect("read"), PromiseCopy::Absent);
}

/// Replacing a copy leaves only the new one.
#[test]
fn a_replaced_copy_holds_the_new_payload() {
    let (dir, _) = written(b"old");
    write_copy(dir.path(), &INC, b"newer", &Footprints::default()).expect("replace");
    assert_eq!(
        read_copy(dir.path()).expect("read"),
        PromiseCopy::Held {
            incarnation: INC,
            payload: b"newer".to_vec()
        }
    );
}

/// Every single damaged byte and every truncation is detected: a damaged
/// copy is never read as some other promise.
#[test]
fn any_damaged_byte_or_truncation_is_detected() {
    let (_dir, bytes) = written(b"a promise of some length");
    for at in 0..bytes.len() {
        let mut flipped = bytes.clone();
        flipped[at] ^= 0x40;
        assert!(
            matches!(decode(&flipped), PromiseCopy::Damaged(_)),
            "byte {at} flipped"
        );
    }
    for len in 0..bytes.len() {
        assert!(
            matches!(decode(&bytes[..len]), PromiseCopy::Damaged(_)),
            "cut at {len}"
        );
    }
}

/// A replacement cut short before its rename leaves its staged file behind:
/// the copy in place is still the previous one, whole, and the next
/// replacement writes over the staged file.
#[test]
fn a_replacement_cut_before_its_rename_leaves_the_previous_copy() {
    let (dir, _) = written(b"previous");
    let staged = dir.path().join(format!("{PROMISE_FILE}.tmp"));
    std::fs::write(&staged, b"torn half of a new cop").expect("a torn staged file");
    assert_eq!(
        read_copy(dir.path()).expect("read"),
        PromiseCopy::Held {
            incarnation: INC,
            payload: b"previous".to_vec()
        }
    );
    write_copy(dir.path(), &INC, b"next", &Footprints::default()).expect("replace");
    assert_eq!(
        read_copy(dir.path()).expect("read"),
        PromiseCopy::Held {
            incarnation: INC,
            payload: b"next".to_vec()
        }
    );
    assert!(!staged.exists());
}

/// A replacement the endpoint has no room for is refused before anything is
/// written, and the copy in place stays whole: a full disk never costs a
/// promise already given.
#[test]
fn a_replacement_without_room_keeps_the_previous_copy() {
    use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
    let dir = tempfile::tempdir().expect("tempdir");
    write_copy(dir.path(), &INC, b"given", &Footprints::default()).expect("first copy");
    // Room for the copy in place, charged by allocation unit, and not for a
    // staged replacement beside it.
    let unit = 4096;
    let endpoint = EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(unit + unit / 2);
    let footprints = Footprints::open(&[endpoint]).expect("footprints");
    let refused = write_copy(dir.path(), &INC, b"a larger promise", &footprints);
    assert!(
        matches!(refused, Err(StorageError::CapacityExhausted { .. })),
        "{refused:?}"
    );
    assert_eq!(
        read_copy(dir.path()).expect("read"),
        PromiseCopy::Held {
            incarnation: INC,
            payload: b"given".to_vec()
        }
    );
}

/// A marker's incarnation survives its text form; an empty marker (written
/// before markers carried one) has none, and anything else is refused.
#[test]
fn an_incarnation_survives_its_marker_text() {
    let incarnation: Incarnation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xff];
    let text = format_incarnation(&incarnation);
    assert_eq!(parse_incarnation(&text).expect("parse"), Some(incarnation));
    assert_eq!(parse_incarnation("").expect("empty"), None);
    assert!(parse_incarnation("not hex").is_err());
    assert!(parse_incarnation(&"z".repeat(32)).is_err());
}
