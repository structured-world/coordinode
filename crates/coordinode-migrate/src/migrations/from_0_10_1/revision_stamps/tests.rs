use super::*;

fn version<'a>(key: &'a [u8], value: &'a [u8], seqno: u64) -> StoredVersion<'a> {
    StoredVersion {
        key,
        value,
        seqno,
        operand: false,
    }
}

fn split(stored: &[u8]) -> (&[u8], u64) {
    let (payload, stamp) = stored.split_at(stored.len() - 8);
    (
        payload,
        u64::from_le_bytes(stamp.try_into().expect("eight bytes")),
    )
}

/// A version the journal holds the write of, at its timestamp with its key
/// and bytes, keeps that commit; any other version, whatever its sequence
/// number or the bytes it ends in, gets the unknown revision, its payload
/// byte for byte.
#[test]
fn a_version_keeps_only_the_commit_the_journal_proves() {
    let mut known = Writes::new();
    known.insert((Partition::Node, b"node:1".to_vec(), 70), b"v".to_vec());

    let proved = stamped(Partition::Node, version(b"node:1", b"v", 70), &known);
    assert_eq!(split(&proved), (&b"v"[..], 70));

    // Another value at that timestamp, the same value at another one.
    for (value, seqno) in [(&b"w"[..], 70), (&b"v"[..], 71)] {
        let stored = stamped(Partition::Node, version(b"node:1", value, seqno), &known);
        assert_eq!(split(&stored), (value, UNKNOWN));
    }

    // An installed row at a sequence number of its own, its last eight
    // bytes anything: its payload kept whole, its revision unknown.
    let long = b"a long payload ending in eight bytes 12345678";
    let stored = stamped(Partition::Schema, version(b"schema:x", long, 9_000), &known);
    assert_eq!(split(&stored), (&long[..], UNKNOWN));

    // A settled row: no commit is ever at zero.
    let mut settled = Writes::new();
    settled.insert((Partition::Node, b"node:2".to_vec(), 0), b"v".to_vec());
    let stored = stamped(Partition::Node, version(b"node:2", b"v", 0), &settled);
    assert_eq!(split(&stored), (&b"v"[..], UNKNOWN));
}

/// A key of the engine's reserved namespace carries no revision: it is
/// written as it was.
#[test]
fn a_reserved_key_is_written_as_it_was() {
    let stored = stamped(
        Partition::Node,
        version(&[0x00, b'c'], b"coverage", 5),
        &Writes::new(),
    );
    assert_eq!(stored, b"coverage");
}
