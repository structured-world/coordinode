use super::*;

fn full() -> Handshake {
    Handshake {
        node_id: 2,
        group_id: GroupId(7),
        pair: VersionPair {
            engine: 3,
            host_epoch: 9,
        },
        group_pair: Some(RecordedPair {
            pair: VersionPair {
                engine: 2,
                host_epoch: 9,
            },
            seq: 120,
        }),
        leader: Some((1, "10.0.0.1:7080".into())),
    }
}

/// The encoding is frozen: these bytes are what every release sends and
/// must keep reading. A change here breaks the only exchange two releases
/// share.
#[test]
fn the_encoding_is_frozen() {
    let mut expected = Vec::new();
    expected.extend_from_slice(b"CNVH");
    expected.extend_from_slice(&2u64.to_le_bytes());
    expected.extend_from_slice(&7u64.to_le_bytes());
    expected.extend_from_slice(&3u32.to_le_bytes());
    expected.extend_from_slice(&9u64.to_le_bytes());
    expected.push(1);
    expected.extend_from_slice(&2u32.to_le_bytes());
    expected.extend_from_slice(&9u64.to_le_bytes());
    expected.extend_from_slice(&120u64.to_le_bytes());
    expected.push(1);
    expected.extend_from_slice(&1u64.to_le_bytes());
    expected.extend_from_slice(&13u16.to_le_bytes());
    expected.extend_from_slice(b"10.0.0.1:7080");
    assert_eq!(full().encode(), expected);
    assert_eq!(Handshake::decode(&expected), Ok(full()));
}

#[test]
fn absent_group_pair_and_leader_round_trip() {
    let h = Handshake {
        group_pair: None,
        leader: None,
        ..full()
    };
    let bytes = h.encode();
    assert_eq!(bytes.len(), 4 + 8 + 8 + 12 + 1 + 1);
    assert_eq!(Handshake::decode(&bytes), Ok(h));
}

/// Damaged or foreign bytes are refused by name, never read as a match.
#[test]
fn malformed_records_are_refused() {
    let bytes = full().encode();
    for cut in 0..bytes.len() {
        assert_eq!(
            Handshake::decode(&bytes[..cut]),
            Err(HandshakeError::Truncated),
            "cut at {cut}"
        );
    }
    let mut foreign = bytes.clone();
    foreign[0] = b'X';
    assert_eq!(Handshake::decode(&foreign), Err(HandshakeError::BadMagic));
    let mut flag = bytes.clone();
    flag[4 + 8 + 8 + 12] = 2;
    assert_eq!(Handshake::decode(&flag), Err(HandshakeError::BadFlag));
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(Handshake::decode(&trailing), Err(HandshakeError::Trailing));
    let mut not_utf8 = bytes;
    let last = not_utf8.len() - 1;
    not_utf8[last] = 0xff;
    assert_eq!(
        Handshake::decode(&not_utf8),
        Err(HandshakeError::BadAddress)
    );
}

/// An address too long for the record is left out, not cut.
#[test]
fn an_oversized_leader_address_is_left_out() {
    let h = Handshake {
        leader: Some((1, "a".repeat(MAX_LEADER_ADDR + 1))),
        ..full()
    };
    let decoded = Handshake::decode(&h.encode()).expect("decodes");
    assert_eq!(decoded.leader, Some((1, String::new())));
}

#[test]
fn pairs_match_only_when_both_parts_are_equal() {
    let p = VersionPair {
        engine: ENGINE_FORMAT_VERSION,
        host_epoch: 0,
    };
    assert_eq!(p, VersionPair::current(0));
    assert_ne!(p, VersionPair { host_epoch: 1, ..p });
    assert_ne!(
        p,
        VersionPair {
            engine: ENGINE_FORMAT_VERSION + 1,
            ..p
        }
    );
}

#[test]
fn the_later_record_wins() {
    let a = RecordedPair {
        pair: VersionPair::current(0),
        seq: 3,
    };
    assert!(RecordedPair { seq: 4, ..a }.is_later_than(&a));
    assert!(!a.is_later_than(&a));
    assert!(!RecordedPair { seq: 2, ..a }.is_later_than(&a));
}
