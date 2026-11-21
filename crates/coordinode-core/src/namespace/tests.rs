use super::*;

/// A name is a lowercase letter then letters, digits, `_` and `-`: one
/// spelling per name, so `Sales` and `sales` cannot name two namespaces.
#[test]
fn names_admit_one_spelling() {
    for name in ["default", "a", "tenant-7", "team_b", "x1-y_2"] {
        assert_eq!(validate_namespace_name(name), Ok(()), "{name}");
    }
    assert_eq!(validate_namespace_name(""), Err(NamespaceNameError::Empty));
    assert_eq!(
        validate_namespace_name("Sales"),
        Err(NamespaceNameError::Start { first: 'S' })
    );
    assert_eq!(
        validate_namespace_name("7th"),
        Err(NamespaceNameError::Start { first: '7' })
    );
    assert_eq!(
        validate_namespace_name("sales.eu"),
        Err(NamespaceNameError::Character { found: '.' })
    );
    assert_eq!(
        validate_namespace_name("café"),
        Err(NamespaceNameError::Character { found: 'é' })
    );
    let long = "a".repeat(MAX_NAMESPACE_NAME_LEN + 1);
    assert_eq!(
        validate_namespace_name(&long),
        Err(NamespaceNameError::TooLong {
            len: MAX_NAMESPACE_NAME_LEN + 1
        })
    );
    assert_eq!(
        validate_namespace_name(&long[..MAX_NAMESPACE_NAME_LEN]),
        Ok(())
    );
}

/// A record decodes back to what it encoded, in either state; bytes no
/// encode wrote (an unknown state, a short token, a name no namespace may
/// have) decode to nothing rather than to a guess.
#[test]
fn records_round_trip_and_refuse_foreign_bytes() {
    for state in [NamespaceState::Live, NamespaceState::Dropping] {
        let record = NamespaceRecord {
            name: "tenant-a".into(),
            state,
            token: [7; NAMESPACE_TOKEN_LEN],
        };
        assert_eq!(NamespaceRecord::decode(&record.encode()), Some(record));
    }
    let good = NamespaceRecord {
        name: "a".into(),
        state: NamespaceState::Live,
        token: [1; NAMESPACE_TOKEN_LEN],
    }
    .encode();
    let mut unknown_state = good.clone();
    unknown_state[0] = 2;
    assert_eq!(NamespaceRecord::decode(&unknown_state), None);
    assert_eq!(NamespaceRecord::decode(&good[..NAMESPACE_TOKEN_LEN]), None);
    let mut bad_name = good;
    bad_name.push(b'.');
    assert_eq!(NamespaceRecord::decode(&bad_name), None);
    assert_eq!(NamespaceRecord::decode(&[]), None);
}

/// Record keys list in id order and name back the id they were made for;
/// a key of another family names none.
#[test]
fn record_keys_order_by_id() {
    let ids = [1u64, 2, 255, 256, u64::MAX].map(NamespaceId::from_raw);
    let keys: Vec<Vec<u8>> = ids.iter().map(|&id| namespace_record_key(id)).collect();
    assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
    for (key, id) in keys.iter().zip(ids) {
        assert_eq!(decode_namespace_record_key(key), Some(id));
    }
    assert_eq!(decode_namespace_record_key(&namespace_name_key("a")), None);
    assert_eq!(decode_namespace_record_key(NAMESPACE_ALLOCATOR_KEY), None);
}

/// Every key the catalog writes is a catalog key, and the deployment's other
/// records, including one that only shares the allocator's spelling as a
/// prefix, are not.
#[test]
fn catalog_keys_are_told_from_other_records() {
    for key in [
        NAMESPACE_ALLOCATOR_KEY.to_vec(),
        namespace_record_key(NamespaceId::from_raw(3)),
        namespace_name_key("tenant-a"),
    ] {
        assert!(is_namespace_catalog_key(&key), "{key:?}");
    }
    for key in [
        &b"ns:nextish"[..],
        b"meta:route",
        b"raft:applied",
        b"ns",
        b"",
    ] {
        assert!(!is_namespace_catalog_key(key), "{key:?}");
    }
}
