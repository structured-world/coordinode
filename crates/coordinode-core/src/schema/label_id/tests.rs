use super::*;

fn id(raw: u32) -> LabelId {
    LabelId::new(raw).unwrap()
}

/// Id 0 is never an identity: records store `NonZeroU32`, and a zero read
/// from storage is corruption, not a label.
#[test]
fn zero_is_not_a_label_id() {
    assert_eq!(LabelId::new(0), None);
    assert_eq!(FIRST_LABEL_ID.get(), 1);
    assert_eq!(id(u32::MAX).next(), None, "the id space ends");
}

/// Every binding state survives the round trip, and the decoder refuses
/// bytes that are not a binding (wrong length, id 0, unknown state) rather
/// than reading them as one.
#[test]
fn binding_round_trips_and_refuses_garbage() {
    for state in [
        BindingState::Implicit,
        BindingState::Declared,
        BindingState::Retired,
    ] {
        let binding = LabelBinding {
            id: id(7),
            revision: 42,
            state,
        };
        assert_eq!(LabelBinding::decode(&binding.encode()), Some(binding));
    }
    let mut bytes = LabelBinding {
        id: id(7),
        revision: 1,
        state: BindingState::Declared,
    }
    .encode();
    bytes[12] = 9;
    assert_eq!(LabelBinding::decode(&bytes), None, "unknown state");
    assert_eq!(LabelBinding::decode(&[0u8; 13]), None, "id 0");
    assert_eq!(
        LabelBinding::decode(&1u64.to_be_bytes()),
        None,
        "a bare revision"
    );
}

/// Only a declared binding has a current definition, and a retired one no
/// longer names a live identity: the name resolves to nothing.
#[test]
fn binding_state_decides_definition_and_liveness() {
    let at = |state| LabelBinding {
        id: id(3),
        revision: 5,
        state,
    };
    assert_eq!(at(BindingState::Declared).declared_revision(), 5);
    assert_eq!(at(BindingState::Implicit).declared_revision(), 0);
    assert_eq!(at(BindingState::Retired).declared_revision(), 0);
    assert_eq!(at(BindingState::Implicit).live_id(), Some(id(3)));
    assert_eq!(at(BindingState::Retired).live_id(), None);
}

/// Identity record keys sort by id, so the newest identity issued is the
/// last record, and decode only keys of their own family.
#[test]
fn identity_keys_sort_by_id_and_decode() {
    assert!(label_id_key(id(255)) < label_id_key(id(256)));
    assert_eq!(decode_label_id_key(&label_id_key(id(9))), Some(id(9)));
    assert_eq!(decode_label_id_key(b"schema:label:User:1"), None);
    assert_eq!(
        decode_label_id_key(b"schema:label_id:\x00\x00\x00\x00"),
        None
    );
}

/// A record stores an id as a plain MessagePack integer.
#[test]
fn label_id_serializes_as_an_integer() {
    let bytes = rmp_serde::to_vec(&id(5)).unwrap();
    assert_eq!(bytes, vec![5]);
    assert_eq!(rmp_serde::from_slice::<LabelId>(&bytes).unwrap(), id(5));
    assert!(rmp_serde::from_slice::<LabelId>(&[0]).is_err());
}
