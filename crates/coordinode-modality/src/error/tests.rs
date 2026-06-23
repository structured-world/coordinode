use super::*;

/// Module docstring promises CapacityExhausted flows through
/// StoreError::Storage with the typed variant preserved (so the
/// gRPC layer can drill into the chain and surface
/// RESOURCE_EXHAUSTED). Verify by constructing one and matching.
#[test]
fn capacity_exhausted_preserved_through_store_error() {
    let storage_err = StorageError::CapacityExhausted {
        endpoint_id: "ep1".to_owned(),
        used_bytes: 100,
        hard_limit_bytes: 100,
    };
    let store_err: StoreError = storage_err.into();
    match store_err {
        StoreError::Storage(StorageError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        }) => {
            assert_eq!(endpoint_id, "ep1");
            assert_eq!(used_bytes, 100);
            assert_eq!(hard_limit_bytes, 100);
        }
        other => panic!("expected wrapped CapacityExhausted, got {other:?}"),
    }
}

/// Decode error variant retains kind + message verbatim.
#[test]
fn decode_error_carries_kind_and_message() {
    let err = StoreError::Decode {
        kind: "test thing",
        message: "boom".to_owned(),
    };
    let rendered = format!("{err}");
    assert!(rendered.contains("test thing"));
    assert!(rendered.contains("boom"));
}

/// Invariant variant exists separately from Storage / Decode so
/// callers can distinguish "I broke a precondition" from "the
/// engine broke".
#[test]
fn invariant_error_is_distinguishable() {
    let err = StoreError::Invariant("bad state".into());
    assert!(matches!(err, StoreError::Invariant(_)));
}
