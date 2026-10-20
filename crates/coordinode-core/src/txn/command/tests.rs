use super::*;

/// An ID survives its byte form exactly.
#[test]
fn an_id_round_trips_through_its_bytes() {
    let id = TransactionId::from_raw([7; TRANSACTION_ID_LEN]);
    assert_eq!(TransactionId::from_bytes(&id.to_bytes()), Ok(id));
}

/// Bytes of another length are refused, not padded or cut into an ID.
#[test]
fn bytes_of_another_length_are_refused() {
    assert_eq!(
        TransactionId::from_bytes(&[1; TRANSACTION_ID_LEN - 1]),
        Err(MalformedTransactionId)
    );
    assert_eq!(
        TransactionId::from_bytes(&[1; TRANSACTION_ID_LEN + 1]),
        Err(MalformedTransactionId)
    );
    assert_eq!(TransactionId::from_bytes(&[]), Err(MalformedTransactionId));
}

/// An ID prints as a hyphenated UUID, the form clients and logs show.
#[test]
fn an_id_prints_as_a_uuid() {
    let id = TransactionId::from_raw([
        0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0x4d, 0xef, 0x81, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
        0xef,
    ]);
    assert_eq!(id.to_string(), "12345678-9abc-4def-8123-456789abcdef");
}
