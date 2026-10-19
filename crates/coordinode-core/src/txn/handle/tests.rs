use super::*;

/// A handle survives its byte form exactly.
#[test]
fn a_handle_round_trips_through_its_bytes() {
    let handle = TransactionHandle::new(Some(DeploymentId([9; 16])), 3, 41, 7);
    let back = TransactionHandle::from_bytes(&handle.to_bytes()).expect("decode");
    assert_eq!(back, handle);
    assert_eq!((back.owner(), back.incarnation(), back.local()), (3, 41, 7));
}

/// Two handles with the same number in different tenures, members or
/// databases are different handles.
#[test]
fn handles_differ_by_scope_owner_and_tenure() {
    let base = TransactionHandle::new(Some(DeploymentId([1; 16])), 1, 5, 1);
    for other in [
        TransactionHandle::new(Some(DeploymentId([2; 16])), 1, 5, 1),
        TransactionHandle::new(None, 1, 5, 1),
        TransactionHandle::new(Some(DeploymentId([1; 16])), 2, 5, 1),
        TransactionHandle::new(Some(DeploymentId([1; 16])), 1, 6, 1),
    ] {
        assert_ne!(other, base);
        assert_ne!(other.to_bytes(), base.to_bytes());
    }
}

/// Bytes of the wrong length or format are refused, not read as a handle.
#[test]
fn malformed_bytes_are_refused() {
    let bytes = TransactionHandle::new(None, 0, 1, 1).to_bytes();
    assert_eq!(
        TransactionHandle::from_bytes(&bytes[..TRANSACTION_HANDLE_LEN - 1]),
        Err(MalformedHandle)
    );
    assert_eq!(TransactionHandle::from_bytes(&[]), Err(MalformedHandle));
    let mut other_format = bytes;
    other_format[0] = 2;
    assert_eq!(
        TransactionHandle::from_bytes(&other_format),
        Err(MalformedHandle)
    );
}
