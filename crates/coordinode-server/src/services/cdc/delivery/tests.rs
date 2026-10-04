use std::sync::Arc;

use prost::Message as _;
use tonic::Code;
use tonic_types::{ErrorDetails, StatusExt};

use super::{Credit, session_error};

/// Credit granted accumulates and is used by what is sent.
#[test]
fn credit_accumulates_and_is_used() {
    let credit = Credit::new(2);
    credit.grant(3);
    assert_eq!(credit.left(), 5);
    credit.take(4);
    assert_eq!(credit.left(), 1);
}

/// A cancelled subscription's waits end at once.
#[tokio::test]
async fn cancelling_wakes_a_wait_for_credit() {
    let credit = Arc::new(Credit::new(0));
    let waiter = Arc::clone(&credit);
    let waiting = tokio::spawn(async move {
        loop {
            let woken = waiter.granted.notified();
            if waiter.is_cancelled() {
                return;
            }
            woken.await;
        }
    });
    tokio::task::yield_now().await;
    credit.cancel();
    tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
        .await
        .expect("the wait ended")
        .expect("task");
}

/// A session error carries the canonical status with the typed details the
/// unary RPC returns, so a client branches on the same ErrorInfo reason.
#[test]
fn a_session_error_carries_the_canonical_status() {
    let status = tonic::Status::with_error_details(
        Code::FailedPrecondition,
        "change stream retention lost",
        ErrorDetails::with_error_info(
            "RETENTION_LOST",
            "coordinode.dev",
            [("requested_index".to_string(), "3".to_string())],
        ),
    );
    let error = session_error(&status);
    assert_eq!(error.code, Code::FailedPrecondition as u32);
    let canonical = error.status.expect("canonical status");
    assert_eq!(canonical.code, Code::FailedPrecondition as i32);
    let roundtrip = tonic::Status::with_details(
        Code::FailedPrecondition,
        canonical.message.clone(),
        canonical.encode_to_vec().into(),
    );
    let info = roundtrip
        .get_details_error_info()
        .expect("the ErrorInfo survives");
    assert_eq!(info.reason, "RETENTION_LOST");
    assert_eq!(info.metadata.get("requested_index"), Some(&"3".to_string()));
}

/// A status built without details still gets a canonical one.
#[test]
fn a_plain_status_gets_a_canonical_one() {
    let error = session_error(&tonic::Status::internal("boom"));
    let canonical = error.status.expect("canonical status");
    assert_eq!(
        (canonical.code, canonical.message.as_str()),
        (Code::Internal as i32, "boom")
    );
}
