use super::*;
use tonic_types::StatusExt;

#[test]
fn the_reason_survives_the_round_trip_a_client_makes() {
    // What a client actually does: take the status, ask for the details, read
    // the reason and the domain. If this breaks, every caller branching on a
    // reason silently falls back to guessing from the message.
    let status = status_with_reason(
        Code::InvalidArgument,
        "Unknown function 'lenght'",
        Reason::UnknownFunction,
        [("function", "lenght".to_string())],
    );

    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(status.message(), "Unknown function 'lenght'");

    let details = status.get_error_details();
    let info = details
        .error_info()
        .expect("the status must carry ErrorInfo");
    assert_eq!(info.reason, "UNKNOWN_FUNCTION");
    assert_eq!(info.domain, ERROR_DOMAIN);
    assert_eq!(
        info.metadata.get("function").map(String::as_str),
        Some("lenght")
    );
}

#[test]
fn a_retryable_reason_says_so_and_the_rest_do_not() {
    // The retry advice is the server's, published so a client does not have to
    // keep its own copy of which of our failures are worth repeating.
    let conflict = status_with_reason(
        Code::Aborted,
        "conflict",
        Reason::TransactionConflict,
        [("transaction_id", "7".to_string())],
    );
    assert!(
        conflict.get_error_details().retry_info().is_some(),
        "a conflict is resolved by re-running the transaction"
    );

    let syntax = status_with_reason(Code::InvalidArgument, "bad", Reason::QuerySyntax, []);
    assert!(
        syntax.get_error_details().retry_info().is_none(),
        "a syntax error fails identically every time"
    );

    // Backpressure is retryable too, but with a NON-zero delay: an immediate
    // retry bounces off the same shedding verdict, so the advice is a floor
    // for the client's backoff (a conflict advises zero, retry-now).
    let shed = status_with_reason(
        Code::ResourceExhausted,
        "shedding",
        Reason::WriteBackpressure,
        [],
    );
    let details = shed.get_error_details();
    let retry = details.retry_info().expect("backpressure is retryable");
    assert!(
        retry.retry_delay.unwrap_or_default() > std::time::Duration::ZERO,
        "backpressure must advise waiting, not retry-now"
    );
}

#[test]
fn every_reason_has_a_distinct_wire_string() {
    // A duplicate would make two situations indistinguishable to a client
    // while looking perfectly fine in the source.
    let all = [
        Reason::QuerySyntax,
        Reason::QuerySemantics,
        Reason::UnknownFunction,
        Reason::DivideByZero,
        Reason::LongOverflow,
        Reason::UnknownTransaction,
        Reason::TransactionConflict,
        Reason::TransactionTooLarge,
        Reason::CapacityExhausted,
        Reason::SchemaViolation,
        Reason::WriteBackpressure,
        Reason::NotLeader,
    ];
    let mut seen = std::collections::BTreeSet::new();
    for reason in all {
        assert!(
            seen.insert(reason.as_str()),
            "duplicate reason string: {}",
            reason.as_str()
        );
        assert!(
            reason
                .as_str()
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '_'),
            "{} is not SCREAMING_SNAKE_CASE",
            reason.as_str()
        );
    }
}

/// A write that reached the wrong node must tell the caller where the right
/// one is, and that a retry is worth making.
///
/// Without the id in the metadata a client can only guess which node to try
/// next, and without the retry advice it cannot tell this apart from a
/// failure that will repeat forever. Both live in the structured details, not
/// in the message, so a client never has to parse prose.
#[test]
fn a_leader_change_carries_the_leader_and_says_retry_now() {
    let redirect = status_with_reason(
        Code::FailedPrecondition,
        "not the leader; leader is node 3",
        Reason::NotLeader,
        [("leader_id", "3".to_string())],
    );
    let details = redirect.get_error_details();
    let info = details.error_info().expect("carries ErrorInfo");
    assert_eq!(info.reason, "NOT_LEADER");
    assert_eq!(
        info.metadata.get("leader_id").map(String::as_str),
        Some("3"),
        "the caller cannot redirect without the leader's id"
    );
    let retry = details.retry_info().expect("a leader change is retryable");
    assert_eq!(
        retry.retry_delay.unwrap_or_default(),
        std::time::Duration::ZERO,
        "waiting does not help; the write has to go to another node"
    );

    // An election in flight leaves no leader to name. The failure is still
    // retryable, and the absent id is what tells the client to poll rather
    // than redirect.
    let electing = status_with_reason(
        Code::FailedPrecondition,
        "not the leader; no leader known yet",
        Reason::NotLeader,
        [],
    );
    let details = electing.get_error_details();
    assert!(
        !details
            .error_info()
            .expect("carries ErrorInfo")
            .metadata
            .contains_key("leader_id"),
        "no leader is known, so none must be claimed"
    );
    assert!(
        details.retry_info().is_some(),
        "an election settles; the write is still worth retrying"
    );
}
