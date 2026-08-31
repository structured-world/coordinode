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
