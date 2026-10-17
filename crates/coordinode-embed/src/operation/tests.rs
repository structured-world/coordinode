use std::collections::HashMap;

use super::*;

/// Two attempts of one statement carry one intent whatever order their
/// parameters were built in; a different statement or value another.
#[test]
fn a_statement_intent_depends_on_text_and_values_only() {
    let params = |pairs: &[(&str, i64)]| -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), Value::Int(*v)))
            .collect()
    };
    let query = "CREATE (:T {a: $a, b: $b})";
    let one = statement_intent(query, Some(&params(&[("a", 1), ("b", 2)])));
    let reordered = statement_intent(query, Some(&params(&[("b", 2), ("a", 1)])));
    assert_eq!(one, reordered);
    assert_ne!(
        one,
        statement_intent(query, Some(&params(&[("a", 1), ("b", 3)])))
    );
    assert_ne!(one, statement_intent("CREATE (:U)", None));
    assert_eq!(
        statement_intent("CREATE (:U)", None),
        statement_intent("CREATE (:U)", Some(&HashMap::new()))
    );
}

/// A transaction's intent is its statements in order.
#[test]
fn a_chain_intent_is_ordered() {
    let a = statement_intent("CREATE (:A)", None);
    let b = statement_intent("CREATE (:B)", None);
    let ab = chain_intent(&chain_intent(&EMPTY_CHAIN, &a), &b);
    let ba = chain_intent(&chain_intent(&EMPTY_CHAIN, &b), &a);
    assert_ne!(ab, ba);
    assert_eq!(ab, chain_intent(&chain_intent(&EMPTY_CHAIN, &a), &b));
}

/// An issuer hands out increasing numbers and stops at the end of the
/// space instead of wrapping onto numbers it handed out.
#[test]
fn an_issuer_numbers_writes_and_never_wraps() {
    let domain = DeploymentId([2; 16]);
    let issuer = OperationIssuer::new(IssuerId([1; 16]), 3, 1, domain);
    let first = issuer.next().unwrap();
    let second = issuer.next().unwrap();
    assert_eq!((first.op.sequence, second.op.sequence), (1, 2));
    assert_eq!(first.op.incarnation, 3);
    assert_eq!(issuer.next_sequence(), 3);

    assert_eq!(first.deployment, domain);

    let last = OperationIssuer::new(IssuerId([1; 16]), 3, u64::MAX, domain);
    assert_eq!(last.next(), Err(IssuerExhausted));
}
