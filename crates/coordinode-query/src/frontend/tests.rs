use super::*;

#[test]
fn cypher_frontend_parses_valid_query() {
    let fe = CypherFrontend::new();
    let parsed = fe.parse("MATCH (n:User) RETURN n").expect("parse");
    assert!(!parsed.canonical.is_empty());
    assert_ne!(parsed.fingerprint, 0);
}

#[test]
fn same_query_has_stable_fingerprint() {
    let fe = CypherFrontend::new();
    let a = fe.parse("MATCH (n:User) RETURN n").expect("parse a");
    let b = fe.parse("MATCH (n:User) RETURN n").expect("parse b");
    assert_eq!(a.fingerprint, b.fingerprint);
    assert_eq!(a.canonical, b.canonical);
}

/// The wait a query names in a hint rides beside its plan; a query without
/// one leaves it to the session, and of two hints the later one wins.
#[test]
fn vector_build_wait_hint_is_carried_beside_the_plan() {
    let fe = CypherFrontend::new();
    let hinted = fe
        .parse("MATCH (n:Doc) RETURN n /*+ vector_build_wait('750ms') */")
        .expect("parse hinted");
    assert_eq!(
        hinted.vector_build_wait,
        Some(core::time::Duration::from_millis(750))
    );

    let plain = fe.parse("MATCH (n:Doc) RETURN n").expect("parse plain");
    assert_eq!(plain.vector_build_wait, None);

    let twice = fe
        .parse(
            "/*+ vector_build_wait('1s') */ MATCH (n:Doc) RETURN n \
             /*+ vector_build_wait('2s') */",
        )
        .expect("parse twice");
    assert_eq!(
        twice.vector_build_wait,
        Some(core::time::Duration::from_secs(2))
    );
}

#[test]
fn parse_error_surfaces_as_frontend_parse_error() {
    let fe = CypherFrontend::new();
    let err = fe.parse("this is not cypher !!!").expect_err("should fail");
    assert!(matches!(err, FrontendError::Parse(_)));
}
