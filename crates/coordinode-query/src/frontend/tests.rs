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

#[test]
fn parse_error_surfaces_as_frontend_parse_error() {
    let fe = CypherFrontend::new();
    let err = fe.parse("this is not cypher !!!").expect_err("should fail");
    assert!(matches!(err, FrontendError::Parse(_)));
}
