use super::*;
use crate::cypher::parse;

/// Normalization replaces literal values with `$` placeholder.
#[test]
fn normalize_strips_literals() {
    let q1 = parse("MATCH (n:User {id: 42}) RETURN n.name").expect("parse q1");
    let q2 = parse("MATCH (n:User {id: 99}) RETURN n.name").expect("parse q2");

    let n1 = normalize(&q1);
    let n2 = normalize(&q2);

    assert_eq!(
        n1, n2,
        "queries differing only by literal should normalize equally"
    );
    assert!(
        n1.contains('$'),
        "normalized form should contain $ placeholder"
    );
    assert!(
        !n1.contains("42"),
        "normalized form should not contain original literal"
    );
}

/// Different query structures produce different fingerprints.
#[test]
fn different_structure_different_fingerprint() {
    let q1 = parse("MATCH (n:User) RETURN n").expect("parse q1");
    let q2 = parse("MATCH (n:User) RETURN n.name").expect("parse q2");

    let (_, fp1) = normalize_and_fingerprint(&q1);
    let (_, fp2) = normalize_and_fingerprint(&q2);

    assert_ne!(
        fp1, fp2,
        "different structures should have different fingerprints"
    );
}

/// Same structure with different literals produces same fingerprint.
#[test]
fn same_structure_same_fingerprint() {
    let q1 = parse("MATCH (n:User {id: 1}) RETURN n").expect("parse q1");
    let q2 = parse("MATCH (n:User {id: 999}) RETURN n").expect("parse q2");

    let (_, fp1) = normalize_and_fingerprint(&q1);
    let (_, fp2) = normalize_and_fingerprint(&q2);

    assert_eq!(
        fp1, fp2,
        "same structure with different literals = same fingerprint"
    );
}

/// Fingerprint is deterministic across calls.
#[test]
fn fingerprint_is_deterministic() {
    let q = parse("MATCH (n:User) WHERE n.age > 25 RETURN n").expect("parse");
    let (_, fp1) = normalize_and_fingerprint(&q);
    let (_, fp2) = normalize_and_fingerprint(&q);
    assert_eq!(fp1, fp2);
}

/// Parameters preserve their names (they're already normalized).
#[test]
fn parameters_preserved() {
    let q = parse("MATCH (n:User {id: $uid}) RETURN n").expect("parse");
    let canonical = normalize(&q);
    assert!(
        canonical.contains("$uid"),
        "parameter names should be preserved"
    );
}

/// Normalization preserves relationship types and labels.
#[test]
fn labels_and_types_preserved() {
    let q = parse("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b").expect("parse");
    let canonical = normalize(&q);
    assert!(canonical.contains("Person"), "labels preserved");
    assert!(canonical.contains("KNOWS"), "relationship types preserved");
}

/// Variable-length paths are preserved in canonical form.
#[test]
fn variable_length_preserved() {
    let q = parse("MATCH (a)-[:KNOWS*2..5]->(b) RETURN b").expect("parse");
    let canonical = normalize(&q);
    assert!(
        canonical.contains("*2..5"),
        "variable-length bounds preserved"
    );
}

/// Write operations normalize correctly.
#[test]
fn create_normalizes() {
    let q1 = parse("CREATE (n:User {name: 'Alice'})").expect("parse q1");
    let q2 = parse("CREATE (n:User {name: 'Bob'})").expect("parse q2");

    let (_, fp1) = normalize_and_fingerprint(&q1);
    let (_, fp2) = normalize_and_fingerprint(&q2);
    assert_eq!(
        fp1, fp2,
        "CREATE with different literals = same fingerprint"
    );
}

/// OPTIONAL MATCH is distinguished from MATCH.
#[test]
fn optional_match_distinct() {
    let q1 = parse("MATCH (n:User) RETURN n").expect("parse q1");
    let q2 =
        parse("MATCH (n:User) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n, m").expect("parse q2");

    let (_, fp1) = normalize_and_fingerprint(&q1);
    let (_, fp2) = normalize_and_fingerprint(&q2);
    assert_ne!(fp1, fp2);
}
