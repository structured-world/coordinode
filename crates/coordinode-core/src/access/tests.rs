use super::*;

/// A principal is both of its parts: the same subject under another trust
/// scope is another principal, and neither part may be empty or longer than
/// a log entry may carry.
#[test]
fn a_principal_is_its_subject_and_trust_scope() {
    let alice = PrincipalId::new("alice", "issuer-a").expect("alice");
    assert_eq!(alice, PrincipalId::new("alice", "issuer-a").expect("again"));
    assert_ne!(
        alice,
        PrincipalId::new("alice", "issuer-b").expect("other scope")
    );
    assert_eq!(
        PrincipalId::new("", "issuer-a"),
        Err(BadPrincipal {
            part: "subject",
            len: 0
        })
    );
    let long = "x".repeat(MAX_PRINCIPAL_PART_BYTES + 1);
    assert_eq!(
        PrincipalId::new("alice", &long),
        Err(BadPrincipal {
            part: "trust scope",
            len: MAX_PRINCIPAL_PART_BYTES + 1
        })
    );
    let edge = "x".repeat(MAX_PRINCIPAL_PART_BYTES);
    assert!(PrincipalId::new(&edge, &edge).is_ok());
}
