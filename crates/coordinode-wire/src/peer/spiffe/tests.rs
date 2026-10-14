#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;

/// The forms the SPIFFE ID standard admits parse, and keep their exact
/// trust domain and path.
#[test]
fn valid_ids_parse_into_trust_domain_and_path() {
    let id = SpiffeId::parse("spiffe://example.org/ns/prod/node-7").expect("valid");
    assert_eq!(id.trust_domain(), "example.org");
    assert_eq!(id.path(), "/ns/prod/node-7");
    assert_eq!(id.as_str(), "spiffe://example.org/ns/prod/node-7");

    let bare = SpiffeId::parse("spiffe://a-b_c.d").expect("an id with no path");
    assert_eq!(bare.trust_domain(), "a-b_c.d");
    assert_eq!(bare.path(), "");

    // Path segments may use upper case, digits and the three symbols.
    SpiffeId::parse("spiffe://td/Node_1.a-b").expect("mixed-case path segment");
}

/// Each rule of the standard that an ID breaks refuses it by name, so a
/// certificate with a malformed SAN never becomes an identity.
#[test]
fn malformed_ids_are_refused_with_their_reason() {
    let cases: &[(&str, SpiffeIdError)] = &[
        ("", SpiffeIdError::Scheme),
        ("https://example.org/a", SpiffeIdError::Scheme),
        ("SPIFFE://example.org/a", SpiffeIdError::Scheme),
        ("spiffe://", SpiffeIdError::EmptyTrustDomain),
        ("spiffe:///a", SpiffeIdError::EmptyTrustDomain),
        (
            "spiffe://Example.org/a",
            SpiffeIdError::TrustDomainChar('E'),
        ),
        (
            "spiffe://example.org:8443/a",
            SpiffeIdError::TrustDomainChar(':'),
        ),
        (
            "spiffe://user@example.org/a",
            SpiffeIdError::TrustDomainChar('@'),
        ),
        ("spiffe://example.org/", SpiffeIdError::TrailingSlash),
        ("spiffe://example.org/a/", SpiffeIdError::TrailingSlash),
        ("spiffe://example.org//a", SpiffeIdError::EmptySegment),
        ("spiffe://example.org/./a", SpiffeIdError::DotSegment),
        ("spiffe://example.org/a/..", SpiffeIdError::DotSegment),
        ("spiffe://example.org/a?x=1", SpiffeIdError::PathChar('?')),
        ("spiffe://example.org/a#f", SpiffeIdError::PathChar('#')),
        ("spiffe://example.org/a%20b", SpiffeIdError::PathChar('%')),
        (
            "spiffe://example.org?x",
            SpiffeIdError::TrustDomainChar('?'),
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(
            SpiffeId::parse(input).as_ref().map(SpiffeId::as_str),
            Err(expected),
            "{input:?}"
        );
    }
}

/// The length limits of the standard: a trust domain of at most 255 bytes
/// and a whole ID of at most 2048.
#[test]
fn ids_past_the_length_limits_are_refused() {
    let domain = "a".repeat(255);
    SpiffeId::parse(&format!("spiffe://{domain}")).expect("255-byte trust domain");
    assert_eq!(
        SpiffeId::parse(&format!("spiffe://{domain}a")).map(|_| ()),
        Err(SpiffeIdError::TrustDomainTooLong)
    );

    let prefix = "spiffe://td/";
    let fits = format!("{prefix}{}", "p".repeat(2048 - prefix.len()));
    SpiffeId::parse(&fits).expect("2048-byte id");
    assert_eq!(
        SpiffeId::parse(&format!("{fits}p")).map(|_| ()),
        Err(SpiffeIdError::TooLong)
    );
}

/// Identity is the whole ID: two members sharing a trust domain, or one ID
/// that is a prefix of another, are different identities.
#[test]
fn distinct_ids_never_alias() {
    let a = SpiffeId::parse("spiffe://td/node/1").expect("a");
    let b = SpiffeId::parse("spiffe://td/node/10").expect("b");
    let c = SpiffeId::parse("spiffe://td/node/1").expect("c");
    let other_domain = SpiffeId::parse("spiffe://td2/node/1").expect("other");
    assert_ne!(a, b);
    assert_eq!(a, c);
    assert_ne!(a, other_domain);
}
