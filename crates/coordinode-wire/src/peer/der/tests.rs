#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use rcgen::{CertificateParams, IsCa, KeyPair, KeyUsagePurpose, SanType};

fn uri(s: &str) -> SanType {
    SanType::URI(s.try_into().expect("ia5 uri"))
}

fn cert_der(params: CertificateParams) -> Vec<u8> {
    let key = KeyPair::generate().expect("key");
    params.self_signed(&key).expect("sign").der().to_vec()
}

/// A leaf's URIs are read in order, beside other name types, and its
/// keyUsage bits are what it was issued with.
#[test]
fn uris_and_key_usage_of_a_leaf_are_read() {
    let mut params = CertificateParams::new(vec!["node.example".to_string()]).expect("params");
    params.subject_alt_names.push(uri("spiffe://td/a"));
    params.subject_alt_names.push(uri("spiffe://td/b"));
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    let fields = leaf_fields(&cert_der(params)).expect("fields");
    assert_eq!(
        fields.uris,
        vec![b"spiffe://td/a".to_vec(), b"spiffe://td/b".to_vec()]
    );
    assert_eq!(
        fields.key_usage,
        Some(KeyUsage {
            digital_signature: true,
            key_cert_sign: false,
            crl_sign: false,
        })
    );
}

/// A signing certificate's keyCertSign and cRLSign bits are seen, and a
/// certificate with no keyUsage reports none rather than an empty set.
#[test]
fn ca_bits_and_absent_key_usage_are_distinguished() {
    let mut ca = CertificateParams::new(Vec::<String>::new()).expect("params");
    ca.is_ca = IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    ca.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    let fields = leaf_fields(&cert_der(ca)).expect("fields");
    let usage = fields.key_usage.expect("key usage present");
    assert!(usage.key_cert_sign && usage.crl_sign && !usage.digital_signature);

    let plain = CertificateParams::new(vec!["x.example".to_string()]).expect("params");
    let fields = leaf_fields(&cert_der(plain)).expect("fields");
    assert_eq!(fields.key_usage, None);
    assert!(fields.uris.is_empty(), "a DNS name is not a URI");
}

/// The end of validity is read in both time forms a certificate uses:
/// UTCTime before 2050 and GeneralizedTime from 2050 on.
#[test]
fn not_after_is_read_in_both_time_forms() {
    let mut params = CertificateParams::new(vec!["x.example".to_string()]).expect("params");
    params.not_after = rcgen::date_time_ymd(2030, 5, 6);
    assert_eq!(
        leaf_fields(&cert_der(params)).expect("fields").not_after,
        1_904_256_000
    );
    let mut params = CertificateParams::new(vec!["x.example".to_string()]).expect("params");
    params.not_after = rcgen::date_time_ymd(2060, 1, 1);
    assert_eq!(
        leaf_fields(&cert_der(params)).expect("fields").not_after,
        2_840_140_800
    );
}

/// UTCTime years 50-99 are the 1900s and 00-49 the 2000s; times that are
/// not UTC, not a real date or in another form are refused.
#[test]
fn certificate_times_follow_rfc_5280() {
    assert_eq!(parse_time(UTC_TIME, b"491231235959Z"), Ok(2_524_607_999));
    assert_eq!(
        parse_time(UTC_TIME, b"500101000000Z"),
        Err(DerError("certificate time out of range"))
    );
    assert_eq!(
        parse_time(UTC_TIME, b"300101000000+"),
        Err(DerError("certificate time not in UTC"))
    );
    assert_eq!(
        parse_time(UTC_TIME, b"301301000000Z"),
        Err(DerError("certificate time out of range"))
    );
    assert_eq!(
        parse_time(GENERALIZED_TIME, b"20300230000000Z"),
        Err(DerError("certificate time out of range"))
    );
    assert_eq!(
        parse_time(GENERALIZED_TIME, b"2030010100000Z"),
        Err(DerError("certificate time in an unsupported form"))
    );
    assert_eq!(
        parse_time(UTC_TIME, b"3001010000a0Z"),
        Err(DerError("non-digit in a certificate time"))
    );
}

/// A URI that is not valid text still counts as a URI, so a certificate
/// cannot hide a second identity from the exactly-one rule.
#[test]
fn a_non_utf8_uri_is_still_counted() {
    // GeneralNames: one [6] URI holding 0xFF.
    let names = [0x30, 0x03, 0x86, 0x01, 0xFF];
    assert_eq!(san_uris(&names).expect("uris"), vec![vec![0xFF]]);
}

/// Encodings DER forbids are refused with their reason rather than read
/// as something else.
#[test]
fn malformed_encodings_are_refused() {
    let cases: &[(&[u8], &str)] = &[
        (&[], "truncated tag"),
        (&[0x30], "truncated length"),
        (&[0x30, 0x80, 0x00, 0x00], "indefinite length"),
        (&[0x30, 0x05, 0x00], "content runs past the end"),
        (&[0x30, 0x81, 0x01, 0x00], "long form for a short length"),
        (&[0x30, 0x82, 0x00, 0x81], "length with a leading zero byte"),
        (
            &[0x30, 0x89, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "length does not fit",
        ),
        (&[0x1F, 0x00], "multi-byte tag"),
    ];
    for (bytes, reason) in cases {
        assert_eq!(
            Reader::new(bytes).next_tlv().map(|_| ()),
            Err(DerError(reason)),
            "{bytes:02x?}"
        );
    }
    // A certificate followed by anything else is not one certificate.
    let mut der = cert_der(CertificateParams::new(vec!["x.example".to_string()]).expect("p"));
    der.push(0);
    assert_eq!(leaf_fields(&der), Err(DerError("trailing bytes")));
}

/// A keyUsage bit string whose unused-bit count is impossible is refused.
#[test]
fn a_bad_key_usage_bit_string_is_refused() {
    assert_eq!(
        key_usage(&[0x03, 0x02, 0x08, 0x80]),
        Err(DerError("keyUsage bit string has a bad unused-bit count"))
    );
    assert_eq!(
        key_usage(&[0x03, 0x01, 0x01]),
        Err(DerError("keyUsage bit string has a bad unused-bit count"))
    );
    assert_eq!(
        key_usage(&[0x03, 0x00]),
        Err(DerError("empty keyUsage bit string"))
    );
}

/// An extension that appears twice is refused (RFC 5280 §4.2), so two
/// subjectAltName extensions cannot carry two identities.
#[test]
fn a_repeated_subject_alt_name_is_refused() {
    let mut params = CertificateParams::new(Vec::<String>::new()).expect("params");
    params.subject_alt_names.push(uri("spiffe://td/a"));
    let der = cert_der(params);
    // Duplicate the subjectAltName extension inside the extensions list.
    let fields = leaf_fields(&der).expect("one san");
    assert_eq!(fields.uris.len(), 1);
    let doubled = double_extension(&der, OID_SUBJECT_ALT_NAME);
    assert_eq!(
        leaf_fields(&doubled),
        Err(DerError("subjectAltName appears twice"))
    );
}

/// Re-encode `der` with the extension `oid` appearing twice. Only the
/// lengths of the enclosing SEQUENCEs change; the signature is left stale,
/// which the field reader does not check.
fn double_extension(der: &[u8], oid: &[u8]) -> Vec<u8> {
    let mut cert = Reader::new(der);
    let mut body = cert.expect(SEQUENCE).expect("cert");
    let (tbs_tag, tbs) = body.next_tlv().expect("tbs");
    let tail = body.rest();
    let mut fields = Reader::new(tbs);
    let mut before = Vec::new();
    let mut extension_list = Vec::new();
    while !fields.is_empty() {
        let start = fields.rest;
        let (tag, content) = fields.next_tlv().expect("field");
        let whole = &start[..start.len() - fields.rest.len()];
        if tag == EXTENSIONS {
            let mut list = Reader::new(content).expect(SEQUENCE).expect("list");
            while !list.is_empty() {
                let start = list.rest;
                let (_, ext) = list.next_tlv().expect("ext");
                let whole = &start[..start.len() - list.rest.len()];
                extension_list.extend_from_slice(whole);
                let mut ext_reader = Reader::new(ext);
                if ext_reader.expect(OID).expect("oid").rest() == oid {
                    extension_list.extend_from_slice(whole);
                }
            }
        } else {
            before.extend_from_slice(whole);
        }
    }
    let extensions = tlv(EXTENSIONS, &tlv(SEQUENCE, &extension_list));
    let tbs = tlv(tbs_tag, &[before, extensions].concat());
    tlv(SEQUENCE, &[tbs, tail.to_vec()].concat())
}

fn tlv(tag: u8, content: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let len = content.len();
    if len < 0x80 {
        out.push(len as u8);
    } else {
        let bytes: Vec<u8> = len
            .to_be_bytes()
            .into_iter()
            .skip_while(|b| *b == 0)
            .collect();
        out.push(0x80 | bytes.len() as u8);
        out.extend(bytes);
    }
    out.extend_from_slice(content);
    out
}
