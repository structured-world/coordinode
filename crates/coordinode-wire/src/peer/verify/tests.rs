#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Mutex;

use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose, SanType,
};

use super::*;

/// A CA that signs leaves: the anchor of one trust domain.
struct Ca {
    der: CertificateDer<'static>,
    issuer: Issuer<'static, KeyPair>,
}

fn ca() -> Ca {
    let key = KeyPair::generate().expect("key");
    let mut params = CertificateParams::new(Vec::<String>::new()).expect("params");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    let der = params.self_signed(&key).expect("ca").der().clone();
    Ca {
        der,
        issuer: Issuer::new(params, key),
    }
}

/// An SVID-shaped leaf with `uris`, signed by `ca`; `tweak` adjusts it.
fn leaf(
    ca: &Ca,
    uris: &[&str],
    tweak: impl FnOnce(&mut CertificateParams),
) -> CertificateDer<'static> {
    let key = KeyPair::generate().expect("key");
    let mut params = CertificateParams::new(Vec::<String>::new()).expect("params");
    params.subject_alt_names = uris
        .iter()
        .map(|u| SanType::URI((*u).try_into().expect("ia5")))
        .collect();
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ClientAuth,
        ExtendedKeyUsagePurpose::ServerAuth,
    ];
    tweak(&mut params);
    params
        .signed_by(&key, &ca.issuer)
        .expect("leaf")
        .der()
        .clone()
}

fn spiffe(domains: &[(&str, &Ca)]) -> PeerProfile {
    PeerProfile::Spiffe {
        bundles: domains
            .iter()
            .map(|(domain, ca)| {
                let anchors =
                    TrustAnchors::from_der(std::slice::from_ref(&ca.der)).expect("anchor");
                ((*domain).to_owned(), anchors)
            })
            .collect(),
    }
}

fn algorithms() -> &'static [&'static dyn SignatureVerificationAlgorithm] {
    let provider = Box::leak(Box::new(crate::tls::rustcrypto_provider()));
    provider.signature_verification_algorithms.all
}

fn now() -> UnixTime {
    UnixTime::now()
}

fn verify(
    profile: &PeerProfile,
    leaf: &CertificateDer<'_>,
    role: PeerRole,
) -> Result<PeerIdentity, PeerError> {
    profile
        .verify(leaf, &[], now(), role, algorithms())
        .map(|peer| peer.identity)
}

fn id(s: &str) -> PeerIdentity {
    PeerIdentity::Spiffe(SpiffeId::parse(s).expect("id"))
}

/// An X.509-SVID signed by its trust domain's CA establishes its whole
/// SPIFFE ID, as a client and as a server.
#[test]
fn a_valid_svid_establishes_its_id_in_both_roles() {
    let ca = ca();
    let profile = spiffe(&[("td.test", &ca)]);
    let svid = leaf(&ca, &["spiffe://td.test/any/opaque/path"], |_| {});
    for role in [PeerRole::Client, PeerRole::Server] {
        assert_eq!(
            verify(&profile, &svid, role),
            Ok(id("spiffe://td.test/any/opaque/path")),
            "{role:?}"
        );
    }
}

/// The trust domain picks the bundle: an ID in a domain with no bundle is
/// refused, and a leaf signed by another domain's CA is refused even when
/// both CAs are trusted, so a shared CA cannot vouch across domains.
#[test]
fn trust_domains_are_checked_against_their_own_bundle() {
    let a = ca();
    let b = ca();
    let profile = spiffe(&[("a.test", &a), ("b.test", &b)]);
    let foreign = leaf(&a, &["spiffe://c.test/x"], |_| {});
    assert_eq!(
        verify(&profile, &foreign, PeerRole::Client),
        Err(PeerError::UnknownTrustDomain("c.test".to_owned()))
    );
    let cross = leaf(&a, &["spiffe://b.test/x"], |_| {});
    assert!(matches!(
        verify(&profile, &cross, PeerRole::Client),
        Err(PeerError::Chain(_))
    ));
}

/// X.509-SVID shape rules: exactly one URI SAN that is a SPIFFE ID, and
/// a key usage that allows signing but not certificate or CRL signing.
#[test]
fn leaves_that_are_not_svids_are_refused_with_their_reason() {
    let ca = ca();
    let profile = spiffe(&[("td.test", &ca)]);
    let check = |l: CertificateDer<'_>| verify(&profile, &l, PeerRole::Client);

    assert_eq!(check(leaf(&ca, &[], |_| {})), Err(PeerError::UriCount(0)));
    assert_eq!(
        check(leaf(
            &ca,
            &["spiffe://td.test/a", "spiffe://td.test/b"],
            |_| {}
        )),
        Err(PeerError::UriCount(2))
    );
    assert_eq!(
        check(leaf(&ca, &["https://td.test/a"], |_| {})),
        Err(PeerError::NotSpiffe(SpiffeIdError::Scheme))
    );
    assert_eq!(
        check(leaf(&ca, &["spiffe://td.test/a"], |p| p.key_usages.clear())),
        Err(PeerError::KeyUsage("keyUsage is missing"))
    );
    assert_eq!(
        check(leaf(&ca, &["spiffe://td.test/a"], |p| {
            p.key_usages = vec![KeyUsagePurpose::KeyEncipherment];
        })),
        Err(PeerError::KeyUsage("digitalSignature is not set"))
    );
    assert_eq!(
        check(leaf(&ca, &["spiffe://td.test/a"], |p| {
            p.key_usages
                .extend([KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign]);
        })),
        Err(PeerError::KeyUsage("keyCertSign or cRLSign is set"))
    );
}

/// The chain itself decides: a CA certificate is no leaf, an expired leaf
/// is refused, and a leaf limited to server use is refused as a client.
#[test]
fn chain_rules_are_enforced_by_the_verifier() {
    let ca = ca();
    let profile = spiffe(&[("td.test", &ca)]);
    let is_ca = leaf(&ca, &["spiffe://td.test/a"], |p| {
        p.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    });
    assert!(matches!(
        verify(&profile, &is_ca, PeerRole::Client),
        Err(PeerError::Chain(_))
    ));
    let expired = leaf(&ca, &["spiffe://td.test/a"], |p| {
        p.not_before = rcgen::date_time_ymd(2000, 1, 1);
        p.not_after = rcgen::date_time_ymd(2001, 1, 1);
    });
    assert!(matches!(
        verify(&profile, &expired, PeerRole::Client),
        Err(PeerError::Chain(_))
    ));
    let server_only = leaf(&ca, &["spiffe://td.test/a"], |p| {
        p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    });
    assert!(matches!(
        verify(&profile, &server_only, PeerRole::Client),
        Err(PeerError::Chain(_))
    ));
    assert_eq!(
        verify(&profile, &server_only, PeerRole::Server),
        Ok(id("spiffe://td.test/a"))
    );
}

/// The validity end of the leaf travels with the identity, so
/// authorization founded on it can end when the certificate does.
#[test]
fn the_leaf_validity_end_travels_with_the_identity() {
    let ca = ca();
    let profile = spiffe(&[("td.test", &ca)]);
    let svid = leaf(&ca, &["spiffe://td.test/a"], |p| {
        p.not_after = rcgen::date_time_ymd(2030, 5, 6);
    });
    let peer = profile
        .verify(&svid, &[], now(), PeerRole::Client, algorithms())
        .expect("verified");
    assert_eq!(peer.not_after, 1_904_256_000);
}

/// A recording host adapter.
#[derive(Debug, Default)]
struct Recorder {
    calls: Mutex<usize>,
    refuse: bool,
}

impl HostIdentityAdapter for Recorder {
    fn identify(
        &self,
        chain: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<HostIdentity, String> {
        *self.calls.lock().expect("lock") += 1;
        if self.refuse {
            return Err("lease token below the highest seen".to_owned());
        }
        Ok(HostIdentity {
            scope: "instance-1".to_owned(),
            subject: format!("part-{}", chain.len()),
        })
    }
}

fn host(ca: &Ca, adapter: Arc<Recorder>) -> PeerProfile {
    PeerProfile::Host {
        anchors: TrustAnchors::from_der(std::slice::from_ref(&ca.der)).expect("anchor"),
        adapter,
    }
}

/// The host profile interprets only chains that verified against its
/// anchors: its adapter is never asked about a chain from elsewhere, and
/// its refusal stands.
#[test]
fn the_host_adapter_sees_only_verified_chains() {
    let trusted = ca();
    let stranger = ca();
    let recorder = Arc::new(Recorder::default());
    let profile = host(&trusted, Arc::clone(&recorder));

    let ok = leaf(&trusted, &[], |_| {});
    assert_eq!(
        verify(&profile, &ok, PeerRole::Client),
        Ok(PeerIdentity::Host(HostIdentity {
            scope: "instance-1".to_owned(),
            subject: "part-1".to_owned(),
        }))
    );
    assert_eq!(*recorder.calls.lock().expect("lock"), 1);

    let foreign = leaf(&stranger, &[], |_| {});
    assert!(matches!(
        verify(&profile, &foreign, PeerRole::Client),
        Err(PeerError::Chain(_))
    ));
    assert_eq!(
        *recorder.calls.lock().expect("lock"),
        1,
        "the adapter never saw the unverified chain"
    );

    let refusing = host(
        &trusted,
        Arc::new(Recorder {
            refuse: true,
            ..Recorder::default()
        }),
    );
    assert_eq!(
        verify(&refusing, &ok, PeerRole::Client),
        Err(PeerError::Host(
            "lease token below the highest seen".to_owned()
        ))
    );
}

/// On a port shared with clients, a certificate naming a SPIFFE ID of a
/// trusted trust domain (in any letter case of the scheme), or one that does
/// not parse, is a node's and is verified as one; an ordinary client
/// certificate, and a workload identity of a trust domain the node has no
/// bundle for, are not claimed.
#[test]
fn node_certificates_are_told_from_client_certificates() {
    let ca = ca();
    let profile = spiffe(&[("td.test", &ca)]);
    let client = {
        let key = KeyPair::generate().expect("key");
        CertificateParams::new(vec!["client.example".to_string()])
            .expect("params")
            .signed_by(&key, &ca.issuer)
            .expect("client")
            .der()
            .clone()
    };
    assert!(!profile.claims_node_identity(&client));
    assert!(profile.claims_node_identity(&leaf(&ca, &["spiffe://td.test/a"], |_| {})));
    assert!(profile.claims_node_identity(&leaf(&ca, &["SPIFFE://td.test/a"], |_| {})));
    assert!(!profile.claims_node_identity(&leaf(&ca, &["spiffe://apps.test/web"], |_| {})));
    assert!(profile.claims_node_identity(&leaf(&ca, &["spiffe://TD.test/a"], |_| {})));
    assert!(profile.claims_node_identity(&leaf(
        &ca,
        &["spiffe://apps.test/web", "spiffe://td.test/a"],
        |_| {}
    )));
}
