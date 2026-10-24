use std::collections::BTreeMap;
use std::path::Path;

use coordinode_test_fixtures::TestPki;

use super::*;
use crate::config::server::PeerEntry;

/// Settings naming node `n`'s credentials and `pki`'s bundle, written into
/// `dir`.
fn settings(pki: &TestPki, dir: &Path, n: u64) -> NodeIdentitySettings {
    let (cert, key) = pki.write_credentials(dir, "node", &pki.spiffe_id(n));
    let bundle = pki.write_bundle(dir);
    NodeIdentitySettings {
        cert: cert.display().to_string(),
        key: key.display().to_string(),
        trust_bundles: BTreeMap::from([(
            pki.trust_domain().to_owned(),
            bundle.display().to_string(),
        )]),
        deployment: None,
        enroll_from: Vec::new(),
    }
}

/// A node's files make a context proving its SPIFFE ID; renewed files
/// written in place take effect on a reload, while a reload of files that do
/// not fit together is refused and the node keeps what it had.
#[test]
fn a_reload_publishes_renewed_files_and_refuses_broken_ones() {
    let pki = TestPki::new();
    coordinode_wire::tls::install_crypto_provider(pki.crypto_provider())
        .expect("test host provider");
    let dir = tempfile::tempdir().expect("tempdir");
    let identity = NodeIdentity::load(settings(&pki, dir.path(), 1)).expect("load");
    assert_eq!(identity.context().identity(), pki.identity(1));
    let generation = identity.context().generation();

    // Renewed in place, under the same names.
    let _ = pki.write_credentials(dir.path(), "node", &pki.spiffe_id(1));
    identity.reload().expect("a renewal");
    assert!(identity.context().generation() > generation);
    let renewed = identity.context().generation();

    // A key that belongs to another certificate.
    let (_, other_key) = pki.write_credentials(dir.path(), "other", &pki.spiffe_id(1));
    std::fs::copy(other_key, dir.path().join("node.key")).expect("swap the key");
    let error = identity.reload().expect_err("the key does not match");
    assert!(error.contains("does not match"), "{error}");
    assert_eq!(identity.context().generation(), renewed, "nothing changed");

    // Another identity is a new enrollment, not a renewal.
    let _ = pki.write_credentials(dir.path(), "node", &pki.spiffe_id(2));
    assert!(identity.reload().is_err());
    assert_eq!(identity.context().identity(), pki.identity(1));
}

/// A joining node needs the deployment it joins and a member to learn the
/// membership from, both well formed.
#[test]
fn a_join_needs_the_deployment_and_a_member_to_enroll_from() {
    let pki = TestPki::new();
    coordinode_wire::tls::install_crypto_provider(pki.crypto_provider())
        .expect("test host provider");
    let dir = tempfile::tempdir().expect("tempdir");
    let mut joiner = settings(&pki, dir.path(), 2);
    let load = |s: &NodeIdentitySettings| NodeIdentity::load(s.clone()).expect("load");

    assert!(load(&joiner).enrollment().is_err(), "no deployment");
    joiner.deployment = Some("not-hex".to_owned());
    assert!(load(&joiner).deployment().is_err());

    let deployment = "00112233445566778899aabbccddeeff";
    joiner.deployment = Some(deployment.to_owned());
    assert!(
        load(&joiner).enrollment().is_err(),
        "no member to enroll from"
    );

    joiner.enroll_from = vec![PeerEntry {
        addr: "db1:7080".to_owned(),
        identity: "not a SPIFFE ID".to_owned(),
    }];
    assert!(load(&joiner).enrollment().is_err());

    joiner.enroll_from = vec![PeerEntry {
        addr: "db1:7080".to_owned(),
        identity: pki.spiffe_id(1),
    }];
    let enrollment = load(&joiner).enrollment().expect("enrollment");
    assert_eq!(enrollment.deployment.to_string(), deployment);
    assert_eq!(enrollment.authorities, vec![pki.target(1, "db1:7080")]);
}

/// A certificate no configured bundle issues is not a node identity, and a
/// node without any bundle could verify no peer: both refuse to start.
#[test]
fn credentials_outside_the_bundles_are_refused() {
    let pki = TestPki::new();
    coordinode_wire::tls::install_crypto_provider(pki.crypto_provider())
        .expect("test host provider");
    let stranger = TestPki::with_trust_domain("elsewhere.test");
    let dir = tempfile::tempdir().expect("tempdir");
    let mut foreign = settings(&pki, dir.path(), 1);
    let (cert, key) = stranger.write_credentials(dir.path(), "foreign", &pki.spiffe_id(1));
    foreign.cert = cert.display().to_string();
    foreign.key = key.display().to_string();
    assert!(NodeIdentity::load(foreign).is_err());

    let mut unbundled = settings(&pki, dir.path(), 1);
    unbundled.trust_bundles.clear();
    assert!(NodeIdentity::load(unbundled).is_err());
}

/// Identity loading never chooses a backend, and succeeds after host selection.
#[test]
fn identity_loading_requires_explicit_host_selection() {
    let pki = TestPki::new();
    let dir = tempfile::tempdir().expect("tempdir");
    let configured = settings(&pki, dir.path(), 1);
    let error = NodeIdentity::load(configured.clone())
        .err()
        .expect("host selection required");
    assert!(error.contains("host must select"), "{error}");
    assert!(coordinode_wire::tls::CryptoProvider::get_default().is_none());
    coordinode_wire::tls::install_crypto_provider(pki.crypto_provider())
        .expect("test host provider");
    let identity = NodeIdentity::load(configured).expect("explicitly configured host");
    assert_eq!(identity.context().identity(), pki.identity(1));
}
