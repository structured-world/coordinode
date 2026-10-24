//! Node credentials for tests that open members of a group: one CA, a
//! trust domain, and an X.509-SVID per node.
//!
//! Node `n` is `spiffe://<trust domain>/node/<n>`, so a test names its
//! members by number. Every call issues fresh credentials for the same
//! identity, which is what a restart with renewed certificates looks like.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use coordinode_wire::peer::{
    HostIdentity, HostIdentityAdapter, PeerConnections, PeerIdentity, PeerProfile, PeerTarget,
    SecurityContext, SpiffeId, TrustAnchors,
};
use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose, SanType,
};
use rustls_pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// A certificate chain, leaf first, and its key.
pub type Credentials = (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>);

/// A test CA issuing node identities in one trust domain.
pub struct TestPki {
    provider: Arc<coordinode_wire::tls::CryptoProvider>,
    trust_domain: String,
    anchor: CertificateDer<'static>,
    anchor_pem: String,
    issuer: Issuer<'static, KeyPair>,
}

impl Default for TestPki {
    fn default() -> Self {
        Self::new()
    }
}

impl TestPki {
    /// A CA for the trust domain `cluster.test`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_trust_domain("cluster.test")
    }

    /// A CA for `trust_domain`. Two of them share no trust.
    ///
    /// # Panics
    ///
    /// The CA cannot be generated.
    #[must_use]
    pub fn with_trust_domain(trust_domain: &str) -> Self {
        let key = KeyPair::generate().expect("CA key");
        let mut params = CertificateParams::new(Vec::<String>::new()).expect("CA params");
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        let ca = params.self_signed(&key).expect("CA");
        Self {
            provider: coordinode_wire::tls::rustcrypto_provider(),
            trust_domain: trust_domain.to_owned(),
            anchor: ca.der().clone(),
            anchor_pem: ca.pem(),
            issuer: Issuer::new(params, key),
        }
    }

    /// The backend explicitly owned by this test host and its security contexts.
    #[must_use]
    pub fn crypto_provider(&self) -> Arc<coordinode_wire::tls::CryptoProvider> {
        Arc::clone(&self.provider)
    }

    /// Write the CA certificate as a PEM trust bundle into `dir`; returns
    /// its path.
    ///
    /// # Panics
    ///
    /// The file cannot be written.
    #[must_use]
    pub fn write_bundle(&self, dir: &Path) -> PathBuf {
        let path = dir.join(format!("{}-bundle.pem", self.trust_domain));
        std::fs::write(&path, &self.anchor_pem).expect("write the trust bundle");
        path
    }

    /// Issue fresh credentials for the URI SAN `uri` and write them as PEM
    /// into `dir`, under `name`; returns the certificate and key paths.
    ///
    /// # Panics
    ///
    /// The certificate cannot be issued or the files cannot be written.
    #[must_use]
    pub fn write_credentials(&self, dir: &Path, name: &str, uri: &str) -> (PathBuf, PathBuf) {
        self.write_credentials_until(dir, name, uri, None)
    }

    /// [`Self::write_credentials`], valid until `not_after` when given; the
    /// files replace any of the same name, as a renewal does.
    ///
    /// # Panics
    ///
    /// The certificate cannot be issued or the files cannot be written.
    #[must_use]
    pub fn write_credentials_until(
        &self,
        dir: &Path,
        name: &str,
        uri: &str,
        not_after: Option<SystemTime>,
    ) -> (PathBuf, PathBuf) {
        let key = KeyPair::generate().expect("node key");
        let cert = self
            .leaf_params(uri, not_after)
            .signed_by(&key, &self.issuer)
            .expect("node certificate");
        let cert_path = dir.join(format!("{name}.crt"));
        let key_path = dir.join(format!("{name}.key"));
        std::fs::write(&cert_path, cert.pem()).expect("write the certificate");
        std::fs::write(&key_path, key.serialize_pem()).expect("write the key");
        (cert_path, key_path)
    }

    /// The parameters of a node certificate for the URI SAN `uri`.
    fn leaf_params(&self, uri: &str, not_after: Option<SystemTime>) -> CertificateParams {
        let mut params = CertificateParams::new(Vec::<String>::new()).expect("node params");
        params.subject_alt_names = vec![SanType::URI(uri.try_into().expect("a URI"))];
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        params.extended_key_usages = vec![
            ExtendedKeyUsagePurpose::ClientAuth,
            ExtendedKeyUsagePurpose::ServerAuth,
        ];
        if let Some(not_after) = not_after {
            params.not_after = not_after.into();
        }
        params
    }

    /// The trust domain.
    #[must_use]
    pub fn trust_domain(&self) -> &str {
        &self.trust_domain
    }

    /// The CA certificate.
    #[must_use]
    pub fn anchor(&self) -> &CertificateDer<'static> {
        &self.anchor
    }

    /// The SPIFFE ID of node `node_id`.
    #[must_use]
    pub fn spiffe_id(&self, node_id: u64) -> String {
        format!("spiffe://{}/node/{node_id}", self.trust_domain)
    }

    /// The identity of node `node_id`.
    ///
    /// # Panics
    ///
    /// The trust domain is not a valid SPIFFE trust domain.
    #[must_use]
    pub fn identity(&self, node_id: u64) -> PeerIdentity {
        PeerIdentity::Spiffe(SpiffeId::parse(&self.spiffe_id(node_id)).expect("a SPIFFE ID"))
    }

    /// The SPIFFE profile trusting this CA for its trust domain alone.
    ///
    /// # Panics
    ///
    /// The CA certificate is not usable as an anchor.
    #[must_use]
    pub fn profile(&self) -> PeerProfile {
        self.profile_with(&[])
    }

    /// The SPIFFE profile trusting this CA and every one of `others`, each
    /// for its own trust domain.
    ///
    /// # Panics
    ///
    /// A CA certificate is not usable as an anchor.
    #[must_use]
    pub fn profile_with(&self, others: &[&TestPki]) -> PeerProfile {
        let bundles = std::iter::once(self)
            .chain(others.iter().copied())
            .map(|pki| {
                let anchors = TrustAnchors::from_der(std::slice::from_ref(&pki.anchor))
                    .expect("a trust anchor");
                (pki.trust_domain.clone(), anchors)
            })
            .collect();
        PeerProfile::Spiffe { bundles }
    }

    /// Fresh credentials for the URI SAN `uri`, valid until `not_after`
    /// when given.
    ///
    /// # Panics
    ///
    /// The certificate cannot be issued.
    #[must_use]
    pub fn credentials(&self, uri: &str, not_after: Option<SystemTime>) -> Credentials {
        let key = KeyPair::generate().expect("node key");
        let cert = self
            .leaf_params(uri, not_after)
            .signed_by(&key, &self.issuer)
            .expect("node certificate");
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.serialize_der()));
        (vec![cert.der().clone()], key)
    }

    /// Fresh credentials of node `node_id`.
    #[must_use]
    pub fn node_credentials(&self, node_id: u64) -> Credentials {
        self.credentials(&self.spiffe_id(node_id), None)
    }

    /// A security context for node `node_id`, trusting this CA.
    #[must_use]
    pub fn context(&self, node_id: u64) -> Arc<SecurityContext> {
        self.context_of(&self.spiffe_id(node_id))
    }

    /// A security context proving the SPIFFE ID `uri`, trusting this CA: a
    /// node under an identity of its own choosing, such as a member that
    /// returns under a new one.
    ///
    /// # Panics
    ///
    /// The context refuses the credentials.
    #[must_use]
    pub fn context_of(&self, uri: &str) -> Arc<SecurityContext> {
        let (chain, key) = self.credentials(uri, None);
        Arc::new(
            SecurityContext::new(self.crypto_provider(), chain, key, self.profile())
                .expect("a node security context"),
        )
    }

    /// The peer connections of node `node_id`, under [`Self::context`].
    #[must_use]
    pub fn connections(&self, node_id: u64) -> Arc<PeerConnections> {
        Arc::new(PeerConnections::new(self.context(node_id)))
    }

    /// The peer connections of a node proving `uri`.
    #[must_use]
    pub fn connections_of(&self, uri: &str) -> Arc<PeerConnections> {
        Arc::new(PeerConnections::new(self.context_of(uri)))
    }

    /// The identity a node proving `uri` has.
    ///
    /// # Panics
    ///
    /// `uri` is not a SPIFFE ID.
    #[must_use]
    pub fn identity_of(uri: &str) -> PeerIdentity {
        PeerIdentity::Spiffe(SpiffeId::parse(uri).expect("a SPIFFE ID"))
    }

    /// Node `node_id`, found at `addr`.
    #[must_use]
    pub fn target(&self, node_id: u64, addr: impl Into<String>) -> PeerTarget {
        PeerTarget {
            addr: addr.into(),
            identity: self.identity(node_id),
        }
    }

    /// The embedding-host profile trusting this CA, identifying nodes as
    /// `adapter` says.
    ///
    /// # Panics
    ///
    /// The CA certificate is not usable as an anchor.
    #[must_use]
    pub fn host_profile(&self, adapter: Arc<UriAdapter>) -> PeerProfile {
        self.host_profile_with(adapter, &[])
    }

    /// The embedding-host profile trusting this CA and every one of
    /// `others`, identifying nodes as `adapter` says.
    ///
    /// # Panics
    ///
    /// A CA certificate is not usable as an anchor.
    #[must_use]
    pub fn host_profile_with(&self, adapter: Arc<UriAdapter>, others: &[&TestPki]) -> PeerProfile {
        let anchors: Vec<CertificateDer<'static>> = std::iter::once(self)
            .chain(others.iter().copied())
            .map(|pki| pki.anchor.clone())
            .collect();
        PeerProfile::Host {
            anchors: TrustAnchors::from_der(&anchors).expect("trust anchors"),
            adapter,
        }
    }

    /// The connections of node `node_id` under the embedding-host profile
    /// of `adapter`.
    ///
    /// # Panics
    ///
    /// The context refuses the credentials.
    #[must_use]
    pub fn host_connections(
        &self,
        node_id: u64,
        adapter: &Arc<UriAdapter>,
    ) -> Arc<PeerConnections> {
        let (chain, key) = self.node_credentials(node_id);
        let context = SecurityContext::new(
            self.crypto_provider(),
            chain,
            key,
            self.host_profile(Arc::clone(adapter)),
        )
        .expect("a node security context");
        Arc::new(PeerConnections::new(Arc::new(context)))
    }

    /// The identity `adapter` gives node `node_id`.
    #[must_use]
    pub fn host_identity(&self, node_id: u64, adapter: &UriAdapter) -> PeerIdentity {
        PeerIdentity::Host(HostIdentity {
            scope: adapter.scope.clone(),
            subject: self.spiffe_id(node_id),
        })
    }
}

/// Serve `routes` on `addr` the way a server's port is served: TLS made
/// under the member's `context`, telling nodes from clients by the identity
/// they prove, and taking clients in plain text. The port is bound before
/// this returns; the server runs until the task is aborted.
///
/// # Panics
///
/// The port cannot be bound or the TLS configuration built.
pub async fn serve_member_port(
    addr: std::net::SocketAddr,
    context: &Arc<SecurityContext>,
    routes: tonic::service::Routes,
) -> tokio::task::JoinHandle<()> {
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind the member port");
    let incoming = coordinode_wire::peer::incoming(
        listener,
        context,
        coordinode_wire::peer::Accept {
            inbound: coordinode_wire::peer::InboundAuth::Shared { clients: None },
            plaintext_clients: true,
            handshake_timeout: coordinode_wire::peer::PeerTimeouts::default().handshake,
        },
    )
    .expect("the port's TLS configuration");
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_routes(routes)
            .serve_with_incoming(incoming)
            .await
            .expect("serve the member port");
    })
}

/// An embedding host's identity adapter for tests: a verified chain's node
/// is the URI its leaf carries, in the adapter's issuer scope, unless the
/// host revoked that URI.
#[derive(Debug)]
pub struct UriAdapter {
    /// The issuer scope every identity it establishes is in.
    pub scope: String,
    // no-std: spin::Mutex; touched once per handshake in tests.
    revoked: std::sync::Mutex<Vec<String>>,
}

impl UriAdapter {
    /// An adapter for issuer scope `scope`.
    #[must_use]
    pub fn new(scope: &str) -> Arc<Self> {
        Arc::new(Self {
            scope: scope.to_owned(),
            revoked: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Stop admitting the node whose leaf carries `uri`, as a host whose
    /// issuer revoked it does from then on.
    pub fn revoke(&self, uri: &str) {
        self.revoked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(uri.to_owned());
    }
}

impl HostIdentityAdapter for UriAdapter {
    fn identify(
        &self,
        chain: &[CertificateDer<'_>],
        _now: rustls_pki_types::UnixTime,
    ) -> Result<HostIdentity, String> {
        let leaf = chain.first().ok_or("no certificate")?;
        let cert = webpki::EndEntityCert::try_from(leaf).map_err(|e| e.to_string())?;
        let subject = cert
            .valid_uri_names()
            .next()
            .ok_or("the leaf carries no URI")?
            .to_owned();
        let revoked = self
            .revoked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&subject);
        if revoked {
            return Err(format!("{subject} is revoked"));
        }
        Ok(HostIdentity {
            scope: self.scope.clone(),
            subject,
        })
    }
}
