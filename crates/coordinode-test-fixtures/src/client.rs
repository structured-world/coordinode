//! An application client of a member's gRPC port over TLS, presenting a
//! certificate of its own when given one.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use rustls_pki_types::{CertificateDer, ServerName, UnixTime};

use crate::pki::Credentials;

/// A channel to `authority` (`host:port`) over TLS, presenting `identity`
/// when given. The server is not verified: these clients test what the
/// server decides about them.
///
/// # Panics
///
/// The TLS configuration cannot be built or `authority` is not an address.
#[must_use]
pub fn tls_client_channel(
    authority: &str,
    identity: Option<Credentials>,
) -> tonic::transport::Channel {
    let provider = coordinode_wire::tls::rustcrypto_provider();
    let builder = rustls::ClientConfig::builder_with_provider(Arc::clone(&provider))
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AnyServer(provider)));
    let mut config = match identity {
        Some((chain, key)) => builder
            .with_client_auth_cert(chain, key)
            .expect("the client certificate"),
        None => builder.with_no_client_auth(),
    };
    config.alpn_protocols = vec![b"h2".to_vec()];
    tonic::transport::Endpoint::from_shared(format!("http://{authority}"))
        .expect("an address")
        .connect_with_connector_lazy(Connector {
            config: Arc::new(config),
            authority: authority.to_owned(),
        })
}

/// Dials the server: TCP, then TLS.
#[derive(Clone)]
struct Connector {
    config: Arc<rustls::ClientConfig>,
    authority: String,
}

type Connection = hyper_util::rt::TokioIo<tokio_rustls::client::TlsStream<tokio::net::TcpStream>>;

impl tower_service::Service<tonic::transport::Uri> for Connector {
    type Response = Connection;
    type Error = io::Error;
    type Future = Pin<Box<dyn Future<Output = io::Result<Connection>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _uri: tonic::transport::Uri) -> Self::Future {
        let this = self.clone();
        Box::pin(async move {
            let tcp = tokio::net::TcpStream::connect(&this.authority).await?;
            let name = ServerName::try_from("member.invalid")
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            let tls = tokio_rustls::TlsConnector::from(this.config)
                .connect(name, tcp)
                .await?;
            Ok(hyper_util::rt::TokioIo::new(tls))
        })
    }
}

/// Accepts whatever certificate the server presents, checking only that it
/// signed the handshake with it.
#[derive(Debug)]
struct AnyServer(Arc<rustls::crypto::CryptoProvider>);

impl rustls::client::danger::ServerCertVerifier for AnyServer {
    fn verify_server_cert(
        &self,
        _: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}
