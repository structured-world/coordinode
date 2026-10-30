//! Connections between nodes under one database's security context.
//!
//! Outbound, [`PeerConnections`] dials a peer expecting one identity: the
//! address is where to find it, the identity is who must answer. Inbound,
//! [`incoming`] accepts connections on a listener and hands each to the
//! server with the node identity it proved, if any, as its
//! [`PeerConnectInfo`]. Either way the identity stays bound to the live
//! connection: once the trust it was verified under changes or its
//! certificate expires, it is verified again, and a connection that no
//! longer verifies stops carrying traffic.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tonic::codegen::http;

use super::context::{InboundAuth, SecurityContext};
use super::verify::{PeerError, PeerIdentity, PeerRole, VerifiedPeer};

/// A peer to dial: where to find it and who it must prove to be.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeerTarget {
    /// Where the peer listens, `host:port`; a scheme in front is ignored.
    pub addr: String,
    /// The identity the peer must prove.
    pub identity: PeerIdentity,
}

/// The node identity a connection's peer proved, kept current for the life
/// of the connection.
pub struct AuthenticatedPeer {
    context: Arc<SecurityContext>,
    chain: Vec<CertificateDer<'static>>,
    role: PeerRole,
    standing: ArcSwap<Standing>,
}

/// The outcome of the last verification and what it holds for.
struct Standing {
    /// The context's trust generation it was verified under.
    generation: u64,
    verdict: Result<VerifiedPeer, PeerError>,
    /// When the certificate stops being valid; the verdict is retaken then.
    until: Option<Instant>,
}

impl fmt::Debug for AuthenticatedPeer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let standing = self.standing.load();
        f.debug_struct("AuthenticatedPeer")
            .field("role", &self.role)
            .field("verdict", &standing.verdict)
            .finish_non_exhaustive()
    }
}

impl AuthenticatedPeer {
    /// The peer that presented `chain` (leaf first) acting as `role`,
    /// verified under `context` now.
    ///
    /// # Errors
    ///
    /// The chain establishes no node identity.
    pub fn establish(
        context: Arc<SecurityContext>,
        chain: Vec<CertificateDer<'static>>,
        role: PeerRole,
    ) -> Result<Arc<Self>, PeerError> {
        let standing = verify(&context, &chain, role);
        standing.verdict.as_ref().map_err(Clone::clone)?;
        Ok(Arc::new(Self {
            context,
            chain,
            role,
            standing: ArcSwap::from_pointee(standing),
        }))
    }

    /// The peer's identity as of now: verified again when the trust it was
    /// verified under has changed since, or its certificate has expired.
    ///
    /// # Errors
    ///
    /// The chain no longer establishes a node identity.
    pub fn identity(&self) -> Result<PeerIdentity, PeerError> {
        self.current().map(|peer| peer.identity)
    }

    /// The verified peer as of now; see [`Self::identity`].
    ///
    /// # Errors
    ///
    /// The chain no longer establishes a node identity.
    pub fn current(&self) -> Result<VerifiedPeer, PeerError> {
        let standing = self.standing.load();
        if standing.generation == self.context.generation()
            && standing.until.is_none_or(|until| Instant::now() < until)
        {
            return standing.verdict.clone();
        }
        let renewed = verify(&self.context, &self.chain, self.role);
        let verdict = renewed.verdict.clone();
        self.standing.store(Arc::new(renewed));
        verdict
    }
}

/// Verify `chain` under the context's current trust, now.
fn verify(
    context: &SecurityContext,
    chain: &[CertificateDer<'static>],
    role: PeerRole,
) -> Standing {
    let generation = context.generation();
    let verdict = match chain.split_first() {
        Some((leaf, intermediates)) => {
            context.verify_peer(leaf, intermediates, UnixTime::now(), role)
        }
        None => Err(PeerError::Chain("no certificate".to_owned())),
    };
    // A deadline past what the clock can represent is none: the trust
    // generation still rechecks it.
    let until = verdict.as_ref().ok().and_then(|peer| {
        // notAfter is the last valid second (RFC 5280 4.1.2.5): invalid
        // from the start of the next one on, measured from the exact clock
        // rather than its whole second.
        let invalid_from =
            UNIX_EPOCH.checked_add(Duration::from_secs(peer.not_after.checked_add(1)?))?;
        let left = invalid_from
            .duration_since(SystemTime::now())
            .unwrap_or_default();
        Instant::now().checked_add(left)
    });
    Standing {
        generation,
        verdict,
        until,
    }
}

/// A stream that carries traffic only while its peer still verifies.
pub struct Guarded<S> {
    inner: S,
    peer: Arc<AuthenticatedPeer>,
}

impl<S> Guarded<S> {
    fn admit(&self) -> io::Result<()> {
        self.peer
            .current()
            .map(|_| ())
            .map_err(|e| io::Error::new(io::ErrorKind::PermissionDenied, e))
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for Guarded<S> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        self.admit()?;
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for Guarded<S> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.admit()?;
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        self.admit()?;
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

/// Why a peer could not be dialed.
#[derive(Debug, thiserror::Error)]
pub enum DialError {
    /// The address is not `host:port`.
    #[error("peer address '{addr}': {reason}")]
    Address {
        /// The address given.
        addr: String,
        /// What is wrong with it.
        reason: String,
    },
    /// The provider offers no TLS protocol version.
    #[error("peer TLS configuration: {0}")]
    Tls(#[from] rustls::Error),
}

/// The bounds of the inter-node channel; what overruns one fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerTimeouts {
    /// A dial: the TCP connection and the TLS handshake (5 s by default).
    pub connect: Duration,
    /// One call to a peer (30 s by default).
    pub call: Duration,
    /// The TLS handshake of an accepted connection (10 s by default).
    pub handshake: Duration,
}

impl Default for PeerTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(5),
            call: Duration::from_secs(30),
            handshake: Duration::from_secs(10),
        }
    }
}

/// One database's connections to its peers, one per peer identity, made
/// under its security context. Every service the database calls a peer on
/// shares the connection to that peer.
pub struct PeerConnections {
    context: Arc<SecurityContext>,
    timeouts: PeerTimeouts,
    // no-std: spin::Mutex; taken once per new client of a peer, never per call.
    channels: parking_lot::Mutex<HashMap<PeerIdentity, (String, tonic::transport::Channel)>>,
}

impl fmt::Debug for PeerConnections {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PeerConnections")
            .field("context", &self.context)
            .field("peers", &self.channels.lock().len())
            .finish()
    }
}

impl PeerConnections {
    /// Connections made under `context`, with the default timeouts.
    #[must_use]
    pub fn new(context: Arc<SecurityContext>) -> Self {
        Self {
            context,
            timeouts: PeerTimeouts::default(),
            channels: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// The same connections bounded by `timeouts`.
    #[must_use]
    pub fn with_timeouts(mut self, timeouts: PeerTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// The security context the connections are made under.
    #[must_use]
    pub fn context(&self) -> &Arc<SecurityContext> {
        &self.context
    }

    /// The bounds the connections are made with.
    #[must_use]
    pub fn timeouts(&self) -> PeerTimeouts {
        self.timeouts
    }

    /// The connection to `target`, made on first use and kept: it connects
    /// on the first call and reconnects by itself after a drop. A peer
    /// found at a new address gets a new connection there.
    ///
    /// # Errors
    ///
    /// The address is not `host:port`, or no TLS configuration can be built.
    pub fn channel(&self, target: &PeerTarget) -> Result<tonic::transport::Channel, DialError> {
        let mut channels = self.channels.lock();
        if let Some((addr, channel)) = channels.get(&target.identity)
            && *addr == target.addr
        {
            return Ok(channel.clone());
        }
        let channel = self.dial(target)?;
        channels.insert(
            target.identity.clone(),
            (target.addr.clone(), channel.clone()),
        );
        Ok(channel)
    }

    /// How many peers a connection is kept to.
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.lock().len()
    }

    /// Whether no connection is kept yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A new lazily connecting channel to `target`.
    fn dial(&self, target: &PeerTarget) -> Result<tonic::transport::Channel, DialError> {
        let authority = authority(&target.addr)?;
        let config = Arc::new(self.context.client_config(target.identity.clone())?);
        // The connector makes the TLS connection itself, so the endpoint
        // names plain HTTP/2: tonic would otherwise try TLS of its own.
        let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{authority}"))
            .map_err(|e| DialError::Address {
                addr: target.addr.clone(),
                reason: e.to_string(),
            })?
            .connect_timeout(self.timeouts.connect)
            .timeout(self.timeouts.call);
        Ok(endpoint.connect_with_connector_lazy(Connector {
            context: Arc::clone(&self.context),
            config,
            authority,
        }))
    }
}

/// The `host:port` of `addr`, without a scheme or a path.
fn authority(addr: &str) -> Result<String, DialError> {
    let bad = |reason: &str| DialError::Address {
        addr: addr.to_owned(),
        reason: reason.to_owned(),
    };
    let rest = addr.split_once("://").map_or(addr, |(_, rest)| rest);
    let authority = rest.split('/').next().unwrap_or_default();
    let uri: http::uri::Authority = authority.parse().map_err(|_| bad("not host:port"))?;
    if uri.port_u16().is_none() {
        return Err(bad("no port"));
    }
    Ok(uri.as_str().to_owned())
}

/// Dials one peer: TCP, then TLS expecting the peer's identity.
#[derive(Clone)]
struct Connector {
    context: Arc<SecurityContext>,
    config: Arc<rustls::ClientConfig>,
    authority: String,
}

/// An outbound connection as hyper reads and writes it.
type Outbound = hyper_util::rt::TokioIo<Guarded<tokio_rustls::client::TlsStream<TcpStream>>>;

impl tower_service::Service<http::Uri> for Connector {
    type Response = Outbound;
    type Error = io::Error;
    type Future = Pin<Box<dyn Future<Output = io::Result<Outbound>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _uri: http::Uri) -> Self::Future {
        let this = self.clone();
        Box::pin(async move {
            connect_tls(this.context, this.config, &this.authority)
                .await
                .map(hyper_util::rt::TokioIo::new)
        })
    }
}

/// Connect to `authority` and complete TLS under `config`, which accepts
/// only the expected peer; the connection is guarded by that peer's
/// standing from then on.
async fn connect_tls(
    context: Arc<SecurityContext>,
    config: Arc<rustls::ClientConfig>,
    authority: &str,
) -> io::Result<Guarded<tokio_rustls::client::TlsStream<TcpStream>>> {
    let tcp = TcpStream::connect(authority).await?;
    tcp.set_nodelay(true)?;
    // The name is a hint for a listener that serves several names; the
    // identity check does not read it.
    let host = authority
        .rsplit_once(':')
        .map_or(authority, |(host, _)| host)
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_owned();
    let name = ServerName::try_from(host)
        .or_else(|_| ServerName::try_from("peer.invalid"))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let tls = tokio_rustls::TlsConnector::from(config)
        .connect(name, tcp)
        .await?;
    let chain = tls
        .get_ref()
        .1
        .peer_certificates()
        .map(<[_]>::to_vec)
        .unwrap_or_default();
    let peer = AuthenticatedPeer::establish(context, chain, PeerRole::Server)
        .map_err(|e| io::Error::new(io::ErrorKind::PermissionDenied, e))?;
    Ok(Guarded { inner: tls, peer })
}

/// Admission to the peer services of one group's member beyond consensus
/// (segment transfer, write routing): who may call them, and what a call to
/// another member carries so that member admits it.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot admit peers to a member's services",
    label = "not a peer admission",
    note = "the consensus layer's admission of a member implements it"
)]
pub trait PeerAdmission: Send + Sync {
    /// Admit the caller of a peer service on the connection `info`
    /// describes: a node whose identity is bound to a member of the group
    /// and of this deployment, in any role. Returns that member's id.
    ///
    /// # Errors
    ///
    /// The refusal to answer the call with.
    fn admit_peer(
        &self,
        info: Option<&PeerConnectInfo>,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<u64, tonic::Status>;

    /// Mark a call to another member so that member can admit it.
    fn stamp(&self, metadata: &mut tonic::metadata::MetadataMap);
}

/// What a server knows about the connection a request arrived on.
#[derive(Debug, Clone)]
pub struct PeerConnectInfo {
    /// The address the connection came from.
    pub remote: Option<SocketAddr>,
    /// The node the connection authenticated as, `None` for a client.
    pub node: Option<Arc<AuthenticatedPeer>>,
}

/// One accepted connection.
pub enum InboundStream {
    /// A node, authenticated by its node identity.
    Node(
        Guarded<tokio_rustls::server::TlsStream<TcpStream>>,
        Option<SocketAddr>,
    ),
    /// A client over TLS, with an ordinary certificate or none.
    ClientTls(
        tokio_rustls::server::TlsStream<TcpStream>,
        Option<SocketAddr>,
    ),
    /// A client in plain text, on a port that accepts it.
    ClientPlain(TcpStream, Option<SocketAddr>),
}

impl tonic::transport::server::Connected for InboundStream {
    type ConnectInfo = PeerConnectInfo;

    fn connect_info(&self) -> PeerConnectInfo {
        match self {
            Self::Node(stream, remote) => PeerConnectInfo {
                remote: *remote,
                node: Some(Arc::clone(&stream.peer)),
            },
            Self::ClientTls(_, remote) | Self::ClientPlain(_, remote) => PeerConnectInfo {
                remote: *remote,
                node: None,
            },
        }
    }
}

impl AsyncRead for InboundStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Node(s, _) => Pin::new(s).poll_read(cx, buf),
            Self::ClientTls(s, _) => Pin::new(s).poll_read(cx, buf),
            Self::ClientPlain(s, _) => Pin::new(s).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for InboundStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Node(s, _) => Pin::new(s).poll_write(cx, buf),
            Self::ClientTls(s, _) => Pin::new(s).poll_write(cx, buf),
            Self::ClientPlain(s, _) => Pin::new(s).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Node(s, _) => Pin::new(s).poll_flush(cx),
            Self::ClientTls(s, _) => Pin::new(s).poll_flush(cx),
            Self::ClientPlain(s, _) => Pin::new(s).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Node(s, _) => Pin::new(s).poll_shutdown(cx),
            Self::ClientTls(s, _) => Pin::new(s).poll_shutdown(cx),
            Self::ClientPlain(s, _) => Pin::new(s).poll_shutdown(cx),
        }
    }
}

/// How a listener accepts connections.
#[derive(Debug, Clone)]
pub struct Accept {
    /// Who may present a certificate, and how it is judged.
    pub inbound: InboundAuth,
    /// Whether a client may connect in plain text. Never a node: an
    /// inter-node service finds no node identity on such a connection.
    pub plaintext_clients: bool,
    /// How long an accepted connection has to complete its TLS handshake.
    pub handshake_timeout: Duration,
}

/// Connections whose handshake has completed and the server has not taken
/// yet; the accept loop waits for room.
const ACCEPT_QUEUE: usize = 64;

/// The first byte of a TLS handshake record (RFC 8446 5.1, ContentType
/// handshake(22)).
const TLS_HANDSHAKE: u8 = 22;

/// The connections arriving on `listener`, each authenticated under
/// `context` as `accept` says, for `Server::serve_with_incoming`. A
/// connection that fails its handshake is dropped and logged; the stream
/// never ends on its own, and dropping it stops accepting and releases the
/// listener.
///
/// # Errors
///
/// The server TLS configuration cannot be built.
pub fn incoming(
    listener: TcpListener,
    context: &Arc<SecurityContext>,
    accept: Accept,
) -> Result<Incoming, rustls::Error> {
    let config = Arc::new(context.server_config(accept.inbound.clone())?);
    let acceptor = tokio_rustls::TlsAcceptor::from(config);
    let (tx, rx) = tokio::sync::mpsc::channel(ACCEPT_QUEUE);
    let context = Arc::clone(context);
    let task = tokio::spawn(async move {
        loop {
            let accepted = tokio::select! {
                () = tx.closed() => break,
                accepted = listener.accept() => accepted,
            };
            let (tcp, remote) = match accepted {
                Ok(pair) => pair,
                Err(e) => {
                    // Out of descriptors and the like: back off rather than
                    // spin, as the stock TCP acceptor does.
                    tracing::warn!(error = %e, "accepting a connection");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
            };
            let tx = tx.clone();
            let acceptor = acceptor.clone();
            let context = Arc::clone(&context);
            let plaintext = accept.plaintext_clients;
            let bound = accept.handshake_timeout;
            tokio::spawn(async move {
                let handshake = authenticate(tcp, remote, &acceptor, context, plaintext);
                match tokio::time::timeout(bound, handshake).await {
                    Ok(Ok(stream)) => {
                        // A closed queue means the server stopped.
                        let _ = tx.send(stream).await;
                    }
                    Ok(Err(e)) => {
                        metrics::counter!("coordinode_peer_connections_refused_total").increment(1);
                        tracing::warn!(%remote, error = %e, "connection refused");
                    }
                    Err(_) => tracing::debug!(%remote, "handshake timed out"),
                }
            });
        }
    });
    Ok(Incoming { rx, task })
}

/// Authenticate one accepted connection.
async fn authenticate(
    tcp: TcpStream,
    remote: SocketAddr,
    acceptor: &tokio_rustls::TlsAcceptor,
    context: Arc<SecurityContext>,
    plaintext_clients: bool,
) -> io::Result<InboundStream> {
    tcp.set_nodelay(true)?;
    if plaintext_clients {
        let mut first = [0u8; 1];
        let read = tcp.peek(&mut first).await?;
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        if first[0] != TLS_HANDSHAKE {
            return Ok(InboundStream::ClientPlain(tcp, Some(remote)));
        }
    }
    let tls = acceptor.accept(tcp).await?;
    let chain = tls.get_ref().1.peer_certificates().map(<[_]>::to_vec);
    let is_node = chain
        .as_ref()
        .and_then(|chain| chain.first())
        .is_some_and(|leaf| context.claims_node_identity(leaf));
    match chain {
        // The handshake verified it as a node already; verifying again
        // here yields the identity the connection carries.
        Some(chain) if is_node => {
            let peer = AuthenticatedPeer::establish(context, chain, PeerRole::Client)
                .map_err(|e| io::Error::new(io::ErrorKind::PermissionDenied, e))?;
            Ok(InboundStream::Node(
                Guarded { inner: tls, peer },
                Some(remote),
            ))
        }
        _ => Ok(InboundStream::ClientTls(tls, Some(remote))),
    }
}

/// Authenticated connections arriving on a listener; see [`incoming`].
pub struct Incoming {
    rx: tokio::sync::mpsc::Receiver<InboundStream>,
    task: tokio::task::JoinHandle<()>,
}

impl futures_util::Stream for Incoming {
    type Item = io::Result<InboundStream>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().rx.poll_recv(cx).map(|next| next.map(Ok))
    }
}

impl Drop for Incoming {
    fn drop(&mut self) {
        // The loop also ends once it sees the queue closed; aborting releases
        // the listener at once, so a restart can bind the port again.
        self.task.abort();
    }
}

#[cfg(test)]
mod tests;
