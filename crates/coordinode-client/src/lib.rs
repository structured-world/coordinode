//! # CoordiNode Rust Client
//!
//! Official Rust gRPC driver for CoordiNode with optional source location
//! tracking for the Query Advisor.
//!
//! ## Quick start
//!
//! ```no_run
//! # tokio_test::block_on(async {
//! use coordinode_client::CoordinodeClient;
//!
//! // A server serves tenant data to a client that names its namespace.
//! let mut client = CoordinodeClient::builder("http://localhost:7080")
//!     .namespace("default")
//!     .build()
//!     .await
//!     .unwrap();
//!
//! let rows = client.execute_cypher("MATCH (n:Person) RETURN n.name LIMIT 5")
//!     .await
//!     .unwrap();
//!
//! for row in rows {
//!     println!("{:?}", row);
//! }
//! # })
//! ```
//!
//! ## Source location tracking
//!
//! Enable `debug_source_tracking` in development to let the server's Query
//! Advisor map slow-query suggestions to exact call sites in your application:
//!
//! ```no_run
//! # tokio_test::block_on(async {
//! use coordinode_client::CoordinodeClient;
//!
//! let mut client = CoordinodeClient::builder("http://localhost:7080")
//!     .namespace("default")          // ← the namespace every call works in
//!     .debug_source_tracking(true)   // ← attach file:line to every query
//!     .app_name("my-service")
//!     .app_version(env!("CARGO_PKG_VERSION"))
//!     .build()
//!     .await
//!     .unwrap();
//!
//! // The statement carries its call site (file "src/api/handlers/feed.rs",
//! // line 47); the session names the application ("my-service") once.
//! let rows = client.execute_cypher("MATCH (u:User)-[:FOLLOWS]->(f) RETURN f LIMIT 10")
//!     .await
//!     .unwrap();
//! # })
//! ```
//!
//! ## Transport
//!
//! Statements travel over one persistent bidirectional session per client,
//! opened on first use: no connection, authentication or stream setup per
//! statement. Where a proxy cannot carry a bidirectional stream, choose one
//! unary call per statement with [`Transport::Unary`].
//!
//! ## Nearest reads
//!
//! A client given several members of a group
//! ([`CoordinodeClientBuilder::endpoints`]) measures how fast each answers
//! when it is built and sends a read a follower may serve to the fastest. A
//! member there whose view does not meet the read's causal bound passes it
//! once to a member that can, within the read's one wait, so the caller
//! makes one call either way:
//!
//! ```no_run
//! # tokio_test::block_on(async {
//! use std::collections::HashMap;
//! use coordinode_client::{CausalToken, CoordinodeClient, ReadPreference};
//!
//! let mut client = CoordinodeClient::builder("http://node1:7080")
//!     .namespace("default")
//!     .endpoints(["http://node2:7080", "http://node3:7080"])
//!     .causal_wait(std::time::Duration::from_millis(500))
//!     .build()
//!     .await
//!     .unwrap();
//! // Refresh a key after a change event committed at `seen`.
//! let seen = CausalToken::from(0);
//! let (rows, _) = client
//!     .execute_causal_read_with_preference(
//!         "MATCH (d:Doc {id: 1}) RETURN d",
//!         HashMap::new(),
//!         seen,
//!         ReadPreference::Nearest,
//!     )
//!     .await
//!     .unwrap();
//! # let _ = rows;
//! # })
//! ```
//!
//! Source tracking relies on Rust's built-in `#[track_caller]` attribute.
//! No macros are needed — simply call the normal client methods and the
//! driver infers the caller's `file:line` automatically.

pub mod config;
pub mod error;
mod routing;
mod session;
mod source;
pub mod subscription;
mod value;

pub use config::{ClientConfig, CoordinodeClientBuilder, Transport};
pub use error::ClientError;
pub use subscription::{
    ChangeAction, ChangeBatch, ChangeDelivery, ChangeEvent, ChangeTarget, ChangeUnit, Cursor,
    Relationship, Retention, SubscribeOptions, Subscription,
};
pub use value::Value;

/// Causal consistency token: the latest operation time a causal session has
/// observed.
///
/// Returned by [`CoordinodeClient::execute_causal_write`] and
/// [`CoordinodeClient::execute_causal_read`], and passed to the next causal
/// read. It wraps an operation time: a write's commit timestamp, or the
/// timestamp a read's view was taken at. A read bounded by it is served only
/// where every commit at or below it is applied, on the leader or a follower
/// alike, which gives read-your-writes and monotonic reads. A session keeps
/// the largest token it has received; tokens order by time, so
/// `a.max(b)` combines two.
///
/// A member that does not meet the bound yet refuses the read with
/// `UNAVAILABLE`, reason `READ_NOT_READY`, unless the client allows a wait
/// ([`CoordinodeClientBuilder::causal_wait`]). The same holds standalone,
/// where the token is the write's commit timestamp, never zero.
///
/// [`CoordinodeClientBuilder::causal_wait`]: config::CoordinodeClientBuilder::causal_wait
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CausalToken(u64);

impl CausalToken {
    /// The operation time, in the server's HLC microseconds.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for CausalToken {
    fn from(operation_time: u64) -> Self {
        Self(operation_time)
    }
}

use std::collections::HashMap;
use std::future::Future;
use std::panic::Location;
use std::sync::Arc;

// Proto modules generated by build.rs from the proto/ submodule.
// Generated code has clippy warnings that are outside our control.
#[allow(clippy::all, dead_code)]
mod proto {
    pub mod common {
        include!(concat!(env!("OUT_DIR"), "/coordinode.v1.common.rs"));
    }
    pub mod graph {
        include!(concat!(env!("OUT_DIR"), "/coordinode.v1.graph.rs"));
    }
    // Replication types (ReadPreference, ReadConcern, WriteConcern).
    // Required because cypher.proto imports consistency.proto and the generated
    // query code resolves ReadConcern/ReadPreference via `super::replication::`.
    pub mod replication {
        include!(concat!(env!("OUT_DIR"), "/coordinode.v1.replication.rs"));
    }
    pub mod query {
        include!(concat!(env!("OUT_DIR"), "/coordinode.v1.query.rs"));
    }
    pub mod session {
        include!(concat!(env!("OUT_DIR"), "/coordinode.v1.session.rs"));
    }
    pub mod operation {
        include!(concat!(env!("OUT_DIR"), "/coordinode.v1.operation.rs"));
    }
}

use proto::{
    query::{ExecuteCypherRequest, cypher_service_client::CypherServiceClient},
    replication::{Journal, ReadConcern, ReadConcernLevel, WriteConcern, WriteConcernMode},
    session::{Execute, client_frame, server_frame},
};

/// Routing preference for read queries in a replicated cluster.
///
/// Mirrors the wire-level `ReadPreference`: [`Primary`] reads always
/// execute on the Raft leader (strongest freshness), the other
/// variants allow follower nodes to serve the read for scale-out.
/// Follower reads observe the follower's applied state; combine with
/// [`CoordinodeClient::execute_causal_read`] when read-your-writes is
/// required.
///
/// [`Primary`]: ReadPreference::Primary
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadPreference {
    /// Read from the Raft leader only (default). Fails on followers.
    #[default]
    Primary,
    /// Prefer the leader, fall back to this node when it is a follower.
    PrimaryPreferred,
    /// Read from a follower only. Fails on the leader.
    Secondary,
    /// Prefer a follower, also accept the leader.
    SecondaryPreferred,
    /// Read from whichever node serves the connection.
    Nearest,
}

impl ReadPreference {
    fn to_proto(self) -> i32 {
        use proto::replication::ReadPreference as Wire;
        match self {
            ReadPreference::Primary => Wire::Primary as i32,
            ReadPreference::PrimaryPreferred => Wire::PrimaryPreferred as i32,
            ReadPreference::Secondary => Wire::Secondary as i32,
            ReadPreference::SecondaryPreferred => Wire::SecondaryPreferred as i32,
            ReadPreference::Nearest => Wire::Nearest as i32,
        }
    }
}

/// A single result row: column name → property value.
pub type Row = HashMap<String, Value>;

/// CoordiNode gRPC client.
///
/// Create with [`CoordinodeClient::connect`] or [`CoordinodeClient::builder`].
/// All query methods are `#[track_caller]` — when `debug_source_tracking` is
/// enabled the caller's `file:line` is automatically sent to the server.
pub struct CoordinodeClient {
    config: ClientConfig,
    channel: tonic::transport::Channel,
    cypher: CypherServiceClient<tonic::transport::Channel>,
    /// The persistent session subscriptions ride, opened on first use.
    session: Option<Arc<session::SessionLink>>,
    /// The endpoints of the group this client reaches, fastest first, when
    /// it was given more than one.
    routes: Option<routing::Routes>,
}

impl CoordinodeClient {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Connect to `endpoint` with default settings (no source tracking).
    ///
    /// Equivalent to `CoordinodeClient::builder(endpoint).build().await`.
    pub async fn connect(endpoint: impl Into<String>) -> Result<Self, ClientError> {
        CoordinodeClientBuilder::new(endpoint).build().await
    }

    /// Return a builder to configure the client before connecting.
    pub fn builder(endpoint: impl Into<String>) -> CoordinodeClientBuilder {
        CoordinodeClientBuilder::new(endpoint)
    }

    /// Internal: connect using a fully-built [`ClientConfig`].
    pub(crate) async fn connect_with_config(config: ClientConfig) -> Result<Self, ClientError> {
        let channel = tonic::transport::Channel::from_shared(config.endpoint.clone())
            .map_err(|e| ClientError::InvalidEndpoint(e.to_string()))?
            .connect()
            .await?;

        let cypher = CypherServiceClient::new(channel.clone());
        let routes = routing::Routes::probe(&config).await;

        Ok(Self {
            config,
            channel,
            cypher,
            session: None,
            routes,
        })
    }

    /// The endpoints this client reaches, fastest first, as it measured them
    /// when it was built; empty when it was given only the one endpoint.
    pub fn endpoints_by_latency(&self) -> Vec<String> {
        self.routes.as_ref().map_or_else(Vec::new, |routes| {
            routes
                .members()
                .iter()
                .map(|route| route.endpoint.clone())
                .collect()
        })
    }

    // ── Change streams ────────────────────────────────────────────────────────

    /// Open a change-stream subscription on the client's persistent session.
    ///
    /// ```no_run
    /// # tokio_test::block_on(async {
    /// use coordinode_client::{CoordinodeClient, Retention, SubscribeOptions};
    ///
    /// let mut client = CoordinodeClient::connect("http://localhost:7080").await.unwrap();
    /// let mut sub = client
    ///     .subscribe(SubscribeOptions::register("audit-sink", Retention::Strict))
    ///     .await
    ///     .unwrap();
    /// while let Some(batch) = sub.next().await.unwrap() {
    ///     for unit in &batch.units {
    ///         println!("{} events at {}", unit.events.len(), unit.log_index);
    ///     }
    ///     // Held: the stream may release what comes before the cursor.
    ///     sub.acknowledge(&batch.cursor).await.unwrap();
    /// }
    /// # })
    /// ```
    ///
    /// # Errors
    ///
    /// The session could not be opened, or the server refused the
    /// subscription (an already live consumer id, an ended incarnation, a
    /// position it no longer holds).
    pub async fn subscribe(
        &mut self,
        options: SubscribeOptions,
    ) -> Result<Subscription, ClientError> {
        Subscription::open(self.session().await?, &options).await
    }

    /// The persistent session, opened now if none is open.
    async fn session(&mut self) -> Result<Arc<session::SessionLink>, ClientError> {
        if let Some(link) = self.session.as_ref().filter(|link| !link.is_closed()) {
            return Ok(Arc::clone(link));
        }
        let link = session::SessionLink::open(self.channel.clone(), &self.config).await?;
        self.session = Some(Arc::clone(&link));
        Ok(link)
    }

    // ── Cypher ────────────────────────────────────────────────────────────────

    /// Execute a Cypher query with no parameters.
    ///
    /// When `debug_source_tracking` is enabled, the caller's source location
    /// is attached to the request automatically via `#[track_caller]`.
    ///
    /// # Note on `#[track_caller]` and async
    ///
    /// Rust does not yet stabilise `#[track_caller]` on `async fn` (rust-lang/rust#110011).
    /// This method is therefore a synchronous `#[track_caller]` wrapper that captures
    /// `Location::caller()` before delegating to the async implementation. From the
    /// caller's perspective the API is unchanged: `client.execute_cypher(...).await`.
    #[track_caller]
    pub fn execute_cypher(
        &mut self,
        query: impl Into<String>,
    ) -> impl Future<Output = Result<Vec<Row>, ClientError>> + '_ {
        // Location is captured HERE — in the synchronous, #[track_caller] wrapper —
        // so it points to the user's call site, not to driver internals.
        let location = if self.config.debug_source_tracking {
            Some(Location::caller())
        } else {
            None
        };
        self.execute_cypher_inner(query.into(), HashMap::new(), location)
    }

    /// Execute a Cypher query with named parameters.
    ///
    /// ```no_run
    /// # tokio_test::block_on(async {
    /// # use std::collections::HashMap;
    /// # use coordinode_client::{CoordinodeClient, Value};
    /// # let mut client = CoordinodeClient::connect("http://localhost:7080").await.unwrap();
    /// let mut params = HashMap::new();
    /// params.insert("id".to_string(), Value::Int(42));
    ///
    /// let rows = client.execute_cypher_with_params(
    ///     "MATCH (u:User {id: $id}) RETURN u.name",
    ///     params,
    /// ).await.unwrap();
    /// # })
    /// ```
    #[track_caller]
    pub fn execute_cypher_with_params(
        &mut self,
        query: impl Into<String>,
        params: HashMap<String, Value>,
    ) -> impl Future<Output = Result<Vec<Row>, ClientError>> + '_ {
        let location = if self.config.debug_source_tracking {
            Some(Location::caller())
        } else {
            None
        };
        self.execute_cypher_inner(query.into(), params, location)
    }

    /// Execute a read query with an explicit [`ReadPreference`].
    ///
    /// [`execute_cypher`] and [`execute_cypher_with_params`] leave the
    /// preference to the server's configured default (the Raft leader unless
    /// the server says otherwise); this method names one, letting a
    /// connection to a follower serve reads ([`ReadPreference::Secondary`] /
    /// [`SecondaryPreferred`] / [`Nearest`]) for read scale-out.
    /// Follower reads observe that follower's applied state — use
    /// [`execute_causal_read`] when the read must observe a specific
    /// prior write.
    ///
    /// [`execute_cypher`]: CoordinodeClient::execute_cypher
    /// [`execute_cypher_with_params`]: CoordinodeClient::execute_cypher_with_params
    /// [`execute_causal_read`]: CoordinodeClient::execute_causal_read
    /// [`SecondaryPreferred`]: ReadPreference::SecondaryPreferred
    /// [`Nearest`]: ReadPreference::Nearest
    #[track_caller]
    pub fn execute_cypher_with_read_preference(
        &mut self,
        query: impl Into<String>,
        params: HashMap<String, Value>,
        read_preference: ReadPreference,
    ) -> impl Future<Output = Result<Vec<Row>, ClientError>> + '_ {
        let location = if self.config.debug_source_tracking {
            Some(Location::caller())
        } else {
            None
        };
        let query = query.into();
        async move {
            let (rows, _applied_index) = self
                .execute_request(query, params, Some(read_preference), None, None, location)
                .await?;
            Ok(rows)
        }
    }

    // ── Causal consistency ────────────────────────────────────────────────────

    /// Execute a write with **majority durability**, returning a causal token.
    ///
    /// The write is acknowledged only after a quorum of replicas holds it
    /// journaled. The returned [`CausalToken`] is its commit timestamp; pass
    /// it to [`execute_causal_read`] for read-your-writes, even when the read
    /// is served by a follower.
    ///
    /// # Errors
    ///
    /// Returns `ClientError::Grpc` if the server rejects the request.
    ///
    /// [`execute_causal_read`]: CoordinodeClient::execute_causal_read
    #[track_caller]
    pub fn execute_causal_write(
        &mut self,
        query: impl Into<String>,
    ) -> impl Future<Output = Result<(Vec<Row>, CausalToken), ClientError>> + '_ {
        let location = if self.config.debug_source_tracking {
            Some(Location::caller())
        } else {
            None
        };
        self.execute_causal_write_inner(query.into(), HashMap::new(), location)
    }

    /// Execute a write with **majority durability** and named parameters.
    ///
    /// Identical to [`execute_causal_write`] but accepts query parameters.
    ///
    /// [`execute_causal_write`]: CoordinodeClient::execute_causal_write
    #[track_caller]
    pub fn execute_causal_write_with_params(
        &mut self,
        query: impl Into<String>,
        params: HashMap<String, Value>,
    ) -> impl Future<Output = Result<(Vec<Row>, CausalToken), ClientError>> + '_ {
        let location = if self.config.debug_source_tracking {
            Some(Location::caller())
        } else {
            None
        };
        self.execute_causal_write_inner(query.into(), params, location)
    }

    /// Execute a read that observes everything up to `after`, and return the
    /// session's token after it.
    ///
    /// The read is served only where every commit at or below `after` is
    /// applied, which gives **read-your-writes** for a token from
    /// [`execute_causal_write`] and **monotonic reads** for one from an
    /// earlier causal read. The returned token is the larger of `after` and
    /// the time this read was taken at: pass it to the next causal read.
    ///
    /// # Errors
    ///
    /// `ClientError::Grpc` with `UNAVAILABLE`, reason `READ_NOT_READY`, when
    /// the serving member does not meet `after` (after the wait the client
    /// allows, see [`CoordinodeClientBuilder::causal_wait`]); any other
    /// refusal of the server.
    ///
    /// [`execute_causal_write`]: CoordinodeClient::execute_causal_write
    /// [`CoordinodeClientBuilder::causal_wait`]: config::CoordinodeClientBuilder::causal_wait
    #[track_caller]
    pub fn execute_causal_read(
        &mut self,
        query: impl Into<String>,
        after: CausalToken,
    ) -> impl Future<Output = Result<(Vec<Row>, CausalToken), ClientError>> + '_ {
        let location = if self.config.debug_source_tracking {
            Some(Location::caller())
        } else {
            None
        };
        self.execute_causal_read_inner(query.into(), HashMap::new(), after, None, location)
    }

    /// Execute a causal read with named parameters.
    ///
    /// Identical to [`execute_causal_read`] but accepts query parameters.
    ///
    /// [`execute_causal_read`]: CoordinodeClient::execute_causal_read
    #[track_caller]
    pub fn execute_causal_read_with_params(
        &mut self,
        query: impl Into<String>,
        params: HashMap<String, Value>,
        after: CausalToken,
    ) -> impl Future<Output = Result<(Vec<Row>, CausalToken), ClientError>> + '_ {
        let location = if self.config.debug_source_tracking {
            Some(Location::caller())
        } else {
            None
        };
        self.execute_causal_read_inner(query.into(), params, after, None, location)
    }

    /// A causal read that names its [`ReadPreference`]: a refresh after a
    /// change event, for example, read where `preference` permits at or
    /// after `after`. A preference that lets a follower serve it goes to the
    /// fastest endpoint this client reaches
    /// ([`CoordinodeClientBuilder::endpoints`]); a member there that is
    /// behind `after` passes it once to a member that can serve it, within
    /// the one wait the client allows, so the caller makes one call either
    /// way. Any later update or deletion is what such a read sees.
    ///
    /// # Errors
    ///
    /// As [`Self::execute_causal_read`].
    ///
    /// [`CoordinodeClientBuilder::endpoints`]: config::CoordinodeClientBuilder::endpoints
    #[track_caller]
    pub fn execute_causal_read_with_preference(
        &mut self,
        query: impl Into<String>,
        params: HashMap<String, Value>,
        after: CausalToken,
        preference: ReadPreference,
    ) -> impl Future<Output = Result<(Vec<Row>, CausalToken), ClientError>> + '_ {
        let location = if self.config.debug_source_tracking {
            Some(Location::caller())
        } else {
            None
        };
        self.execute_causal_read_inner(query.into(), params, after, Some(preference), location)
    }

    // ── Config access ─────────────────────────────────────────────────────────

    /// Returns `true` when source location tracking is enabled.
    pub fn source_tracking_enabled(&self) -> bool {
        self.config.debug_source_tracking
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    async fn execute_cypher_inner(
        &mut self,
        query: String,
        params: HashMap<String, Value>,
        location: Option<&'static Location<'static>>,
    ) -> Result<Vec<Row>, ClientError> {
        // Settings left unset: the server applies its configured defaults.
        let (rows, _applied_index) = self
            .execute_request(query, params, None, None, None, location)
            .await?;
        Ok(rows)
    }

    async fn execute_causal_write_inner(
        &mut self,
        query: String,
        params: HashMap<String, Value>,
        location: Option<&'static Location<'static>>,
    ) -> Result<(Vec<Row>, CausalToken), ClientError> {
        // Named explicitly rather than left to the server default: a causal
        // write's position is only a promise when a majority holds it journaled.
        let write_concern = Some(WriteConcern {
            w: Some(proto::replication::write_concern::W::Mode(
                WriteConcernMode::Majority as i32,
            )),
            journal: Journal::Journal as i32,
            timeout_ms: 0,
        });
        let (rows, operation_time) = self
            .execute_request(query, params, None, None, write_concern, location)
            .await?;
        Ok((rows, CausalToken(operation_time)))
    }

    async fn execute_causal_read_inner(
        &mut self,
        query: String,
        params: HashMap<String, Value>,
        after: CausalToken,
        preference: Option<ReadPreference>,
        location: Option<&'static Location<'static>>,
    ) -> Result<(Vec<Row>, CausalToken), ClientError> {
        let read_concern = Some(ReadConcern {
            level: ReadConcernLevel::Majority as i32,
            after_ts: after.as_u64(),
            after_position: None,
            wait_ms: self
                .config
                .causal_wait
                .map_or(0, |w| u32::try_from(w.as_millis()).unwrap_or(u32::MAX)),
            at_timestamp: 0,
        });
        let (rows, operation_time) = self
            .execute_request(query, params, preference, read_concern, None, location)
            .await?;
        Ok((rows, after.max(CausalToken(operation_time))))
    }

    /// Low-level request executor shared by all public methods.
    ///
    /// Returns the decoded rows **and** the receipt's operation time from
    /// `QueryStats` so causal methods can wrap it in a [`CausalToken`]. An
    /// absent `preference` and concern leave the setting to the server. A
    /// read whose preference lets this client choose where it goes, when the
    /// client reaches several endpoints, goes as one unary call there; the
    /// rest take the client's transport.
    async fn execute_request(
        &mut self,
        query: String,
        params: HashMap<String, Value>,
        preference: Option<ReadPreference>,
        read_concern: Option<ReadConcern>,
        write_concern: Option<WriteConcern>,
        location: Option<&'static Location<'static>>,
    ) -> Result<(Vec<Row>, u64), ClientError> {
        let proto_params: HashMap<_, _> = params
            .into_iter()
            .map(|(k, v)| (k, value::to_proto(v)))
            .collect();
        let read_preference = preference.map_or(0, ReadPreference::to_proto);

        let routed = preference.and_then(|preference| {
            self.routes
                .as_ref()
                .and_then(|routes| routes.for_read(preference))
        });
        if let Some((route, direct)) = routed {
            let request = ExecuteCypherRequest {
                query: query.clone(),
                parameters: proto_params.clone(),
                read_preference,
                read_concern: read_concern.clone(),
                write_concern,
                ..Default::default()
            };
            let mut cypher = route.cypher;
            match self.unary(&mut cypher, request, location).await {
                // A leader named on a forwarded answer that no longer
                // carries the read: it goes on as if none had been named, to
                // the fastest endpoint or the one the client was built for.
                Err(ClientError::Grpc(status))
                    if direct && status.code() == tonic::Code::Unavailable =>
                {
                    if let Some(routes) = self.routes.as_mut() {
                        routes.forget_direct();
                    }
                    return Box::pin(
                        self.execute_request(
                            query,
                            proto_params
                                .into_iter()
                                .map(|(k, v)| (k, value::from_proto(v)))
                                .collect(),
                            preference,
                            read_concern,
                            write_concern,
                            location,
                        ),
                    )
                    .await;
                }
                answered => return answered,
            }
        }

        if self.config.transport == Transport::Session {
            let link = self.session().await?;
            return execute_on_session(
                &link,
                Execute {
                    query,
                    parameters: proto_params,
                    // Auto-commit, as on the unary path.
                    txid: Vec::new(),
                    read_concern,
                    write_concern,
                    read_preference: (read_preference != 0).then_some(read_preference),
                    source: location.map(source::statement_source),
                    // The driver does not number its writes yet.
                    operation: None,
                },
            )
            .await;
        }

        let request = ExecuteCypherRequest {
            query,
            parameters: proto_params,
            read_preference,
            read_concern,
            write_concern,
            // Vector settings are named in a hint in the query, or by the
            // server's defaults.
            vector_consistency: 0,
            vector_build_wait_ms: None,
            // The server's memory limit; a session's `SET query_memory_limit_mb`
            // names another.
            query_memory_limit_mb: None,
            // The driver does not number its writes yet.
            operation: None,
        };
        let mut cypher = self.cypher.clone();
        self.unary(&mut cypher, request, location).await
    }

    /// Send `request` as one unary call through `cypher`, take the routing
    /// hints its response carries, and decode its rows and operation time.
    async fn unary(
        &mut self,
        cypher: &mut CypherServiceClient<tonic::transport::Channel>,
        request: ExecuteCypherRequest,
        location: Option<&'static Location<'static>>,
    ) -> Result<(Vec<Row>, u64), ClientError> {
        let mut request = tonic::Request::new(request);
        self.config.name_namespace(request.metadata_mut());
        if let Some(loc) = location {
            source::inject_grpc_metadata(request.metadata_mut(), loc, &self.config);
        }

        let response = cypher.execute_cypher(request).await?;
        if let Some(routes) = self.routes.as_mut() {
            routes.observe(response.metadata());
        }
        let body = response.into_inner();

        let operation_time = operation_time(body.stats.as_ref());

        let columns = body.columns;
        let rows = body
            .rows
            .into_iter()
            .map(|row| {
                columns
                    .iter()
                    .zip(row.values)
                    .map(|(col, pv)| (col.clone(), value::from_proto(pv)))
                    .collect::<Row>()
            })
            .collect();

        Ok((rows, operation_time))
    }
}

/// The operation time a statement's receipt names; zero when it names none.
fn operation_time(stats: Option<&proto::query::QueryStats>) -> u64 {
    stats
        .and_then(|s| s.receipt.as_ref())
        .map_or(0, |r| r.operation_time)
}

/// Rows of one statement buffered in the session before its reader is waited
/// on: a few batches, enough that the stream does not stall on a reader that
/// is merely collecting.
const STATEMENT_BATCHES: usize = 16;

/// Run `execute` on the session and collect its cursor: the column header,
/// every row batch, and the applied index its end reports.
async fn execute_on_session(
    link: &session::SessionLink,
    execute: Execute,
) -> Result<(Vec<Row>, u64), ClientError> {
    let (id, mut answers) = link.request(STATEMENT_BATCHES);
    let collected = async {
        link.send(id, client_frame::Op::Execute(execute)).await?;
        let mut columns: Vec<String> = Vec::new();
        let mut rows: Vec<Row> = Vec::new();
        loop {
            let frame = answers.recv().await.ok_or(ClientError::SessionClosed)?;
            match frame.event {
                Some(server_frame::Event::CursorOpen(open)) => columns = open.columns,
                Some(server_frame::Event::Rows(batch)) => {
                    rows.extend(batch.rows.into_iter().map(|row| {
                        columns
                            .iter()
                            .zip(row.values)
                            .map(|(col, pv)| (col.clone(), value::from_proto(pv)))
                            .collect::<Row>()
                    }));
                }
                Some(server_frame::Event::CursorEnd(end)) => {
                    return Ok((rows, operation_time(end.stats.as_ref())));
                }
                Some(server_frame::Event::Error(error)) => {
                    return Err(ClientError::Grpc(session::error_status(error)));
                }
                other => return Err(ClientError::UnexpectedAnswer(format!("{other:?}"))),
            }
        }
    }
    .await;
    link.forget(id);
    collected
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
