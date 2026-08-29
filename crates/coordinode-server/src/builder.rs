//! [`ServerBuilder`]: the seam a downstream distribution links against.
//!
//! The CE binary builds an empty builder, so its behaviour is exactly what it
//! was before the seam existed. A downstream distribution registers extra gRPC
//! services, query-extension handlers, background tasks, a placement strategy
//! and extra serve modes, then calls [`ServerBuilder::run`] with the same
//! parsed [`crate::cli::Command`].
//!
//! Registration happens once, at process start, so trait objects are the right
//! shape here. Nothing registered is consulted per result row: query
//! extensions are resolved by name while the plan is built, and the gRPC
//! providers only contribute routes to the router that is assembled once
//! before the server starts accepting connections.

use std::collections::BTreeMap;
use std::sync::Arc;

use coordinode_cluster::{ClusterTopology, ShardRouting};
use coordinode_query::executor::runner::ExtensionHandler;
use tonic::service::RoutesBuilder;

/// The live server, handed to everything registered on a [`ServerBuilder`].
///
/// Constructed once the storage engine, Raft node and database are open, and
/// borrowed by each extension point in turn. Accessors rather than public
/// fields, so the set can grow without breaking a downstream distribution.
pub struct ServerContext {
    node_id: u64,
    data_dir: String,
    cluster_mode: bool,
    max_request_bytes: usize,
    database: Arc<parking_lot::RwLock<coordinode_embed::Database>>,
    engine: Arc<coordinode_storage::engine::core::StorageEngine>,
    raft_node: Arc<coordinode_raft::cluster::RaftNode>,
    session_registry: Arc<coordinode_session::SessionRegistry>,
    routing: Arc<dyn ShardRouting>,
    topology: Arc<dyn ClusterTopology>,
}

impl ServerContext {
    /// Raft node id of this process.
    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    /// Root data directory as resolved from the config gate.
    pub fn data_dir(&self) -> &str {
        &self.data_dir
    }

    /// Whether peers are configured. `false` is the standalone single-node
    /// Raft case, where no inter-node service is registered.
    pub fn cluster_mode(&self) -> bool {
        self.cluster_mode
    }

    /// Decoded-size ceiling to apply to every gRPC service, so a registered
    /// service inherits the same unbounded-allocation guard as the built-in
    /// ones.
    pub fn max_request_bytes(&self) -> usize {
        self.max_request_bytes
    }

    /// The shared database handle backing every protocol frontend.
    pub fn database(&self) -> &Arc<parking_lot::RwLock<coordinode_embed::Database>> {
        &self.database
    }

    /// The storage engine underneath the database.
    pub fn engine(&self) -> &Arc<coordinode_storage::engine::core::StorageEngine> {
        &self.engine
    }

    /// This node's Raft handle, for read fences and membership queries.
    pub fn raft_node(&self) -> &Arc<coordinode_raft::cluster::RaftNode> {
        &self.raft_node
    }

    /// Live session registry, the source behind SHOW SESSIONS.
    pub fn session_registry(&self) -> &Arc<coordinode_session::SessionRegistry> {
        &self.session_registry
    }

    /// Routing strategy in force, as set by [`ServerBuilder::with_placement`].
    pub fn routing(&self) -> &Arc<dyn ShardRouting> {
        &self.routing
    }

    /// Cluster topology in force, as set by [`ServerBuilder::with_placement`].
    pub fn topology(&self) -> &Arc<dyn ClusterTopology> {
        &self.topology
    }
}

/// Contributes gRPC services to the router shared on the main port.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be registered as a gRPC service provider",
    label = "does not implement `GrpcServiceProvider`",
    note = "implement `register` and add each service with `RoutesBuilder::add_service`, then pass it to `ServerBuilder::register_grpc_service`"
)]
pub trait GrpcServiceProvider: Send + Sync {
    /// Add services to `routes`. Called once, before the server binds.
    ///
    /// Apply `ctx.max_request_bytes()` via `max_decoding_message_size` to each
    /// service, matching the built-in ones.
    fn register(&self, ctx: &ServerContext, routes: &mut RoutesBuilder);
}

/// A long-running task started once the server is up.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be registered as a background task",
    label = "does not implement `BackgroundTask`",
    note = "implement `start` to spawn the work, then pass it to `ServerBuilder::register_background_task`"
)]
pub trait BackgroundTask: Send + Sync {
    /// Spawn the task. Called once, after the database is open and before the
    /// server binds. Must not block: spawn and return.
    fn start(&self, ctx: &ServerContext);
}

/// Implements a serve mode beyond the built-in `full`.
///
/// Registering a mode is what makes its name a legal value of the `mode`
/// config key. Per the config-driven control decision the mode is never a
/// dedicated CLI flag: it is read from the config file like every other
/// runtime policy.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be registered as a serve mode",
    label = "does not implement `ServeModeHandler`",
    note = "implement `start` to apply the mode to the running node, then pass it to `ServerBuilder::register_serve_mode`"
)]
pub trait ServeModeHandler: Send + Sync {
    /// Bring the node into this mode. Called once, after the database is open
    /// and before the server binds.
    fn start(&self, ctx: &ServerContext) -> Result<(), Box<dyn std::error::Error>>;
}

/// Assembles the server, optionally extended, and runs a parsed command.
///
/// A default builder is exactly the CE server:
///
/// ```no_run
/// # async fn f() -> Result<(), Box<dyn std::error::Error>> {
/// use coordinode_server::{cli, ServerBuilder};
///
/// ServerBuilder::new().run(cli::parse_args()).await
/// # }
/// ```
#[derive(Default)]
pub struct ServerBuilder {
    pub(crate) grpc_services: Vec<Arc<dyn GrpcServiceProvider>>,
    pub(crate) query_extensions: Vec<(String, Arc<dyn ExtensionHandler>)>,
    pub(crate) background_tasks: Vec<Arc<dyn BackgroundTask>>,
    pub(crate) serve_modes: BTreeMap<String, Arc<dyn ServeModeHandler>>,
    pub(crate) placement: Option<(Arc<dyn ShardRouting>, Arc<dyn ClusterTopology>)>,
}

impl ServerBuilder {
    /// A builder with nothing registered, which serves exactly like CE.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider of additional gRPC services on the main port.
    pub fn register_grpc_service(mut self, provider: Arc<dyn GrpcServiceProvider>) -> Self {
        self.grpc_services.push(provider);
        self
    }

    /// Register an extension-op handler under `name`.
    ///
    /// The query engine resolves the handler by name once, while the plan is
    /// built, and never per result row.
    pub fn register_query_extension(
        mut self,
        name: impl Into<String>,
        handler: Arc<dyn ExtensionHandler>,
    ) -> Self {
        self.query_extensions.push((name.into(), handler));
        self
    }

    /// Register a background task to start with the server.
    pub fn register_background_task(mut self, task: Arc<dyn BackgroundTask>) -> Self {
        self.background_tasks.push(task);
        self
    }

    /// Replace the placement strategy the server publishes to its extensions.
    ///
    /// Without this the server uses the single-shard, single-node strategy.
    pub fn with_placement(
        mut self,
        routing: Arc<dyn ShardRouting>,
        topology: Arc<dyn ClusterTopology>,
    ) -> Self {
        self.placement = Some((routing, topology));
        self
    }

    /// Make `name` a legal value of the `mode` config key, handled by
    /// `handler`. Last registration under a name wins.
    pub fn register_serve_mode(
        mut self,
        name: impl Into<String>,
        handler: Arc<dyn ServeModeHandler>,
    ) -> Self {
        self.serve_modes.insert(name.into(), handler);
        self
    }

    /// Run a parsed command. `serve` starts the extended server; every other
    /// command ignores the registrations and behaves as it does in CE.
    pub async fn run(self, command: crate::cli::Command) -> Result<(), Box<dyn std::error::Error>> {
        crate::run_with(self, command).await
    }
}

impl ServerContext {
    /// Assemble the context. Crate-internal: only the serve path builds one.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        node_id: u64,
        data_dir: String,
        cluster_mode: bool,
        max_request_bytes: usize,
        database: Arc<parking_lot::RwLock<coordinode_embed::Database>>,
        engine: Arc<coordinode_storage::engine::core::StorageEngine>,
        raft_node: Arc<coordinode_raft::cluster::RaftNode>,
        session_registry: Arc<coordinode_session::SessionRegistry>,
        routing: Arc<dyn ShardRouting>,
        topology: Arc<dyn ClusterTopology>,
    ) -> Self {
        Self {
            node_id,
            data_dir,
            cluster_mode,
            max_request_bytes,
            database,
            engine,
            raft_node,
            session_registry,
            routing,
            topology,
        }
    }
}
