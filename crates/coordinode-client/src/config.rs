//! Client configuration and builder.

/// Configuration for a [`super::CoordinodeClient`] connection.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// gRPC endpoint to connect to (e.g. `"http://localhost:7080"`).
    pub(crate) endpoint: String,

    /// When `true`, every query automatically attaches the caller's source
    /// location (`file:line`) as gRPC metadata. The server records this in the
    /// query advisor, which can then map suggestions to exact application code.
    ///
    /// Default: `false`. Enable in development/staging; keep disabled in
    /// production to avoid any overhead.
    pub(crate) debug_source_tracking: bool,

    /// Optional application name sent alongside the source location.
    /// Appears in query advisor output as "App: my-service".
    pub(crate) app_name: String,

    /// Optional application version sent alongside the source location.
    pub(crate) app_version: String,
}

impl ClientConfig {
    /// Returns a builder targeting the given endpoint.
    pub fn builder(endpoint: impl Into<String>) -> CoordinodeClientBuilder {
        CoordinodeClientBuilder::new(endpoint)
    }
}

/// Builder for [`super::CoordinodeClient`].
///
/// # Examples
///
/// ```no_run
/// # tokio_test::block_on(async {
/// use coordinode_client::CoordinodeClient;
///
/// let mut client = CoordinodeClient::builder("http://localhost:7080")
///     .debug_source_tracking(true)
///     .app_name("my-service")
///     .app_version("v1.2.3")
///     .build()
///     .await
///     .unwrap();
/// # })
/// ```
pub struct CoordinodeClientBuilder {
    pub(crate) endpoint: String,
    pub(crate) debug_source_tracking: bool,
    pub(crate) app_name: String,
    pub(crate) app_version: String,
}

impl CoordinodeClientBuilder {
    /// Create a new builder targeting the given endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            debug_source_tracking: false,
            app_name: String::new(),
            app_version: String::new(),
        }
    }

    /// Enable or disable source location tracking.
    ///
    /// When enabled, every Cypher, vector-search, and text-search call
    /// automatically attaches the caller's `file:line` to the request as gRPC
    /// metadata. The server uses this in the query advisor to surface
    /// suggestions with precise source-code context.
    ///
    /// Relies on Rust's `#[track_caller]` attribute — no macros required.
    /// Overhead: ~2 ns per call (static location read + 2 metadata inserts).
    pub fn debug_source_tracking(mut self, enabled: bool) -> Self {
        self.debug_source_tracking = enabled;
        self
    }

    /// Set the application name included in source-tracking metadata.
    pub fn app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = name.into();
        self
    }

    /// Set the application version included in source-tracking metadata.
    pub fn app_version(mut self, version: impl Into<String>) -> Self {
        self.app_version = version.into();
        self
    }

    /// Connect to the server and return a ready [`super::CoordinodeClient`].
    pub async fn build(self) -> Result<super::CoordinodeClient, super::ClientError> {
        let config = ClientConfig {
            endpoint: self.endpoint.clone(),
            debug_source_tracking: self.debug_source_tracking,
            app_name: self.app_name,
            app_version: self.app_version,
        };
        super::CoordinodeClient::connect_with_config(config).await
    }
}
