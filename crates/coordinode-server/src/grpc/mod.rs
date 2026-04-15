//! gRPC server middleware and utilities (:7080).
//!
//! # NodeInfoLayer
//!
//! `NodeInfoLayer` is a tower `Layer` that injects CoordiNode node-identity
//! headers into every gRPC response. It sits between the tonic transport and
//! the gRPC service implementations.
//!
//! ## CE headers (injected on every response)
//!
//! | Header | CE value | Notes |
//! |--------|----------|-------|
//! | `x-coordinode-node` | node_id (decimal) | Always set |
//! | `x-coordinode-hops` | `0` | No routing in CE; always served locally |
//! | `x-coordinode-load` | `0` | Load tracking deferred to R151 |
//!
//! EE header `x-coordinode-shard-hint` is **not** added by CE code.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use tower::{Layer, Service};

// ── Constants ──────────────────────────────────────────────────────────────────

const HEADER_NODE: &str = "x-coordinode-node";
const HEADER_HOPS: &str = "x-coordinode-hops";
const HEADER_LOAD: &str = "x-coordinode-load";

// ── NodeInfoLayer ──────────────────────────────────────────────────────────────

/// Tower layer that injects CoordiNode node-identity into every gRPC response.
///
/// Apply once at the `Server::builder()` level:
///
/// ```rust,ignore
/// Server::builder()
///     .layer(NodeInfoLayer::new(node_id))
///     .add_service(cypher_svc)
///     .serve_with_shutdown(addr, shutdown)
///     .await?;
/// ```
#[derive(Clone)]
pub struct NodeInfoLayer {
    node_id: u64,
}

impl NodeInfoLayer {
    /// Create a layer that tags every response with `node_id`.
    pub fn new(node_id: u64) -> Self {
        Self { node_id }
    }
}

impl<S> Layer<S> for NodeInfoLayer {
    type Service = NodeInfoService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        NodeInfoService {
            inner,
            node_id: self.node_id,
        }
    }
}

// ── NodeInfoService ────────────────────────────────────────────────────────────

/// Service wrapper produced by [`NodeInfoLayer`].
#[derive(Clone)]
pub struct NodeInfoService<S> {
    inner: S,
    node_id: u64,
}

impl<S, ReqBody, ResBody> Service<http::Request<ReqBody>> for NodeInfoService<S>
where
    S: Service<http::Request<ReqBody>, Response = http::Response<ResBody>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = NodeInfoFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<ReqBody>) -> Self::Future {
        NodeInfoFuture {
            inner: self.inner.call(req),
            node_id: self.node_id,
        }
    }
}

// ── NodeInfoFuture ─────────────────────────────────────────────────────────────

pin_project_lite::pin_project! {
    /// Future returned by [`NodeInfoService`].
    pub struct NodeInfoFuture<F> {
        #[pin]
        inner: F,
        node_id: u64,
    }
}

impl<F, B, E> Future for NodeInfoFuture<F>
where
    F: Future<Output = Result<http::Response<B>, E>>,
{
    type Output = Result<http::Response<B>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.inner.poll(cx) {
            Poll::Ready(Ok(mut response)) => {
                let headers = response.headers_mut();

                // x-coordinode-node: this node's numeric ID
                let node_str = this.node_id.to_string();
                if let Ok(v) = http::HeaderValue::from_str(&node_str) {
                    headers.insert(http::header::HeaderName::from_static(HEADER_NODE), v);
                }

                // x-coordinode-hops: 0 (CE has no routing — always local)
                headers.insert(
                    http::header::HeaderName::from_static(HEADER_HOPS),
                    http::HeaderValue::from_static("0"),
                );

                // x-coordinode-load: 0 (load tracking deferred to R151)
                headers.insert(
                    http::header::HeaderName::from_static(HEADER_LOAD),
                    http::HeaderValue::from_static("0"),
                );

                Poll::Ready(Ok(response))
            }
            other => other,
        }
    }
}
