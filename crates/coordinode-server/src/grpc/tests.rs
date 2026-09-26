use std::convert::Infallible;
use std::future::{Ready, ready};
use std::task::{Context, Poll};

use tower::{Layer, Service};

use super::{HEADER_HOPS, HEADER_NODE, NodeInfoLayer};

/// Answers every request with a fixed set of response headers.
struct Respond(Vec<(&'static str, &'static str)>);

impl Service<http::Request<()>> for Respond {
    type Response = http::Response<()>;
    type Error = Infallible;
    type Future = Ready<Result<Self::Response, Infallible>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: http::Request<()>) -> Self::Future {
        let mut response = http::Response::new(());
        for (name, value) in &self.0 {
            response.headers_mut().insert(
                http::header::HeaderName::from_static(name),
                http::HeaderValue::from_static(value),
            );
        }
        ready(Ok(response))
    }
}

async fn call(inner: Respond) -> http::Response<()> {
    let mut service = NodeInfoLayer::new(7).layer(inner);
    service
        .call(http::Request::new(()))
        .await
        .expect("infallible")
}

/// A response served locally reports zero hops and this node's id.
#[tokio::test]
async fn a_local_response_reports_zero_hops() {
    let response = call(Respond(Vec::new())).await;
    assert_eq!(response.headers()[HEADER_HOPS], "0");
    assert_eq!(response.headers()[HEADER_NODE], "7");
}

/// A request forwarded to the leader comes back with the hop count the
/// forwarding handler set; the layer must not overwrite it with zero.
#[tokio::test]
async fn a_forwarded_response_keeps_its_hop_count() {
    let response = call(Respond(vec![(HEADER_HOPS, "1")])).await;
    assert_eq!(response.headers()[HEADER_HOPS], "1");
}
