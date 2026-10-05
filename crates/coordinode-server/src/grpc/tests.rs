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

/// Every call is counted under its method, with its duration, and a call
/// refused with a status in the headers is counted as an error with its code.
#[test]
fn calls_are_counted_by_method_and_refusals_by_code() {
    use std::future::Future as _;

    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        for (status, method) in [
            (None, "/coordinode.v1.query.CypherService/ExecuteCypher"),
            (
                Some("5"),
                "/coordinode.v1.query.CypherService/ExecuteCypher",
            ),
            (None, "/coordinode.v1.session.SessionService/Session"),
        ] {
            let headers = status.map(|s| vec![("grpc-status", s)]).unwrap_or_default();
            let mut service = NodeInfoLayer::new(7).layer(Respond(headers));
            let request = http::Request::builder()
                .uri(method)
                .body(())
                .expect("request");
            let mut call = std::pin::pin!(service.call(request));
            let mut cx = Context::from_waker(std::task::Waker::noop());
            assert!(call.as_mut().poll(&mut cx).is_ready());
        }
    });
    let text = handle.render();
    assert!(
        text.contains(r#"coordinode_grpc_requests_total{method="/coordinode.v1.query.CypherService/ExecuteCypher"} 2"#),
        "{text}"
    );
    assert!(
        text.contains(r#"coordinode_grpc_requests_total{method="/coordinode.v1.session.SessionService/Session"} 1"#),
        "{text}"
    );
    assert!(
        text.contains(r#"coordinode_grpc_errors_total{method="/coordinode.v1.query.CypherService/ExecuteCypher",code="5"} 1"#),
        "{text}"
    );
    assert!(text.contains("coordinode_grpc_duration_seconds"), "{text}");
}

/// A request forwarded to the leader comes back with the hop count the
/// forwarding handler set; the layer must not overwrite it with zero.
#[tokio::test]
async fn a_forwarded_response_keeps_its_hop_count() {
    let response = call(Respond(vec![(HEADER_HOPS, "1")])).await;
    assert_eq!(response.headers()[HEADER_HOPS], "1");
}
