use super::{Readiness, respond};

/// `/ready` answers from the readiness flag: 503 until the server starts
/// serving and again once shutdown begins, so health checks and balancers
/// never route to a node that cannot take the request.
#[test]
fn ready_follows_the_flag() {
    let readiness = Readiness::default();
    let (status, _, body) = respond("/ready", readiness.get(), String::new);
    assert_eq!(status, "503 Service Unavailable");
    assert_eq!(body, r#"{"ready":false}"#);

    readiness.set(true);
    let (status, _, body) = respond("/ready", readiness.get(), String::new);
    assert_eq!(status, "200 OK");
    assert_eq!(body, r#"{"ready":true}"#);

    readiness.set(false);
    let (status, _, _) = respond("/ready", readiness.get(), String::new);
    assert_eq!(status, "503 Service Unavailable");
}

/// Liveness does not depend on readiness: a starting or draining process is
/// alive and must not be restarted for it.
#[test]
fn health_answers_whether_or_not_the_node_is_ready() {
    for ready in [false, true] {
        let (status, _, _) = respond("/health", ready, String::new);
        assert_eq!(status, "200 OK", "ready = {ready}");
    }
}

#[test]
fn metrics_are_rendered_only_for_the_metrics_path() {
    let (status, content_type, body) = respond("/metrics", false, || "m 1\n".to_string());
    assert_eq!(status, "200 OK");
    assert_eq!(content_type, "text/plain; charset=utf-8");
    assert_eq!(body, "m 1\n");

    let (status, _, _) = respond("/nope", true, || unreachable!("not asked for metrics"));
    assert_eq!(status, "404 Not Found");
}
