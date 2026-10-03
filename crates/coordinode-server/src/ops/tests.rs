use super::{Readiness, respond};

/// `/ready` answers from the readiness flag: 503 until the server starts
/// serving and again once shutdown begins, so health checks and balancers
/// never route to a node that cannot take the request.
#[test]
fn ready_follows_the_flag() {
    let readiness = Readiness::default();
    let (status, _, body) = respond("/ready", readiness.get(), String::new, String::new);
    assert_eq!(status, "503 Service Unavailable");
    assert_eq!(body, r#"{"ready":false}"#);

    readiness.set(true);
    let (status, _, body) = respond("/ready", readiness.get(), String::new, String::new);
    assert_eq!(status, "200 OK");
    assert_eq!(body, r#"{"ready":true}"#);

    readiness.set(false);
    let (status, _, _) = respond("/ready", readiness.get(), String::new, String::new);
    assert_eq!(status, "503 Service Unavailable");
}

/// A node whose consensus stopped on a fatal error commits nothing: `/ready`
/// answers 503 from then on, even when the server is (or later starts)
/// serving.
#[test]
fn ready_stays_down_once_consensus_failed() {
    let readiness = Readiness::default();
    readiness.set(true);
    readiness.consensus_failed();
    let (status, _, _) = respond("/ready", readiness.get(), String::new, String::new);
    assert_eq!(status, "503 Service Unavailable");

    readiness.set(true);
    assert!(!readiness.get(), "serving again does not lift the failure");
}

/// Liveness does not depend on readiness: a starting or draining process is
/// alive and must not be restarted for it.
#[test]
fn health_answers_whether_or_not_the_node_is_ready() {
    for ready in [false, true] {
        let (status, _, _) = respond("/health", ready, String::new, String::new);
        assert_eq!(status, "200 OK", "ready = {ready}");
    }
}

#[test]
fn metrics_are_rendered_only_for_the_metrics_path() {
    let (status, content_type, body) = respond(
        "/metrics",
        false,
        || "m 1\n".to_string(),
        || unreachable!("not asked for the version"),
    );
    assert_eq!(status, "200 OK");
    assert_eq!(content_type, "text/plain; charset=utf-8");
    assert_eq!(body, "m 1\n");

    let (status, _, _) = respond(
        "/nope",
        true,
        || unreachable!("not asked for metrics"),
        || unreachable!("not asked for the version"),
    );
    assert_eq!(status, "404 Not Found");
}

/// The version report is served whether or not the node is ready: a member
/// that does not run its group's version is exactly the one whose operator
/// needs to read why.
#[test]
fn version_is_served_whether_or_not_the_node_is_ready() {
    for ready in [false, true] {
        let (status, content_type, body) = respond(
            "/version",
            ready,
            || unreachable!("not asked for metrics"),
            || r#"{"node_id":1}"#.to_string(),
        );
        assert_eq!(status, "200 OK");
        assert_eq!(content_type, "application/json");
        assert_eq!(body, r#"{"node_id":1}"#);
    }
}
