use std::sync::Arc;

use parking_lot::Mutex;

use super::{
    BlobGcRequest, BlobGcView, Readiness, ReloadHook, blob_gc_response, reload_response, respond,
};

/// A view that records what it was asked and answers with its name.
fn recording_view() -> (BlobGcView, Arc<Mutex<Vec<BlobGcRequest>>>) {
    let asked = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&asked);
    let view: BlobGcView = Arc::new(move |request| {
        seen.lock().push(request);
        Ok(format!("{request:?}"))
    });
    (view, asked)
}

/// `GET /blob-gc` reports and `POST /blob-gc/run` runs a pass, reconciling
/// only when asked; the wrong method is refused before the collector is
/// asked anything, and an unknown path under the prefix is not found.
#[test]
fn blob_gc_reports_on_get_and_runs_on_post() {
    let (view, asked) = recording_view();
    let (status, content_type, _) = blob_gc_response("GET", "/blob-gc", &view);
    assert_eq!((status, content_type), ("200 OK", "application/json"));
    let (status, _, _) = blob_gc_response("POST", "/blob-gc/run", &view);
    assert_eq!(status, "200 OK");
    let (status, _, _) = blob_gc_response("POST", "/blob-gc/run?reconcile=true", &view);
    assert_eq!(status, "200 OK");
    assert_eq!(
        *asked.lock(),
        vec![
            BlobGcRequest::Status,
            BlobGcRequest::Run { reconcile: false },
            BlobGcRequest::Run { reconcile: true },
        ]
    );

    let (view, asked) = recording_view();
    assert_eq!(
        blob_gc_response("GET", "/blob-gc/run", &view).0,
        "405 Method Not Allowed",
        "a GET runs nothing"
    );
    assert_eq!(
        blob_gc_response("POST", "/blob-gc", &view).0,
        "405 Method Not Allowed"
    );
    assert_eq!(
        blob_gc_response("GET", "/blob-gc/other", &view).0,
        "404 Not Found"
    );
    assert!(asked.lock().is_empty());
}

/// A collector that fails says so as an error the caller sees.
#[test]
fn blob_gc_reports_the_collector_failure() {
    let failing: BlobGcView = Arc::new(|_| Err("the store is damaged".to_owned()));
    let (status, _, body) = blob_gc_response("POST", "/blob-gc/run", &failing);
    assert_eq!(status, "500 Internal Server Error");
    assert!(body.contains("the store is damaged"), "{body}");
}

/// `POST /reload` runs the reload and reports its outcome; a refused reload
/// is an error the caller sees, any other method changes nothing, and a node
/// without a node identity has nothing to reload.
#[test]
fn reload_runs_only_on_post_and_reports_its_outcome() {
    let ok: ReloadHook = Arc::new(|| Ok(()));
    let (status, _, body) = reload_response("POST", Some(&ok));
    assert_eq!(status, "200 OK");
    assert_eq!(body, r#"{"reloaded":true}"#);

    let refused: ReloadHook = Arc::new(|| Err("the key does not match".to_owned()));
    let (status, _, body) = reload_response("POST", Some(&refused));
    assert_eq!(status, "500 Internal Server Error");
    assert!(body.contains("the key does not match"), "{body}");

    let untouched: ReloadHook = Arc::new(|| unreachable!("a GET reloads nothing"));
    let (status, _, _) = reload_response("GET", Some(&untouched));
    assert_eq!(status, "405 Method Not Allowed");

    let (status, _, _) = reload_response("POST", None);
    assert_eq!(status, "404 Not Found");
}

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
