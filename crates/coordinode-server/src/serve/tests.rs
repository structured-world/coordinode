use super::*;

/// The REST proxy dials the gRPC listener where it is bound: an IPv6 or a
/// single-interface listener is not reachable at 127.0.0.1, and one bound to
/// every interface is reached on the loopback of its own family.
#[cfg(feature = "rest-proxy")]
#[test]
fn the_rest_proxy_dials_the_grpc_listener_where_it_is_bound() {
    let at = |s: &str| local_upstream(s.parse().expect("socket address"));
    assert_eq!(at("[::1]:7080"), "http://[::1]:7080");
    assert_eq!(at("10.1.2.3:7080"), "http://10.1.2.3:7080");
    assert_eq!(at("0.0.0.0:7080"), "http://127.0.0.1:7080");
    assert_eq!(at("[::]:7080"), "http://[::1]:7080");
}

/// A stopping server waits for its connections only as long as the drain
/// window: serving that never finishes (a peer holding a stream open) is cut
/// once the window has passed since the stop, never before the stop, and
/// serving that finishes on its own returns at once.
#[tokio::test]
async fn the_drain_window_bounds_a_stop() {
    use std::time::Duration;

    let grace = Duration::from_millis(300);
    let (stop, stopping) = tokio::sync::watch::channel(false);
    let held_open = std::future::pending::<Result<(), ()>>();
    let draining = tokio::spawn(drain_within(held_open, stopping, grace));
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(!draining.is_finished(), "no stop, no cut");
    stop.send_replace(true);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!draining.is_finished(), "cut before the window passed");
    let cut = tokio::time::timeout(Duration::from_secs(5), draining)
        .await
        .expect("cut once the window passed")
        .expect("join");
    assert_eq!(cut, Ok(()));

    let (_stop, stopping) = tokio::sync::watch::channel(true);
    let drained = std::future::ready(Ok::<(), ()>(()));
    assert_eq!(drain_within(drained, stopping, grace).await, Ok(()));
}

/// A fatal consensus error starts the shutdown that ends the process, and
/// nothing else does: a watcher that ends without one (a normal stop) never
/// triggers it.
#[tokio::test]
async fn only_a_reported_fatal_error_starts_the_shutdown() {
    use std::time::Duration;

    let (tx, mut rx) = tokio::sync::watch::channel(None::<String>);
    let waiting = tokio::spawn(async move { fatal_reported(&mut rx).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!waiting.is_finished(), "no error, no shutdown");
    tx.send_replace(Some("sync the raft log: I/O error".into()));
    tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .expect("a reported error starts the shutdown")
        .expect("join");

    let (tx, mut rx) = tokio::sync::watch::channel(None::<String>);
    drop(tx);
    let normal_end = tokio::time::timeout(Duration::from_millis(50), fatal_reported(&mut rx));
    assert!(
        normal_end.await.is_err(),
        "a watcher gone without an error is not one"
    );
}
