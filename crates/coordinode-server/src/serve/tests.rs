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
