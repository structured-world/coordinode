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
