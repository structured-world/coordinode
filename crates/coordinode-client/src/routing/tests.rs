use std::time::Duration;

use tonic::metadata::MetadataMap;
use tonic::transport::Channel;

use super::*;

/// An endpoint that is never dialled: routing decisions need none.
fn route(endpoint: &str, rtt_ms: u64, node: u64) -> Route {
    let channel = Channel::from_shared(endpoint.to_owned())
        .unwrap()
        .connect_lazy();
    Route {
        endpoint: endpoint.to_owned(),
        cypher: CypherServiceClient::new(channel),
        rtt: Duration::from_millis(rtt_ms),
        node: Some(node),
    }
}

fn answered(hops: &str, leader: Option<&str>) -> MetadataMap {
    let mut metadata = MetadataMap::new();
    metadata.insert(HOPS_HEADER, hops.parse().unwrap());
    if let Some(leader) = leader {
        metadata.insert(LEADER_HEADER, leader.parse().unwrap());
    }
    metadata
}

/// Two clients reaching the same members in different latency orders each
/// send a read a follower may serve to their own fastest; a read only the
/// leader may serve, with no leader named yet, takes the endpoint the client
/// was built for.
#[tokio::test]
async fn each_client_reads_from_its_own_fastest_member() {
    let near_three = Routes::measured(vec![
        route("http://127.0.0.1:1", 2, 3),
        route("http://127.0.0.1:2", 9, 1),
    ]);
    let near_one = Routes::measured(vec![
        route("http://127.0.0.1:2", 1, 1),
        route("http://127.0.0.1:1", 7, 3),
    ]);
    for (routes, fastest) in [(&near_three, 3), (&near_one, 1)] {
        for preference in [
            ReadPreference::Nearest,
            ReadPreference::Secondary,
            ReadPreference::SecondaryPreferred,
            ReadPreference::PrimaryPreferred,
        ] {
            let (to, direct) = routes.for_read(preference).expect("a route");
            assert_eq!(to.node, Some(fastest));
            assert!(!direct);
        }
        assert!(routes.for_read(ReadPreference::Primary).is_none());
    }
}

/// An answer that took a hop and names a leader this client reaches sends
/// the next read the leader may serve straight there; one only a follower
/// may serve still goes to the fastest member. An answer served where it
/// arrived moves nothing whatever leader it names, a leader this client
/// does not reach leaves it on its route, and a direct route that failed is
/// forgotten.
#[tokio::test]
async fn a_forwarded_answer_names_where_the_next_read_goes() {
    let mut routes = Routes::measured(vec![
        route("http://127.0.0.1:1", 2, 3),
        route("http://127.0.0.1:2", 9, 1),
    ]);
    routes.observe(&answered("0", Some("1")));
    assert_eq!(
        routes.for_read(ReadPreference::Nearest).unwrap().0.node,
        Some(3),
        "a local answer's leader hint moves nothing"
    );

    routes.observe(&answered("1", Some("7")));
    assert_eq!(
        routes.for_read(ReadPreference::Nearest).unwrap().0.node,
        Some(3),
        "a private leader leaves the client on its route"
    );

    routes.observe(&answered("1", Some("1")));
    let (to, direct) = routes.for_read(ReadPreference::Nearest).unwrap();
    assert_eq!((to.node, direct), (Some(1), true), "straight to the leader");
    let (to, direct) = routes.for_read(ReadPreference::Primary).unwrap();
    assert_eq!((to.node, direct), (Some(1), true));
    let (to, direct) = routes.for_read(ReadPreference::Secondary).unwrap();
    assert_eq!(
        (to.node, direct),
        (Some(3), false),
        "a follower-only read never goes to the leader"
    );

    routes.forget_direct();
    assert_eq!(
        routes.for_read(ReadPreference::Nearest).unwrap().0.node,
        Some(3)
    );
    assert!(routes.for_read(ReadPreference::Primary).is_none());
}
