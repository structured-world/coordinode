//! A read reaching a member behind the bound it asks for is served in the
//! client's one call: the member sends it once, before it runs, to a member
//! its preference permits, and the response says it took one hop.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[allow(dead_code)]
#[path = "support/members.rs"]
mod members;

use std::time::{Duration, Instant};

use coordinode_raft::cluster::nemesis;
use coordinode_server::proto::{query, replication};

use members::three_members;

fn hops(metadata: &tonic::metadata::MetadataMap) -> Option<String> {
    metadata
        .get("x-coordinode-hops")
        .map(|v| v.to_str().unwrap_or_default().to_owned())
}

async fn client(
    port: u16,
) -> query::cypher_service_client::CypherServiceClient<tonic::transport::Channel> {
    query::cypher_service_client::CypherServiceClient::connect(format!("http://127.0.0.1:{port}"))
        .await
        .unwrap()
}

fn fresh_read(
    after_ts: u64,
    preference: replication::ReadPreference,
    wait_ms: u32,
) -> query::ExecuteCypherRequest {
    query::ExecuteCypherRequest {
        query: "MATCH (n:Refreshed) RETURN n.k AS k".to_string(),
        read_preference: preference as i32,
        read_concern: Some(replication::ReadConcern {
            level: replication::ReadConcernLevel::Majority as i32,
            after_ts,
            wait_ms,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// A TCP relay in front of `upstream` that holds every chunk it carries,
/// either way, for `delay`: an endpoint the fixture makes far away. Returns
/// the port it listens on.
async fn far_away(upstream: u16, delay: Duration) -> u16 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let port = coordinode_test_fixtures::alloc_port();
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap();
    tokio::spawn(async move {
        while let Ok((client, _)) = listener.accept().await {
            let Ok(server) = tokio::net::TcpStream::connect(("127.0.0.1", upstream)).await else {
                continue;
            };
            let (client_read, client_write) = client.into_split();
            let (server_read, server_write) = server.into_split();
            let relay = |mut from: tokio::net::tcp::OwnedReadHalf,
                         mut to: tokio::net::tcp::OwnedWriteHalf| async move {
                let mut buf = vec![0u8; 64 * 1024];
                while let Ok(n) = from.read(&mut buf).await {
                    if n == 0 {
                        break;
                    }
                    tokio::time::sleep(delay).await;
                    if to.write_all(&buf[..n]).await.is_err() {
                        break;
                    }
                }
            };
            tokio::spawn(relay(client_read, server_write));
            tokio::spawn(relay(server_read, client_write));
        }
    });
    port
}

/// Two clients reaching the same members in different latency orders each
/// read from their own nearest; a refresh after a change at C, from a client
/// whose nearest member is held below C, is answered in that one call, and
/// a client that knows only that member as its ingress receives the same.
#[tokio::test(flavor = "multi_thread")]
async fn each_client_refreshes_through_its_own_nearest_member() {
    use coordinode_client::{CausalToken, CoordinodeClient, ReadPreference};

    let members = three_members().await;
    let (leader_port, near_port) = (members[0].1, members[2].1);
    let url = |port: u16| format!("http://127.0.0.1:{port}");
    let far = Duration::from_millis(80);
    let far_leader = far_away(leader_port, far).await;
    let far_near = far_away(near_port, far).await;

    // Client A is near member 3, client B near member 1.
    let mut near_three = CoordinodeClient::builder(url(far_leader))
        .namespace("default")
        .endpoints([url(near_port)])
        .causal_wait(Duration::from_secs(5))
        .build()
        .await
        .unwrap();
    let near_one = CoordinodeClient::builder(url(leader_port))
        .namespace("default")
        .endpoints([url(far_near)])
        .build()
        .await
        .unwrap();
    assert_eq!(near_three.endpoints_by_latency()[0], url(near_port));
    assert_eq!(near_one.endpoints_by_latency()[0], url(leader_port));

    nemesis::isolate(3, &[1, 2]);
    let mut writer = CoordinodeClient::builder(url(leader_port))
        .namespace("default")
        .build()
        .await
        .unwrap();
    let (_, c) = writer
        .execute_causal_write("CREATE (:Invalidated {k: 1})")
        .await
        .expect("the change at C");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let served = writer
            .execute_causal_read_with_preference(
                "MATCH (n:Invalidated) RETURN n.k AS k",
                std::collections::HashMap::new(),
                c,
                ReadPreference::Primary,
            )
            .await;
        if served.is_ok() {
            break;
        }
        assert!(Instant::now() < deadline, "the leader serves C");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let (rows, after) = near_three
        .execute_causal_read_with_preference(
            "MATCH (n:Invalidated) RETURN n.k AS k",
            std::collections::HashMap::new(),
            c,
            ReadPreference::Nearest,
        )
        .await
        .expect("one call, through the near member");
    assert_eq!(rows.len(), 1, "the refresh sees C");
    assert!(after >= c);

    let mut ingress_only = CoordinodeClient::builder(url(near_port))
        .namespace("default")
        .transport(coordinode_client::Transport::Unary)
        .build()
        .await
        .unwrap();
    let (rows, _) = ingress_only
        .execute_causal_read_with_preference(
            "MATCH (n:Invalidated) RETURN n.k AS k",
            std::collections::HashMap::new(),
            CausalToken::from(c.as_u64()),
            ReadPreference::Nearest,
        )
        .await
        .expect("the same answer through the one ingress");
    assert_eq!(rows.len(), 1);

    nemesis::heal();
}

/// A consumer that refreshes current state on each committed change and
/// acknowledges only what it published: the refresh, a causal read bounded
/// by the change's commit, is served through the consumer's nearest member
/// though it is held below the change, and sees a later update too. A
/// refresh refused leaves the cursor unacknowledged, and the registration
/// resumed redelivers the same unit from it, in order.
#[tokio::test(flavor = "multi_thread")]
async fn a_refresh_then_acknowledge_consumer_publishes_before_it_acknowledges() {
    use coordinode_client::{
        CausalToken, CoordinodeClient, ReadPreference, Retention, SubscribeOptions,
    };
    use std::collections::HashMap;

    let members = three_members().await;
    let (leader_port, near_port) = (members[0].1, members[2].1);
    let url = |port: u16| format!("http://127.0.0.1:{port}");
    nemesis::isolate(3, &[1, 2]);

    let mut writer = CoordinodeClient::builder(url(leader_port))
        .namespace("default")
        .build()
        .await
        .unwrap();
    let mut stream = CoordinodeClient::builder(url(leader_port))
        .namespace("default")
        .build()
        .await
        .unwrap();
    let mut refresher = CoordinodeClient::builder(url(near_port))
        .namespace("default")
        .endpoints([url(leader_port)])
        .causal_wait(Duration::from_secs(10))
        .transport(coordinode_client::Transport::Unary)
        .build()
        .await
        .unwrap();
    let mut subscription = stream
        .subscribe(SubscribeOptions::register("refresher", Retention::Strict))
        .await
        .expect("a subscription");
    let incarnation = subscription.incarnation();

    // What the consumer publishes: the refreshed value per key.
    let mut published: HashMap<i64, i64> = HashMap::new();
    fn int(value: &coordinode_client::Value) -> i64 {
        match value {
            coordinode_client::Value::Int(i) => *i,
            other => panic!("an integer, got {other:?}"),
        }
    }
    async fn refresh(
        client: &mut CoordinodeClient,
        at: u64,
    ) -> Result<Vec<coordinode_client::Row>, coordinode_client::ClientError> {
        client
            .execute_causal_read_with_preference(
                "MATCH (d:Doc) RETURN d.id AS id, d.v AS v",
                HashMap::new(),
                CausalToken::from(at),
                ReadPreference::Nearest,
            )
            .await
            .map(|(rows, _)| rows)
    }

    // A change at C1, and a later update before the consumer refreshes.
    let (_, c1) = writer
        .execute_causal_write("CREATE (:Doc {id: 1, v: 1})")
        .await
        .unwrap();
    let (_, c2) = writer
        .execute_causal_write("MATCH (d:Doc {id: 1}) SET d.v = 2")
        .await
        .unwrap();
    // Entries carrying no unit of the stream come as empty batches.
    let mut seen = Vec::new();
    let batch = loop {
        let batch = subscription.next().await.unwrap().expect("a batch");
        seen.extend(batch.units.iter().cloned());
        if seen.iter().any(|u| u.commit_ts == c1.as_u64()) {
            break batch;
        }
    };
    let first = seen
        .iter()
        .find(|u| u.commit_ts == c1.as_u64())
        .expect("the change at C1")
        .clone();
    let rows = refresh(&mut refresher, first.commit_ts)
        .await
        .expect("the refresh, one call through the near member");
    for row in rows {
        published.insert(int(&row["id"]), int(&row["v"]));
    }
    assert!(
        matches!(published.get(&1), Some(1 | 2)),
        "any view at or after C1 answers, the later update's or not: {published:?}"
    );
    // The update's own event, refreshed at its commit, sees it.
    for row in refresh(&mut refresher, c2.as_u64()).await.unwrap() {
        published.insert(int(&row["id"]), int(&row["v"]));
    }
    assert_eq!(published.get(&1), Some(&2), "the later update");
    // A deletion's refresh finds the key gone.
    let (_, gone) = writer
        .execute_causal_write("MATCH (d:Doc {id: 1}) DELETE d")
        .await
        .unwrap();
    let rows = refresh(&mut refresher, gone.as_u64()).await.unwrap();
    assert!(
        rows.iter().all(|row| int(&row["id"]) != 1),
        "the deletion is what the refresh sees"
    );
    // Published: now the cursor covering it is acknowledged.
    subscription.acknowledge(&batch.cursor).await.unwrap();

    // A change at C3 whose refresh is refused: nothing is acknowledged.
    let (_, c3) = writer
        .execute_causal_write("CREATE (:Doc {id: 2, v: 1})")
        .await
        .unwrap();
    let mut pending = Vec::new();
    while !pending
        .iter()
        .any(|u: &coordinode_client::ChangeUnit| u.commit_ts == c3.as_u64())
    {
        let batch = subscription.next().await.unwrap().expect("a batch");
        pending.extend(batch.units);
    }
    let unit = pending
        .iter()
        .find(|u| u.commit_ts == c3.as_u64())
        .unwrap()
        .clone();
    let refused = refresh(&mut refresher, u64::MAX).await;
    assert!(refused.is_err(), "a refresh no member can serve is refused");
    assert!(!published.contains_key(&2), "nothing published");
    drop(subscription);

    // Resumed, the registration redelivers the unit from the cursor last
    // acknowledged; refreshed and published, it is acknowledged.
    let mut resumed = stream
        .subscribe(SubscribeOptions::resume("refresher", incarnation))
        .await
        .expect("the registration resumed");
    let mut redelivered = Vec::new();
    let cursor = loop {
        let batch = resumed.next().await.unwrap().expect("a batch");
        redelivered.extend(batch.units);
        if redelivered.iter().any(|u| u.commit_ts == c3.as_u64()) {
            break batch.cursor;
        }
    };
    let again = redelivered
        .iter()
        .find(|u| u.commit_ts == c3.as_u64())
        .unwrap();
    assert_eq!(
        (again.log_index, again.unit, &again.events),
        (unit.log_index, unit.unit, &unit.events),
        "the same unit, the same events"
    );
    assert!(
        redelivered.iter().all(|u| u.commit_ts > c1.as_u64()),
        "nothing acknowledged is delivered again"
    );
    let rows = refresh(&mut refresher, again.commit_ts).await.unwrap();
    for row in rows {
        published.insert(int(&row["id"]), int(&row["v"]));
    }
    assert_eq!(published.get(&2), Some(&1));
    resumed.acknowledge(&cursor).await.unwrap();

    nemesis::heal();
}

/// A refresh after a change at C, sent to a member held below C: one client
/// call, one hop, the change visible. A read anyone may serve goes to the
/// leader even with no wait allowed; a read only a follower may serve goes
/// to the other follower and never to the leader.
#[tokio::test(flavor = "multi_thread")]
async fn a_refresh_reaching_a_member_behind_it_is_served_in_one_call() {
    let members = three_members().await;
    let (leader_port, near_port) = (members[0].1, members[2].1);
    // The near member hears nothing of consensus from here on; the members'
    // command channel still reaches it.
    nemesis::isolate(3, &[1, 2]);
    let written = client(leader_port)
        .await
        .execute_cypher(coordinode_test_fixtures::client::named(
            query::ExecuteCypherRequest {
                query: "CREATE (:Refreshed {k: 1})".to_string(),
                ..Default::default()
            },
        ))
        .await
        .expect("the change at C")
        .into_inner();
    let c = written
        .stats
        .and_then(|stats| stats.receipt)
        .expect("a receipt")
        .operation_time;
    let deadline = Instant::now() + Duration::from_secs(30);
    while members[1]
        .0
        .db
        .write()
        .execute_cypher("MATCH (n:Refreshed) RETURN n")
        .unwrap()
        .is_empty()
    {
        assert!(Instant::now() < deadline, "member 2 applies C");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // The leader serves C at once from here on: a zero wait is then about
    // the near member alone.
    let mut leader = client(leader_port).await;
    while leader
        .execute_cypher(coordinode_test_fixtures::client::named(fresh_read(
            c,
            replication::ReadPreference::Primary,
            0,
        )))
        .await
        .is_err()
    {
        assert!(Instant::now() < deadline, "the leader serves C");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let mut near = client(near_port).await;
    let nearest = near
        .execute_cypher(coordinode_test_fixtures::client::named(fresh_read(
            c,
            replication::ReadPreference::Nearest,
            0,
        )))
        .await
        .expect("served after one hop");
    assert_eq!(hops(nearest.metadata()).as_deref(), Some("1"));
    assert_eq!(nearest.get_ref().rows.len(), 1, "the refresh sees C");

    let strict = near
        .execute_cypher(coordinode_test_fixtures::client::named(fresh_read(
            c,
            replication::ReadPreference::Secondary,
            10_000,
        )))
        .await
        .expect("served by the other follower");
    assert_eq!(hops(strict.metadata()).as_deref(), Some("1"));
    assert_eq!(strict.get_ref().rows.len(), 1);
    assert!(
        !strict
            .get_ref()
            .stats
            .as_ref()
            .is_some_and(|stats| stats.served_by_leader),
        "a follower-only read never reads from the leader"
    );

    // A target that cannot serve either refuses as not ready, and its
    // refusal says it took one hop; it sends the read on to no one.
    let refused = near
        .execute_cypher(coordinode_test_fixtures::client::named(fresh_read(
            u64::MAX,
            replication::ReadPreference::Secondary,
            0,
        )))
        .await
        .expect_err("no member meets the bound");
    assert_eq!(refused.code(), tonic::Code::Unavailable);
    assert!(
        refused.message().contains("not ready"),
        "{}",
        refused.message()
    );
    assert_eq!(hops(refused.metadata()).as_deref(), Some("1"));

    // The call's deadline goes with the read: earlier than its wait, it ends
    // the forwarded read as a deadline, not after the whole wait.
    let mut timed = coordinode_test_fixtures::client::named(fresh_read(
        u64::MAX,
        replication::ReadPreference::Secondary,
        30_000,
    ));
    timed.set_timeout(Duration::from_millis(500));
    let started = Instant::now();
    let ended = near
        .execute_cypher(timed)
        .await
        .expect_err("the deadline ends it");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "ended by its deadline, not its wait: {:?}",
        started.elapsed()
    );
    assert!(
        matches!(
            ended.code(),
            tonic::Code::DeadlineExceeded | tonic::Code::Cancelled
        ),
        "{ended:?}"
    );

    // A member whose view meets the bound serves it where it arrives.
    let mut ready = client(members[1].1).await;
    let local = ready
        .execute_cypher(coordinode_test_fixtures::client::named(fresh_read(
            c,
            replication::ReadPreference::Nearest,
            0,
        )))
        .await
        .expect("served locally");
    assert_ne!(hops(local.metadata()).as_deref(), Some("1"), "no hop");
    assert_eq!(local.get_ref().rows.len(), 1);

    nemesis::heal();
}
