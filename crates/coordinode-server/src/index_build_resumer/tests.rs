use std::time::Instant;

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_modality::{IndexState, IndexStore as _, LocalIndexStore};
use coordinode_query::index::{BuildPhase, BuildState, BuildStatus};
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_test_fixtures::alloc_port;

use super::*;

struct Member {
    db: Arc<parking_lot::RwLock<Database>>,
    node: Arc<RaftNode>,
    _dir: tempfile::TempDir,
}

async fn open_member(node_id: u64, port: u16, founder: bool) -> Member {
    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).unwrap());
    let listen: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let node = Arc::new(if founder {
        RaftNode::open_cluster(
            node_id,
            Arc::clone(&engine),
            listen,
            format!("http://127.0.0.1:{port}"),
        )
        .await
        .unwrap()
    } else {
        RaftNode::open_joining(node_id, Arc::clone(&engine), listen)
            .await
            .unwrap()
    });
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft())));
    let db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();
    let db = Arc::new(parking_lot::RwLock::new(db));
    spawn(Arc::clone(&db), Arc::clone(&node));
    Member {
        db,
        node,
        _dir: dir,
    }
}

/// The build of the index named `name` as `member` lists it, once `accept`
/// holds for it.
async fn await_build(
    member: &Member,
    name: &str,
    accept: impl Fn(&BuildStatus) -> bool,
) -> BuildStatus {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let found = {
            let db = member.db.read();
            let id = LocalIndexStore::new(db.engine())
                .resolve_name(name)
                .unwrap();
            db.index_build_status()
                .unwrap()
                .into_iter()
                .find(|s| s.record.as_ref().is_some_and(|r| Some(r.index) == id) && accept(s))
        };
        if let Some(status) = found {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "the build of {name} never got there"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn is_published(s: &BuildStatus) -> bool {
    s.record.as_ref().map(|r| &r.state) == Some(&BuildState::Published)
}

async fn await_leader(member: &Member) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while member.node.current_leader() != Some(member.node.node_id()) {
        assert!(Instant::now() < deadline, "never led");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Start `CREATE INDEX <name> ON :User(<property>)` on `member` in a thread
/// of its own, as a client whose statement waits for the build.
fn create_index(
    member: &Member,
    name: &str,
    property: &str,
) -> std::thread::JoinHandle<Result<(), String>> {
    let db = Arc::clone(&member.db);
    let statement = format!("CREATE INDEX {name} ON :User({property})");
    std::thread::spawn(move || {
        db.read()
            .execute_cypher_shared(&statement, None, None, None, None)
            .map(|_| ())
            .map_err(|e| e.to_string())
    })
}

/// Hand the lead from `from`, whose build of `name` waits behind a
/// transaction older than its index, to `to`, and wait for `to` to publish
/// the index. The executor on `from` stays fenced off; its statement learns
/// the published outcome once the older transaction ends.
async fn hand_over_a_held_build(from: &Member, to: &Member, name: &str, property: &str) {
    let older = from.db.read().begin_transaction();
    let created = create_index(from, name, property);
    let held = await_build(from, name, |s| {
        s.phase == Some(BuildPhase::AwaitingOlderTransactions)
    })
    .await;
    await_build(to, name, |s| {
        s.generation == held.generation
            && matches!(
                s.record.as_ref().map(|r| &r.state),
                Some(BuildState::Running { .. })
            )
    })
    .await;

    from.node
        .transfer_leadership_to(to.node.node_id())
        .await
        .unwrap();
    await_build(to, name, is_published).await;
    from.db.read().rollback_transaction(older).unwrap();
    let statement = tokio::task::spawn_blocking(move || created.join().unwrap())
        .await
        .unwrap();
    assert_eq!(
        statement,
        Ok(()),
        "the old leader's statement learns the outcome"
    );

    for member in [from, to] {
        await_build(member, name, is_published).await;
        let db = member.db.read();
        let store = LocalIndexStore::new(db.engine());
        let id = store.resolve_name(name).unwrap().unwrap();
        assert_eq!(
            store.load_definition(id).unwrap().unwrap().state,
            IndexState::Ready
        );
    }
}

/// A build the leader held when it handed the lead over is finished by the
/// new leader. The lead then moves back with another build held, and the
/// first member, leading for the second time, finishes that one: the
/// resumer looks on every lead, not only the first.
#[tokio::test(flavor = "multi_thread")]
async fn every_new_leader_finishes_the_build_the_old_one_held() {
    let (p1, p2) = (alloc_port(), alloc_port());
    let m1 = open_member(1, p1, true).await;
    let m2 = open_member(2, p2, false).await;
    await_leader(&m1).await;
    m1.node
        .add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .unwrap();
    m1.node.change_membership(vec![1, 2]).await.unwrap();
    m1.db
        .write()
        .execute_cypher(
            "CREATE (:User {email: 'a@x', name: 'A'}), (:User {email: 'b@x', name: 'B'})",
        )
        .unwrap();

    hand_over_a_held_build(&m1, &m2, "user_email", "email").await;
    await_leader(&m2).await;
    hand_over_a_held_build(&m2, &m1, "user_name", "name").await;

    let rows = m1
        .db
        .write()
        .execute_cypher("MATCH (u:User) WHERE u.email = 'a@x' AND u.name = 'A' RETURN u")
        .unwrap();
    assert_eq!(rows.len(), 1);
}
