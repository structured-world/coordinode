//! A replicated embedded database, through the entry point a host opens:
//! growth in place, refusals, leadership, failover and election preference.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::budget::CancelFlag;
use coordinode_core::graph::types::Value;
use coordinode_core::txn::command::{ActingContext, AdmissionSignal, TransactionId};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_embed::backup::restore::{RestoreError, RestoreOptions};
use coordinode_embed::backup::{BackupFormat, export};
use coordinode_embed::db::draw_transaction_id;
use coordinode_embed::replication::admission::Requested;
use coordinode_embed::replication::ingress::{Placement, Routed, Statement};
use coordinode_embed::replication::routing::{
    RoutedAnswer, RoutedBegin, RoutedCommit, RoutedStatement,
};
use coordinode_embed::replication::{
    ElectionPreference, Founding, MemberOptions, PeerTimeouts, Peering, ReplicatedDatabase,
    ReplicationError,
};
use coordinode_embed::{Database, DatabaseError};
use coordinode_raft::cluster::nemesis;
use coordinode_raft::read_fence::ReadPreference;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::alloc_port;

/// The CA the members of these tests are issued their identities by.
static PKI: std::sync::LazyLock<coordinode_test_fixtures::TestPki> =
    std::sync::LazyLock::new(coordinode_test_fixtures::TestPki::new);

const TEST_TIMEOUT: Duration = Duration::from_secs(180);

fn storage(dir: &Path) -> StorageConfig {
    storage_limited(dir, 0)
}

/// [`storage`] with every file of the member under `dir` held to
/// `hard_limit` bytes; `0` for no limit.
fn storage_limited(dir: &Path, hard_limit: u64) -> StorageConfig {
    StorageConfig::with_endpoints(vec![
        EndpointConfig::new("default", dir, Media::Hdd, Durability::Durable, Tier::Warm)
            .with_hard_limit_bytes(hard_limit),
    ])
}

fn addr(port: u16) -> String {
    format!("127.0.0.1:{port}")
}

/// Open member `id` over `dir`, serving its peers on `port`.
async fn open(
    id: u64,
    dir: &Path,
    port: u16,
    founding: Founding,
    preference: ElectionPreference,
) -> Result<ReplicatedDatabase, ReplicationError> {
    open_under(id, dir, port, founding, preference, PKI.context(id)).await
}

/// [`open`] proving the identity and trusting the peers `security` says.
async fn open_under(
    id: u64,
    dir: &Path,
    port: u16,
    founding: Founding,
    preference: ElectionPreference,
    security: Arc<coordinode_embed::replication::SecurityContext>,
) -> Result<ReplicatedDatabase, ReplicationError> {
    open_over(&storage(dir), id, port, founding, preference, security).await
}

/// Open member `id` over the store `storage` describes, serving its peers on
/// `port`.
async fn open_over(
    storage: &StorageConfig,
    id: u64,
    port: u16,
    founding: Founding,
    preference: ElectionPreference,
    security: Arc<coordinode_embed::replication::SecurityContext>,
) -> Result<ReplicatedDatabase, ReplicationError> {
    let mut options = MemberOptions::new(id);
    options.preference = preference;
    options.peering = Some(Peering {
        security,
        advertise: addr(port),
        founding,
        timeouts: PeerTimeouts::default(),
    });
    ReplicatedDatabase::open(storage, options, Some(addr(port).parse().unwrap())).await
}

async fn wait_until(what: &str, cond: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting: {what}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// How many nodes with `label` `db` holds.
fn count(db: &ReplicatedDatabase, label: &str) -> usize {
    db.database()
        .write()
        .execute_cypher(&format!("MATCH (n:{label}) RETURN n"))
        .map(|rows| rows.len())
        .unwrap_or(0)
}

/// A group of members `1..=n` over `dirs`, member 1 forming it and the others
/// joining empty and promoted to voters; `last` stands for election last.
async fn group(
    dirs: &[tempfile::TempDir],
    last: Option<u64>,
) -> (Vec<ReplicatedDatabase>, Vec<u16>) {
    group_limited(dirs, last, 0).await
}

/// [`group`] with every member's files held to `hard_limit` bytes; `0` for
/// no limit.
async fn group_limited(
    dirs: &[tempfile::TempDir],
    last: Option<u64>,
    hard_limit: u64,
) -> (Vec<ReplicatedDatabase>, Vec<u16>) {
    let ports: Vec<u16> = dirs.iter().map(|_| alloc_port()).collect();
    let preference = |id: u64| {
        if last == Some(id) {
            ElectionPreference::Last
        } else {
            ElectionPreference::Ordinary
        }
    };
    let founder = open_over(
        &storage_limited(dirs[0].path(), hard_limit),
        1,
        ports[0],
        Founding::Form { deployment: None },
        preference(1),
        PKI.context(1),
    )
    .await
    .unwrap();
    let enrollment = founder.enrollment(addr(ports[0])).expect("enrollment");
    let mut members = vec![founder];
    for (i, dir) in dirs.iter().enumerate().skip(1) {
        let id = i as u64 + 1;
        members.push(
            open_over(
                &storage_limited(dir.path(), hard_limit),
                id,
                ports[i],
                Founding::Join(enrollment.clone()),
                preference(id),
                PKI.context(id),
            )
            .await
            .unwrap(),
        );
        members[0]
            .add_learner(id, addr(ports[i]), PKI.identity(id))
            .await
            .unwrap();
        members[0].promote(id).await.unwrap();
    }
    (members, ports)
}

/// Text of `len` bytes no codec shrinks, distinct per `seed`.
fn incompressible_text(seed: u64, len: usize) -> String {
    use std::fmt::Write;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = String::with_capacity(len + 8);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        write!(&mut out, "{:08x}", state as u32).unwrap();
    }
    out
}

/// A group whose members' files are held to a hard limit refuses writes
/// once the limit is reached, typed, with its consensus running; under a
/// limit lowered below what the members hold, every write is refused
/// through the leader and through a follower alike, nothing is deleted, and
/// every member keeps answering reads: from the leader, from a follower's
/// own copy, and at read concern majority. A raised limit admits writes at
/// once.
#[tokio::test(flavor = "multi_thread")]
async fn a_full_group_refuses_writes_and_keeps_serving_reads() {
    use coordinode_core::txn::read_concern::ReadConcernLevel;
    // Twice the smallest supported limit, so the members can be lowered to
    // the minimum below what they hold.
    const LIMIT: u64 = 2 * coordinode_storage::engine::config::MIN_HARD_LIMIT_BYTES;
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group_limited(&dirs, None, LIMIT).await;
        followers_reach_the_leader(&members).await;

        let mut accepted = 0i64;
        let mut refused = None;
        for key in 0..1024u64 {
            let statement = format!(
                "CREATE (:Full {{key: {key}, payload: '{}'}})",
                incompressible_text(key, 32 * 1024)
            );
            match members[0].execute(Statement::new(&statement)).await {
                Ok(_) => accepted += 1,
                Err(e) => {
                    refused = Some(e);
                    break;
                }
            }
        }
        let refused = refused.expect("the group never refused a write past its limit");
        assert!(
            matches!(refused, DatabaseError::CapacityExhausted { .. }),
            "the refusal is typed, the leader's consensus untouched: {refused:?}"
        );
        assert!(accepted > 0, "the group took writes before it filled");

        // Room comes back as the group reclaims its log, so a limit lowered
        // below what every member holds is what keeps the group full: it
        // deletes nothing and refuses every write, through the leader and
        // through a follower alike.
        for member in &members {
            member
                .database()
                .read()
                .set_hard_limit(
                    "default",
                    coordinode_storage::engine::config::MIN_HARD_LIMIT_BYTES,
                    None,
                )
                .expect("lower the limit");
        }
        for (member, path) in [(&members[0], "the leader"), (&members[1], "a follower")] {
            let refused = member
                .execute(Statement::new("CREATE (:Full {key: -1})"))
                .await;
            assert!(
                matches!(refused, Err(DatabaseError::CapacityExhausted { .. })),
                "a write through {path} is refused, typed: {refused:?}"
            );
        }

        let rows = |answer: Routed<RoutedAnswer>| match answer.value.rows.as_slice() {
            [row] => match row.as_slice() {
                [Value::Int(n)] => *n,
                other => panic!("count row {other:?}"),
            },
            other => panic!("count rows {other:?}"),
        };
        let count_query = "MATCH (n:Full) RETURN count(n) AS n";
        let from_leader = members[0]
            .execute(Statement::new(count_query))
            .await
            .expect("the leader answers a read when full");
        assert_eq!(rows(from_leader), accepted);
        for (member, concern) in [
            (&members[1], None),
            (&members[2], Some(ReadConcernLevel::Majority)),
        ] {
            wait_until("the follower applied every accepted write", || {
                label_count(member, "Full") == accepted as u64
            })
            .await;
            let answer = member
                .execute(Statement {
                    requested: Requested {
                        read_concern: concern,
                        read_preference: Some(ReadPreference::Nearest),
                        ..Requested::default()
                    },
                    ..Statement::new(count_query)
                })
                .await
                .expect("a full follower answers a read from its own copy");
            assert_eq!(answer.forwarded_to, None, "served by the follower");
            assert_eq!(rows(answer), accepted);
        }

        // A raised limit admits writes again, at once, with nothing lost.
        for member in &members {
            member
                .database()
                .read()
                .set_hard_limit("default", 2 * LIMIT, None)
                .expect("raise the limit");
        }
        members[0]
            .execute(Statement::new("CREATE (:Full {key: 2000000})"))
            .await
            .expect("a raised limit admits writes");
        let after = members[0]
            .execute(Statement::new(count_query))
            .await
            .expect("read after the limit was raised");
        assert_eq!(rows(after), accepted + 1);

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A store written by the embedded library without replication opens as the
/// member a group is formed around, and what it held reaches a member that
/// joins it empty: growth on the same directory, no export or import. A
/// member that holds data of its own is refused as a joiner.
#[tokio::test(flavor = "multi_thread")]
async fn a_store_written_without_replication_grows_into_a_group_in_place() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        {
            let mut alone = Database::open(dirs[0].path()).unwrap();
            alone
                .execute_cypher("CREATE (:Before {k: 1}), (:Before {k: 2})")
                .unwrap();
        }
        // The third directory holds data of its own.
        {
            let mut other = Database::open(dirs[2].path()).unwrap();
            other.execute_cypher("CREATE (:Elsewhere {k: 1})").unwrap();
        }

        let (p1, p2, p3) = (alloc_port(), alloc_port(), alloc_port());
        let founder = open(
            1,
            dirs[0].path(),
            p1,
            Founding::Form { deployment: None },
            ElectionPreference::Ordinary,
        )
        .await
        .unwrap();
        assert_eq!(count(&founder, "Before"), 2, "the store keeps its data");
        let enrollment = founder.enrollment(addr(p1)).expect("enrollment");

        let joiner = open(
            2,
            dirs[1].path(),
            p2,
            Founding::Join(enrollment.clone()),
            ElectionPreference::Ordinary,
        )
        .await
        .unwrap();
        founder
            .add_learner(2, addr(p2), PKI.identity(2))
            .await
            .unwrap();
        founder.promote(2).await.unwrap();
        wait_until("the joiner holds the founder's data", || {
            count(&joiner, "Before") == 2
        })
        .await;

        founder
            .database()
            .write()
            .execute_cypher("CREATE (:After {k: 1})")
            .unwrap();
        wait_until("a write after the growth reaches the joiner", || {
            count(&joiner, "After") == 1
        })
        .await;

        let refused = open(
            3,
            dirs[2].path(),
            p3,
            Founding::Join(enrollment),
            ElectionPreference::Ordinary,
        )
        .await;
        assert!(
            matches!(refused, Err(ReplicationError::Consensus(_))),
            "a joiner holding data is refused, got {:?}",
            refused.as_ref().err()
        );

        joiner.shutdown().await.unwrap();
        founder.shutdown().await.unwrap();
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A member that is not the leader never applies a caller's write, whatever
/// the write concern: a w:0 write on a follower is refused naming the leader
/// and lands on no member.
#[tokio::test(flavor = "multi_thread")]
async fn a_fire_and_forget_write_on_a_follower_lands_nowhere() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        wait_until("member 2 knows the leader", || {
            members[1].leader() == Some(1)
        })
        .await;

        for concern in [WriteConcern::w0(), WriteConcern::majority()] {
            let mut follower = members[1].database().write();
            follower.set_write_concern(concern);
            let refused = follower.execute_cypher("CREATE (:Stray {k: 1})");
            // A statement is refused by its executor, a transaction's commit
            // by the database: either names the leader.
            assert!(
                matches!(
                    refused,
                    Err(DatabaseError::NotLeader { leader_id: Some(1) })
                ),
                "{concern} on a follower is refused naming the leader, got {:?}",
                refused.as_ref().err()
            );
        }
        // Any entry would have reached both members by now.
        members[0]
            .database()
            .write()
            .execute_cypher("CREATE (:Marker {k: 1})")
            .unwrap();
        wait_until("the marker reaches the follower", || {
            count(&members[1], "Marker") == 1
        })
        .await;
        for m in &members {
            assert_eq!(count(m, "Stray"), 0, "no member holds the refused write");
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// The host hands leadership to a named member, as a laptop about to sleep
/// would; the call returns once that member leads and serves writes, and the
/// leadership stream reports the change.
#[tokio::test(flavor = "multi_thread")]
async fn leadership_moves_to_a_named_member_and_the_stream_reports_it() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        let mut changes = members[1].leadership();
        let first = changes.next().await;
        assert_eq!(
            first.leader,
            Some(1),
            "the stream starts at the current leader"
        );

        members[0].transfer_leadership(2).await.unwrap();
        let moved = changes.next().await;
        assert_eq!(moved.leader, Some(2));
        assert!(moved.term > first.term);
        assert!(members[1].is_leader().await);
        members[1]
            .database()
            .write()
            .execute_cypher("CREATE (:OnTheNewLeader {k: 1})")
            .unwrap();

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// An open transaction belongs to the lead it was begun in. A follower opens
/// none and names the leader instead; once the lead moves, the old leader
/// drops its transaction and neither it nor the new leader knows it any
/// more, nothing of it lands anywhere, and only the new leader opens
/// transactions.
#[tokio::test(flavor = "multi_thread")]
async fn a_transaction_ends_with_the_lead_it_was_begun_in() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        wait_until("member 2 knows the leader", || {
            members[1].leader() == Some(1)
        })
        .await;
        wait_until("member 2 refuses BEGIN naming the leader", || {
            matches!(
                members[1].database().read().begin_transaction(),
                Err(DatabaseError::NotLeader { leader_id: Some(1) })
            )
        })
        .await;

        let handle = {
            let leader = members[0].database().read();
            let handle = leader.begin_transaction().expect("the leader begins");
            leader
                .execute_in_transaction(handle, "CREATE (:OldLead {k: 1})", None)
                .expect("statement");
            handle
        };

        members[0].transfer_leadership(2).await.unwrap();
        // The tenure follows the leadership asynchronously; until it moves a
        // statement still runs on the context.
        wait_until("the old leader drops its transaction", || {
            matches!(
                members[0].database().read().execute_in_transaction(
                    handle,
                    "RETURN 1 AS one",
                    None
                ),
                Err(DatabaseError::UnknownTransaction(_))
            )
        })
        .await;
        assert!(matches!(
            members[0].database().read().commit_transaction(handle),
            Err(DatabaseError::UnknownTransaction(h)) if h == handle
        ));
        assert!(
            matches!(
                members[1].database().read().commit_transaction(handle),
                Err(DatabaseError::UnknownTransaction(h)) if h == handle
            ),
            "the new leader never held it"
        );
        assert!(matches!(
            members[0].database().read().begin_transaction(),
            Err(DatabaseError::NotLeader { leader_id: Some(2) })
        ));

        {
            let new_leader = members[1].database().read();
            let fresh = new_leader
                .begin_transaction()
                .expect("the new leader begins");
            new_leader
                .execute_in_transaction(fresh, "CREATE (:NewLead {k: 1})", None)
                .expect("statement");
            new_leader.commit_transaction(fresh).expect("commit");
        }
        wait_until("the new leader's write reaches member 1", || {
            count(&members[0], "NewLead") == 1
        })
        .await;
        for m in &members {
            assert_eq!(
                count(m, "OldLead"),
                0,
                "the aborted transaction applied nothing"
            );
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A leader cut off from its group is replaced, the group keeps taking
/// writes, and the old leader returns as a follower holding what was written
/// without it. A follower stopped and reopened on its directory catches up.
#[tokio::test(flavor = "multi_thread")]
async fn a_group_fails_over_and_its_members_rejoin() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (mut members, ports) = group(&dirs, None).await;
        members[0]
            .database()
            .write()
            .execute_cypher("CREATE (:Before {k: 1})")
            .unwrap();

        nemesis::isolate(1, &[2, 3]);
        wait_until("another member leads", || {
            matches!(members[1].leader(), Some(2 | 3))
        })
        .await;
        let new = (members[1].leader().unwrap() - 1) as usize;
        members[new]
            .database()
            .write()
            .execute_cypher("CREATE (:During {k: 1})")
            .unwrap();
        nemesis::heal();
        wait_until("the old leader holds the write made without it", || {
            count(&members[0], "During") == 1
        })
        .await;
        assert_ne!(members[0].leader(), Some(1), "it returns as a follower");

        // A follower stops, misses a write, and is reopened on its directory.
        let follower = (0..3).find(|&i| i != new).unwrap();
        let stopped = members.remove(follower);
        stopped.shutdown().await.unwrap();
        let leader = members
            .iter()
            .find(|m| m.leader() == Some(m.node_id()))
            .expect("a leader");
        leader
            .database()
            .write()
            .execute_cypher("CREATE (:Missed {k: 1})")
            .unwrap();
        let id = follower as u64 + 1;
        let founding = if id == 1 {
            Founding::Form { deployment: None }
        } else {
            Founding::Join(
                leader
                    .enrollment(addr(ports[(leader.node_id() - 1) as usize]))
                    .unwrap(),
            )
        };
        let reopened = open(
            id,
            dirs[follower].path(),
            ports[follower],
            founding,
            ElectionPreference::Ordinary,
        )
        .await
        .unwrap();
        wait_until("the reopened member catches up", || {
            count(&reopened, "Missed") == 1
        })
        .await;

        reopened.shutdown().await.unwrap();
        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// Every file under `dir` with its bytes, sorted by path.
fn listing(dir: &Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut all = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in std::fs::read_dir(&at).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                all.push((path, bytes));
            }
        }
    }
    all.sort();
    all
}

/// A member that lost every copy of its vote does not open through the host
/// entry point: the open fails with the typed refusal.
/// The leader replaces it through the handle with a new member under a new
/// node id on its own directory; a repeated call changes nothing, the old
/// identity cannot come back, and the old member's directory stays as it
/// was for whoever inspects it.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_without_its_vote_is_refused_and_replaced_through_the_handle() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (mut members, ports) = group(&dirs, None).await;
        members[0]
            .database()
            .write()
            .execute_cypher("CREATE (:Before {k: 1})")
            .unwrap();
        wait_until("member 3 holds the write", || {
            count(&members[2], "Before") == 1
        })
        .await;

        let lost = members.remove(2);
        lost.shutdown().await.unwrap();
        let dir = dirs[2].path();
        std::fs::remove_dir_all(dir.join(Partition::Raft.name())).unwrap();
        std::fs::remove_dir_all(dir.join("local-journal")).unwrap();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("consensus-vote"))
            {
                std::fs::remove_file(path).unwrap();
            }
        }
        let leader = &members[0];
        let refused = open(
            3,
            dir,
            ports[2],
            Founding::Join(leader.enrollment(addr(ports[0])).unwrap()),
            ElectionPreference::Ordinary,
        )
        .await;
        assert!(
            matches!(
                refused,
                Err(ReplicationError::Consensus(
                    coordinode_raft::cluster::RaftNodeError::PromiseUnqualified(_)
                ))
            ),
            "a member whose vote no copy proves must not open: {:?}",
            refused.err()
        );
        // The store as the refusal left it (its trees opened, so their
        // manifests were rewritten): nothing after this may touch it.
        let damaged = listing(dir);

        let new_dir = tempfile::tempdir().unwrap();
        let new_port = alloc_port();
        let replacement = open(
            4,
            new_dir.path(),
            new_port,
            Founding::Join(leader.enrollment(addr(ports[0])).unwrap()),
            ElectionPreference::Ordinary,
        )
        .await
        .unwrap();
        // The same identity is never let back.
        let same = leader.replace(3, 3, addr(ports[2]), PKI.identity(3)).await;
        assert!(
            matches!(same, Err(ReplicationError::Consensus(_))),
            "a member is not replaced by itself: {same:?}"
        );
        leader
            .replace(3, 4, addr(new_port), PKI.identity(4))
            .await
            .expect("the replacement completes");
        leader
            .replace(3, 4, addr(new_port), PKI.identity(4))
            .await
            .expect("a repeated replacement changes nothing");
        wait_until("member 4 holds the history", || {
            count(&replacement, "Before") == 1
        })
        .await;
        leader
            .database()
            .write()
            .execute_cypher("CREATE (:After {k: 1})")
            .unwrap();
        wait_until("member 4 takes new writes", || {
            count(&replacement, "After") == 1
        })
        .await;
        let back = leader.add_learner(3, addr(ports[2]), PKI.identity(3)).await;
        assert!(
            matches!(back, Err(ReplicationError::Consensus(_))),
            "the old identity is retired: {back:?}"
        );
        assert!(
            listing(dir) == damaged,
            "the replacement left the old directory as it was"
        );

        replacement.shutdown().await.unwrap();
        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A dump in `format` of a database holding two `:U` nodes under a unique
/// index and an edge with a property between them.
fn dump(format: BackupFormat) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let mut source = Database::open(dir.path()).unwrap();
    source
        .execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .unwrap();
    source
        .execute_cypher("CREATE (:U {email: 'a@x'})-[:KNOWS {since: 1}]->(:U {email: 'b@x'})")
        .unwrap();
    let interner = source.interner().unwrap();
    let snapshot = source.engine().snapshot();
    let mut buf = Vec::new();
    match format {
        BackupFormat::Json => {
            export::export_json(source.engine(), &interner, 1, &snapshot, &mut buf)
        }
        _ => export::export_binary(source.engine(), &interner, 1, &snapshot, &mut buf),
    }
    .unwrap();
    buf
}

/// The index entries `db` holds.
fn index_entries(db: &ReplicatedDatabase) -> usize {
    db.member()
        .engine()
        .prefix_scan(Partition::Idx, b"")
        .unwrap()
        .count()
}

/// A restore on a follower is refused naming the leader and writes nothing;
/// on the leader it reaches every member: the records, the edge and the
/// entries of the index built over them, so a member that leads later
/// enforces the index over the restored values.
#[tokio::test(flavor = "multi_thread")]
async fn a_restore_reaches_every_member_and_is_refused_on_a_follower() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        for format in [BackupFormat::Binary, BackupFormat::Json] {
            let dump = dump(format);
            let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
            let (members, _ports) = group(&dirs, None).await;
            wait_until("member 2 knows the leader", || {
                members[1].leader() == Some(1)
            })
            .await;

            let refused =
                members[1]
                    .database()
                    .write()
                    .restore(format, &dump, &RestoreOptions::default());
            assert!(
                matches!(refused, Err(RestoreError::NotLeader { leader_id: Some(1) })),
                "{format:?}: a restore on a follower is refused naming the leader, got {:?}",
                refused.as_ref().err()
            );

            members[0]
                .database()
                .write()
                .restore(format, &dump, &RestoreOptions::default())
                .unwrap_or_else(|e| panic!("{format:?}: restore on the leader: {e}"));
            let edges = |db: &ReplicatedDatabase| {
                db.database()
                    .write()
                    .execute_cypher("MATCH (:U)-[r:KNOWS]->(:U) RETURN r.since")
                    .map(|rows| rows.len())
                    .unwrap_or(0)
            };
            wait_until("the restored graph reaches the follower", || {
                count(&members[1], "U") == 2 && edges(&members[1]) == 1
            })
            .await;
            let held = index_entries(&members[0]);
            assert!(held > 0, "{format:?}: the leader holds index entries");
            wait_until("the index entries reach the follower", || {
                index_entries(&members[1]) == held
            })
            .await;

            members[0].transfer_leadership(2).await.unwrap();
            let duplicate = members[1]
                .database()
                .write()
                .execute_cypher("CREATE (:U {email: 'b@x'})");
            assert!(
                duplicate.is_err(),
                "{format:?}: the new leader's index holds a restored value"
            );

            for m in members.into_iter().rev() {
                m.shutdown().await.unwrap();
            }
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A member that stands last does not win an election while another member
/// can. When the only other live member lags, it wins (it holds the most
/// complete log), replicates to the lagging member, and hands leadership to
/// it once it has caught up.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_that_stands_last_leads_only_when_no_other_can() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, Some(3)).await;

        // The leader goes away: of the two left, the ordinary one wins.
        nemesis::isolate(1, &[2, 3]);
        wait_until("a new leader", || {
            members[1].leader().is_some_and(|l| l != 1)
        })
        .await;
        assert_eq!(members[1].leader(), Some(2), "the ordinary member wins");
        nemesis::heal();
        wait_until("member 1 follows 2", || members[0].leader() == Some(2)).await;

        // Member 1 falls behind while 2 and 3 commit without it.
        nemesis::isolate(1, &[2, 3]);
        for k in 0..5 {
            members[1]
                .database()
                .write()
                .execute_cypher(&format!("CREATE (:Missed {{k: {k}}})"))
                .unwrap();
        }
        // Then the leader goes away and 1 returns: 1 cannot win with its
        // shorter log, so 3 stands at last and wins.
        let mut seen_by_3 = members[2].leadership();
        nemesis::heal();
        nemesis::isolate(2, &[1, 3]);
        let mut three_led = false;
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            let change = tokio::time::timeout(
                deadline.saturating_duration_since(Instant::now()),
                seen_by_3.next(),
            )
            .await
            .expect("leadership settles on member 1");
            three_led |= change.leader == Some(3);
            if change.leader == Some(1) {
                break;
            }
        }
        assert!(three_led, "member 3 led while member 1 lagged");
        wait_until("member 1 holds every write", || {
            count(&members[0], "Missed") == 5
        })
        .await;
        nemesis::heal();

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    nemesis::heal();
    assert!(result.is_ok(), "TIMED OUT");
}

/// Growing a standalone store into a group keeps its operations: the group
/// keeps the store's domain, so a descriptor issued before the growth still
/// resolves, its retry runs nothing, an unused number of the same open
/// incarnation still writes once, and a member that joins holds the same
/// outcomes.
#[tokio::test(flavor = "multi_thread")]
async fn descriptors_issued_standalone_stay_valid_in_the_grown_group() {
    use coordinode_embed::db::StatementOptions;
    use coordinode_embed::operation::{IssuerId, OperationDescriptor, OperationOutcome, Resolve};

    fn run(db: &Database, op: &OperationDescriptor, query: &str) -> Result<(), DatabaseError> {
        db.execute_cypher_shared_with(
            query,
            None,
            None,
            &StatementOptions {
                operation: Some(*op),
                ..StatementOptions::default()
            },
        )
        .map(drop)
    }

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
        let (done, later) = {
            let alone = Database::open(dirs[0].path()).unwrap();
            let issuer = alone.enroll_issuer(IssuerId([0x29; 16])).unwrap();
            let done = issuer.next().unwrap();
            let later = issuer.next().unwrap();
            run(&alone, &done, "CREATE (:Early {k: 1})").unwrap();
            (done, later)
        };

        let (p1, p2) = (alloc_port(), alloc_port());
        let founder = open(
            1,
            dirs[0].path(),
            p1,
            Founding::Form { deployment: None },
            ElectionPreference::Ordinary,
        )
        .await
        .unwrap();
        let enrollment = founder.enrollment(addr(p1)).expect("enrollment");
        let joiner = open(
            2,
            dirs[1].path(),
            p2,
            Founding::Join(enrollment),
            ElectionPreference::Ordinary,
        )
        .await
        .unwrap();
        founder
            .add_learner(2, addr(p2), PKI.identity(2))
            .await
            .unwrap();
        founder.promote(2).await.unwrap();

        {
            let leader = founder.database().write();
            assert!(matches!(
                leader.resolve_operation(&done, Resolve::Lookup).unwrap(),
                OperationOutcome::Committed { .. }
            ));
            assert!(matches!(
                run(&leader, &done, "CREATE (:Early {k: 1})"),
                Err(DatabaseError::OperationSettled { .. })
            ));
            run(&leader, &later, "CREATE (:Later {k: 1})").unwrap();
        }
        wait_until("the joiner holds the later write", || {
            count(&joiner, "Later") == 1
        })
        .await;
        for m in [&founder, &joiner] {
            let db = m.database().write();
            for op in [&done, &later] {
                assert!(matches!(
                    db.resolve_operation(op, Resolve::Lookup).unwrap(),
                    OperationOutcome::Committed { .. }
                ));
            }
            drop(db);
            assert_eq!(count(m, "Early"), 1);
        }

        joiner.shutdown().await.unwrap();
        founder.shutdown().await.unwrap();
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A host's append-only journal writer, as a host retries it: each entry is
/// one record written in an interactive transaction under the entry's
/// operation. A commit that fails or goes unanswered is never written again
/// blindly: the writer resolves the same operation at whoever leads, and
/// writes the entry again, under a new number, only once the first is proven
/// never to take effect. Driven through the faults that leave a commit's
/// outcome unknown (its leader cut off mid-commit, a lost reply, a change of
/// leader before the retry), every entry is in the history exactly once.
#[tokio::test(flavor = "multi_thread")]
async fn a_journal_writer_resolves_an_unknown_commit_instead_of_writing_again() {
    use coordinode_embed::operation::{
        IssuerId, OperationDescriptor, OperationIssuer, OperationOutcome, Resolve,
    };

    /// Write entry `k` under `op` at `member`: begin, create, commit.
    fn write(
        member: &ReplicatedDatabase,
        k: i64,
        op: &OperationDescriptor,
    ) -> Result<(), DatabaseError> {
        let db = member.database().write();
        let txn = db.begin_transaction()?;
        db.execute_in_transaction(txn, &format!("CREATE (:Entry {{k: {k}}})"), None)?;
        db.commit_transaction_as(txn, op).map(drop)
    }

    /// The member that leads now, once one does.
    async fn leading(members: &[ReplicatedDatabase]) -> &ReplicatedDatabase {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(m) = members.iter().find(|m| m.leader() == Some(m.node_id())) {
                return m;
            }
            assert!(Instant::now() < deadline, "no member leads");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// The writer's retry after a failed commit of entry `k` under `op`:
    /// resolve the operation at the leader; write again under a new number
    /// only when it is proven aborted.
    async fn recover(
        members: &[ReplicatedDatabase],
        issuer: &OperationIssuer,
        k: i64,
        op: OperationDescriptor,
    ) {
        let mut op = op;
        loop {
            let leader = leading(members).await;
            let resolved = leader
                .database()
                .write()
                .resolve_operation(&op, Resolve::Exclude);
            match resolved {
                Ok(OperationOutcome::Committed { .. }) => return,
                Ok(OperationOutcome::Aborted) => {
                    op = issuer.next().unwrap();
                    match write(leader, k, &op) {
                        Ok(()) => return,
                        // Lost to a change of leader: resolve that one too.
                        Err(_) => continue,
                    }
                }
                Ok(other) => panic!("an exclusion answers finally, got {other:?}"),
                // The leader moved under the call: ask the next one.
                Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
            }
        }
    }

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        let issuer = members[0]
            .database()
            .write()
            .enroll_issuer(IssuerId([0x6a; 16]))
            .unwrap();
        // A bounded wait, so a commit its leader cannot get acknowledged
        // fails instead of hanging; the write may still commit afterwards.
        for m in &members {
            m.database()
                .write()
                .set_write_concern(WriteConcern::majority_with_timeout(2000));
        }

        // Entry 1: the leader is cut off while it commits. The outcome is
        // unknown to the writer; it resolves instead of writing again.
        let op = issuer.next().unwrap();
        nemesis::isolate(1, &[2, 3]);
        let failed = write(&members[0], 1, &op);
        nemesis::heal();
        assert!(
            failed.is_err(),
            "a leader cut off from its group cannot have a commit acknowledged"
        );
        recover(&members, &issuer, 1, op).await;

        // Entry 2: committed, but the reply is lost. The writer runs the
        // whole write again under the same operation: nothing runs, and the
        // answer is the commit the first attempt made.
        let leader = leading(&members).await;
        let op = issuer.next().unwrap();
        write(leader, 2, &op).unwrap();
        match write(leader, 2, &op) {
            Err(DatabaseError::OperationSettled {
                outcome: OperationOutcome::Committed { .. },
                intent_differs: false,
                ..
            }) => {}
            other => panic!("a retry of a committed entry runs nothing, got {other:?}"),
        }

        // Entry 3: committed, then leadership moves before the retry, which
        // reaches the new leader under the same operation.
        let leader = leading(&members).await;
        let op = issuer.next().unwrap();
        write(leader, 3, &op).unwrap();
        let next = members
            .iter()
            .map(|m| m.node_id())
            .find(|&id| id != leader.node_id())
            .unwrap();
        leader.transfer_leadership(next).await.unwrap();
        let leader = leading(&members).await;
        assert!(matches!(
            write(leader, 3, &op),
            Err(DatabaseError::OperationSettled {
                outcome: OperationOutcome::Committed { .. },
                ..
            })
        ));

        for m in &members {
            for k in 1..=3 {
                wait_until("every member holds each entry", || {
                    m.database()
                        .write()
                        .execute_cypher(&format!("MATCH (e:Entry {{k: {k}}}) RETURN e"))
                        .map(|rows| rows.len())
                        .unwrap_or(0)
                        >= 1
                })
                .await;
            }
            assert_eq!(count(m, "Entry"), 3, "each entry is in the history once");
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    nemesis::heal();
    assert!(result.is_ok(), "TIMED OUT");
}

/// How many `label` nodes `db`'s label counts hold: an additive operand, so
/// it shows a second application.
fn label_count(db: &ReplicatedDatabase, label: &str) -> u64 {
    db.database()
        .read()
        .label_counts()
        .unwrap()
        .get(label)
        .copied()
        .unwrap_or(0)
}

/// A schema package published on the leader is proved there and reaches
/// every member through the group's log: the label in force STRICT, its
/// uniqueness active. Publishing it again changes nothing. A follower
/// refuses a publication as not the leader, and nothing of it lands.
#[tokio::test(flavor = "multi_thread")]
async fn a_schema_package_is_published_through_the_group() {
    use coordinode_core::schema::definition::{
        ConstraintKind, ConstraintState, LabelSchema, PropertyDef, PropertyType, SchemaMode,
    };
    use coordinode_embed::{Intent, LabelDeclaration, PackageConstraint, SchemaPackage};

    let package = || {
        let mut user = LabelSchema::new_node_id("User");
        user.set_mode(SchemaMode::Strict);
        user.add_property(PropertyDef::new("email", PropertyType::String));
        SchemaPackage {
            labels: vec![LabelDeclaration {
                schema: user,
                intent: Intent::Ensure,
            }],
            constraints: vec![PackageConstraint {
                name: Some("user_email".into()),
                label: "User".into(),
                properties: vec!["email".into()],
                kind: ConstraintKind::Unique,
                scope: None,
                on_duplicate_rename: None,
            }],
            ..SchemaPackage::default()
        }
    };
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        members[0]
            .database()
            .write()
            .execute_cypher("CREATE (:User {email: 'a@x'})")
            .unwrap();

        let refused = members[1].database().read().publish_schema(package());
        assert!(
            matches!(
                refused,
                Err(DatabaseError::NotLeader { leader_id: Some(1) })
            ),
            "{refused:?}"
        );

        {
            let leader = members[0].database().read();
            let published = leader.publish_schema(package()).unwrap();
            assert!(published.committed);
            let ready = leader
                .await_schema(&published, Duration::from_secs(60))
                .unwrap();
            assert!(ready.all_ready(), "{ready:?}");
            assert!(!leader.publish_schema(package()).unwrap().committed);
        }
        let in_force = |m: &ReplicatedDatabase| {
            let db = m.database().read();
            let strict = db.label_schemas().unwrap().iter().any(|s| {
                s.name == "User" && s.mode == SchemaMode::Strict && s.candidate().is_none()
            });
            let active = db.constraints().unwrap().iter().any(|c| {
                c.constraint.name == "user_email" && c.constraint.state == ConstraintState::Active
            });
            strict && active
        };
        for m in &members {
            wait_until("the schema is in force on the member", || in_force(m)).await;
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A j:memory write on the leader is answered once the leader's log holds
/// it, and goes through the group's log like any write: it reaches every
/// member exactly once, its label count included.
#[tokio::test(flavor = "multi_thread")]
async fn a_memory_write_reaches_every_member_once() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        {
            let mut leader = members[0].database().write();
            leader.set_write_concern(WriteConcern::memory());
            leader.execute_cypher("CREATE (:Volatile {k: 1})").unwrap();
        }
        for m in &members {
            wait_until("the write reaches the member", || count(m, "Volatile") == 1).await;
            assert_eq!(label_count(m, "Volatile"), 1);
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// The leader loses the group right after it answered a j:memory write of a
/// client operation, which no majority may hold yet. If the write is lost
/// with that leadership, as the concern allows, the operation is resolved,
/// not re-run: the new leader excludes it, and then no member ever holds its
/// effect. If a majority got it first, the operation resolves committed and
/// every member holds the effect once.
#[tokio::test(flavor = "multi_thread")]
async fn a_memory_write_lost_with_its_leader_is_excluded_not_rerun() {
    use coordinode_embed::db::StatementOptions;
    use coordinode_embed::operation::{IssuerId, OperationOutcome, Resolve};

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        let op = {
            let leader = members[0].database().write();
            let op = leader
                .enroll_issuer(IssuerId([0x5c; 16]))
                .unwrap()
                .next()
                .unwrap();
            leader
                .execute_cypher_shared_with(
                    "CREATE (:Orphan {k: 1})",
                    None,
                    None,
                    &StatementOptions {
                        operation: Some(op),
                        write_concern: Some(WriteConcern::memory()),
                        ..StatementOptions::default()
                    },
                )
                .unwrap();
            op
        };
        nemesis::isolate(1, &[2, 3]);
        let survivors = &members[1..];
        let deadline = Instant::now() + Duration::from_secs(60);
        let leader = loop {
            if let Some(m) = survivors.iter().find(|m| m.leader() == Some(m.node_id())) {
                break m;
            }
            assert!(Instant::now() < deadline, "no survivor leads");
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        let outcome = leader
            .database()
            .write()
            .resolve_operation(&op, Resolve::Exclude)
            .unwrap();
        nemesis::heal();
        let expected = match outcome {
            OperationOutcome::Aborted => 0,
            OperationOutcome::Committed { .. } => 1,
            other => panic!("an exclusion answers finally, got {other:?}"),
        };
        // A marker written after the exclusion reaches every member last, so
        // the old leader's entry, had it survived, would have landed before it.
        leader
            .database()
            .write()
            .execute_cypher("CREATE (:After {k: 1})")
            .unwrap();
        for m in &members {
            wait_until("the marker reaches the member", || count(m, "After") == 1).await;
            assert_eq!(count(m, "Orphan"), expected, "{outcome:?}");
            assert_eq!(label_count(m, "Orphan"), expected as u64, "{outcome:?}");
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    nemesis::heal();
    assert!(result.is_ok(), "TIMED OUT");
}

/// A bound write's outcome lives in the group's history, not with the member
/// that led when it ran: after leadership moves, the new leader reports the
/// commit, answers a retry without running it, and excludes a number the old
/// leader handed out but never used. Exclusion is the log's decision, so a
/// follower refuses it, and every member reports the same outcomes.
#[tokio::test(flavor = "multi_thread")]
async fn an_operation_keeps_its_outcome_across_a_leader_change() {
    use coordinode_embed::db::StatementOptions;
    use coordinode_embed::operation::{IssuerId, OperationDescriptor, OperationOutcome, Resolve};

    fn run(db: &Database, op: &OperationDescriptor) -> Result<(), DatabaseError> {
        db.execute_cypher_shared_with(
            "CREATE (:Bound {k: 1})",
            None,
            None,
            &StatementOptions {
                operation: Some(*op),
                ..StatementOptions::default()
            },
        )
        .map(drop)
    }

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        let (done, unused) = {
            let leader = members[0].database().write();
            let issuer = leader.enroll_issuer(IssuerId([0x17; 16])).unwrap();
            let done = issuer.next().unwrap();
            let unused = issuer.next().unwrap();
            run(&leader, &done).unwrap();
            (done, unused)
        };

        members[0].transfer_leadership(2).await.unwrap();
        assert!(members[1].is_leader().await);
        {
            let new_leader = members[1].database().write();
            assert!(matches!(
                new_leader
                    .resolve_operation(&done, Resolve::Lookup)
                    .unwrap(),
                OperationOutcome::Committed { .. }
            ));
            assert!(matches!(
                run(&new_leader, &done),
                Err(DatabaseError::OperationSettled {
                    outcome: OperationOutcome::Committed { .. },
                    intent_differs: false,
                    ..
                })
            ));
        }
        wait_until("member 1 follows 2", || members[0].leader() == Some(2)).await;
        assert!(matches!(
            members[0]
                .database()
                .write()
                .resolve_operation(&unused, Resolve::Exclude),
            Err(DatabaseError::NotLeader { .. })
        ));
        assert_eq!(
            members[1]
                .database()
                .write()
                .resolve_operation(&unused, Resolve::Exclude)
                .unwrap(),
            OperationOutcome::Aborted
        );
        assert!(matches!(
            run(&members[1].database().write(), &unused),
            Err(DatabaseError::OperationSettled {
                outcome: OperationOutcome::Aborted,
                ..
            })
        ));

        for m in &members {
            wait_until("every member holds the exclusion", || {
                m.database()
                    .write()
                    .resolve_operation(&unused, Resolve::Lookup)
                    .is_ok_and(|o| o == OperationOutcome::Aborted)
            })
            .await;
            assert!(matches!(
                m.database()
                    .write()
                    .resolve_operation(&done, Resolve::Lookup)
                    .unwrap(),
                OperationOutcome::Committed { .. }
            ));
            assert_eq!(count(m, "Bound"), 1, "the write took effect once");
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// Open `dir` without replication and write to it; the refusal it meets.
async fn open_outside_the_group(dir: &Path) -> Option<DatabaseError> {
    let path = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        Database::open(&path)
            .and_then(|mut db| db.execute_cypher("CREATE (:Outside {k: 1})"))
            .err()
    })
    .await
    .unwrap()
}

/// A store whose history is a group's log is opened only as a member: the
/// non-replicated open refuses it, naming why, and writes nothing, so no
/// write lands outside the log and the member reopens with everything it
/// held. The trees alone show it when the member-local journal is gone.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_store_is_refused_by_the_non_replicated_open() {
    use coordinode_storage::error::StorageError;

    let dir = tempfile::tempdir().unwrap();
    let member = ReplicatedDatabase::open(&storage(dir.path()), MemberOptions::new(1), None)
        .await
        .unwrap();
    member
        .database()
        .write()
        .execute_cypher("CREATE (:InTheLog {k: 1})")
        .unwrap();
    member.shutdown().await.unwrap();

    let refused = open_outside_the_group(dir.path()).await;
    assert!(
        matches!(
            refused.as_ref().and_then(DatabaseError::storage_cause),
            Some(StorageError::GroupMemberStore {
                evidence: "its log is bound to a consensus group",
                ..
            })
        ),
        "a member's store opened without its group, got {refused:?}"
    );
    std::fs::remove_file(dir.path().join("consensus-log")).unwrap();
    let refused = open_outside_the_group(dir.path()).await;
    assert!(
        matches!(
            refused.as_ref().and_then(DatabaseError::storage_cause),
            Some(StorageError::GroupMemberStore {
                evidence: "its trees hold entries of a group's log",
                ..
            })
        ),
        "a member's store without its marker, got {refused:?}"
    );

    let member = ReplicatedDatabase::open(&storage(dir.path()), MemberOptions::new(1), None)
        .await
        .unwrap();
    assert_eq!(
        count(&member, "InTheLog"),
        1,
        "the member keeps its history"
    );
    assert_eq!(
        count(&member, "Outside"),
        0,
        "no write landed outside the log"
    );
    member
        .database()
        .write()
        .execute_cypher("CREATE (:Later {k: 1})")
        .unwrap();
    member.shutdown().await.unwrap();
}

/// Wait until every follower knows member 1 leads and where it is: a member
/// that has heard the leader's heartbeat before applying the membership that
/// names its address has nowhere to send a command yet, and says so.
async fn followers_reach_the_leader(members: &[ReplicatedDatabase]) {
    for m in &members[1..] {
        wait_until("the follower knows the leader and its address", || {
            m.leader() == Some(1) && m.member().node().member_target(1).is_some()
        })
        .await;
    }
}

/// A transaction command a host sends through `ingress`.
fn in_transaction(id: TransactionId, query: &str) -> RoutedStatement {
    RoutedStatement {
        actor: ActingContext::Trusted,
        query: query.to_owned(),
        params: Default::default(),
        settings: Default::default(),
        transaction: Some(id),
        operation: None,
        deadline: None,
        source: None,
        start: None,
    }
}

fn begin_of(id: TransactionId) -> RoutedBegin {
    RoutedBegin {
        transaction: id,
        actor: ActingContext::Trusted,
        read_concern: None,
        write_concern: None,
        timeout_outcome: None,
        operation: None,
    }
}

fn commit_of(id: TransactionId) -> RoutedCommit {
    RoutedCommit {
        actor: ActingContext::Trusted,
        transaction: id,
        operation: None,
        deadline: None,
        expect: Vec::new(),
    }
}

/// Run `query` in transaction `id` through `member`'s ingress.
async fn run_in(
    member: &ReplicatedDatabase,
    id: TransactionId,
    query: &str,
) -> Result<Routed<RoutedAnswer>, DatabaseError> {
    member
        .ingress()
        .execute_in_transaction(
            in_transaction(id, query),
            &AdmissionSignal::new(|| {}),
            &CancelFlag::new(),
            Placement::Anywhere,
        )
        .await
}

/// A host works through whichever member it opened: a write on a follower
/// runs at the leader and lands on every member, a read the follower may not
/// serve is answered by the leader, and the answer is the one the leader
/// gave. A leader-local write is never routed: off the leader it is refused
/// naming the leader and lands nowhere.
#[tokio::test(flavor = "multi_thread")]
async fn a_host_works_through_any_member() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;

        let written = members[1]
            .execute(Statement::new("CREATE (:ViaFollower {k: 7})"))
            .await
            .expect("a follower's write runs at the leader");
        assert_eq!(written.value.stats.nodes_created, 1);
        assert!(
            written.value.stats.commit_ts > 0,
            "the leader's commit is reported"
        );
        // The answer says it was forwarded and names the leader.
        assert_eq!(written.forwarded_to, Some(1));
        assert_eq!(written.leader, Some(1));
        for m in &members {
            wait_until("the routed write reaches every member", || {
                count(m, "ViaFollower") == 1
            })
            .await;
        }

        // The default preference reads from the leader: the follower routes.
        let read = members[2]
            .execute(Statement::new("MATCH (n:ViaFollower) RETURN n.k AS k"))
            .await
            .expect("a follower's read is answered");
        assert_eq!(read.value.columns, vec!["k".to_owned()]);
        assert_eq!(read.value.rows, vec![vec![Value::Int(7)]]);
        assert_eq!(read.forwarded_to, Some(1));

        // A read this member may serve is served here, and says so.
        let near = members[2]
            .execute(Statement {
                requested: Requested {
                    read_preference: Some(ReadPreference::Nearest),
                    ..Requested::default()
                },
                ..Statement::new("MATCH (n:ViaFollower) RETURN n.k AS k")
            })
            .await
            .expect("a nearest read on a follower");
        assert_eq!(near.forwarded_to, None, "served by the member reached");
        assert_eq!(near.leader, Some(1), "still names the leader");
        assert_eq!(near.value.rows, vec![vec![Value::Int(7)]]);

        let refused = members[1]
            .execute(Statement {
                placement: Placement::LeaderLocal,
                ..Statement::new("CREATE (:Stray {k: 1})")
            })
            .await;
        assert!(
            matches!(
                refused,
                Err(DatabaseError::NotLeader { leader_id: Some(1) })
            ),
            "a leader-local write off the leader, got {refused:?}"
        );
        members[0]
            .execute(Statement {
                placement: Placement::LeaderLocal,
                ..Statement::new("CREATE (:OnTheLeader {k: 1})")
            })
            .await
            .expect("a leader-local write on the leader");
        for m in &members {
            wait_until("the leader-local write reaches every member", || {
                count(m, "OnTheLeader") == 1
            })
            .await;
            assert_eq!(count(m, "Stray"), 0, "the refused write landed nowhere");
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A member that has just joined may know who leads before it learns where
/// the leader is; a command it could not send then goes again once it can,
/// within a bounded series, so the host need not wait for the membership to
/// settle. A command that may have run is never sent twice: one write lands
/// once.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_that_just_joined_sends_once_it_knows_where_the_leader_is() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        // Only the leader's id, not its address: the command may find the
        // address unknown and has to go again.
        wait_until("member 3 knows who leads", || {
            members[2].leader() == Some(1)
        })
        .await;

        let written = members[2]
            .execute(Statement::new("CREATE (:JustJoined {k: 1})"))
            .await
            .expect("sent once the leader's address is known");
        assert_eq!(written.forwarded_to, Some(1));
        let begun = members[2]
            .ingress()
            .begin(begin_of(draw_transaction_id()), Placement::Anywhere)
            .await
            .expect("a transaction through the new member");
        assert_eq!(begun.forwarded_to, Some(1));
        for m in &members {
            wait_until("the write reaches every member", || {
                count(m, "JustJoined") == 1
            })
            .await;
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A write through a follower's ingress whose answer the host never heard,
/// with the lead moving meanwhile, takes effect once: sent again under its
/// operation it is answered with the outcome, never run twice; any member
/// resolves it; an operation never sent is excluded and then refused. A
/// command that reaches the old leader directly is refused before it is
/// admitted, naming the new leader, and runs nowhere.
#[tokio::test(flavor = "multi_thread")]
async fn a_routed_write_takes_effect_once_across_a_leader_change() {
    use coordinode_embed::operation::{
        IssuerId, OperationDescriptor, OperationId, OperationOutcome, Resolve,
    };
    use coordinode_embed::replication::routing::{RoutedOperationAnswer, RoutedOperationCall};

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        let ingress = members[2].ingress();

        let issuer = IssuerId([0x2b; 16]);
        let enrolled = ingress
            .operation(
                ActingContext::Trusted,
                RoutedOperationCall::Enroll(issuer),
                Placement::Anywhere,
            )
            .await
            .expect("enrolled through a follower");
        let RoutedOperationAnswer::Enrolled {
            incarnation,
            next_sequence,
            deployment,
        } = enrolled.value
        else {
            panic!("an enrollment answers with its incarnation: {enrolled:?}");
        };
        let descriptor = |sequence| OperationDescriptor {
            op: OperationId {
                issuer,
                incarnation,
                sequence,
            },
            deployment,
        };
        let (sent, unsent) = (descriptor(next_sequence), descriptor(next_sequence + 1));
        let write = |op| Statement {
            requested: Requested {
                operation: Some(op),
                ..Requested::default()
            },
            ..Statement::new("CREATE (:Once {k: 1})")
        };

        ingress.execute(write(sent)).await.expect("the write");
        // The host never heard the answer, and the lead moves.
        members[0].transfer_leadership(2).await.unwrap();
        wait_until("every member follows 2", || {
            members.iter().all(|m| m.leader() == Some(2))
                && members[2].member().node().member_target(2).is_some()
        })
        .await;

        let again = ingress.execute(write(sent)).await;
        assert!(
            matches!(
                again,
                Err(DatabaseError::OperationSettled {
                    outcome: OperationOutcome::Committed { .. },
                    intent_differs: false,
                    ..
                })
            ),
            "sent again, the write answers its outcome: {again:?}"
        );
        let resolved = members[0]
            .ingress()
            .operation(
                ActingContext::Trusted,
                RoutedOperationCall::Resolve(sent, Resolve::Lookup),
                Placement::Anywhere,
            )
            .await
            .expect("resolved through the old leader");
        assert!(matches!(
            resolved.value,
            RoutedOperationAnswer::Outcome(OperationOutcome::Committed { .. })
        ));
        assert_eq!(resolved.forwarded_to, Some(2));

        let excluded = ingress
            .operation(
                ActingContext::Trusted,
                RoutedOperationCall::Resolve(unsent, Resolve::Exclude),
                Placement::Anywhere,
            )
            .await
            .expect("excluded");
        assert_eq!(
            excluded.value,
            RoutedOperationAnswer::Outcome(OperationOutcome::Aborted)
        );
        assert!(matches!(
            ingress.execute(write(unsent)).await,
            Err(DatabaseError::OperationSettled {
                outcome: OperationOutcome::Aborted,
                ..
            })
        ));
        // An operation of another deployment is refused before it runs.
        let foreign = OperationDescriptor {
            deployment: coordinode_core::group::DeploymentId([0xee; 16]),
            ..descriptor(next_sequence + 2)
        };
        let refused = ingress.execute(write(foreign)).await;
        assert!(
            matches!(refused, Err(DatabaseError::ForeignOperation { .. })),
            "another deployment's operation: {refused:?}"
        );

        // Straight to the member that stopped leading: refused before it is
        // admitted, naming the leader, and never passed on.
        let stale = ingress
            .admitter()
            .router(1)
            .expect("a router to member 1")
            .execute(
                RoutedStatement {
                    actor: ActingContext::Trusted,
                    query: "CREATE (:Stale {k: 1})".into(),
                    params: Default::default(),
                    settings: Default::default(),
                    transaction: None,
                    operation: None,
                    deadline: None,
                    source: None,
                    start: None,
                },
                &AdmissionSignal::new(|| {}),
            )
            .await;
        assert!(
            matches!(stale, Err(DatabaseError::NotLeader { leader_id: Some(2) })),
            "a stale target names the leader: {stale:?}"
        );

        members[1]
            .execute(Statement::new("CREATE (:Marker {k: 1})"))
            .await
            .expect("a marker on the new leader");
        for m in &members {
            wait_until("the marker reaches every member", || {
                count(m, "Marker") == 1
            })
            .await;
            assert_eq!(count(m, "Once"), 1, "the write took effect once");
            assert_eq!(count(m, "Stale"), 0, "the stale command ran nowhere");
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A transaction belongs to the caller that opened it, whichever member a
/// command comes through: a command acting for another subject, issuer or
/// namespace under the same ID is refused as if no such transaction were
/// open, before it reads or writes anything, and the owner's transaction
/// carries on untouched. Knowing an ID grants nothing.
#[tokio::test(flavor = "multi_thread")]
async fn a_transaction_answers_only_the_caller_that_opened_it() {
    use coordinode_core::txn::command::Principal;

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        let acting = |subject: &str, trust_scope: &str, namespace: &str| {
            ActingContext::Principal(Principal {
                subject: subject.into(),
                trust_scope: trust_scope.into(),
                namespace: namespace.into(),
            })
        };
        let alice = acting("alice", "idp", "app");
        let id = draw_transaction_id();
        let as_actor = |actor: &ActingContext, query: &str| RoutedStatement {
            actor: actor.clone(),
            ..in_transaction(id, query)
        };
        let run = |member: &ReplicatedDatabase, statement: RoutedStatement| {
            let ingress = member.ingress().clone();
            async move {
                ingress
                    .execute_in_transaction(
                        statement,
                        &AdmissionSignal::new(|| {}),
                        &CancelFlag::new(),
                        Placement::Anywhere,
                    )
                    .await
            }
        };

        members[1]
            .ingress()
            .begin(
                RoutedBegin {
                    actor: alice.clone(),
                    ..begin_of(id)
                },
                Placement::Anywhere,
            )
            .await
            .expect("alice begins through follower 2");
        run(&members[1], as_actor(&alice, "CREATE (:Owned {k: 1})"))
            .await
            .expect("alice writes");

        for intruder in [
            acting("mallory", "idp", "app"),
            acting("alice", "other-idp", "app"),
            acting("alice", "idp", "other-app"),
            ActingContext::Trusted,
        ] {
            let read = run(
                &members[2],
                as_actor(&intruder, "MATCH (n:Owned) RETURN count(n) AS c"),
            )
            .await;
            assert!(
                matches!(read, Err(DatabaseError::UnknownTransaction(t)) if t == id),
                "{intruder:?} reads nothing of it: {read:?}"
            );
            let commit = members[2]
                .ingress()
                .commit(
                    RoutedCommit {
                        actor: intruder.clone(),
                        ..commit_of(id)
                    },
                    Placement::Anywhere,
                )
                .await;
            assert!(
                matches!(commit, Err(DatabaseError::UnknownTransaction(t)) if t == id),
                "{intruder:?} commits nothing: {commit:?}"
            );
        }

        let own = run(
            &members[2],
            as_actor(&alice, "MATCH (n:Owned) RETURN count(n) AS c"),
        )
        .await
        .expect("alice carries on through another member");
        assert_eq!(own.value.rows, vec![vec![Value::Int(1)]]);
        members[2]
            .ingress()
            .commit(
                RoutedCommit {
                    actor: alice.clone(),
                    ..commit_of(id)
                },
                Placement::Anywhere,
            )
            .await
            .expect("alice commits");
        for m in &members {
            wait_until("the owner's write reaches every member", || {
                count(m, "Owned") == 1
            })
            .await;
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A transaction's commands pipelined through followers run in the order
/// they were sent: each is sent once the one before it has its place at the
/// leader, not once it answered, alternating between two members, and each
/// sees exactly the commands sent before it.
#[tokio::test(flavor = "multi_thread")]
async fn pipelined_commands_run_in_the_order_they_were_sent() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        let id = draw_transaction_id();
        members[1]
            .ingress()
            .begin(begin_of(id), Placement::Anywhere)
            .await
            .expect("begun");

        let mut answers = Vec::new();
        for k in 0..12_i64 {
            let ingress = members[1 + (k as usize % 2)].ingress().clone();
            let placed = Arc::new(tokio::sync::Notify::new());
            let signal = {
                let placed = Arc::clone(&placed);
                AdmissionSignal::new(move || placed.notify_one())
            };
            let mut statement = in_transaction(
                id,
                "MATCH (n:Seq) WITH count(n) AS i CREATE (:Seq {i: i, k: $k})",
            );
            statement.params.insert("k".into(), Value::Int(k));
            answers.push(tokio::spawn(async move {
                ingress
                    .execute_in_transaction(
                        statement,
                        &signal,
                        &CancelFlag::new(),
                        Placement::Anywhere,
                    )
                    .await
            }));
            // The next command goes once this one has its place, before it
            // has answered.
            placed.notified().await;
        }
        for answer in answers {
            answer.await.unwrap().expect("each command ran");
        }
        let order = run_in(
            &members[2],
            id,
            "MATCH (n:Seq) WHERE n.i <> n.k RETURN count(n) AS out_of_order",
        )
        .await
        .expect("read back");
        assert_eq!(
            order.value.rows,
            vec![vec![Value::Int(0)]],
            "every command saw exactly the ones sent before it"
        );
        members[1]
            .ingress()
            .commit(commit_of(id), Placement::Anywhere)
            .await
            .expect("committed");
        for m in &members {
            wait_until("the pipeline's writes reach every member", || {
                count(m, "Seq") == 12
            })
            .await;
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// One transaction through two followers: begun through one, written and
/// read back through the other, committed through the first. Its writes are
/// its own until the commit, then land on every member once. A start naming
/// the open ID is refused, a second commit finds nothing, and a leader-local
/// start off the leader is refused naming the leader.
#[tokio::test(flavor = "multi_thread")]
async fn a_transaction_runs_through_two_followers() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        let id = draw_transaction_id();

        members[1]
            .ingress()
            .begin(begin_of(id), Placement::Anywhere)
            .await
            .expect("begun through follower 2");
        run_in(&members[2], id, "CREATE (:InTxn {k: 1})")
            .await
            .expect("written through follower 3");
        let own = run_in(&members[2], id, "MATCH (n:InTxn) RETURN count(n) AS c")
            .await
            .expect("read back through follower 3");
        assert_eq!(
            own.value.rows,
            vec![vec![Value::Int(1)]],
            "it reads its own write"
        );
        assert_eq!(own.forwarded_to, Some(1), "run at the leader that holds it");
        assert_eq!(
            count(&members[0], "InTxn"),
            0,
            "nothing applied before commit"
        );

        let taken = members[2]
            .ingress()
            .begin(begin_of(id), Placement::Anywhere)
            .await;
        assert!(
            matches!(taken, Err(DatabaseError::TransactionIdInUse(t)) if t == id),
            "a start naming an open ID, got {taken:?}"
        );

        let receipt = members[1]
            .ingress()
            .commit(commit_of(id), Placement::Anywhere)
            .await
            .expect("committed through follower 2");
        assert!(receipt.value.commit_ts.as_raw() > 0);
        assert_eq!(receipt.forwarded_to, Some(1));
        assert_eq!(receipt.leader, Some(1));
        for m in &members {
            wait_until("the transaction reaches every member", || {
                count(m, "InTxn") == 1
            })
            .await;
        }
        let again = members[2]
            .ingress()
            .commit(commit_of(id), Placement::Anywhere)
            .await;
        assert!(
            matches!(again, Err(DatabaseError::UnknownTransaction(t)) if t == id),
            "a committed transaction is gone, got {again:?}"
        );

        let local = members[1]
            .ingress()
            .begin(begin_of(draw_transaction_id()), Placement::LeaderLocal)
            .await;
        assert!(
            matches!(local, Err(DatabaseError::NotLeader { leader_id: Some(1) })),
            "a leader-local start off the leader, got {local:?}"
        );

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A transaction does not outlive the lead it was begun in, whichever member
/// the host reaches it through: after the lead moves the new leader answers
/// that it knows no such transaction, the old leader leading again does not
/// revive it, and nothing of it lands anywhere.
#[tokio::test(flavor = "multi_thread")]
async fn a_routed_transaction_ends_with_its_lead_and_stays_ended() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        let id = draw_transaction_id();
        members[1]
            .ingress()
            .begin(begin_of(id), Placement::Anywhere)
            .await
            .expect("begun through follower 2");
        run_in(&members[1], id, "CREATE (:LostWithItsLead {k: 1})")
            .await
            .expect("written through follower 2");

        members[0].transfer_leadership(3).await.unwrap();
        wait_until("member 2 follows the new leader", || {
            members[1].leader() == Some(3)
        })
        .await;
        let gone = run_in(&members[1], id, "RETURN 1 AS one").await;
        assert!(
            matches!(gone, Err(DatabaseError::UnknownTransaction(t)) if t == id),
            "the new leader never held it, got {gone:?}"
        );

        members[2].transfer_leadership(1).await.unwrap();
        wait_until("member 2 follows member 1 again", || {
            members[1].leader() == Some(1)
        })
        .await;
        let still = members[1]
            .ingress()
            .commit(commit_of(id), Placement::Anywhere)
            .await;
        assert!(
            matches!(still, Err(DatabaseError::UnknownTransaction(t)) if t == id),
            "leading again does not revive it, got {still:?}"
        );
        members[0]
            .execute(Statement::new("CREATE (:Marker {k: 1})"))
            .await
            .expect("a marker through the leader");
        for m in &members {
            wait_until("the marker reaches every member", || {
                count(m, "Marker") == 1
            })
            .await;
            assert_eq!(count(m, "LostWithItsLead"), 0, "nothing of it landed");
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A transaction a host opened through a follower and then left idle ends at
/// the leader with its timeout outcome, without any further command: one
/// begun to roll back leaves nothing and answers a late commit as expired;
/// one begun to commit at expiry lands on every member exactly once, and its
/// operation resolves to that commit through any member.
#[tokio::test(flavor = "multi_thread")]
async fn an_idle_routed_transaction_ends_with_its_timeout_outcome() {
    use coordinode_core::txn::command::TimeoutOutcome;
    use coordinode_embed::TransactionEnd;
    use coordinode_embed::operation::{
        IssuerId, OperationDescriptor, OperationId, OperationOutcome, Resolve,
    };
    use coordinode_embed::replication::routing::{RoutedOperationAnswer, RoutedOperationCall};

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        for m in &members {
            m.database()
                .write()
                .set_interactive_idle_timeout(Duration::from_millis(300));
        }
        let ingress = members[1].ingress();

        let issuer = IssuerId([0x3c; 16]);
        let enrolled = ingress
            .operation(
                ActingContext::Trusted,
                RoutedOperationCall::Enroll(issuer),
                Placement::Anywhere,
            )
            .await
            .expect("enrolled through a follower");
        let RoutedOperationAnswer::Enrolled {
            incarnation,
            next_sequence,
            deployment,
        } = enrolled.value
        else {
            panic!("an enrollment answers with its incarnation: {enrolled:?}");
        };
        let op = OperationDescriptor {
            op: OperationId {
                issuer,
                incarnation,
                sequence: next_sequence,
            },
            deployment,
        };

        let rolled = draw_transaction_id();
        ingress
            .begin(begin_of(rolled), Placement::Anywhere)
            .await
            .expect("begun to roll back");
        run_in(&members[1], rolled, "CREATE (:RolledAtExpiry {k: 1})")
            .await
            .expect("written");
        let committed = draw_transaction_id();
        ingress
            .begin(
                RoutedBegin {
                    timeout_outcome: Some(TimeoutOutcome::Commit),
                    operation: Some(op),
                    ..begin_of(committed)
                },
                Placement::Anywhere,
            )
            .await
            .expect("begun to commit at expiry");
        run_in(&members[1], committed, "CREATE (:CommittedAtExpiry {k: 1})")
            .await
            .expect("written");

        // Nothing more is sent: the leader's reaper ends both.
        for m in &members {
            wait_until("the transaction committed at expiry lands", || {
                count(m, "CommittedAtExpiry") == 1
            })
            .await;
        }
        let late = ingress.commit(commit_of(rolled), Placement::Anywhere).await;
        assert!(
            matches!(
                late,
                Err(DatabaseError::TransactionAborted {
                    end: TransactionEnd::IdleExpired,
                    ..
                })
            ),
            "a late commit of the rolled-back one, got {late:?}"
        );
        let resolved = members[2]
            .ingress()
            .operation(
                ActingContext::Trusted,
                RoutedOperationCall::Resolve(op, Resolve::Lookup),
                Placement::Anywhere,
            )
            .await
            .expect("resolved through another follower");
        assert!(
            matches!(
                resolved.value,
                RoutedOperationAnswer::Outcome(OperationOutcome::Committed { .. })
            ),
            "the operation resolves to the commit at expiry: {resolved:?}"
        );

        members[0]
            .execute(Statement::new("CREATE (:Marker {k: 1})"))
            .await
            .expect("a marker through the leader");
        for m in &members {
            wait_until("the marker reaches every member", || {
                count(m, "Marker") == 1
            })
            .await;
            assert_eq!(
                count(m, "RolledAtExpiry"),
                0,
                "the rolled-back one left nothing"
            );
            assert_eq!(count(m, "CommittedAtExpiry"), 1, "committed once");
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// Node `id`'s security context under an embedding host's profile, which
/// identifies peers as `adapter` says.
fn host_context(
    id: u64,
    adapter: &Arc<coordinode_test_fixtures::pki::UriAdapter>,
) -> Arc<coordinode_embed::replication::SecurityContext> {
    let (chain, key) = PKI.node_credentials(id);
    Arc::new(
        coordinode_embed::replication::SecurityContext::new(
            PKI.crypto_provider(),
            chain,
            key,
            PKI.host_profile(Arc::clone(adapter)),
        )
        .expect("a node security context"),
    )
}

/// A group of members `1..=dirs.len()` under the host profile of `adapter`.
async fn host_group(
    dirs: &[tempfile::TempDir],
    adapter: &Arc<coordinode_test_fixtures::pki::UriAdapter>,
) -> Vec<ReplicatedDatabase> {
    let ports: Vec<u16> = dirs.iter().map(|_| alloc_port()).collect();
    let founder = open_under(
        1,
        dirs[0].path(),
        ports[0],
        Founding::Form { deployment: None },
        ElectionPreference::Ordinary,
        host_context(1, adapter),
    )
    .await
    .unwrap();
    let enrollment = founder.enrollment(addr(ports[0])).expect("enrollment");
    let mut members = vec![founder];
    for (i, dir) in dirs.iter().enumerate().skip(1) {
        let id = i as u64 + 1;
        members.push(
            open_under(
                id,
                dir.path(),
                ports[i],
                Founding::Join(enrollment.clone()),
                ElectionPreference::Ordinary,
                host_context(id, adapter),
            )
            .await
            .unwrap(),
        );
        members[0]
            .add_learner(id, addr(ports[i]), PKI.host_identity(id, adapter))
            .await
            .unwrap();
        members[0].promote(id).await.unwrap();
    }
    members
}

/// The credentials a routed transaction travels under change while it is
/// open. A member renewing its certificate keeps sending the transaction's
/// commands; once the host revokes a member, the leader stops taking its
/// commands on the connection already made, so a statement sent through it
/// runs nowhere, and the transaction carries on through another member
/// with nothing of that statement in it.
#[tokio::test(flavor = "multi_thread")]
async fn a_routed_transaction_outlives_a_renewal_and_a_revoked_member() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let adapter = coordinode_test_fixtures::pki::UriAdapter::new("host-routing");
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let members = host_group(&dirs, &adapter).await;
        followers_reach_the_leader(&members).await;
        let id = draw_transaction_id();
        members[1]
            .ingress()
            .begin(begin_of(id), Placement::Anywhere)
            .await
            .expect("begun through follower 2");
        run_in(&members[1], id, "CREATE (:Credentials {step: 'before'})")
            .await
            .expect("written before the renewal");

        let (chain, key) = PKI.node_credentials(2);
        members[1]
            .security()
            .expect("peers")
            .update_credentials(chain, key)
            .expect("a renewal keeping the identity");
        run_in(&members[1], id, "CREATE (:Credentials {step: 'renewed'})")
            .await
            .expect("written after the renewal");

        adapter.revoke(&PKI.spiffe_id(2));
        members[0].security().expect("peers").reverify_peers();
        let refused = run_in(&members[1], id, "CREATE (:Credentials {step: 'revoked'})").await;
        assert!(
            matches!(
                refused,
                Err(DatabaseError::PeerUnreachable { .. }
                    | DatabaseError::PeerOutcomeUnknown { .. })
            ),
            "a revoked member's command, got {refused:?}"
        );

        let receipt = members[2]
            .ingress()
            .commit(commit_of(id), Placement::Anywhere)
            .await
            .expect("committed through follower 3");
        assert_eq!(receipt.forwarded_to, Some(1));
        let rows = members[0]
            .database()
            .write()
            .execute_cypher("MATCH (n:Credentials) RETURN n.step AS step ORDER BY step")
            .unwrap();
        let steps: Vec<_> = rows.iter().map(|row| row.get("step").cloned()).collect();
        assert_eq!(
            steps,
            vec![
                Some(Value::String("before".into())),
                Some(Value::String("renewed".into())),
            ],
            "the revoked member's statement ran nowhere"
        );

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A write that reaches a follower goes to the leader before anything of it
/// runs there: the follower's admission names the leader for it, while a
/// read the follower may serve runs where it is. Running it first and
/// learning from consensus that it cannot be taken would cost the whole
/// statement twice.
#[tokio::test(flavor = "multi_thread")]
async fn a_write_on_a_follower_is_sent_to_the_leader_before_it_runs() {
    use coordinode_embed::replication::admission::{Admission, Requested};

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        let admitter = members[1].ingress().admitter().clone();
        // A session that reads from the nearest member sends its writes with
        // the same settings.
        let nearest = Requested {
            read_preference: Some(ReadPreference::Nearest),
            ..Requested::default()
        };

        for write in [
            "CREATE (:Forwarded {k: 1})",
            "MATCH (n:Forwarded) SET n.k = 2",
            "CALL { CREATE (:Forwarded {k: 3}) }",
        ] {
            let admitted = admitter
                .admit(write, &nearest, false)
                .await
                .expect("admitted");
            assert!(
                matches!(admitted, Admission::Forward(1)),
                "{write}: {admitted:?}"
            );
        }
        let read = admitter
            .admit("MATCH (n:Forwarded) RETURN n", &nearest, false)
            .await
            .expect("admitted");
        assert!(matches!(read, Admission::Run(_)), "{read:?}");

        // Routed once already, a write the member cannot take is refused
        // naming the leader, never routed on.
        let routed = admitter
            .admit("CREATE (:Forwarded {k: 4})", &nearest, true)
            .await;
        assert!(
            matches!(routed, Err(DatabaseError::NotLeader { leader_id: Some(1) })),
            "{routed:?}"
        );

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A transaction opened through a follower opens with its first statement:
/// the statement carries the start, and the leader opens the transaction and
/// runs the statement in it in one call, with no begin before it. A second
/// start under the same ID is refused before its statement runs, and the
/// transaction it names carries on untouched.
#[tokio::test(flavor = "multi_thread")]
async fn a_transaction_opens_with_its_first_statement() {
    use coordinode_embed::replication::routing::TransactionStart;

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        let id = draw_transaction_id();
        let starting = |query: &str| RoutedStatement {
            start: Some(TransactionStart::default()),
            ..in_transaction(id, query)
        };
        let ingress = members[1].ingress();

        let first = ingress
            .execute_in_transaction(
                starting("CREATE (:Started {k: 1})"),
                &AdmissionSignal::new(|| {}),
                &CancelFlag::new(),
                Placement::Anywhere,
            )
            .await
            .expect("opened by its first statement");
        assert_eq!(first.forwarded_to, Some(1));

        let again = ingress
            .execute_in_transaction(
                starting("CREATE (:Started {k: 2})"),
                &AdmissionSignal::new(|| {}),
                &CancelFlag::new(),
                Placement::Anywhere,
            )
            .await;
        assert!(
            matches!(again, Err(DatabaseError::TransactionIdInUse(t)) if t == id),
            "a second start under an open ID, got {again:?}"
        );

        let own = run_in(&members[2], id, "MATCH (n:Started) RETURN count(n) AS c")
            .await
            .expect("read back through another follower");
        assert_eq!(own.value.rows, vec![vec![Value::Int(1)]]);
        members[2]
            .ingress()
            .commit(commit_of(id), Placement::Anywhere)
            .await
            .expect("committed");
        for m in &members {
            wait_until("the transaction reaches every member", || {
                count(m, "Started") == 1
            })
            .await;
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A leader cut off from its group, once the others elected another, takes
/// nothing more: a write, a transaction's statement and its commit through
/// it are refused, none applies there, and none reaches the group when the
/// link returns. A read it may serve itself is still answered, without
/// waiting for a leader.
#[tokio::test(flavor = "multi_thread")]
async fn a_leader_cut_off_from_its_group_takes_nothing_more() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (members, _ports) = group(&dirs, None).await;
        followers_reach_the_leader(&members).await;
        members[0]
            .execute(Statement::new("CREATE (:BeforeTheSplit {k: 1})"))
            .await
            .expect("written while the group stands");
        let id = draw_transaction_id();
        members[0]
            .ingress()
            .begin(begin_of(id), Placement::Anywhere)
            .await
            .expect("begun on the leader before the split");

        nemesis::isolate(1, &[2, 3]);
        wait_until("the others elect another leader", || {
            matches!(members[1].leader(), Some(2 | 3))
        })
        .await;
        let cut_off = &members[0];

        // Refused as off the leader, typed, so the host knows it ran nothing.
        let write = cut_off
            .execute(Statement::new("CREATE (:CutOff {k: 1})"))
            .await;
        assert!(
            matches!(write, Err(DatabaseError::NotLeader { .. })),
            "the cut-off leader takes no write: {write:?}"
        );
        let statement = run_in(cut_off, id, "CREATE (:CutOff {k: 2})").await;
        assert!(
            matches!(statement, Err(DatabaseError::NotLeader { .. })),
            "nor a transaction's statement: {statement:?}"
        );
        // The refused statement ended its transaction: nothing commits.
        let commit = cut_off
            .ingress()
            .commit(commit_of(id), Placement::Anywhere)
            .await;
        assert!(
            matches!(
                commit,
                Err(DatabaseError::TransactionAborted { .. } | DatabaseError::UnknownTransaction(_))
            ),
            "a transaction begun before the split does not commit there: {commit:?}"
        );

        let near = cut_off
            .execute(Statement {
                requested: Requested {
                    read_preference: Some(ReadPreference::Nearest),
                    ..Requested::default()
                },
                ..Statement::new("MATCH (n:BeforeTheSplit) RETURN n.k AS k")
            })
            .await
            .expect("a read the member serves itself");
        assert_eq!(near.forwarded_to, None);
        assert_eq!(near.value.rows, vec![vec![Value::Int(1)]]);

        nemesis::heal();
        let leader = members[1].leader().expect("the group leads");
        members[(leader - 1) as usize]
            .execute(Statement::new("CREATE (:Marker {k: 1})"))
            .await
            .expect("a marker on the new leader");
        for m in &members {
            wait_until("the marker reaches every member", || {
                count(m, "Marker") == 1
            })
            .await;
            assert_eq!(
                count(m, "CutOff"),
                0,
                "nothing of the cut-off leader landed"
            );
        }

        for m in members.into_iter().rev() {
            m.shutdown().await.unwrap();
        }
    })
    .await;
    nemesis::heal();
    assert!(result.is_ok(), "TIMED OUT");
}
