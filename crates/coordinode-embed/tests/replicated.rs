//! A replicated embedded database, through the entry point a host opens:
//! growth in place, refusals, leadership, failover and election preference.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::time::{Duration, Instant};

use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_embed::backup::restore::{RestoreError, RestoreOptions};
use coordinode_embed::backup::{BackupFormat, export};
use coordinode_embed::replication::{
    ElectionPreference, Founding, MemberOptions, PeerTimeouts, Peering, ReplicatedDatabase,
    ReplicationError,
};
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::executor::runner::ExecutionError;
use coordinode_raft::cluster::nemesis;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::alloc_port;

/// The CA the members of these tests are issued their identities by.
static PKI: std::sync::LazyLock<coordinode_test_fixtures::TestPki> =
    std::sync::LazyLock::new(coordinode_test_fixtures::TestPki::new);

const TEST_TIMEOUT: Duration = Duration::from_secs(180);

fn storage(dir: &Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
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
    let mut options = MemberOptions::new(id);
    options.preference = preference;
    options.peering = Some(Peering {
        security: PKI.context(id),
        advertise: addr(port),
        founding,
        timeouts: PeerTimeouts::default(),
    });
    ReplicatedDatabase::open(&storage(dir), options, Some(addr(port).parse().unwrap())).await
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
    let ports: Vec<u16> = dirs.iter().map(|_| alloc_port()).collect();
    let preference = |id: u64| {
        if last == Some(id) {
            ElectionPreference::Last
        } else {
            ElectionPreference::Ordinary
        }
    };
    let founder = open(
        1,
        dirs[0].path(),
        ports[0],
        Founding::Form { deployment: None },
        preference(1),
    )
    .await
    .unwrap();
    let enrollment = founder.enrollment(addr(ports[0])).expect("enrollment");
    let mut members = vec![founder];
    for (i, dir) in dirs.iter().enumerate().skip(1) {
        let id = i as u64 + 1;
        members.push(
            open(
                id,
                dir.path(),
                ports[i],
                Founding::Join(enrollment.clone()),
                preference(id),
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
                    Err(DatabaseError::NotLeader { leader_id: Some(1) }
                        | DatabaseError::Execution(ExecutionError::NotLeader {
                            leader_id: Some(1)
                        }))
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
        let txn = db.begin_transaction();
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
