//! What routing a command to the leader costs against running it at the
//! leader, on a three-member group on this host: latency (P50/P99/P999) and
//! allocations per command for an autonomous write, a read the leader must
//! serve, and a transaction of one statement, each sent through the leader
//! and through a follower in alternating order. Then a large answer read
//! slowly through a follower: the heap it takes stays a few batches,
//! whatever the answer's size.
//!
//! Both members run in this process, so a routed command's allocations
//! include the leader's share; the difference between the arms is what the
//! route adds.
#![allow(clippy::expect_used, clippy::print_stdout, clippy::unwrap_used)]

use std::alloc::System;
use std::path::Path;
use std::time::{Duration, Instant};

use coordinode_core::txn::command::{ActingContext, AdmissionSignal};
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_embed::db::draw_transaction_id;
use coordinode_embed::replication::admission::Requested;
use coordinode_embed::replication::ingress::{Placement, Statement};
use coordinode_embed::replication::routing::{
    RoutedBegin, RoutedCommit, RoutedSettings, RoutedStatement,
};
use coordinode_embed::replication::{
    ElectionPreference, Founding, MemberOptions, PeerTimeouts, Peering, ReplicatedDatabase,
};
use coordinode_raft::read_fence::ReadPreference;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Bytes allocated and not yet freed since `region` began. A reallocation
/// is already counted in both totals by its growth or shrinkage.
fn held(region: &Region<'_, System>) -> isize {
    let change = region.change();
    change.bytes_allocated as isize - change.bytes_deallocated as isize
}

/// Commands per arm.
const ROUNDS: usize = 1000;

/// Nodes the large answer returns (`ROUTING_BENCH_ROWS` overrides it, to see
/// the slow consumer's heap stay flat as the answer grows), and the bytes of
/// each one's payload.
const LARGE_ROWS: usize = 40_000;
const PAYLOAD: usize = 512;

fn large_rows() -> usize {
    std::env::var("ROUTING_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(LARGE_ROWS)
}

static PKI: std::sync::LazyLock<coordinode_test_fixtures::TestPki> =
    std::sync::LazyLock::new(coordinode_test_fixtures::TestPki::new);

/// One arm's measurements.
#[derive(Default)]
struct Arm {
    times: Vec<Duration>,
    allocs: usize,
    bytes: usize,
}

impl Arm {
    async fn measure<F: std::future::Future<Output = ()>>(&mut self, run: F) {
        let region = Region::new(GLOBAL);
        let start = Instant::now();
        run.await;
        self.times.push(start.elapsed());
        let change = region.change();
        self.allocs += change.allocations;
        self.bytes += change.bytes_allocated;
    }

    fn report(mut self, label: &str) {
        let n = self.times.len();
        self.times.sort_unstable();
        let at = |q: usize| self.times[(n - 1) * q / 1000];
        println!(
            "{label}: n={n} p50={:?} p99={:?} p999={:?} allocs/op={} bytes/op={}",
            at(500),
            at(990),
            at(999),
            self.allocs / n,
            self.bytes / n
        );
    }
}

fn storage(dir: &Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Ssd,
        Durability::Durable,
        Tier::Warm,
    )])
}

async fn open(id: u64, dir: &Path, port: u16, founding: Founding) -> ReplicatedDatabase {
    let mut options = MemberOptions::new(id);
    options.preference = ElectionPreference::Ordinary;
    options.peering = Some(Peering {
        security: PKI.context(id),
        advertise: format!("127.0.0.1:{port}"),
        founding,
        timeouts: PeerTimeouts::default(),
    });
    ReplicatedDatabase::open(
        &storage(dir),
        options,
        Some(format!("127.0.0.1:{port}").parse().unwrap()),
    )
    .await
    .expect("open a member")
}

/// Members 1..=3, member 1 leading, every follower knowing where it is.
async fn group(dirs: &[tempfile::TempDir]) -> Vec<ReplicatedDatabase> {
    let ports: Vec<u16> = dirs
        .iter()
        .map(|_| coordinode_test_fixtures::alloc_port())
        .collect();
    let founder = open(
        1,
        dirs[0].path(),
        ports[0],
        Founding::Form { deployment: None },
    )
    .await;
    let enrollment = founder
        .enrollment(format!("127.0.0.1:{}", ports[0]))
        .expect("enrollment");
    let mut members = vec![founder];
    for (i, dir) in dirs.iter().enumerate().skip(1) {
        let id = i as u64 + 1;
        members.push(open(id, dir.path(), ports[i], Founding::Join(enrollment.clone())).await);
        members[0]
            .add_learner(id, format!("127.0.0.1:{}", ports[i]), PKI.identity(id))
            .await
            .expect("add");
        members[0].promote(id).await.expect("promote");
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    while !members[1..]
        .iter()
        .all(|m| m.leader() == Some(1) && m.member().node().member_target(1).is_some())
    {
        assert!(Instant::now() < deadline, "the followers find the leader");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    members
}

/// A read the leader has to serve: a follower may not answer it.
fn leader_read(k: usize) -> Statement {
    Statement {
        requested: Requested {
            read_concern: Some(ReadConcernLevel::Majority),
            read_preference: Some(ReadPreference::Primary),
            ..Requested::default()
        },
        ..Statement::new(format!("MATCH (n:Bench {{k: {k}}}) RETURN n.k AS k"))
    }
}

async fn transaction(member: &ReplicatedDatabase, k: usize) {
    let id = draw_transaction_id();
    let ingress = member.ingress();
    ingress
        .begin(
            RoutedBegin {
                transaction: id,
                actor: ActingContext::Trusted,
                read_concern: None,
                write_concern: None,
                timeout_outcome: None,
                operation: None,
            },
            Placement::Anywhere,
        )
        .await
        .expect("begin");
    ingress
        .execute_in_transaction(
            RoutedStatement {
                actor: ActingContext::Trusted,
                query: format!("CREATE (:InTxn {{k: {k}}})"),
                params: Default::default(),
                settings: RoutedSettings::default(),
                transaction: Some(id),
                operation: None,
                deadline: None,
                source: None,
            },
            &AdmissionSignal::new(|| {}),
            &coordinode_core::budget::CancelFlag::new(),
            Placement::Anywhere,
        )
        .await
        .expect("statement");
    ingress
        .commit(
            RoutedCommit {
                actor: ActingContext::Trusted,
                transaction: id,
                operation: None,
                deadline: None,
                expect: Vec::new(),
            },
            Placement::Anywhere,
        )
        .await
        .expect("commit");
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
    let members = group(&dirs).await;
    let (leader, follower) = (&members[0], &members[1]);

    let (mut write_direct, mut write_routed) = (Arm::default(), Arm::default());
    let (mut read_direct, mut read_routed) = (Arm::default(), Arm::default());
    let (mut txn_direct, mut txn_routed) = (Arm::default(), Arm::default());
    for round in 0..ROUNDS {
        // Neither arm always runs first.
        let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        for arm in order {
            let (member, write, read, txn) = if arm == 0 {
                (leader, &mut write_direct, &mut read_direct, &mut txn_direct)
            } else {
                (
                    follower,
                    &mut write_routed,
                    &mut read_routed,
                    &mut txn_routed,
                )
            };
            let k = round * 2 + arm;
            write
                .measure(async {
                    member
                        .ingress()
                        .execute(Statement::new(format!("CREATE (:Bench {{k: {k}}})")))
                        .await
                        .expect("write");
                })
                .await;
            read.measure(async {
                member
                    .ingress()
                    .execute(leader_read(k))
                    .await
                    .expect("read");
            })
            .await;
            txn.measure(transaction(member, k)).await;
        }
    }
    write_direct.report("write at the leader");
    write_routed.report("write through a follower");
    read_direct.report("leader read at the leader");
    read_routed.report("leader read through a follower");
    txn_direct.report("transaction at the leader");
    txn_routed.report("transaction through a follower");

    slow_consumer(leader, follower).await;

    for m in members.into_iter().rev() {
        m.shutdown().await.expect("shutdown");
    }
}

/// A large answer read through a follower by a consumer that takes a batch
/// at a time with a pause between: the heap held at the peak, above what was
/// live before the read, against the answer's size.
async fn slow_consumer(leader: &ReplicatedDatabase, follower: &ReplicatedDatabase) {
    use coordinode_core::graph::types::Value;
    // Payloads that barely compress, so the answer's size on the wire is
    // close to its size here.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut payload = || {
        (0..PAYLOAD)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let n = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 58) as u8;
                char::from(
                    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"[n as usize],
                )
            })
            .collect::<String>()
    };
    const SEED_CHUNK: usize = 2000;
    for base in (0..large_rows()).step_by(SEED_CHUNK) {
        let mut statement = Statement::new(
            "UNWIND range(0, size($ps) - 1) AS i CREATE (:Large {k: $base + i, payload: $ps[i]})",
        );
        statement
            .params
            .insert("base".into(), Value::Int(base as i64));
        statement.params.insert(
            "ps".into(),
            Value::Array((0..SEED_CHUNK).map(|_| Value::String(payload())).collect()),
        );
        leader.ingress().execute(statement).await.expect("seed");
    }
    let query = "MATCH (n:Large) RETURN n.k AS k, n.payload AS payload";
    let routed = || RoutedStatement {
        actor: ActingContext::Trusted,
        query: query.to_owned(),
        params: Default::default(),
        settings: RoutedSettings {
            read_concern: Some(ReadConcernLevel::Majority),
            ..RoutedSettings::default()
        },
        transaction: None,
        operation: None,
        deadline: None,
        source: None,
    };
    let router = follower
        .ingress()
        .admitter()
        .router(1)
        .expect("a router to the leader");

    assert!(
        leader.database().read().keyset_pageable(query),
        "the leader answers it page by page"
    );
    // Once whole first, so the storage's caches are warm and what follows
    // measures the answer, not the cache filling.
    drop(
        router
            .execute(routed(), &AdmissionSignal::new(|| {}))
            .await
            .expect("warm-up"),
    );

    let region = Region::new(GLOBAL);
    let start = Instant::now();
    let mut rows = router
        .execute_rows(routed(), &AdmissionSignal::new(|| {}))
        .await
        .expect("answering");
    let mut read = 0;
    let mut slow_peak = 0;
    loop {
        let piece = rows.next_rows(256).await.expect("rows");
        // Sampled at each piece: the leader has had a pause to run ahead.
        slow_peak = slow_peak.max(held(&region));
        if piece.is_empty() {
            break;
        }
        read += piece.len();
        drop(piece);
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    println!(
        "slow consumer through a follower: rows={read} answer~{} KiB elapsed={:?} peak heap above start={} KiB",
        read * (PAYLOAD + 16) / 1024,
        start.elapsed(),
        slow_peak / 1024
    );
    drop(rows);

    // For scale: the same answer taken whole holds all of it at once.
    let region = Region::new(GLOBAL);
    let start = Instant::now();
    let whole = router
        .execute(routed(), &AdmissionSignal::new(|| {}))
        .await
        .expect("answer");
    println!(
        "whole answer collected at the ingress: rows={} elapsed={:?} heap held by it={} KiB",
        whole.rows.len(),
        start.elapsed(),
        held(&region) / 1024
    );
}
