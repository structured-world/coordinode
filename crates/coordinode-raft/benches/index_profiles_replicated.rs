//! What each index maintenance profile costs a replicated write.
//!
//! A three-member group on this host commits the same unit, a node record
//! with one indexed value, maintained RESOLVED (the entry travels in the unit)
//! and DERIVED (the change travels, and every member derives the entry). For
//! each profile: commit latency through the leader, as a distribution, and
//! the bytes the leader's Raft log grew by per entry, which every member
//! stores and receives.
//!
//! Run: cargo bench -p coordinode-raft --bench index_profiles_replicated

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::{IndexInterpretation, KEY_CODEC, PropertyRef};
use coordinode_core::txn::proposal::{
    DerivedIndexWork, DerivedSource, IndexBinding, Mutation, PartitionId, ProposalIdGenerator,
    ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_test_fixtures::alloc_port;

/// Units per profile.
const UNITS: u64 = 2_000;

struct Member {
    node: RaftNode,
    engine: Arc<StorageEngine>,
    _dir: tempfile::TempDir,
}

fn open_engine(dir: &Path) -> Arc<StorageEngine> {
    Arc::new(
        StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]))
        .expect("open"),
    )
}

async fn group() -> Vec<Member> {
    let ports = [alloc_port(), alloc_port(), alloc_port()];
    let mut members = Vec::new();
    for (i, port) in ports.iter().enumerate() {
        let dir = tempfile::tempdir().expect("dir");
        let engine = open_engine(dir.path());
        let listen = format!("127.0.0.1:{port}").parse().expect("addr");
        let id = i as u64 + 1;
        let node = if id == 1 {
            RaftNode::open_cluster(
                id,
                Arc::clone(&engine),
                listen,
                format!("http://127.0.0.1:{port}"),
            )
            .await
        } else {
            RaftNode::open_joining(id, Arc::clone(&engine), listen).await
        }
        .expect("member");
        members.push(Member {
            node,
            engine,
            _dir: dir,
        });
    }
    let leader = &members[0].node;
    for _ in 0..150 {
        if leader.is_leader().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    for (id, port) in [(2u64, ports[1]), (3, ports[2])] {
        leader
            .add_node(id, format!("http://127.0.0.1:{port}"))
            .await
            .expect("add");
    }
    leader
        .change_membership(vec![1, 2, 3])
        .await
        .expect("membership");
    members
}

fn binding() -> IndexBinding {
    IndexBinding {
        epoch: 1,
        interpretation: IndexInterpretation {
            codec: KEY_CODEC,
            generation: coordinode_core::index::identity::GenerationId::from_raw(1),
            unique: false,
            sparse: false,
            properties: vec![PropertyRef {
                field: Some(1),
                name: "email".into(),
            }],
            filter: None,
        },
    }
}

/// The unit inserting user `n`, its index maintained as `derived` says.
fn unit(id_gen: &ProposalIdGenerator, n: u64, derived: bool) -> RaftProposal {
    let email = format!("user-{n}@example.com");
    let mut record = NodeRecord::new("User");
    record.set(1, Value::String(email.clone()));
    let node = Mutation::Put {
        partition: PartitionId::Node,
        key: encode_node_key(0, NodeId::from_raw(n)),
        value: record.to_msgpack().expect("record"),
    };
    let index = if derived {
        Mutation::Derive(DerivedIndexWork {
            binding: binding(),
            node_id: n,
            valid_from: None,
            old: None,
            new: DerivedSource::UnitRecord(0),
        })
    } else {
        let effect = binding()
            .interpretation
            .membership_effects(
                coordinode_core::index::derive::EntryOwner::node(n),
                None,
                Some(&[Value::String(email)]),
            )
            .pop()
            .expect("one entry");
        Mutation::Put {
            partition: PartitionId::Idx,
            key: effect.key,
            value: effect.value.expect("a put"),
        }
    };
    let ts = 1_000_000 + n;
    RaftProposal {
        id: id_gen.next(),
        mutations: vec![node, index],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts - 1),
        bypass_rate_limiter: false,
    }
}

/// Bytes of the Raft log segments under `data_dir`.
fn log_bytes(data_dir: &Path) -> u64 {
    fn walk(dir: &Path) -> u64 {
        std::fs::read_dir(dir).map_or(0, |entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| {
                    let path = e.path();
                    if path.is_dir() {
                        walk(&path)
                    } else {
                        e.metadata().map_or(0, |m| m.len())
                    }
                })
                .sum()
        })
    }
    walk(&data_dir.join("oplog"))
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

async fn run(members: &[Member], label: &str, first: u64, derived: bool) {
    let leader = &members[0];
    let id_gen = ProposalIdGenerator::with_base((first + 1) << 40);
    let before = log_bytes(leader.engine.data_dir());
    let pipeline = leader.node.pipeline();
    let started = Instant::now();
    let mut samples = Vec::with_capacity(UNITS as usize);
    for n in first..first + UNITS {
        let proposal = unit(&id_gen, n, derived);
        let t = Instant::now();
        tokio::task::block_in_place(|| pipeline.propose_and_wait(&proposal)).expect("commit");
        samples.push(t.elapsed());
    }
    let wall = started.elapsed();
    let grown = log_bytes(leader.engine.data_dir()) - before;
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    println!(
        "{label:<10} n={UNITS:>5}  {:>7.0} units/s  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}  log bytes/entry={:>6.1}",
        UNITS as f64 / wall.as_secs_f64(),
        total / UNITS as u32,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
        grown as f64 / UNITS as f64,
    );
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let members = group().await;
        println!("== index maintenance profiles, three members on one host ==");
        // A warm-up round first, so neither profile pays the group's start.
        run(&members, "warm-up", 0, false).await;
        for round in 0..2u64 {
            let base = (round + 1) * 10 * UNITS;
            run(&members, "RESOLVED", base, false).await;
            run(&members, "DERIVED", base + 5 * UNITS, true).await;
        }
        for member in &members {
            member.node.shutdown().await.expect("shutdown");
        }
    });
}
