use super::*;

#[test]
fn every_process_draws_its_own_proposal_id_range() {
    // Proposal ids must not repeat across incarnations: the state machine
    // re-applies the old log on restart and drops a proposal whose id and
    // size it saw, so a generator restarting at a fixed base silently loses
    // the first writes after every restart.
    let bases: std::collections::HashSet<u64> = (0..64).map(|_| fresh_proposal_id_base()).collect();
    assert_eq!(bases.len(), 64, "each draw is a fresh point");
}

#[test]
fn proposal_id_monotonic() {
    let gen = ProposalIdGenerator::new();
    let id1 = gen.next();
    let id2 = gen.next();
    let id3 = gen.next();
    assert_eq!(id1.as_raw(), 1);
    assert_eq!(id2.as_raw(), 2);
    assert_eq!(id3.as_raw(), 3);
}

#[test]
fn proposal_id_concurrent() {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    let gen = Arc::new(ProposalIdGenerator::new());
    let mut handles = Vec::new();

    for _ in 0..4 {
        let gen = Arc::clone(&gen);
        handles.push(std::thread::spawn(move || {
            (0..1000).map(|_| gen.next().as_raw()).collect::<Vec<_>>()
        }));
    }

    let mut all: BTreeSet<u64> = BTreeSet::new();
    for h in handles {
        for id in h.join().expect("thread panicked") {
            assert!(all.insert(id), "duplicate proposal ID: {id}");
        }
    }
    assert_eq!(all.len(), 4000);
}

#[test]
fn mutation_size_estimate() {
    let proposal = RaftProposal {
        id: ProposalId::from_raw(1),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: vec![0; 10],
                value: vec![0; 100],
            },
            Mutation::Delete {
                partition: PartitionId::Adj,
                key: vec![0; 20],
            },
        ],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // 24 + (1 + 10 + 100) + (1 + 20) = 156
    assert_eq!(proposal.size_estimate(), 156);
}

#[test]
fn empty_proposal() {
    let proposal = RaftProposal {
        id: ProposalId::from_raw(1),
        mutations: vec![],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };
    assert_eq!(proposal.mutation_count(), 0);
    assert_eq!(proposal.size_estimate(), 24);
}

#[test]
fn proposal_id_display() {
    let id = ProposalId::from_raw(42);
    assert_eq!(format!("{id}"), "prop:42");
}

#[test]
fn write_concern_timeout_is_cloneable() {
    let err = ProposalError::WriteConcernTimeout { timeout_ms: 3000 };
    let cloned = err.clone();
    assert_eq!(format!("{err}"), format!("{cloned}"));
}

#[test]
fn write_concern_timeout_distinct_from_retry_timeout() {
    let wc_timeout = ProposalError::WriteConcernTimeout { timeout_ms: 5000 };
    let retry_timeout = ProposalError::Timeout { retries: 3 };

    let wc_msg = format!("{wc_timeout}");
    let retry_msg = format!("{retry_timeout}");

    // Both are timeout-related but have distinct messages
    assert!(wc_msg.contains("write concern"), "wc: {wc_msg}");
    assert!(retry_msg.contains("retries"), "retry: {retry_msg}");
    assert_ne!(wc_msg, retry_msg);
}

fn sample_proposal() -> RaftProposal {
    RaftProposal {
        id: ProposalId::from_raw(7),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:7".to_vec(),
                value: b"v".to_vec(),
            },
            Mutation::Delete {
                partition: PartitionId::Idx,
                key: b"idx:x".to_vec(),
            },
        ],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(90),
        bypass_rate_limiter: false,
    }
}

/// A proposal serializes as one frame (a msgpack binary), inside any
/// enclosing structure, and comes back unchanged.
#[test]
fn a_proposal_serializes_as_its_frame() {
    let proposal = sample_proposal();
    let bytes = rmp_serde::to_vec(&vec![proposal.clone()]).expect("serialize");
    // An array of one element, whose element is a bin holding the frame.
    let frame = crate::txn::frame::encode_proposal(&proposal).expect("encode");
    assert_eq!(bytes[0], 0x91, "an array of one");
    assert!(
        matches!(bytes[1], 0xC4..=0xC6),
        "the proposal is a msgpack binary"
    );
    assert!(bytes.ends_with(&frame));
    let back: Vec<RaftProposal> = rmp_serde::from_slice(&bytes).expect("deserialize");
    assert_eq!(back, vec![proposal]);
}

/// The shape proposals had before frames, positional or keyed.
#[derive(Serialize)]
struct FieldWiseShape<'a> {
    id: ProposalId,
    mutations: &'a [Mutation],
    commit_ts: Timestamp,
    start_ts: Timestamp,
    bypass_rate_limiter: bool,
}

/// A log tail written before frames still replays: the field-wise form, as
/// an array or as a map, deserializes to the same proposal.
#[test]
fn a_field_wise_proposal_from_before_frames_still_reads() {
    let proposal = sample_proposal();
    let shape = FieldWiseShape {
        id: proposal.id,
        mutations: &proposal.mutations,
        commit_ts: proposal.commit_ts,
        start_ts: proposal.start_ts,
        bypass_rate_limiter: proposal.bypass_rate_limiter,
    };
    let positional = rmp_serde::to_vec(&shape).expect("positional");
    let keyed = rmp_serde::to_vec_named(&shape).expect("keyed");
    let from_positional: RaftProposal =
        rmp_serde::from_slice(&positional).expect("read positional");
    let from_keyed: RaftProposal = rmp_serde::from_slice(&keyed).expect("read keyed");
    assert_eq!(from_positional, proposal);
    assert_eq!(from_keyed, proposal);
}
