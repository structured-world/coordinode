use super::*;

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
