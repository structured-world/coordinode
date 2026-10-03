use super::*;
use crate::txn::proposal::{Mutation, PartitionId, ProposalError, ProposalOutcome};

/// Mock pipeline that records proposals.
struct RecordingPipeline {
    proposals: Mutex<Vec<RaftProposal>>,
}

impl RecordingPipeline {
    fn new() -> Self {
        Self {
            proposals: Mutex::new(Vec::new()),
        }
    }

    fn proposal_count(&self) -> usize {
        self.proposals.lock().unwrap().len()
    }

    fn total_mutations(&self) -> usize {
        self.proposals
            .lock()
            .unwrap()
            .iter()
            .map(|p| p.mutations.len())
            .sum()
    }
}

impl ProposalPipeline for RecordingPipeline {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        self.proposals.lock().unwrap().push(proposal.clone());
        Ok(ProposalOutcome::local())
    }
}

fn test_mutation(key: &str) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Node,
        key: key.as_bytes().to_vec(),
        value: vec![1, 2, 3],
    }
}

fn test_entry(n_mutations: usize, ts: u64) -> DrainEntry {
    let mutations: Vec<_> = (0..n_mutations)
        .map(|i| test_mutation(&format!("key_{i}")))
        .collect();
    DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
}

/// A drained batch of consecutive point deletes is coalesced into a range delete
/// in the submitted proposal: the producer wiring, end to end.
#[test]
fn drain_coalesces_consecutive_deletes_into_range() {
    let buf = DrainBuffer::new(1 << 20);
    let dels: Vec<Mutation> = (0..8u64)
        .map(|i| Mutation::Delete {
            partition: PartitionId::EdgeProp,
            key: i.to_be_bytes().to_vec(),
        })
        .collect();
    buf.append(DrainEntry::new(
        dels,
        Timestamp::from_raw(100),
        Timestamp::from_raw(1),
    ))
    .expect("append");

    let pipeline = Arc::new(RecordingPipeline::new());
    let id_gen = ProposalIdGenerator::new();
    drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

    let proposals = pipeline.proposals.lock().unwrap();
    assert_eq!(proposals.len(), 1);
    // The 8 point deletes became one range delete.
    assert_eq!(proposals[0].mutations.len(), 1);
    assert!(matches!(
        &proposals[0].mutations[0],
        Mutation::RemoveRange {
            partition: PartitionId::EdgeProp,
            ..
        }
    ));
}

#[test]
fn buffer_append_and_take() {
    let buf = DrainBuffer::new(1024 * 1024);
    assert!(buf.is_empty());

    buf.append(test_entry(3, 100)).unwrap();
    buf.append(test_entry(2, 200)).unwrap();
    assert_eq!(buf.len(), 2);
    assert!(!buf.is_empty());

    let entries = buf.take_all();
    assert_eq!(entries.len(), 2);
    assert!(buf.is_empty());
    assert_eq!(buf.used_bytes(), 0);
}

#[test]
fn buffer_capacity_backpressure() {
    // Tiny buffer — 50 bytes.
    let buf = DrainBuffer::new(50);

    // First entry fits.
    buf.append(test_entry(1, 100)).unwrap();

    // Second entry should be rejected (exceeds capacity).
    let result = buf.append(test_entry(100, 200));
    assert!(result.is_err());

    // Buffer still has the first entry.
    assert_eq!(buf.len(), 1);
}

#[test]
fn drain_once_submits_proposals() {
    let buf = DrainBuffer::new(1024 * 1024);
    let pipeline = Arc::new(RecordingPipeline::new());
    let id_gen = ProposalIdGenerator::new();

    buf.append(test_entry(5, 100)).unwrap();
    buf.append(test_entry(3, 200)).unwrap();

    drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

    // Both entries should be batched into one proposal (8 mutations < 10K).
    assert_eq!(pipeline.proposal_count(), 1);
    assert_eq!(pipeline.total_mutations(), 8);
    assert!(buf.is_empty());
}

#[test]
fn drain_once_respects_batch_max() {
    let buf = DrainBuffer::new(1024 * 1024);
    let pipeline = Arc::new(RecordingPipeline::new());
    let id_gen = ProposalIdGenerator::new();

    // 3 entries × 5 mutations = 15 mutations. batch_max = 7.
    buf.append(test_entry(5, 100)).unwrap();
    buf.append(test_entry(5, 200)).unwrap();
    buf.append(test_entry(5, 300)).unwrap();

    drain_once(&buf, pipeline.as_ref(), &id_gen, 7, None);

    // Should produce 2 proposals: [5, 5+5] won't fit → [5], [5, 5]
    // Actually: first=5 (fits), second=5 (5+5=10>7 → flush first, then second),
    // third=5 (5+5=10>7 → flush second, then third). So: [5], [5], [5] = 3.
    assert_eq!(pipeline.proposal_count(), 3);
    assert_eq!(pipeline.total_mutations(), 15);
}

#[test]
fn drain_once_empty_buffer_noop() {
    let buf = DrainBuffer::new(1024 * 1024);
    let pipeline = Arc::new(RecordingPipeline::new());
    let id_gen = ProposalIdGenerator::new();

    drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

    assert_eq!(pipeline.proposal_count(), 0);
}

#[test]
fn drain_entry_preserves_commit_ts() {
    let buf = DrainBuffer::new(1024 * 1024);
    let pipeline = Arc::new(RecordingPipeline::new());
    let id_gen = ProposalIdGenerator::new();

    buf.append(test_entry(2, 100)).unwrap();
    buf.append(test_entry(2, 500)).unwrap();

    drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

    let proposals = pipeline.proposals.lock().unwrap();
    // Batched into one proposal — commit_ts should be the max (500).
    assert_eq!(proposals[0].commit_ts.as_raw(), 500);
}

#[test]
fn drain_proposals_bypass_rate_limiter() {
    let buf = DrainBuffer::new(1024 * 1024);
    let pipeline = Arc::new(RecordingPipeline::new());
    let id_gen = ProposalIdGenerator::new();

    buf.append(test_entry(1, 100)).unwrap();
    drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

    let proposals = pipeline.proposals.lock().unwrap();
    assert!(proposals[0].bypass_rate_limiter);
}

#[test]
fn drain_handle_start_and_shutdown() {
    let buffer = Arc::new(DrainBuffer::new(1024 * 1024));
    let pipeline = Arc::new(RecordingPipeline::new());
    let id_gen = Arc::new(ProposalIdGenerator::new());

    buffer.append(test_entry(3, 100)).unwrap();

    let config = DrainConfig {
        interval_ms: 10, // Short interval for test
        batch_max: 10_000,
        capacity_bytes: 1024 * 1024,
    };

    let mut handle = DrainHandle::start(
        Arc::clone(&buffer),
        Arc::clone(&pipeline) as Arc<dyn ProposalPipeline>,
        Arc::clone(&id_gen),
        config,
        None,
    );

    // Give drain thread time to process.
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Shutdown should flush remaining entries.
    handle.shutdown();

    // The 3-mutation entry should have been drained.
    assert!(pipeline.proposal_count() >= 1);
    assert!(buffer.is_empty());
}

mod derived {
    use super::*;
    use crate::graph::node::NodeRecord;
    use crate::graph::types::Value;
    use crate::index::derive::{IndexInterpretation, KEY_CODEC, PropertyRef, entry, resolve_unit};
    use crate::index::encoding::encode_tuple;
    use crate::txn::proposal::{DerivedIndexWork, DerivedSource, IndexBinding};
    use std::collections::BTreeMap;

    fn unique_email() -> IndexInterpretation {
        IndexInterpretation {
            codec: KEY_CODEC,
            name: "u_email".into(),
            unique: true,
            sparse: false,
            properties: vec![PropertyRef {
                field: Some(1),
                name: "email".into(),
            }],
            filter: None,
        }
    }

    /// The entry key and value of `node_id` holding `email`.
    fn claim(node_id: u64, email: &str) -> (Vec<u8>, Vec<u8>) {
        let tuple = encode_tuple(&[Value::String(email.into())]).expect("tuple");
        entry(
            "u_email",
            true,
            &tuple,
            crate::index::derive::EntryOwner::node(node_id),
        )
    }

    /// A unit that writes `node_id`'s record with `email` and derives its
    /// entry from that record, as a committed transaction seals it.
    fn derived_entry(node_id: u64, email: &str, ts: u64) -> DrainEntry {
        let mut record = NodeRecord::new("U");
        record.set(1, Value::String(email.into()));
        let mutations = vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:0:{node_id}").into_bytes(),
                value: record.to_msgpack().expect("record"),
            },
            Mutation::Derive(DerivedIndexWork {
                binding: IndexBinding {
                    epoch: 1,
                    interpretation: unique_email(),
                },
                node_id,
                valid_from: None,
                old: None,
                new: DerivedSource::UnitRecord(0),
            }),
        ];
        DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
    }

    /// The index entries a member holds after applying `proposals` in order.
    fn applied(proposals: &[RaftProposal]) -> BTreeMap<Vec<u8>, Vec<u8>> {
        let mut idx = BTreeMap::new();
        for proposal in proposals {
            crate::txn::frame::encode_proposal(proposal).expect("the unit reaches a log");
            for mutation in resolve_unit(&proposal.mutations, usize::MAX)
                .expect("every member derives the unit")
                .iter()
            {
                match mutation {
                    Mutation::Put {
                        partition: PartitionId::Idx,
                        key,
                        value,
                    } => {
                        idx.insert(key.clone(), value.clone());
                    }
                    Mutation::Delete {
                        partition: PartitionId::Idx,
                        key,
                    } => {
                        idx.remove(key);
                    }
                    _ => {}
                }
            }
        }
        idx
    }

    /// Each drained unit's DERIVED work keeps deriving from its own record:
    /// batched after another unit, a record position would name the other
    /// unit's record and index the wrong value for the node.
    #[test]
    fn drained_derived_work_derives_from_its_own_record() {
        let buf = DrainBuffer::new(1 << 20);
        buf.append(derived_entry(1, "a@x", 100)).expect("a");
        buf.append(derived_entry(2, "b@x", 200)).expect("b");
        let pipeline = RecordingPipeline::new();
        drain_once(&buf, &pipeline, &ProposalIdGenerator::new(), 10_000, None);

        let proposals = pipeline.proposals.lock().unwrap();
        assert_eq!(
            applied(&proposals),
            BTreeMap::from([claim(1, "a@x"), claim(2, "b@x")])
        );
    }

    /// A unit whose index moved back to RESOLVED after an earlier drained
    /// unit derived an entry applies after it: the earlier claim is removed
    /// by the later delete, as it is when the units apply one by one.
    #[test]
    fn a_later_resolved_unit_applies_after_earlier_derived_work() {
        let buf = DrainBuffer::new(1 << 20);
        buf.append(derived_entry(7, "a@x", 100)).expect("claim");
        let (key, _) = claim(7, "a@x");
        buf.append(DrainEntry::new(
            vec![Mutation::Delete {
                partition: PartitionId::Idx,
                key,
            }],
            Timestamp::from_raw(200),
            Timestamp::from_raw(1),
        ))
        .expect("release");
        let pipeline = RecordingPipeline::new();
        drain_once(&buf, &pipeline, &ProposalIdGenerator::new(), 10_000, None);

        let proposals = pipeline.proposals.lock().unwrap();
        assert_eq!(applied(&proposals), BTreeMap::new());
    }

    /// Units without DERIVED work still batch into one proposal.
    #[test]
    fn units_without_derived_work_still_batch() {
        let buf = DrainBuffer::new(1 << 20);
        buf.append(test_entry(2, 100)).expect("a");
        buf.append(derived_entry(1, "a@x", 150)).expect("derived");
        buf.append(test_entry(2, 200)).expect("b");
        buf.append(test_entry(3, 300)).expect("c");
        let pipeline = RecordingPipeline::new();
        drain_once(&buf, &pipeline, &ProposalIdGenerator::new(), 10_000, None);

        let sizes: Vec<usize> = pipeline
            .proposals
            .lock()
            .unwrap()
            .iter()
            .map(|p| p.mutations.len())
            .collect();
        assert_eq!(sizes, vec![2, 2, 5]);
    }
}

/// Failed pipeline should not crash the drain thread.
#[test]
fn drain_tolerates_pipeline_errors() {
    struct FailingPipeline;
    impl ProposalPipeline for FailingPipeline {
        fn propose_and_wait(&self, _: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
            Err(ProposalError::Storage("test error".into()))
        }
    }

    let buf = DrainBuffer::new(1024 * 1024);
    let pipeline = Arc::new(FailingPipeline);
    let id_gen = ProposalIdGenerator::new();

    buf.append(test_entry(3, 100)).unwrap();

    // Should not panic — errors are logged and swallowed.
    drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

    // Buffer should be drained even though pipeline failed.
    assert!(buf.is_empty());
}
