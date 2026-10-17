use super::*;
use tempfile::TempDir;

const ID: u64 = 7;

fn test_entry(n: usize, ts: u64) -> DrainEntry {
    let mutations = (0..n)
        .map(|i| Mutation::Put {
            partition: PartitionId::Node,
            key: format!("key_{i}").into_bytes(),
            value: vec![i as u8, 42],
        })
        .collect();
    DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
}

/// The mark an appended entry carries for buffer `id`, entry `seq`.
fn mark(id: u64, seq: u64) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Schema,
        key: DRAINED_KEY.to_vec(),
        value: drained_value(id, seq),
    }
}

/// Append `entry` through `wb`, and the entry as the drain receives it.
fn append(wb: &NvmeWriteBuffer, entry: DrainEntry) -> DrainEntry {
    let drain = DrainBuffer::new(1 << 20);
    wb.append(entry, &drain).unwrap();
    drain.take_all().pop().expect("in the drain buffer")
}

fn recovered(dir: &Path) -> RecoveredBuffer {
    NvmeWriteBuffer::recover(dir).unwrap().expect("entries")
}

/// Every mutation kind survives the round trip whole, so a recovered entry is
/// applied exactly as the original would have been.
#[test]
fn every_mutation_kind_round_trips() {
    use coordinode_core::txn::proposal::MetadataCommand;
    let entry = DrainEntry::new(
        vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"k".to_vec(),
                value: vec![1],
            },
            Mutation::Delete {
                partition: PartitionId::Adj,
                key: b"adj:key".to_vec(),
            },
            Mutation::Merge {
                partition: PartitionId::Counter,
                key: b"c".to_vec(),
                operand: vec![1, 2, 3, 4],
            },
            Mutation::RemoveRange {
                partition: PartitionId::Idx,
                start: b"a".to_vec(),
                end: b"b".to_vec(),
            },
            Mutation::Command(MetadataCommand::RegisterFields {
                names: vec!["a".into(), "b".into()],
            }),
        ],
        Timestamp::from_raw(400),
        Timestamp::from_raw(3),
    );
    let encoded = encode_entry(9, &entry).unwrap();
    let (seq, decoded) = decode_entry(&encoded[4..]).unwrap();
    assert_eq!(seq, 9);
    assert_eq!(decoded.mutations, entry.mutations);
    assert_eq!(decoded.commit_ts.as_raw(), 400);
    assert_eq!(decoded.start_ts.as_raw(), 3);
}

/// Appended entries are numbered in order, each carries the mark of its own
/// number (what the log records when it applies the entry), and each reaches
/// the drain exactly as it was written to the file.
#[test]
fn appended_entries_are_numbered_and_marked() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path(), ID).unwrap();
    let first = append(&wb, test_entry(2, 100));
    let second = append(&wb, test_entry(3, 200));
    assert_eq!(first.mutations.last(), Some(&mark(ID, 1)));
    assert_eq!(second.mutations.last(), Some(&mark(ID, 2)));

    let buffer = recovered(dir.path());
    assert_eq!(buffer.id, ID);
    let seqs: Vec<u64> = buffer.entries.iter().map(|(s, _)| *s).collect();
    assert_eq!(seqs, vec![1, 2]);
    assert_eq!(buffer.entries[0].1.mutations, first.mutations);
    assert_eq!(buffer.entries[1].1.commit_ts.as_raw(), 200);
}

/// A write the drain buffer has no room for is refused before it reaches the
/// file: a refused write must not come back when the store reopens. Its
/// number is not used up either.
#[test]
fn a_write_refused_for_room_is_not_in_the_file() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path(), ID).unwrap();
    let full = DrainBuffer::new(1);
    let refused = wb.append(test_entry(1, 100), &full).unwrap_err();
    assert!(matches!(refused, AppendError::Full(_)), "{refused:?}");
    assert!(full.is_empty());
    assert_eq!(full.used_bytes(), 0, "the reserved room was given back");
    assert!(NvmeWriteBuffer::recover(dir.path()).unwrap().is_none());

    let accepted = append(&wb, test_entry(1, 200));
    assert_eq!(accepted.mutations.last(), Some(&mark(ID, 1)));
}

/// The mark of a client operation's entry goes before its binding, which has
/// to end the unit.
#[test]
fn the_mark_goes_before_an_operation_binding() {
    use coordinode_core::txn::operation::{IssuerId, OperationCommand, OperationId};
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path(), ID).unwrap();
    let binding = Mutation::Operation(OperationCommand::Exclude(OperationId {
        issuer: IssuerId([1; 16]),
        incarnation: 1,
        sequence: 1,
    }));
    let mut entry = test_entry(1, 100);
    entry.mutations.push(binding.clone());
    let entry = append(&wb, entry);
    let n = entry.mutations.len();
    assert_eq!(entry.mutations[n - 2], mark(ID, 1));
    assert_eq!(entry.mutations[n - 1], binding);
}

/// The recorded mark says through which entry the buffer was applied; a
/// record about another buffer says nothing about this one.
#[test]
fn the_drained_record_is_read_for_its_own_buffer_only() {
    assert_eq!(drained_through(&drained_value(ID, 5), ID), Some(5));
    assert_eq!(drained_through(&drained_value(ID + 1, 5), ID), None);
    assert_eq!(drained_through(b"short", ID), None);
}

/// An empty directory, and one whose files hold no entry, recover nothing.
#[test]
fn nothing_to_recover() {
    let dir = TempDir::new().unwrap();
    assert!(NvmeWriteBuffer::recover(dir.path()).unwrap().is_none());
    drop(NvmeWriteBuffer::open(dir.path(), ID).unwrap());
    assert!(NvmeWriteBuffer::recover(dir.path()).unwrap().is_none());
}

/// Recovery reads and changes nothing; the caller discards the files once
/// their entries are dealt with.
#[test]
fn recovery_keeps_the_files_until_discarded() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path(), ID).unwrap();
    append(&wb, test_entry(1, 100));
    drop(wb);

    assert_eq!(recovered(dir.path()).entries.len(), 1);
    assert_eq!(
        recovered(dir.path()).entries.len(),
        1,
        "read twice, same entries"
    );
    NvmeWriteBuffer::discard(dir.path()).unwrap();
    assert!(NvmeWriteBuffer::recover(dir.path()).unwrap().is_none());
}

/// A checkpoint taken by the drain keeps its entries until the drain
/// completes it; entries appended meanwhile go to a fresh file of the same
/// buffer, continuing its numbers.
#[test]
fn a_checkpoint_keeps_its_entries_until_completed() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path(), ID).unwrap();
    append(&wb, test_entry(2, 100));
    let token = wb.begin_drain();
    append(&wb, test_entry(1, 300));

    // A crash here leaves both files: every entry is recovered, in order.
    let seqs: Vec<u64> = recovered(dir.path())
        .entries
        .iter()
        .map(|(s, _)| *s)
        .collect();
    assert_eq!(seqs, vec![1, 2]);

    wb.complete_drain(token);
    let buffer = recovered(dir.path());
    assert_eq!(buffer.entries.len(), 1);
    assert_eq!(buffer.entries[0].0, 2);
    assert!(find_draining_files(dir.path()).unwrap().is_empty());
}

/// A file without the numbered header cannot say which of its entries were
/// applied: it is refused and left where it is, not replayed or deleted.
#[test]
fn an_unnumbered_file_is_refused_and_kept() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(CURRENT_FILE);
    std::fs::write(&path, [20u8, 0, 0, 0, 1, 2, 3, 4, 5]).unwrap();
    let refused = NvmeWriteBuffer::recover(dir.path()).unwrap_err();
    assert!(
        matches!(refused, RecoverError::Unnumbered { .. }),
        "{refused:?}"
    );
    assert!(path.exists());
}

/// Files of two buffers in one directory are refused rather than merged.
#[test]
fn files_of_two_buffers_are_refused() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path(), ID).unwrap();
    append(&wb, test_entry(1, 100));
    wb.begin_drain();
    drop(wb);
    let other = NvmeWriteBuffer::open(dir.path(), ID + 1).unwrap();
    append(&other, test_entry(1, 200));
    let refused = NvmeWriteBuffer::recover(dir.path()).unwrap_err();
    assert!(
        matches!(refused, RecoverError::MixedBuffers { .. }),
        "{refused:?}"
    );
}

/// An entry cut short by a crash mid-append ends the read; the complete
/// entries before it are recovered.
#[test]
fn a_torn_last_entry_is_skipped() {
    let dir = TempDir::new().unwrap();
    let wb = NvmeWriteBuffer::open(dir.path(), ID).unwrap();
    append(&wb, test_entry(1, 100));
    drop(wb);
    let path = dir.path().join(CURRENT_FILE);
    let mut data = std::fs::read(&path).unwrap();
    data.extend_from_slice(&[0xFF, 0xFF, 0x00]);
    std::fs::write(&path, data).unwrap();

    let buffer = recovered(dir.path());
    assert_eq!(buffer.entries.len(), 1);
    assert_eq!(buffer.entries[0].1.commit_ts.as_raw(), 100);
}

#[test]
fn partition_roundtrip() {
    let partitions = [
        PartitionId::Node,
        PartitionId::Adj,
        PartitionId::EdgeProp,
        PartitionId::Blob,
        PartitionId::BlobRef,
        PartitionId::Schema,
        PartitionId::Idx,
        PartitionId::Counter,
        PartitionId::VectorF32,
        PartitionId::Registry,
    ];
    for p in partitions {
        assert_eq!(u8_to_partition(partition_to_u8(p)), Some(p));
    }
    assert_eq!(u8_to_partition(200), None);
}
