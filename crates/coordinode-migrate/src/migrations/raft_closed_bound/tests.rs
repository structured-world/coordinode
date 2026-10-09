use coordinode_raft::storage::{Entry, Request};
use openraft::entry::EntryPayload;

use super::upgrade_entry;
use crate::test_support::{current_entry, old_bytes, proposal};

/// An entry from before the closed bound becomes a current entry with the
/// same log id and proposals and a closed bound of 0: the one error the
/// current engine stops on is the one this rewrites.
#[test]
fn an_old_entry_reads_as_current_with_a_zero_bound() {
    let entry = current_entry(7, Request::batch(vec![proposal(1, 100), proposal(2, 101)]));
    let old = old_bytes(&entry);
    let refused = rmp_serde::from_slice::<Entry>(&old).expect_err("the old shape is refused");
    assert!(
        refused.to_string().contains("invalid length 1"),
        "{refused}"
    );

    let new = upgrade_entry(&old)
        .expect("upgrade")
        .expect("an old entry is rewritten");
    let read: Entry = rmp_serde::from_slice(&new).expect("reads as current");
    assert_eq!(read.log_id, entry.log_id);
    let EntryPayload::Normal(request) = read.payload else {
        unreachable!("a Normal entry stays Normal")
    };
    assert_eq!(request.proposals, vec![proposal(1, 100), proposal(2, 101)]);
    assert_eq!(request.closed_below, 0);
}

/// An entry without proposals in its envelope (they travel as unit frames)
/// upgrades the same way.
#[test]
fn an_old_envelope_without_proposals_upgrades() {
    let entry = current_entry(3, Request::closing(0));
    let new = upgrade_entry(&old_bytes(&entry))
        .expect("upgrade")
        .expect("rewritten");
    assert_eq!(rmp_serde::from_slice::<Entry>(&new).expect("reads"), entry);
}

/// A current entry, with any bound, is left as it is.
#[test]
fn a_current_entry_is_left_alone() {
    let entry = current_entry(9, Request::closing(12_345));
    let bytes = rmp_serde::to_vec(&entry).expect("encode");
    assert_eq!(upgrade_entry(&bytes).expect("survey"), None);
}

/// Membership and blank entries carry no request and are current already.
#[test]
fn entries_without_a_request_are_left_alone() {
    use openraft::entry::RaftEntry as _;
    let blank = Entry::new_blank(openraft::LogId::new(
        coordinode_raft::storage::CommittedLeaderId {
            term: 2,
            node_id: 1,
        },
        1,
    ));
    let bytes = rmp_serde::to_vec(&blank).expect("encode");
    assert_eq!(upgrade_entry(&bytes).expect("survey"), None);
}

/// Bytes that are neither the old shape nor current stop the migration with
/// what was found, instead of being rewritten into something else.
#[test]
fn an_unknown_shape_is_refused_not_rewritten() {
    // Not msgpack at all: nothing to read.
    let err = upgrade_entry(&[]).expect_err("empty");
    assert!(format!("{err:#}").contains("not msgpack"), "{err:#}");
    // A msgpack value that is no entry (0xc1 reads as nil).
    let err = upgrade_entry(&[0xc1]).expect_err("nil");
    assert!(format!("{err:#}").contains("not a log entry"), "{err:#}");

    // A Normal entry whose request has three fields.
    let entry = current_entry(4, Request::closing(0));
    let current = rmp_serde::to_vec(&entry).expect("encode");
    let mut value = rmpv::decode::read_value(&mut &current[..]).expect("decode");
    if let rmpv::Value::Array(fields) = &mut value {
        if let Some(rmpv::Value::Map(payload)) = fields.get_mut(1) {
            if let [(_, rmpv::Value::Array(request))] = payload.as_mut_slice() {
                request.push(rmpv::Value::from(1u64));
            }
        }
    }
    let mut three = Vec::new();
    rmpv::encode::write_value(&mut three, &value).expect("encode");
    let err = upgrade_entry(&three).expect_err("unknown shape");
    assert!(format!("{err:#}").contains("3 fields"), "{err:#}");

    // A map where an entry array belongs.
    let mut map = Vec::new();
    rmpv::encode::write_value(&mut map, &rmpv::Value::Map(vec![])).expect("encode");
    let err = upgrade_entry(&map).expect_err("not an entry");
    assert!(format!("{err:#}").contains("not a log entry"), "{err:#}");
}
