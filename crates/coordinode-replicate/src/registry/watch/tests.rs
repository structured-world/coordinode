use std::sync::Arc;

use super::Watches;

fn relayed() -> Arc<Watches> {
    let watches = Arc::new(Watches::default());
    watches.set_relayed(true);
    watches
}

/// A watch reports a change on its first call, then only after a write to
/// its own record applies: writes to other records do not concern it.
#[test]
fn a_watch_reports_writes_to_its_own_record() {
    let watches = relayed();
    let mut watch = watches.watch(b"consumer:a".to_vec());
    assert!(watch.changed(), "the first call");
    assert!(!watch.changed(), "nothing applied since");

    watches.applied(&[b"consumer:b".to_vec()]);
    assert!(!watch.changed(), "another record changed");

    watches.applied(&[b"consumer:b".to_vec(), b"consumer:a".to_vec()]);
    assert!(watch.changed(), "its record changed");
    assert!(!watch.changed(), "reported once");
}

/// Writes whose keys are not known may have changed any record.
#[test]
fn unknown_writes_reach_every_watch() {
    let watches = relayed();
    let mut a = watches.watch(b"consumer:a".to_vec());
    let mut b = watches.watch(b"consumer:b".to_vec());
    assert!(a.changed() && b.changed());
    watches.applied_unknown();
    assert!(a.changed() && b.changed());
}

/// Without the relay nothing would report a write, so every call is a
/// change: the reader then checks each time, as it would without a watch.
#[test]
fn without_the_relay_every_call_is_a_change() {
    let watches = Arc::new(Watches::default());
    let mut watch = watches.watch(b"consumer:a".to_vec());
    assert!(watch.changed());
    assert!(watch.changed());
}

/// Watches of one record share its generation, and the entry goes with the
/// last of them.
#[test]
fn the_last_watch_of_a_record_removes_its_entry() {
    let watches = relayed();
    let mut first = watches.watch(b"consumer:a".to_vec());
    let mut second = watches.watch(b"consumer:a".to_vec());
    assert!(first.changed() && second.changed());
    watches.applied(&[b"consumer:a".to_vec()]);
    assert!(first.changed() && second.changed(), "both see the write");

    drop(first);
    assert_eq!(watches.by_key.lock().len(), 1, "the other still watches");
    drop(second);
    assert!(watches.by_key.lock().is_empty());
}
