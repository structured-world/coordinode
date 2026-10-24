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

/// The relay delivers a write's notice some time after the write applied, in
/// apply order. A watch made right after its record was written can hear of
/// that write only after its first call: a change reported then may be that
/// earlier write, and the next write to the record is a change of its own.
/// Once the notice of a later write to another record is heard, every
/// earlier notice has arrived.
#[test]
fn a_notice_of_a_write_before_the_watch_can_arrive_after_its_first_call() {
    let watches = relayed();
    let (mine, other) = (b"consumer:mine".to_vec(), b"consumer:other".to_vec());
    // Both records were written; their notices are still on the way.
    let mut watch = watches.watch(mine.clone());
    let mut later = watches.watch(other.clone());
    assert!(watch.changed() && later.changed(), "the first calls");

    // The relay delivers the notice of the registration of `mine` now.
    watches.applied(std::slice::from_ref(&mine));
    // Taken for the next write to `mine`, this change would be the wrong one.
    assert!(watch.changed(), "the earlier write, heard late");
    // The next write to `mine` is reported again, on its own.
    watches.applied(std::slice::from_ref(&mine));
    assert!(watch.changed(), "the next write, heard after it");
    assert!(!watch.changed(), "nothing more applied to it");

    // A write to `other`, written after both registrations: once its notice
    // is heard, no notice of an earlier write is still on the way.
    watches.applied(std::slice::from_ref(&other));
    assert!(later.changed());
    assert!(!watch.changed());
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

/// A reader waiting on its watch is woken by a write to its record: a stream
/// parked until the next applied entry must not miss the end of its
/// registration because the relay bumped the generation after it looked.
#[tokio::test]
async fn a_waiting_watch_is_woken_by_a_write_to_its_record() {
    let watches = relayed();
    let mut watch = watches.watch(b"consumer:a".to_vec());
    assert!(watch.changed());
    let relay = Arc::clone(&watches);
    let waiter = tokio::spawn(async move {
        watch.wait().await;
        watch.changed()
    });
    tokio::task::yield_now().await;
    relay.applied(&[b"consumer:b".to_vec()]);
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished(), "another record's write");
    relay.applied(&[b"consumer:a".to_vec()]);
    let changed = tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
        .await
        .expect("woken")
        .expect("task");
    assert!(changed);
}

/// A write whose keys are not known wakes every waiting watch.
#[tokio::test]
async fn an_unknown_write_wakes_a_waiting_watch() {
    let watches = relayed();
    let mut watch = watches.watch(b"consumer:a".to_vec());
    assert!(watch.changed());
    let relay = Arc::clone(&watches);
    let waiter = tokio::spawn(async move { watch.wait().await });
    tokio::task::yield_now().await;
    relay.applied_unknown();
    tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
        .await
        .expect("woken")
        .expect("task");
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
