use std::path::Path;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};

const UNIT: u64 = 4096;

fn footprints(dir: &Path, limit: u64, reserve: u64) -> Footprints {
    Footprints::open(&[
        EndpointConfig::new("ep", dir, Media::Ssd, Durability::Durable, Tier::Warm)
            .with_hard_limit_bytes(limit)
            .with_recovery_reserve_bytes(reserve),
    ])
    .unwrap()
}

fn entry(index: u64, term: u64, bytes: u64) -> GrantedEntry {
    GrantedEntry { index, term, bytes }
}

/// Stage `entries` and append each to the log, as a request the log takes.
fn take(f: &Footprints, entries: &[GrantedEntry]) -> Result<(), StorageError> {
    f.stage_granted(entries)?;
    for e in entries {
        f.granted_appended(e.index, e.term, e.bytes);
    }
    Ok(())
}

/// A member's store of its grant record, as a restart finds it.
#[derive(Default)]
struct Disk(std::cell::Cell<GrantRecord>);

impl Disk {
    /// A report after `last` once a raise it asked for reached the disk: the
    /// raise is stored at once here, and the report taken again.
    fn report(&self, f: &Footprints, last: u64) -> GrantReport {
        let (report, raise) = f.grant_report(last);
        let Some(raised) = raise else {
            return report;
        };
        self.0.set(raised);
        f.grant_horizon_stored(raised, true);
        let (report, again) = f.grant_report(last);
        assert_eq!(again, None, "one raise reaches as far as the pool");
        report
    }

    /// A restart of the member: a fresh pool opened with what is stored, the
    /// log applied through `applied`, the group's applied client bytes
    /// `counted` (through, total), and `entries`, the client entries the log
    /// holds past `applied`.
    fn restart(
        &self,
        dir: &Path,
        applied: u64,
        counted: (u64, u64),
        entries: &[GrantedEntry],
    ) -> Footprints {
        let f = footprints(dir, 100 * UNIT, 20 * UNIT);
        let opened = GrantRecord {
            epoch: self.0.get().epoch + 1,
            ..self.0.get()
        };
        self.0.set(opened);
        f.recover_granted_start(opened, applied, counted.0, counted.1);
        f.recover_granted(entries).unwrap();
        f.recover_granted_owed().unwrap();
        f
    }
}

/// The pool is client capacity admitted ahead: other client writes cannot
/// take it, and the recovery reserve is not part of it.
#[test]
fn a_grant_is_held_against_other_client_writes() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 20 * UNIT);
    f.set_grant_target(70 * UNIT);
    f.top_up_grant();
    assert_eq!(f.granted_room(), 70 * UNIT);
    assert!(
        f.admit(11 * UNIT, Admit::Client).is_err(),
        "only 10 units left"
    );
    f.set_grant_target(200 * UNIT);
    f.top_up_grant();
    assert!(f.granted_room() <= 80 * UNIT, "what fits, no more");
    assert!(f.admit(UNIT, Admit::Recovery).is_ok(), "the reserve stays");
    let held = f.granted_room();
    f.set_grant_target(UNIT);
    f.top_up_grant();
    assert_eq!(
        f.granted_room(),
        held,
        "a smaller target takes nothing back"
    );
}

/// Entries are paid once by index and term when the log appends them: a
/// retransmission pays nothing, an applied entry pays nothing, and a
/// request refused as a whole sets nothing aside.
#[test]
fn entries_are_paid_once_when_appended() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    // A batch of twice the target: 10 units.
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    take(&f, &[entry(5, 1, 2 * UNIT), entry(6, 1, 3 * UNIT)]).unwrap();
    assert_eq!(f.granted_room(), 5 * UNIT);
    take(&f, &[entry(5, 1, 2 * UNIT), entry(6, 1, 3 * UNIT)]).unwrap();
    assert_eq!(f.granted_room(), 5 * UNIT, "a retransmission pays nothing");
    assert!(
        take(&f, &[entry(7, 1, 8 * UNIT)]).is_err(),
        "more than granted"
    );
    assert_eq!(
        f.granted_room(),
        5 * UNIT,
        "a refused request sets nothing aside"
    );
    f.applied(6, 6, 5 * UNIT);
    assert_eq!(
        f.get("ep").unwrap().admitted(),
        5 * UNIT,
        "applied room is back"
    );
    take(&f, &[entry(4, 1, UNIT)]).unwrap();
    assert_eq!(f.granted_room(), 5 * UNIT, "an applied entry pays nothing");
}

/// The consensus core may append what a request queued after the request's
/// waiter went away: what was set aside stays until the log shows its fate,
/// so the late append is paid. A second request carrying the same entry
/// sets nothing aside again, and the one set-aside is paid once. Before,
/// the waiter's end gave the room back and the append then paid nothing.
#[test]
fn a_waiter_going_away_does_not_refund_a_queued_append() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    // A batch of twice the target: 10 units.
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    f.stage_granted(&[entry(1, 1, 10 * UNIT)]).unwrap();
    // The first waiter is gone; the leader sends the entry again.
    f.stage_granted(&[entry(1, 1, 10 * UNIT)]).unwrap();
    assert_eq!(f.granted_room(), 0);
    // The core appends what the first request queued.
    f.granted_appended(1, 1, 10 * UNIT);
    assert_eq!(f.granted_room(), 0, "the late append is paid");
    assert_eq!(f.get("ep").unwrap().admitted(), 10 * UNIT, "and paid once");
}

/// What was set aside goes back only on what the log shows: another term's
/// entry appended at that index, or the log applied past it. A truncation
/// frees exactly what the cut suffix paid and keeps what is set aside for
/// the entries replacing it.
#[test]
fn only_the_log_frees_what_was_set_aside_or_paid() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    // A batch of twice the target: 10 units.
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    take(&f, &[entry(5, 1, 2 * UNIT), entry(6, 1, 3 * UNIT)]).unwrap();
    assert_eq!(f.granted_room(), 5 * UNIT);

    // A stale leader's request for indexes 6 and 8 that the log rejects.
    f.stage_granted(&[entry(6, 2, UNIT), entry(8, 2, UNIT)])
        .unwrap();
    assert_eq!(f.granted_room(), 3 * UNIT, "set aside, fate unknown");

    // The real leader cuts from 6 and appends its own entry there.
    f.stage_granted(&[entry(6, 3, UNIT)]).unwrap();
    f.granted_truncated(6);
    f.granted_appended(6, 3, UNIT);
    assert_eq!(
        f.granted_room(),
        7 * UNIT,
        "old 6 freed, both stale set-asides freed (term 3 is in the log), the new 6 paid"
    );

    // A set-aside the log never appends goes back once it is applied past.
    f.stage_granted(&[entry(9, 3, UNIT)]).unwrap();
    f.applied(9, 6, 3 * UNIT);
    assert_eq!(
        f.granted_room(),
        7 * UNIT,
        "the never-appended 9 is refunded"
    );
    // 5 and 6 landed: their room left the pool for the data they hold.
    assert_eq!(
        f.get("ep").unwrap().admitted(),
        7 * UNIT,
        "nothing else lost"
    );
}

/// A second report of the same room after an entry the first paid for
/// reaches further: the leader may spend the new room on an entry that has
/// not arrived when the member restarts. The restart holds it again. Before,
/// an equal room was not stored again, and the restart held nothing.
#[test]
fn an_equal_room_reported_later_reaches_further() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 20 * UNIT);
    let disk = Disk::default();
    // A batch of twice the target: 10 units.
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    assert_eq!(disk.report(&f, 0).available, 10 * UNIT);
    take(&f, &[entry(1, 1, 10 * UNIT)]).unwrap();
    f.top_up_grant();
    assert_eq!(disk.report(&f, 1).available, 10 * UNIT);
    assert_eq!(disk.0.get().horizon, 20 * UNIT);

    let again = tempfile::tempdir().unwrap();
    let g = disk.restart(again.path(), 0, (0, 0), &[entry(1, 1, 10 * UNIT)]);
    assert_eq!(g.granted_room(), 10 * UNIT, "the second report's room");
    assert_eq!(
        g.get("ep").unwrap().admitted(),
        20 * UNIT,
        "and the unapplied entry"
    );
}

/// Room set aside for an entry that never reached the log is still owed
/// after a restart, whatever a later report said: the leader sent it and
/// sends it again. Before, a lower report replaced the promise and the
/// restart refused the resent entry.
#[test]
fn a_set_aside_entry_lost_in_a_restart_is_still_owed() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 20 * UNIT);
    let disk = Disk::default();
    // A batch of twice the target: 10 units.
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    disk.report(&f, 0);
    f.stage_granted(&[entry(1, 1, 10 * UNIT)]).unwrap();
    assert_eq!(disk.report(&f, 0).available, 0);

    let again = tempfile::tempdir().unwrap();
    let g = disk.restart(again.path(), 0, (0, 0), &[]);
    assert_eq!(g.granted_room(), 10 * UNIT);
    take(&g, &[entry(1, 1, 10 * UNIT)]).unwrap();
}

/// An entry applied before a restart took its share of what was reported:
/// the group's applied total counts it, whether the log still holds it or
/// was purged past it, and the restart owes the rest, nothing more. Before,
/// the restart read only the unapplied entries and held the whole report
/// again.
#[test]
fn an_applied_entry_spent_its_share_across_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 20 * UNIT);
    let disk = Disk::default();
    // A batch of twice the target: 10 units.
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    disk.report(&f, 0);
    take(&f, &[entry(1, 1, 6 * UNIT)]).unwrap();
    f.applied(1, 1, 6 * UNIT);

    let again = tempfile::tempdir().unwrap();
    let g = disk.restart(again.path(), 1, (1, 6 * UNIT), &[]);
    assert_eq!(g.granted_room(), 4 * UNIT);
    assert_eq!(
        g.get("ep").unwrap().admitted(),
        4 * UNIT,
        "nothing held for the applied entry"
    );
}

/// A snapshot installed past client entries this member never received
/// carries the group's applied total: what those entries took of a report
/// is spent, live and after a restart. Without the total the member owed
/// the whole report again on top of the snapshot's data.
#[test]
fn a_snapshot_spends_what_its_entries_took() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 20 * UNIT);
    let disk = Disk::default();
    // A batch of twice the target: 10 units.
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    assert_eq!(disk.report(&f, 0).available, 10 * UNIT);
    // The leader wrote entry 1 (10 units) against it; this member gets a
    // snapshot through 1 instead of the entry.
    f.applied(1, 1, 10 * UNIT);
    let (report, _) = f.grant_report(1);
    assert_eq!(report.available, 0, "the reported room is spent");

    let again = tempfile::tempdir().unwrap();
    let g = disk.restart(again.path(), 1, (1, 10 * UNIT), &[]);
    assert_eq!(g.get("ep").unwrap().admitted(), 0, "nothing owed");
}

/// A truncation takes the cut entries' bytes out of the log: what they
/// took of a report is owed again, after a restart too.
#[test]
fn a_truncated_entry_gives_its_share_back() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 20 * UNIT);
    let disk = Disk::default();
    // A batch of twice the target: 10 units.
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    disk.report(&f, 0);
    take(&f, &[entry(1, 1, 6 * UNIT)]).unwrap();
    f.granted_truncated(1);
    assert_eq!(f.granted_room(), 10 * UNIT);
    assert_eq!(disk.report(&f, 0).available, 10 * UNIT);
    assert_eq!(disk.0.get().horizon, 10 * UNIT, "reaches no further");

    let again = tempfile::tempdir().unwrap();
    let g = disk.restart(again.path(), 0, (0, 0), &[]);
    assert_eq!(g.granted_room(), 10 * UNIT);
}

/// A report never reaches past the stored horizon: while a raise is being
/// stored (or failed to be), reports carry the stored reach, and only one
/// raise is asked for at a time. The reply is never held for the store.
#[test]
fn a_report_reaches_no_further_than_what_is_stored() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    let disk = Disk::default();
    // Batches of twice the target: 4 units, then up to 10.
    f.set_grant_target(2 * UNIT);
    f.top_up_grant();
    disk.report(&f, 0);
    f.set_grant_target(5 * UNIT);
    f.top_up_grant();
    let (r, raise) = f.grant_report(0);
    assert_eq!(r.available, 4 * UNIT, "the stored reach, not the pool");
    let raised = raise.expect("a raise to store");
    assert_eq!(raised.horizon, 10 * UNIT);
    let (r, again) = f.grant_report(0);
    assert_eq!(
        (r.available, again),
        (4 * UNIT, None),
        "one raise at a time"
    );
    take(&f, &[entry(1, 1, 3 * UNIT)]).unwrap();
    f.grant_horizon_stored(raised, false);
    let (r, retry) = f.grant_report(1);
    assert_eq!(r.available, UNIT, "a failed store reaches no further");
    let retry = retry.expect("asked again");
    f.grant_horizon_stored(retry, true);
    assert_eq!(f.grant_report(1).0.available, 7 * UNIT);
}

/// What the pool keeps per entry is bounded by the entries not yet applied,
/// never by the log: a long applied tail the log has not purged yet leaves
/// nothing behind, whatever the grant. Before, the pool indexed every client
/// entry until the log was purged past it.
#[test]
fn the_pool_keeps_only_the_unapplied_entries() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    f.set_grant_target(UNIT);
    f.top_up_grant();
    let mut total = 0;
    for index in 1..=10_000u64 {
        take(&f, &[entry(index, 1, 64)]).unwrap();
        total += 64;
        if index % 50 == 0 {
            f.applied(index, index, total);
            {
                let grants = f.grants.lock();
                assert!(grants.logged.is_empty() && grants.paid.is_empty());
            }
            // As every reply to the leader does.
            f.top_up_grant();
        }
    }
    take(&f, &[entry(10_001, 1, 64)]).unwrap();
    let grants = f.grants.lock();
    assert_eq!((grants.logged.len(), grants.paid.len()), (1, 1));
    assert_eq!(grants.cum(), total + 64);
}

/// A horizon whose store hangs holds nothing up: reports, set-asides,
/// appends and applies go on (the pool's lock is not held across the
/// store), no second store is asked for, and the stored horizon bounds
/// every report until the hung one lands.
#[test]
fn a_hung_store_holds_up_neither_reports_nor_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    f.set_grant_target(10 * UNIT);
    f.top_up_grant();
    let (_, raise) = f.grant_report(0);
    let hung = raise.expect("a raise to store");
    let started = std::time::Instant::now();
    for index in 1..=1_000u64 {
        let (report, again) = f.grant_report(index - 1);
        assert_eq!((report.available, again), (0, None));
        take(&f, &[entry(index, 1, 8)]).unwrap();
        f.applied(index, index, index * 8);
        f.top_up_grant();
    }
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "the log waited on the store"
    );
    f.grant_horizon_stored(hung, true);
    assert!(
        f.grant_report(1_000).0.available > 0,
        "reaches it once stored"
    );
}

/// Steady client writes cost at most one synced store per report that
/// reaches past the stored horizon, and none while one is in flight: with
/// a report after every entry and every store landing at once, a store per
/// entry, the bound; slower stores coalesce.
#[test]
fn steady_writes_store_at_most_once_per_report() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    f.set_grant_target(10 * UNIT);
    f.top_up_grant();
    let mut stores = 0u64;
    let mut in_flight = None;
    for index in 1..=1_000u64 {
        let (_, raise) = f.grant_report(index - 1);
        if let Some(raised) = raise {
            assert!(in_flight.is_none(), "a second store while one is in flight");
            stores += 1;
            in_flight = Some(raised);
        }
        // Every other store lands before the next report.
        if index % 2 == 0
            && let Some(raised) = in_flight.take()
        {
            f.grant_horizon_stored(raised, true);
        }
        take(&f, &[entry(index, 1, 64)]).unwrap();
        f.applied(index, index, index * 64);
        f.top_up_grant();
    }
    assert!(stores <= 500, "{stores} stores for 1000 reports");
}

/// Run `entries` entries of `bytes` each through `f` as a member does in
/// steady writes (a report before each, its raise stored at once, the
/// entry appended and applied, the pool refilled); returns how many
/// horizons were stored and the most the pool held, free and for entries.
fn steady(f: &Footprints, entries: u64, bytes: u64) -> (u64, u64) {
    let disk = Disk::default();
    let mut stores = 0;
    let mut peak = 0;
    for index in 1..=entries {
        let (_, raise) = f.grant_report(index - 1);
        if let Some(raised) = raise {
            stores += 1;
            disk.0.set(raised);
            f.grant_horizon_stored(raised, true);
        }
        take(f, &[entry(index, 1, bytes)]).unwrap();
        let (free, held) = f.granted_held();
        peak = peak.max(free + held);
        f.applied(index, index, index * bytes);
        f.top_up_grant();
    }
    (stores, peak)
}

/// The pool refills in batches: a horizon is stored once per target's
/// worth of entries, not once per entry, across many refills, and the pool
/// never holds more than twice its target. Before, it refilled up to its
/// target after every entry, and every report stored a further horizon.
#[test]
fn refills_come_in_batches_of_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 1_000 * UNIT, 0);
    f.set_grant_target(10 * UNIT);
    f.top_up_grant();
    // 50 refills' worth of entries.
    let (stores, peak) = steady(&f, 500, UNIT);
    assert!(
        stores <= 500 / 10 + 2,
        "{stores} horizons stored for 500 entries"
    );
    assert!(peak <= 20 * UNIT, "the pool held {peak} bytes");
}

/// An endpoint short of a whole batch grants what fits: the smaller grant
/// stands, refills go on as room allows, and the recovery reserve is never
/// taken.
#[test]
fn a_short_endpoint_grants_what_fits() {
    // In steps of the smallest grant asked for.
    const STEP: u64 = super::GRANT_STEP;
    let dir = tempfile::tempdir().unwrap();
    // 27 steps of client room above a 20-step reserve, 20 of them taken by
    // another client write: 7 left for a batch of 20.
    let f = footprints(dir.path(), 47 * STEP, 20 * STEP);
    let other = f.admit(20 * STEP, Admit::Client).unwrap();
    f.set_grant_target(10 * STEP);
    f.top_up_grant();
    let granted = f.granted_room();
    assert!(
        granted > 0 && granted <= 7 * STEP,
        "what fits of the 7 steps left: {granted}"
    );
    assert!(f.admit(STEP, Admit::Recovery).is_ok(), "the reserve stays");
    drop(other);
    f.top_up_grant();
    assert!(f.granted_room() > granted, "refilled once room came back");
    assert!(f.granted_room() <= 20 * STEP);
}

/// A target of 0 holds nothing; a lowered target takes back nothing granted,
/// and refills only once the pool is below it.
#[test]
fn a_zero_or_lowered_target() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    f.set_grant_target(0);
    f.top_up_grant();
    assert_eq!(f.granted_room(), 0);
    f.set_grant_target(10 * UNIT);
    f.top_up_grant();
    assert_eq!(f.granted_room(), 20 * UNIT);
    f.set_grant_target(2 * UNIT);
    f.top_up_grant();
    assert_eq!(f.granted_room(), 20 * UNIT, "nothing taken back");
    take(&f, &[entry(1, 1, 17 * UNIT)]).unwrap();
    f.top_up_grant();
    assert_eq!(f.granted_room(), 3 * UNIT, "at or above the lowered target");
    take(&f, &[entry(2, 1, 2 * UNIT)]).unwrap();
    f.top_up_grant();
    assert_eq!(
        f.granted_room(),
        4 * UNIT,
        "refilled to twice the lowered target"
    );
}

/// Between refills the pool holds at least its target, so an entry the
/// size of the target always fits the grant the leader saw: a batch is
/// never spent below the target before the next one comes.
#[test]
fn an_entry_of_the_target_fits_between_refills() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 0);
    f.set_grant_target(10 * UNIT);
    f.top_up_grant();
    take(&f, &[entry(1, 1, 9 * UNIT)]).unwrap();
    f.top_up_grant();
    assert_eq!(f.granted_room(), 11 * UNIT, "no refill above the target");
    take(&f, &[entry(2, 1, 10 * UNIT)]).unwrap();
}

/// Committed entries take the grant while it holds them all, else the
/// recovery reserve for exactly the entries not paid yet, once; room the
/// reserve gave that the log never takes goes back to the footprint, not
/// into the pool, and the recovery check runs only when the reserve is used.
#[test]
fn committed_entries_take_the_grant_or_the_reserve_once() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 50 * UNIT);
    f.set_grant_target(2 * UNIT);
    f.top_up_grant();
    let ep = f.get("ep").unwrap();
    let no_check = || -> Result<(), StorageError> { panic!("the grant held them") };
    f.stage_settled(&[entry(1, 1, 3 * UNIT)], no_check).unwrap();
    assert_eq!(f.granted_room(), UNIT, "from the grant");
    f.granted_appended(1, 1, 3 * UNIT);

    let before = ep.admitted();
    let mut checked = 0;
    f.stage_settled(&[entry(1, 1, 3 * UNIT), entry(2, 1, 5 * UNIT)], || {
        checked += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(checked, 1);
    assert_eq!(ep.admitted() - before, 5 * UNIT, "the unpaid entry only");
    assert_eq!(f.granted_room(), UNIT, "the grant untouched");
    // A repeat takes nothing.
    f.stage_settled(&[entry(2, 1, 5 * UNIT)], || panic!("nothing new"))
        .unwrap();

    // Entry 2 is never appended; applied past it, its reserve room goes
    // back to the footprint, not into the pool.
    f.applied(2, 2, 8 * UNIT);
    assert_eq!(f.granted_room(), UNIT, "the pool did not grow");
    assert_eq!(ep.admitted(), before - 3 * UNIT, "both rooms released");
}

/// The recovery check of committed entries (a free-space reading that may
/// stall on the filesystem) holds nothing up: while it is stopped, reports,
/// appends, applies and truncations of the same pool go on. An entry that
/// lands meanwhile is not paid again when the check returns, and a check
/// that refuses leaves no new hold. Before, the check ran under the pool's
/// lock and every one of them waited for it.
#[test]
fn a_stopped_recovery_check_holds_up_nothing() {
    use std::sync::mpsc;
    use std::time::Duration;
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path(), 100 * UNIT, 50 * UNIT);
    f.set_grant_target(UNIT);
    f.top_up_grant();
    let ep = f.get("ep").unwrap().clone();
    let before = ep.admitted();

    let (entered_tx, entered) = mpsc::channel::<()>();
    let (release, release_rx) = mpsc::channel::<Result<(), StorageError>>();
    let checking = {
        let f = f.clone();
        std::thread::spawn(move || {
            f.stage_settled(&[entry(2, 1, 5 * UNIT)], || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap()
            })
        })
    };
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("the check started");

    // Meanwhile: a report, the entry appended and applied, a truncation.
    let (done_tx, done) = mpsc::channel::<()>();
    let others = {
        let f = f.clone();
        std::thread::spawn(move || {
            let _ = f.grant_report(1);
            f.granted_appended(2, 1, 5 * UNIT);
            f.applied(2, 2, 5 * UNIT);
            f.granted_truncated(3);
            done_tx.send(()).unwrap();
        })
    };
    let went_on = done.recv_timeout(Duration::from_secs(2)).is_ok();
    release.send(Ok(())).unwrap();
    checking.join().unwrap().unwrap();
    others.join().unwrap();
    assert!(went_on, "the pool waited on the recovery check");
    assert_eq!(
        ep.admitted(),
        before,
        "the entry applied meanwhile is not paid again"
    );

    // A refusing check leaves no new hold.
    let refused = f.stage_settled(&[entry(3, 1, 5 * UNIT)], || {
        Err(StorageError::Io("no room on the disk".into()))
    });
    assert!(refused.is_err());
    assert_eq!(ep.admitted(), before, "nothing held after a refusal");
}

/// Without a hard limit the pool is unbounded and holds nothing.
#[test]
fn no_limit_grants_without_bound() {
    let f = Footprints::default();
    take(&f, &[entry(1, 1, u64::MAX / 2)]).unwrap();
    assert_eq!(f.granted_room(), u64::MAX);
}
