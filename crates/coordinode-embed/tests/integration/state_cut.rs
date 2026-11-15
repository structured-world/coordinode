//! State recovery through a fixed cut, end to end: a consumer copies a
//! scope of the cut's view into a new generation of its destination, reads
//! the continuation until it joins, and publishes the replacement and its
//! progress in one destination write. Rows deleted at the source, and rows
//! the destination held before, are absent from the replacement; power
//! lost on either side of the publication leaves one complete projection.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::budget::QueryBudget;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::Journal;
use coordinode_embed::Database;
use coordinode_embed::consumer::{CutResume, CutScope, StateCut};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::PowerRig;

const NODES: &[u8] = b"node:";
const CURRENT: &[u8] = b"projection:current";

type Rows = BTreeMap<Vec<u8>, Vec<u8>>;

fn scope(prefix: &[u8]) -> CutScope {
    CutScope::rows(Partition::Node, prefix.to_vec()).expect("a row scope")
}

/// A destination keeping each generation of its projection under its own
/// prefix and naming the published one, with its progress, in one record:
/// publishing a generation is one journalled write.
struct Destination {
    engine: Arc<StorageEngine>,
}

impl Destination {
    fn open(rig: &PowerRig) -> Self {
        Self {
            engine: Arc::new(
                StorageEngine::open_embedded(&rig.config(), Arc::new(TimestampOracle::new()))
                    .expect("destination"),
            ),
        }
    }

    fn write(&self, mutations: &[Mutation]) {
        commit(&self.engine, mutations);
    }

    /// The published generation and progress, if any.
    fn current(&self) -> Option<(u64, u64)> {
        let record = self.engine.get(Partition::Node, CURRENT).expect("read")?;
        let generation = u64::from_be_bytes(record[..8].try_into().expect("generation"));
        let progress = u64::from_be_bytes(record[8..16].try_into().expect("progress"));
        Some((generation, progress))
    }

    fn prefix(generation: u64) -> Vec<u8> {
        format!("g{generation:020}:").into_bytes()
    }

    /// The rows of the published generation.
    fn rows(&self) -> Rows {
        let Some((generation, _)) = self.current() else {
            return Rows::new();
        };
        let prefix = Self::prefix(generation);
        self.engine
            .snapshot_prefix_scan(&self.engine.snapshot(), Partition::Node, &prefix)
            .expect("scan")
            .into_iter()
            .map(|(key, value)| (key[prefix.len()..].to_vec(), value.to_vec()))
            .collect()
    }

    /// Write `rows` as `generation`, unpublished.
    fn stage(&self, generation: u64, rows: &Rows) {
        let prefix = Self::prefix(generation);
        let puts: Vec<Mutation> = rows
            .iter()
            .map(|(key, value)| {
                let mut stored = prefix.clone();
                stored.extend_from_slice(key);
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: stored,
                    value: value.clone(),
                }
            })
            .collect();
        if !puts.is_empty() {
            self.write(&puts);
        }
    }

    /// Publish `generation` with `progress`: one write.
    fn publish(&self, generation: u64, progress: u64) {
        let mut record = generation.to_be_bytes().to_vec();
        record.extend_from_slice(&progress.to_be_bytes());
        self.write(&[Mutation::Put {
            partition: PartitionId::Node,
            key: CURRENT.to_vec(),
            value: record,
        }]);
    }
}

/// Commit `mutations` to `engine`'s own journal at its clock's next stamp.
fn commit(engine: &StorageEngine, mutations: &[Mutation]) {
    let ts = engine.oracle().expect("an embedded engine").next().as_raw();
    engine
        .commit_logged(mutations, ts, Journal::Journal)
        .expect("commit");
}

fn source_rows(db: &Database, prefix: &[u8]) -> Rows {
    let engine = db.engine();
    engine
        .snapshot_prefix_scan(&engine.snapshot(), Partition::Node, prefix)
        .expect("source")
        .into_iter()
        .map(|(key, value)| (key, value.to_vec()))
        .collect()
}

/// The budget each read of these tests runs within.
fn budget() -> QueryBudget {
    QueryBudget::new(1 << 20)
}

/// Copy the cut's view, two keys a page.
fn copy_view(db: &Database, cut: &StateCut) -> Rows {
    let mut rows = Rows::new();
    let mut after: Option<Vec<u8>> = None;
    let budget = budget();
    loop {
        let page = db
            .state_cut_scan(cut, after.as_deref(), 2, &budget)
            .expect("page");
        let Some(last) = page.last_key() else {
            return rows;
        };
        after = Some(last.to_vec());
        rows.extend(page.rows().iter().cloned());
    }
}

/// Apply the continuation to `rows`, reading until the join is complete
/// and the log holds nothing more; where ordinary reading resumes.
fn read_continuation(db: &Database, cut: &mut StateCut, rows: &mut Rows) -> CutResume {
    let budget = budget();
    loop {
        let before = cut.next();
        let batch = db.state_cut_tail(cut, 4, &budget).expect("tail");
        for unit in batch.units() {
            for m in &unit.mutations {
                match m {
                    Mutation::Put { key, value, .. } => {
                        rows.insert(key.clone(), value.clone());
                    }
                    Mutation::Delete { key, .. } => {
                        rows.remove(key);
                    }
                    other => panic!("a node scope carries no {other:?}"),
                }
            }
        }
        if let Some(resume) = batch.joined()
            && cut.next() == before
        {
            return resume.clone();
        }
    }
}

fn open_source(dir: &std::path::Path) -> Database {
    let mut db = Database::open(dir).expect("source");
    for id in 1..=3 {
        db.execute_cypher(&format!("CREATE (:User {{id: {id}}})"))
            .expect("create");
    }
    db
}

/// A write inside the join while the next cut of `db` is taken.
fn write_inside_the_join(db: &Database, key: &'static [u8]) {
    let engine = db.engine_shared();
    let mut once = true;
    db.engine().set_cut_bound_hook(Some(Box::new(move || {
        if std::mem::take(&mut once) {
            commit(
                &engine,
                &[Mutation::Put {
                    partition: PartitionId::Node,
                    key: key.to_vec(),
                    value: b"joined".to_vec(),
                }],
            );
        }
    })));
}

fn rig_with_stale_projection() -> (PowerRig, Destination) {
    let rig = PowerRig::new();
    let destination = Destination::open(&rig);
    let mut stale = Rows::new();
    stale.insert(b"node:stale".to_vec(), b"old".to_vec());
    destination.stage(1, &stale);
    destination.publish(1, 0);
    (rig, destination)
}

/// A rebuild through a cut with a write inside the join and a create and a
/// delete after the cut: nothing is publishable before the join completes,
/// and the published replacement equals the source, holding neither the
/// deleted row nor the destination's earlier rows.
#[test]
fn a_rebuild_through_a_cut_replaces_the_destination_with_the_source() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open_source(&dir.path().join("source"));
    let (_rig, destination) = rig_with_stale_projection();

    write_inside_the_join(&db, b"node:joined");
    let mut cut = db
        .state_cut(scope(NODES), Duration::from_secs(10))
        .expect("cut");
    db.engine().set_cut_bound_hook(None);
    assert!(cut.join_end() > cut.first(), "the join holds a position");
    assert!(cut.joined().is_none(), "no publication before the join");
    db.execute_cypher("CREATE (:User {id: 4})").expect("create");
    db.execute_cypher("MATCH (u:User {id: 1}) DETACH DELETE u")
        .expect("delete");
    let mut rows = copy_view(&db, &cut);
    assert!(
        !rows.contains_key(b"node:joined".as_slice()),
        "the write inside the join is above the view"
    );
    let resume = read_continuation(&db, &mut cut, &mut rows);
    let progress = db.resume_position(&resume, &scope(NODES)).expect("resume");

    destination.stage(2, &rows);
    destination.publish(2, progress);
    assert_eq!(destination.rows(), source_rows(&db, NODES));
    assert!(destination.rows().contains_key(b"node:joined".as_slice()));
    assert!(!destination.rows().contains_key(b"node:stale".as_slice()));
    assert_eq!(destination.current(), Some((2, progress)));
}

/// Power lost after the replacement is staged and before it is published
/// leaves the previous projection and progress; lost after the publication,
/// the new one, complete.
#[test]
fn power_lost_on_either_side_of_the_publication_leaves_one_projection() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_source(&dir.path().join("source"));
    let (rig, destination) = rig_with_stale_projection();
    let stale = destination.rows();
    let mut cut = db
        .state_cut(scope(NODES), Duration::from_secs(10))
        .expect("cut");
    let mut rows = copy_view(&db, &cut);
    let resume = read_continuation(&db, &mut cut, &mut rows);
    let progress = db.resume_position(&resume, &scope(NODES)).expect("resume");

    destination.stage(2, &rows);
    rig.cut(destination.engine);
    let destination = Destination::open(&rig);
    assert_eq!(destination.current(), Some((1, 0)), "not published");
    assert_eq!(destination.rows(), stale);

    destination.publish(2, progress);
    rig.cut(destination.engine);
    let destination = Destination::open(&rig);
    assert_eq!(destination.current(), Some((2, progress)));
    assert_eq!(destination.rows(), source_rows(&db, NODES));
}

/// An empty scope rebuilds to absence: the destination's earlier rows go.
#[test]
fn an_empty_scope_rebuilds_to_absence() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_source(&dir.path().join("source"));
    let rig = PowerRig::new();
    let destination = Destination::open(&rig);
    let mut stale = Rows::new();
    stale.insert(b"node:none:stale".to_vec(), b"old".to_vec());
    destination.stage(1, &stale);
    destination.publish(1, 0);
    let mut cut = db
        .state_cut(scope(b"node:none:"), Duration::from_secs(10))
        .expect("cut");
    let mut rows = copy_view(&db, &cut);
    let resume = read_continuation(&db, &mut cut, &mut rows);
    destination.stage(2, &rows);
    destination.publish(
        2,
        db.resume_position(&resume, &scope(b"node:none:"))
            .expect("resume"),
    );
    assert!(
        destination.rows().is_empty(),
        "nothing of the scope remains"
    );
}

/// A scope a row copy cannot recover is refused before any cut is taken:
/// an index is not recovered as rows.
#[test]
fn a_scope_a_row_copy_cannot_recover_is_refused() {
    assert!(CutScope::rows(Partition::Idx, Vec::new()).is_err());
    assert!(CutScope::rows(Partition::Schema, Vec::new()).is_err());
}

/// A view does not survive its database: the database reopens while a cut
/// taken of it is still held, and the reopened one refuses the old cut and
/// gives a new one with a bound of its own.
#[test]
fn a_view_is_lost_with_its_database_and_a_new_cut_is_taken() {
    let dir = tempfile::tempdir().unwrap();
    let at = dir.path().join("source");
    let db = open_source(&at);
    let mut old = db
        .state_cut(scope(NODES), Duration::from_secs(10))
        .expect("cut");
    drop(db);
    let mut db = Database::open(&at).expect("reopen while the old cut is held");
    db.execute_cypher("CREATE (:User {id: 5})").expect("create");
    let new = db
        .state_cut(scope(NODES), Duration::from_secs(10))
        .expect("new cut");
    assert!(new.bound() > old.bound(), "a view of its own");
    assert_eq!(copy_view(&db, &new), source_rows(&db, NODES));
    let budget = budget();
    assert!(
        db.state_cut_scan(&old, None, 2, &budget).is_err(),
        "the lost view is not served from the reopened database"
    );
    assert!(
        db.state_cut_tail(&mut old, 4, &budget).is_err(),
        "nor is its continuation"
    );
}

/// A cut belongs to its history: read against another database, its
/// continuation is refused rather than joined to a log it was not taken of.
#[test]
fn a_cut_read_against_another_history_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let a = open_source(&dir.path().join("a"));
    let b = open_source(&dir.path().join("b"));
    let mut cut = a
        .state_cut(scope(NODES), Duration::from_secs(10))
        .expect("cut");
    b.state_cut(scope(NODES), Duration::from_secs(10))
        .expect("b founds its history");
    let before = cut.next();
    assert!(
        b.state_cut_tail(&mut cut, 4, &budget()).is_err(),
        "another history"
    );
    assert_eq!(cut.next(), before, "the refusal does not move it");
}

/// Where reading resumes after a join names its history and scope: given
/// back for them, refused in another database's log or for another scope,
/// rather than read as a number of a log it does not count.
#[test]
fn a_resume_position_is_given_back_only_for_its_history_and_scope() {
    let dir = tempfile::tempdir().unwrap();
    let a = open_source(&dir.path().join("a"));
    let b = open_source(&dir.path().join("b"));
    b.state_cut(scope(NODES), Duration::from_secs(10))
        .expect("b founds its history");
    let mut cut = a
        .state_cut(scope(NODES), Duration::from_secs(10))
        .expect("cut");
    let mut rows = copy_view(&a, &cut);
    let resume = read_continuation(&a, &mut cut, &mut rows);
    assert_eq!(
        a.resume_position(&resume, &scope(NODES)).expect("resume"),
        cut.next()
    );
    assert!(
        a.resume_position(&resume, &scope(b"node:other:")).is_err(),
        "another scope"
    );
    assert!(
        b.resume_position(&resume, &scope(NODES)).is_err(),
        "another history"
    );
}

/// A read refused by its budget does not move the continuation: a later read
/// with room takes the same entries.
#[test]
fn a_read_refused_by_its_budget_leaves_the_continuation_where_it_was() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open_source(&dir.path().join("source"));
    write_inside_the_join(&db, b"node:joined");
    let mut cut = db
        .state_cut(scope(NODES), Duration::from_secs(10))
        .expect("cut");
    db.engine().set_cut_bound_hook(None);
    db.execute_cypher("CREATE (:User {id: 4})").expect("create");
    let before = cut.next();
    let tiny = QueryBudget::new(4);
    assert!(db.state_cut_tail(&mut cut, 4, &tiny).is_err(), "refused");
    assert_eq!(cut.next(), before, "the refusal does not move it");
    let mut rows = copy_view(&db, &cut);
    read_continuation(&db, &mut cut, &mut rows);
    assert_eq!(rows, source_rows(&db, NODES), "nothing was skipped");
}
