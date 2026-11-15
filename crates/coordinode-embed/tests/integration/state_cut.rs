//! State recovery through a fixed cut, end to end: a consumer copies a
//! scope of the cut's view into a new generation of its destination, reads
//! the continuation until it joins, and publishes the replacement and its
//! progress in one destination write. Rows deleted at the source, and rows
//! the destination held before, are absent from the replacement; a crash
//! on either side of the publication leaves one complete projection.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::time::Duration;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_embed::Database;
use coordinode_embed::consumer::StateCut;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

const NODES: &[u8] = b"node:";
const CURRENT: &[u8] = b"projection:current";

type Rows = BTreeMap<Vec<u8>, Vec<u8>>;

/// A destination keeping each generation of its projection under its own
/// prefix and naming the published one, with its progress, in one record:
/// publishing a generation is one write.
struct Destination {
    engine: StorageEngine,
}

impl Destination {
    fn open(dir: &std::path::Path) -> Self {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        Self {
            engine: StorageEngine::open(&config).expect("destination"),
        }
    }

    /// The published generation and progress, if any.
    fn current(&self) -> Option<(u64, u64)> {
        let record = self.engine.get(Partition::Schema, CURRENT).expect("read")?;
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
        for (key, value) in rows {
            let mut stored = prefix.clone();
            stored.extend_from_slice(key);
            self.engine
                .put(Partition::Node, &stored, value)
                .expect("stage");
        }
        self.engine.persist().expect("persist");
    }

    /// Publish `generation` with `progress`: one write.
    fn publish(&self, generation: u64, progress: u64) {
        let mut record = generation.to_be_bytes().to_vec();
        record.extend_from_slice(&progress.to_be_bytes());
        self.engine
            .put(Partition::Schema, CURRENT, &record)
            .expect("publish");
        self.engine.persist().expect("persist");
    }
}

fn source_rows(db: &Database, scope: &[u8]) -> Rows {
    let engine = db.engine();
    engine
        .snapshot_prefix_scan(&engine.snapshot(), Partition::Node, scope)
        .expect("source")
        .into_iter()
        .map(|(key, value)| (key, value.to_vec()))
        .collect()
}

/// Copy `scope` of the cut's view, two keys a page.
fn copy_view(db: &Database, cut: &StateCut, scope: &[u8]) -> Rows {
    let mut rows = Rows::new();
    let mut after: Option<Vec<u8>> = None;
    loop {
        let page = db
            .state_cut_scan(cut, Partition::Node, scope, after.as_deref(), 2)
            .expect("page");
        let Some((last, _)) = page.last() else {
            return rows;
        };
        after = Some(last.clone());
        rows.extend(page);
    }
}

/// Apply the continuation's node changes under `scope` to `rows`, reading
/// until the join is complete and the journal holds nothing more.
fn read_continuation(db: &Database, cut: &mut StateCut, scope: &[u8], rows: &mut Rows) {
    loop {
        let before = cut.next();
        let batch = db.state_cut_tail(cut, 4).expect("tail");
        for unit in &batch.units {
            for m in &unit.mutations {
                match m {
                    Mutation::Put {
                        partition: PartitionId::Node,
                        key,
                        value,
                    } if key.starts_with(scope) => {
                        rows.insert(key.clone(), value.clone());
                    }
                    Mutation::Delete {
                        partition: PartitionId::Node,
                        key,
                    } if key.starts_with(scope) => {
                        rows.remove(key);
                    }
                    _ => {}
                }
            }
        }
        if batch.joined.is_some() && cut.next() == before {
            return;
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

/// A rebuild through a cut, with a create and a delete racing the copy:
/// the published replacement equals the source, holds neither the deleted
/// row nor the destination's earlier rows, and is published only once the
/// join is complete.
#[test]
fn a_rebuild_through_a_cut_replaces_the_destination_with_the_source() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open_source(&dir.path().join("source"));
    let destination = Destination::open(&dir.path().join("destination"));
    let mut stale = Rows::new();
    stale.insert(b"node:stale".to_vec(), b"old".to_vec());
    destination.stage(1, &stale);
    destination.publish(1, 0);

    let mut cut = db.state_cut(Duration::from_secs(10)).expect("cut");
    db.execute_cypher("CREATE (:User {id: 4})").expect("create");
    db.execute_cypher("MATCH (u:User {id: 1}) DETACH DELETE u")
        .expect("delete");
    if cut.join_end() > cut.first() {
        assert!(cut.joined().is_none(), "no publication before the join");
    }
    let mut rows = copy_view(&db, &cut, NODES);
    read_continuation(&db, &mut cut, NODES, &mut rows);
    let progress = cut.joined().expect("joined");

    destination.stage(2, &rows);
    destination.publish(2, progress);
    assert_eq!(destination.rows(), source_rows(&db, NODES));
    assert!(!destination.rows().contains_key(b"node:stale".as_slice()));
    assert_eq!(destination.current(), Some((2, progress)));
}

/// A destination closed before the publication keeps its previous complete
/// projection and progress; closed after it, the new one.
#[test]
fn a_destination_closed_on_either_side_of_the_publication_holds_one_projection() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_source(&dir.path().join("source"));
    let at = dir.path().join("destination");
    let mut stale = Rows::new();
    stale.insert(b"node:stale".to_vec(), b"old".to_vec());
    {
        let destination = Destination::open(&at);
        destination.stage(1, &stale);
        destination.publish(1, 0);
    }
    let mut cut = db.state_cut(Duration::from_secs(10)).expect("cut");
    let mut rows = copy_view(&db, &cut, NODES);
    read_continuation(&db, &mut cut, NODES, &mut rows);
    let progress = cut.joined().expect("joined");
    {
        let destination = Destination::open(&at);
        destination.stage(2, &rows);
    }
    let destination = Destination::open(&at);
    assert_eq!(destination.current(), Some((1, 0)), "not yet published");
    assert_eq!(destination.rows(), stale);
    destination.publish(2, progress);
    drop(destination);
    let destination = Destination::open(&at);
    assert_eq!(destination.current(), Some((2, progress)));
    assert_eq!(destination.rows(), source_rows(&db, NODES));
}

/// An empty scope rebuilds to absence: the destination's earlier rows go.
#[test]
fn an_empty_scope_rebuilds_to_absence() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_source(&dir.path().join("source"));
    let destination = Destination::open(&dir.path().join("destination"));
    let mut stale = Rows::new();
    stale.insert(b"node:none:stale".to_vec(), b"old".to_vec());
    destination.stage(1, &stale);
    destination.publish(1, 0);
    let scope = b"node:none:";
    let mut cut = db.state_cut(Duration::from_secs(10)).expect("cut");
    let mut rows = copy_view(&db, &cut, scope);
    read_continuation(&db, &mut cut, scope, &mut rows);
    destination.stage(2, &rows);
    destination.publish(2, cut.joined().expect("joined"));
    assert!(
        destination.rows().is_empty(),
        "nothing of the scope remains"
    );
}

/// A view does not survive its database: the database reopens while a cut
/// taken of it is still held, and the reopened one gives a new cut with a
/// bound of its own, never the lost view again under old positions.
#[test]
fn a_view_is_lost_with_its_database_and_a_new_cut_is_taken() {
    let dir = tempfile::tempdir().unwrap();
    let at = dir.path().join("source");
    let db = open_source(&at);
    let old = db.state_cut(Duration::from_secs(10)).expect("cut");
    drop(db);
    let mut db = Database::open(&at).expect("reopen while the old cut is held");
    db.execute_cypher("CREATE (:User {id: 5})").expect("create");
    let new = db.state_cut(Duration::from_secs(10)).expect("new cut");
    assert!(new.bound() > old.bound(), "a view of its own");
    let rows = copy_view(&db, &new, NODES);
    assert_eq!(rows, source_rows(&db, NODES));
    let mut old = old;
    assert!(
        db.state_cut_scan(&old, Partition::Node, NODES, None, 2)
            .is_err(),
        "the lost view is not served from the reopened database"
    );
    assert!(
        db.state_cut_tail(&mut old, 4).is_err(),
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
    let mut cut = a.state_cut(Duration::from_secs(10)).expect("cut");
    b.state_cut(Duration::from_secs(10))
        .expect("b founds its history");
    assert!(b.state_cut_tail(&mut cut, 4).is_err(), "another history");
}
