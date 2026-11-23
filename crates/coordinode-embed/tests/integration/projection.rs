//! A projection kept in one database from another's change feed: the
//! effects the host derives and the progress through the feed commit
//! together or not at all, a batch is applied once, and a transaction
//! arriving in fragments takes effect whole, after a reopen too.

#![allow(clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use coordinode_core::budget::QueryBudget;
use coordinode_core::graph::types::Value;
use coordinode_embed::changes::{
    ChangeBatch, ChangeCursor, ChangeInterest, OwnedTarget, ProjectionTransaction, UnitChanges,
};
use coordinode_embed::{Database, DatabaseError};

fn run(db: &mut Database, query: &str) {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn source(dir: &std::path::Path) -> Database {
    let mut db = Database::open(dir).expect("open source");
    run(&mut db, "CREATE NODE TYPE Person WITH (name: STRING)");
    db
}

/// Mirror each node and relationship event of `unit` as a `Seen` node keyed
/// by its target: a put or patch merges it, a removal deletes it.
fn mirror(tx: &ProjectionTransaction<'_>, unit: &UnitChanges) -> Result<(), DatabaseError> {
    for event in &unit.events {
        let key = match &event.target {
            OwnedTarget::Node(key) => format!("{key:02x?}"),
            OwnedTarget::Relationship { source, target, .. } => format!("{source}->{target}"),
            _ => continue,
        };
        let params = HashMap::from([("key".to_string(), Value::String(key))]);
        let query = match event.action {
            coordinode_embed::changes::Action::Remove => "MATCH (s:Seen {key: $key}) DELETE s",
            _ => "MERGE (:Seen {key: $key})",
        };
        tx.execute(query, Some(params))?;
    }
    Ok(())
}

/// Record each event of `unit` as a `Seen` node keyed by its identity, what
/// it does aside.
fn record(tx: &ProjectionTransaction<'_>, unit: &UnitChanges) -> Result<(), DatabaseError> {
    for event in &unit.events {
        let key = format!(
            "{}:{}:{}",
            event.id.position.index, event.id.position.sub, event.id.ordinal
        );
        let params = HashMap::from([("key".to_string(), Value::String(key))]);
        tx.execute("MERGE (:Seen {key: $key})", Some(params))?;
    }
    Ok(())
}

fn seen(db: &mut Database) -> usize {
    db.execute_cypher("MATCH (s:Seen) RETURN s")
        .expect("seen")
        .len()
}

/// A copy of the next batch from `cursor`, whole units, which the test keeps
/// as its own memory to hand over more than once.
fn read(db: &Database, cursor: &ChangeCursor) -> ChangeBatch {
    let budget = QueryBudget::new(1 << 24);
    db.read_changes(cursor, 64, &budget, None)
        .expect("read")
        .batch()
        .clone()
}

/// A batch's effects and the projection's progress commit together: a batch
/// whose host step fails applies nothing and leaves the progress, the same
/// batch applied again takes effect once, and a batch applied twice is
/// refused the second time.
#[test]
fn effects_and_progress_commit_together_and_once() {
    let src_dir = tempfile::tempdir().expect("tempdir");
    let dst_dir = tempfile::tempdir().expect("tempdir");
    let mut src = source(src_dir.path());
    let mut dst = Database::open(dst_dir.path()).expect("open destination");
    let start = src
        .change_cursor(&ChangeInterest::default())
        .expect("cursor");
    let started = dst.start_projection("people", &start).expect("start");
    assert_eq!(started.applied, start);

    run(
        &mut src,
        "CREATE (:Person {name: 'a'}), (:Person {name: 'b'})",
    );
    let batch = read(&src, &start);

    // The host fails after writing one effect: none of it lands.
    let failed = dst.apply_changes("people", &batch, |tx, unit| {
        mirror(tx, unit)?;
        Err(DatabaseError::Other("the host stopped".into()))
    });
    assert!(failed.is_err());
    assert_eq!(seen(&mut dst), 0);
    let progress = dst
        .projection_progress("people")
        .expect("read")
        .expect("kept");
    assert_eq!(progress.applied, start);

    let applied = dst.apply_changes("people", &batch, mirror).expect("apply");
    assert_eq!(applied.applied, batch.cursor);
    assert_eq!(applied.received, batch.cursor);
    assert!(!applied.has_staged());
    assert_eq!(seen(&mut dst), 2);

    // The same batch again: refused, nothing applied twice.
    let again = dst.apply_changes("people", &batch, mirror);
    assert!(
        matches!(again, Err(DatabaseError::ProjectionOutOfStep { .. })),
        "{again:?}"
    );
    assert_eq!(seen(&mut dst), 2);

    // Deletions follow from where it stands.
    run(&mut src, "MATCH (n:Person {name: 'a'}) DELETE n");
    let next = read(&src, &applied.received);
    dst.apply_changes("people", &next, mirror).expect("apply");
    assert_eq!(seen(&mut dst), 1);

    assert!(matches!(
        dst.apply_changes("nobody", &batch, mirror),
        Err(DatabaseError::UnknownProjection(_))
    ));
}

/// The fragments of one source transaction, read as the stream service
/// reads them: from the source's journal, within the smallest budget a read
/// from `cursor` keeps an event in, so the large transaction's events do not
/// fit one read. Each with the cursor it was read from. Read as a follower
/// of the cursor's interest, as a stream reads, each fragment the one a
/// read alone gets.
fn fragments(src: &Database, cursor: &ChangeCursor) -> Vec<(ChangeCursor, ChangeBatch)> {
    let follower = src.follow_changes(cursor).expect("follow");
    let read = |at: &ChangeCursor, bytes: u64| {
        let followed = src
            .engine()
            .read_changes(at, 1_000, &QueryBudget::new(bytes), Some(&follower))
            .map(|(batch, _charge)| batch);
        let alone = src
            .engine()
            .read_changes(at, 1_000, &QueryBudget::new(bytes), None)
            .map(|(batch, _charge)| batch);
        assert_eq!(
            followed.as_ref().ok(),
            alone.as_ref().ok(),
            "a follower's fragment is the one read alone"
        );
        followed
    };
    let budget_bytes = (1..)
        .map(|step| step * 500)
        .find(|&bytes| read(cursor, bytes).is_ok())
        .expect("a budget a read keeps an event in");
    let mut out = Vec::new();
    let mut at = cursor.clone();
    loop {
        let batch = read(&at, budget_bytes).expect("read");
        let moved = batch.cursor != at;
        let from = std::mem::replace(&mut at, batch.cursor.clone());
        if batch.units.is_empty() {
            return out;
        }
        out.push((from, batch));
        if !moved {
            return out;
        }
    }
}

/// A transaction arriving in fragments is staged as received, its progress
/// received past each fragment and applied before it, its effects nowhere;
/// the fragment completing it applies every staged fragment and itself in
/// one commit. A reopen between fragments loses none of them.
#[test]
fn a_transaction_in_fragments_takes_effect_whole() {
    let src_dir = tempfile::tempdir().expect("tempdir");
    let dst_dir = tempfile::tempdir().expect("tempdir");
    let mut src = source(src_dir.path());
    let start = linked_hub(&mut src);
    let pieces = fragments(&src, &start);
    let staged_pieces = pieces
        .iter()
        .filter(|(_, batch)| batch.units.iter().any(|u| !u.completes_unit()))
        .count();
    assert!(staged_pieces > 1, "{} staged pieces", staged_pieces);

    {
        let dst = Database::open(dst_dir.path()).expect("open destination");
        dst.start_projection("people", &start).expect("start");
    }
    let mut applied_calls = Vec::new();
    for (from, batch) in &pieces {
        // A destination opened again for every fragment.
        let mut dst = Database::open(dst_dir.path()).expect("reopen destination");
        let complete = batch.units.iter().all(UnitChanges::completes_unit);
        if complete {
            // The host fails on the last fragment, after the staged ones were
            // handed to it: they stay staged, nothing takes effect, and the
            // same batch applies whole next.
            let failed = dst.apply_changes("people", batch, |tx, unit| {
                record(tx, unit)?;
                if unit.completes_unit() {
                    return Err(DatabaseError::Other("the host stopped".into()));
                }
                Ok(())
            });
            assert!(failed.is_err());
            assert_eq!(seen(&mut dst), 0);
            let stands = dst
                .projection_progress("people")
                .expect("read")
                .expect("kept");
            assert_eq!(&stands.received, from);
            assert!(stands.has_staged());
        }
        let progress = dst
            .apply_changes("people", batch, |tx, unit| {
                applied_calls.push((unit.first, unit.through));
                record(tx, unit)
            })
            .expect("apply");
        assert_eq!(progress.received, batch.cursor);
        if complete {
            assert_eq!(progress.applied, batch.cursor);
            assert_eq!(seen(&mut dst), LINKS);
        } else {
            assert!(progress.has_staged());
            assert_eq!(seen(&mut dst), 0, "nothing takes effect before the end");
        }
    }
    // Every fragment applied once, in order, covering the transaction.
    let projected = pieces
        .iter()
        .flat_map(|(_, batch)| batch.units.iter())
        .find(|u| u.projected > 0)
        .expect("the transaction")
        .projected;
    let mut covered = 0;
    for (first, through) in &applied_calls {
        assert_eq!(*first, covered);
        covered = *through;
    }
    assert_eq!(covered, projected);
}

/// Nodes the large transaction of these tests removes.
const LINKS: usize = 300;

/// Write [`LINKS`] members of a type implying many labels, then return a
/// cursor before one transaction deleting them all, which the commit packs
/// into runs of deletes: a transaction whose few stored operations project
/// to many events, each routed by far more labels than its row stores.
fn linked_hub(src: &mut Database) -> ChangeCursor {
    let implied: Vec<String> = (1..=30).map(|i| format!("Group{i}")).collect();
    run(
        src,
        &format!(
            "CREATE NODE TYPE Member IMPLIES ({}) WITH (name: STRING)",
            implied.join(", ")
        ),
    );
    run(
        src,
        &format!("UNWIND range(1, {LINKS}) AS i CREATE (:Member {{name: 'm' + toString(i)}})"),
    );
    let start = src
        .change_cursor(&ChangeInterest::default())
        .expect("cursor");
    run(src, "MATCH (m:Member) DELETE m");
    start
}

/// `projection`'s committed progress, which a refusal must leave as it was.
fn stands(db: &Database, projection: &str) -> (ChangeCursor, ChangeCursor) {
    let progress = db
        .projection_progress(projection)
        .expect("read")
        .expect("kept");
    (progress.received, progress.applied)
}

/// A batch that does not continue where the projection stands is refused
/// before the host is called: one after a skipped batch, one read for
/// another interest from the same place, and a fragment after a skipped
/// fragment of the same transaction. Nothing is applied, staged or moved.
#[test]
fn a_batch_that_skips_or_reads_otherwise_is_refused() {
    let src_dir = tempfile::tempdir().expect("tempdir");
    let dst_dir = tempfile::tempdir().expect("tempdir");
    let mut src = source(src_dir.path());
    let mut dst = Database::open(dst_dir.path()).expect("open destination");
    let start = src
        .change_cursor(&ChangeInterest::default())
        .expect("cursor");
    dst.start_projection("people", &start).expect("start");

    run(&mut src, "CREATE (:Person {name: 'a'})");
    let first = read(&src, &start);
    run(&mut src, "CREATE (:Person {name: 'b'})");
    let second = read(&src, &first.cursor);
    let refuses = |dst: &Database, batch: &ChangeBatch| {
        let before = stands(dst, "people");
        let mut called = false;
        let refused = dst.apply_changes("people", batch, |_, _| {
            called = true;
            Ok(())
        });
        assert!(
            matches!(refused, Err(DatabaseError::ProjectionOutOfStep { .. })),
            "{refused:?}"
        );
        assert!(!called, "refused before the host is called");
        assert_eq!(stands(dst, "people"), before);
    };
    // The second batch handed over as if it followed the start.
    refuses(&dst, &second);
    // The first batch, read for another interest.
    let other = src
        .change_cursor_at(
            start.position(),
            &ChangeInterest {
                labels: Some(vec!["Person".into()]),
                edge_types: None,
                blobs: false,
            },
        )
        .expect("cursor");
    refuses(&dst, &read(&src, &other));
    assert_eq!(seen(&mut dst), 0);
}

/// A fragment after a skipped fragment of the same transaction is refused
/// before the host is called, the fragments staged so far kept as they
/// were.
#[test]
fn a_fragment_after_a_skipped_one_is_refused() {
    let src_dir = tempfile::tempdir().expect("tempdir");
    let dst_dir = tempfile::tempdir().expect("tempdir");
    let mut src = source(src_dir.path());
    let dst = Database::open(dst_dir.path()).expect("open destination");
    let start = linked_hub(&mut src);
    let pieces = fragments(&src, &start);
    assert!(pieces.len() >= 3, "{} pieces", pieces.len());
    assert!(
        !pieces[0].1.units.iter().all(UnitChanges::completes_unit),
        "the first piece stops inside the transaction"
    );
    dst.start_projection("people", &start).expect("start");
    dst.apply_changes("people", &pieces[0].1, record)
        .expect("the first fragment");
    let before = stands(&dst, "people");
    // The third piece handed over in place of the second.
    let mut called = false;
    let refused = dst.apply_changes("people", &pieces[2].1, |_, _| {
        called = true;
        Ok(())
    });
    assert!(
        matches!(refused, Err(DatabaseError::ProjectionOutOfStep { .. })),
        "{refused:?}"
    );
    assert!(!called);
    assert_eq!(stands(&dst, "people"), before);
}
