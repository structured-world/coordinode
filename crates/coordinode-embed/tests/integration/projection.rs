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
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};

fn run(db: &mut Database, query: &str) {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn source(dir: &std::path::Path) -> Database {
    let mut db = Database::open(dir).expect("open source");
    run(&mut db, "CREATE NODE TYPE Person WITH (name: STRING)");
    db
}

/// Mirror each node event of `unit` as a `Seen` node keyed by the node's
/// key bytes: a put or patch merges it, a removal deletes it.
fn mirror(tx: &ProjectionTransaction<'_>, unit: &UnitChanges) -> Result<(), DatabaseError> {
    for event in &unit.events {
        let OwnedTarget::Node(key) = &event.target else {
            continue;
        };
        let params = HashMap::from([("key".to_string(), Value::String(format!("{key:02x?}")))]);
        let query = match event.action {
            coordinode_embed::changes::Action::Remove => "MATCH (s:Seen {key: $key}) DELETE s",
            _ => "MERGE (:Seen {key: $key})",
        };
        tx.execute(query, Some(params))?;
    }
    Ok(())
}

fn seen(db: &mut Database) -> usize {
    db.execute_cypher("MATCH (s:Seen) RETURN s")
        .expect("seen")
        .len()
}

/// The next batch from `cursor`, whole units.
fn read(db: &Database, cursor: &ChangeCursor) -> ChangeBatch {
    let budget = QueryBudget::new(1 << 24);
    db.read_changes(cursor, 64, &budget).expect("read").batch
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
    let failed = dst.apply_changes("people", &start, &batch, |tx, unit| {
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

    let applied = dst
        .apply_changes("people", &start, &batch, mirror)
        .expect("apply");
    assert_eq!(applied.applied, batch.cursor);
    assert_eq!(applied.received, batch.cursor);
    assert!(!applied.has_staged());
    assert_eq!(seen(&mut dst), 2);

    // The same batch again: refused, nothing applied twice.
    let again = dst.apply_changes("people", &start, &batch, mirror);
    assert!(
        matches!(again, Err(DatabaseError::ProjectionOutOfStep { .. })),
        "{again:?}"
    );
    assert_eq!(seen(&mut dst), 2);

    // Deletions follow from where it stands.
    run(&mut src, "MATCH (n:Person {name: 'a'}) DELETE n");
    let next = read(&src, &applied.received);
    dst.apply_changes("people", &applied.received, &next, mirror)
        .expect("apply");
    assert_eq!(seen(&mut dst), 1);

    assert!(matches!(
        dst.apply_changes("nobody", &start, &batch, mirror),
        Err(DatabaseError::UnknownProjection(_))
    ));
}

/// The fragments of one source transaction, read as the stream service
/// reads them: from the source's journal as stored, within `budget_bytes`
/// of events a read, starting at `cursor`. Each with the cursor it was read
/// from.
fn fragments(
    src: &Database,
    src_dir: &std::path::Path,
    cursor: &ChangeCursor,
    budget_bytes: u64,
) -> Vec<(ChangeCursor, ChangeBatch)> {
    let mut tailer = OplogTailer::new(
        &[src_dir.join("oplog").join("0")],
        ResumeToken::from_start(0),
    )
    .expect("tailer")
    .keeping_frames();
    let entries: Vec<_> = tailer
        .read_next(1_000, &CdcFilters::default(), u64::MAX)
        .expect("entries")
        .into_iter()
        .map(|(entry, _)| entry)
        .filter(|entry| entry.index >= cursor.position())
        .collect();
    let next = entries.last().expect("an entry").index + 1;
    let mut out = Vec::new();
    let mut at = cursor.clone();
    loop {
        let budget = QueryBudget::new(budget_bytes);
        let (batch, _charge) = src
            .engine()
            .changes_of_entries(
                &at,
                entries.iter().filter(|e| e.index >= at.position()),
                next,
                &budget,
            )
            .expect("read");
        let moved = batch.cursor != at;
        let from = std::mem::replace(&mut at, batch.cursor.clone());
        out.push((from, batch));
        if !moved || at.position() >= next {
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
    let start = src
        .change_cursor(&ChangeInterest::default())
        .expect("cursor");
    run(
        &mut src,
        "UNWIND range(1, 40) AS i CREATE (:Person {name: 'p' + toString(i)})",
    );
    let pieces = fragments(&src, src_dir.path(), &start, 2_000);
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
        let progress = dst
            .apply_changes("people", from, batch, |tx, unit| {
                applied_calls.push((unit.first, unit.through));
                mirror(tx, unit)
            })
            .expect("apply");
        assert_eq!(progress.received, batch.cursor);
        let complete = batch.units.iter().all(UnitChanges::completes_unit);
        if complete {
            assert_eq!(progress.applied, batch.cursor);
            assert_eq!(seen(&mut dst), 40);
        } else {
            assert!(progress.has_staged());
            assert_eq!(seen(&mut dst), 0, "nothing takes effect before the end");
        }
    }
    // Every fragment applied once, in order, covering the transaction.
    let mut covered = 0;
    for (first, through) in &applied_calls {
        assert_eq!(*first, covered);
        covered = *through;
    }
    assert_eq!(covered, 40);
}
