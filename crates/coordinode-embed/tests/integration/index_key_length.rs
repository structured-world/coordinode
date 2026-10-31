//! A value whose index entry would be longer than a key may be is refused
//! wherever an entry is made: by a write, by an index build over stored
//! nodes, and by a change of what a type implies. The refusal is typed and
//! leaves what was stored and the view readers had as they were, across a
//! reopen.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::{EntryError, KeyTooLong, MAX_INDEX_KEY_LEN};
use coordinode_core::index::encoding::{encode_tuple, entry_key_len};
use coordinode_embed::{Database, DatabaseError};

fn run(db: &mut Database, query: &str) -> Vec<std::collections::BTreeMap<String, Value>> {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
}

fn count(db: &mut Database, query: &str) -> usize {
    run(db, query).len()
}

fn refused(db: &mut Database, query: &str) -> DatabaseError {
    match db.execute_cypher(query) {
        Ok(rows) => panic!(
            "{} was accepted: {} rows",
            &query[..query.len().min(80)],
            rows.len()
        ),
        Err(e) => e,
    }
}

/// The longest title whose entry in a one-property index fits a key.
fn longest_title() -> usize {
    let overhead = encode_tuple(&[Value::String(String::new())])
        .expect("empty")
        .len();
    MAX_INDEX_KEY_LEN - entry_key_len(0, false) - overhead
}

/// The refusal is the typed one, for a key of `len` bytes when given.
fn key_refusal(err: &DatabaseError, len: Option<usize>) {
    match err {
        DatabaseError::IndexEntry(EntryError::KeyTooLong(KeyTooLong { len: got })) => {
            assert!(*got > MAX_INDEX_KEY_LEN, "{got}");
            if let Some(len) = len {
                assert_eq!(*got, len);
            }
        }
        other => panic!("not the typed key refusal: {other:?}"),
    }
}

fn typed(err: &DatabaseError) {
    key_refusal(err, None);
}

/// A title whose entry is exactly as long as a key may be is written and
/// found through the index; one a byte longer, or one of zero bytes the
/// encoding escapes past the limit, is refused with the typed error, and
/// nothing of the refused write is stored: before and after a reopen.
#[test]
fn a_write_past_the_longest_key_is_refused_at_its_boundary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let longest = longest_title();
    {
        let mut db = Database::open(dir.path()).expect("open");
        run(&mut db, "CREATE NODE TYPE Doc WITH (title: STRING)");
        run(&mut db, "CREATE INDEX doc_title ON :Doc(title)");

        let at_limit = "k".repeat(longest);
        run(&mut db, &format!("CREATE (:Doc {{title: '{at_limit}'}})"));
        assert_eq!(
            count(
                &mut db,
                &format!("MATCH (d:Doc {{title: '{at_limit}'}}) RETURN d")
            ),
            1
        );

        let past = "p".repeat(longest + 1);
        key_refusal(
            &refused(&mut db, &format!("CREATE (:Doc {{title: '{past}'}})")),
            Some(MAX_INDEX_KEY_LEN + 1),
        );
        // Each zero byte takes two bytes in the key.
        let zeros = "\\u0000".repeat(longest / 2 + 1);
        typed(&refused(
            &mut db,
            &format!("CREATE (:Doc {{title: '{zeros}'}})"),
        ));
        // An update to a title past the limit is refused the same way.
        typed(&refused(
            &mut db,
            &format!("MATCH (d:Doc) SET d.title = '{past}' RETURN d"),
        ));
        assert_eq!(count(&mut db, "MATCH (d:Doc) RETURN d"), 1);
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(count(&mut db, "MATCH (d:Doc) RETURN d"), 1);
    run(&mut db, "CREATE (:Doc {title: 'short'})");
    assert_eq!(count(&mut db, "MATCH (d:Doc {title: 'short'}) RETURN d"), 1);
}

/// A compound key of two values each well within a key, together past it,
/// is refused the same way.
#[test]
fn a_compound_entry_past_the_longest_key_is_refused() {
    let mut db = Database::open_in_memory().expect("open");
    run(
        &mut db,
        "CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE (p.first, p.last) IS NODE KEY",
    );
    let half = "h".repeat(longest_title() / 2 + 1);
    typed(&refused(
        &mut db,
        &format!("CREATE (:Person {{first: '{half}', last: '{half}'}})"),
    ));
    run(&mut db, "CREATE (:Person {first: 'Ada', last: 'Byron'})");
    assert_eq!(count(&mut db, "MATCH (p:Person) RETURN p"), 1);
}

/// An index created over a stored node whose title is past the limit is
/// not built: the build reports the typed refusal instead of failing the
/// process, the node stays, and writes go on.
#[test]
fn an_index_build_over_a_value_past_the_longest_key_fails_typed() {
    let mut db = Database::open_in_memory().expect("open");
    run(&mut db, "CREATE NODE TYPE Doc WITH (title: STRING)");
    let past = "p".repeat(longest_title() + 1);
    run(&mut db, &format!("CREATE (:Doc {{title: '{past}'}})"));
    let outcome = db.execute_cypher("CREATE INDEX doc_title ON :Doc(title)");
    // The statement's answer, and the outcome its build record keeps.
    key_refusal(&outcome.expect_err("the build took the value"), None);
    let kept = db
        .index_build_status()
        .expect("builds")
        .iter()
        .find_map(coordinode_query::index::BuildStatus::index_entry_refusal);
    assert!(
        matches!(kept, Some(EntryError::KeyTooLong(_))),
        "the build record keeps {kept:?}"
    );
    assert_eq!(count(&mut db, "MATCH (d:Doc) RETURN d"), 1);
    run(&mut db, "CREATE (:Doc {title: 'short'})");
    assert_eq!(count(&mut db, "MATCH (d:Doc) RETURN d"), 2);
}

/// A change of what a type implies placing a stored title past the limit
/// into an index is refused with the typed error: the type keeps implying
/// what it did, its nodes are not found under the label, the next change
/// runs, and so after a reopen. A large value no index reads does not stop
/// a change.
#[test]
fn an_implication_change_placing_a_value_past_the_longest_key_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        run(&mut db, "CREATE NODE TYPE Thing WITH (title: STRING)");
        run(
            &mut db,
            "CREATE NODE TYPE Task WITH (title: STRING, body: STRING)",
        );
        run(
            &mut db,
            "CREATE NODE TYPE Note WITH (title: STRING, body: STRING)",
        );
        run(&mut db, "CREATE INDEX thing_title ON :Thing(title)");
        let past = "p".repeat(longest_title() + 1);
        run(&mut db, &format!("CREATE (:Task {{title: '{past}'}})"));
        let body = "b".repeat(200_000);
        run(
            &mut db,
            &format!("CREATE (:Note {{title: 'n', body: '{body}'}})"),
        );

        typed(&refused(
            &mut db,
            "ALTER NODE TYPE Task SET IMPLIES (Thing)",
        ));
        assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 0);

        run(&mut db, "ALTER NODE TYPE Note SET IMPLIES (Thing)");
        assert_eq!(count(&mut db, "MATCH (n:Thing {title: 'n'}) RETURN n"), 1);
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 1);
    assert_eq!(count(&mut db, "MATCH (n:Task) RETURN n"), 1);
    // The refused change's record keeps the typed refusal across the reopen.
    assert!(
        kept_entry_refusals(&db) >= 1,
        "no certification keeps the typed refusal"
    );
    typed(&refused(
        &mut db,
        "ALTER NODE TYPE Task SET IMPLIES (Thing)",
    ));
}

/// How many concluded certifications keep a typed index refusal.
fn kept_entry_refusals(db: &Database) -> usize {
    use coordinode_modality::index::{IndexStore as _, LocalIndexStore};
    use coordinode_modality::index_def::{BuildState, FailureCause};
    LocalIndexStore::new(db.engine())
        .list_certifications()
        .expect("certifications")
        .iter()
        .filter(|r| {
            matches!(
                r.state,
                BuildState::Failed {
                    cause: Some(FailureCause::IndexEntry(EntryError::KeyTooLong(_))),
                    ..
                }
            )
        })
        .count()
}

/// Under a uniqueness, a change placing a stored title past the limit is
/// refused with the typed error before anything is staged: the removal from
/// the other scopes that comes first writes no ownership revision for a value
/// no entry can hold. Retried, it is refused the same way; once the long
/// title is gone it goes through, and so it stays after a reopen.
#[test]
fn a_unique_change_over_a_value_past_the_longest_key_is_refused_and_retried() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        run(&mut db, "CREATE NODE TYPE Thing WITH (title: STRING)");
        run(&mut db, "CREATE NODE TYPE Task WITH (title: STRING)");
        run(
            &mut db,
            "CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE",
        );
        let past = "p".repeat(70 * 1024);
        run(&mut db, &format!("CREATE (:Task {{title: '{past}'}})"));
        run(&mut db, "CREATE (:Task {title: 'short'})");

        typed(&refused(
            &mut db,
            "ALTER NODE TYPE Task SET IMPLIES (Thing)",
        ));
        typed(&refused(
            &mut db,
            "ALTER NODE TYPE Task SET IMPLIES (Thing)",
        ));
        assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 0);

        run(
            &mut db,
            "MATCH (t:Task) WHERE size(t.title) > 1000 DELETE t",
        );
        run(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
        assert_eq!(
            count(&mut db, "MATCH (n:Thing {title: 'short'}) RETURN n"),
            1
        );
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing {title: 'short'}) RETURN n"),
        1
    );
    typed(&refused(
        &mut db,
        &format!("CREATE (:Thing {{title: '{}'}})", "q".repeat(70 * 1024)),
    ));
}
