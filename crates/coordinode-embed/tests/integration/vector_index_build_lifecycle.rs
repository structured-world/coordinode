//! The lifecycle of an asynchronous vector-index build: who owns it, when it
//! stops, and what it is allowed to write.
//!
//! A build outliving the statement that started it used to write the index
//! definition from a detached thread — progress checkpoints every thousand
//! nodes, then a terminal state. Any later statement touching that index drew
//! its read snapshot before those writes landed, so conflict detection read
//! them as a concurrent transaction and rejected the statement. Both tests
//! here exercise that boundary from the two directions it can be crossed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;

/// Back-to-back CREATE / DROP of the same index must never collide with its
/// own build. The loop is what makes it a test: the failure is a race, and a
/// single pass hits it only when the machine is loaded enough for the build's
/// write to slip past the next statement's snapshot.
#[test]
fn create_drop_cycle_never_conflicts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open db");

    for i in 0..500 {
        let name = format!("idx_{i}");
        db.execute_cypher(&format!(
            "CREATE VECTOR INDEX {name} ON :Item(embedding) OPTIONS {{metric: \"cosine\"}}"
        ))
        .unwrap_or_else(|e| panic!("create #{i}: {e:?}"));
        db.execute_cypher(&format!("DROP VECTOR INDEX {name}"))
            .unwrap_or_else(|e| panic!("drop #{i}: {e:?}"));
    }
}

/// Dropping an index whose build is still running must cancel that build, and
/// the cancelled build must not resurrect the definition it was working on:
/// reopening the database has to find the index gone.
#[test]
fn drop_cancels_a_running_build_and_it_stays_dropped() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open db");

        // Enough vectors that the backfill is still scanning when the DROP
        // arrives — a build that finished first would prove nothing.
        db.execute_cypher(
            "UNWIND range(1, 4000) AS i \
             CREATE (:Item {embedding: [toFloat(i), 1.0, 2.0]})",
        )
        .expect("seed vectors");

        db.execute_cypher(
            "CREATE VECTOR INDEX live_build ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
        )
        .expect("create vector index");

        assert!(
            !db.index_builds().is_empty(),
            "the build should still be running for this test to mean anything"
        );

        db.execute_cypher("DROP VECTOR INDEX live_build")
            .expect("drop while building");

        assert!(
            db.index_builds().is_empty(),
            "DROP returned while its index's build was still running"
        );

        // A cancelled build that was going to write anyway would do it within
        // this window; the reopen below is what catches it if it did.
        std::thread::sleep(std::time::Duration::from_millis(300));
    }

    let mut reopened = Database::open(dir.path()).expect("reopen db");
    // Re-creating under the same name proves the old definition is really
    // gone: a resurrected one would still occupy (label, property).
    reopened
        .execute_cypher(
            "CREATE VECTOR INDEX live_build ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
        )
        .expect("re-create after the cancelled build");
}

/// Deterministic 8-dimensional vector for row `i`, each coordinate an
/// independent uniform draw (splitmix64). Coordinates derived from one hash
/// of `i` put the rows on a low-dimensional curve, where an approximate
/// index misses its own points for reasons that have nothing to do with
/// what is tested here.
fn spread_vector(i: usize) -> String {
    (0..8u64)
        .map(|d| {
            let mut z = (i as u64)
                .wrapping_mul(8)
                .wrapping_add(d)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            format!("{:.6}", (z >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn create_items(db: &mut Database, range: std::ops::Range<usize>) {
    let rows = range
        .map(|i| format!("{{ext_id: {i}, embedding: [{}]}}", spread_vector(i)))
        .collect::<Vec<_>>()
        .join(", ");
    db.execute_cypher(&format!(
        "UNWIND [{rows}] AS row CREATE (:Item {{ext_id: row.ext_id, embedding: row.embedding}})"
    ))
    .expect("create items");
}

/// The node id of every item written as a row from `from` on, by row, in one
/// scan.
fn node_ids_from(db: &mut Database, from: usize) -> Vec<(usize, u64)> {
    let mut ids: Vec<(usize, u64)> = db
        .execute_cypher(&format!(
            "MATCH (n:Item) WHERE n.ext_id >= {from} RETURN n.ext_id AS row, id(n) AS id"
        ))
        .expect("look up items")
        .iter()
        .map(|r| {
            let field = |name: &str| r.get(name).and_then(|v| v.as_int()).expect(name);
            (
                usize::try_from(field("row")).expect("rows are not negative"),
                u64::try_from(field("id")).expect("node ids are not negative"),
            )
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// Statements that keep committing for the whole length of a build, each in
/// its own transaction, some before the handover and some after: every
/// vector they wrote is in the index's graph once the worker has folded what
/// applied. Membership is checked on the graph itself: a nearest-neighbour
/// search is approximate and misses a point now and then (1 in 24,900 seen)
/// for reasons unrelated to whether the build kept the write.
#[test]
fn every_write_across_a_build_is_in_the_index() {
    const N: usize = 6000;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open db");
    for start in (0..N).step_by(500) {
        create_items(&mut db, start..start + 500);
    }
    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) \
         OPTIONS {m: 16, ef_construction: 100, metric: \"euclidean\", dimensions: 8, \
         online_during_build: \"partial-recall\"}",
    )
    .expect("create vector index");

    let mut late = N;
    let mut during = 0;
    while !db.index_builds().is_empty() {
        create_items(&mut db, late..late + 20);
        late += 20;
        during += 1;
    }
    assert!(during > 1, "the build finished before the writes started");

    let ids = node_ids_from(&mut db, N);
    assert_eq!(ids.len(), late - N, "every row written is in the store");
    let registry = db.vector_index_registry();
    // The embedded database keeps its nodes on shard 1.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !registry.delta(1).is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "the vector worker did not fold the applied writes within 60 s"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let handle = registry.get("Item", "embedding").expect("the index");
    let graph = handle.read().expect("the graph");
    let missing: Vec<usize> = ids
        .iter()
        .filter(|&&(_, id)| !graph.contains(id))
        .map(|&(i, _)| i)
        .collect();
    drop(graph);
    assert!(
        missing.is_empty(),
        "{} of {} vectors written across the build are not in the index: {missing:?}",
        missing.len(),
        late - N
    );
    let plan = db
        .explain_cypher(&format!(
            "MATCH (n:Item) WITH *, vector_distance(n.embedding, [{}]) AS d \
             ORDER BY d ASC LIMIT 1 RETURN n.ext_id",
            spread_vector(0)
        ))
        .expect("explain");
    assert!(
        plan.contains("HnswScan"),
        "searched without the index:\n{plan}"
    );
}

/// The bound on waiting for a building index belongs to the caller: the
/// session's bound applies to a query that names none, and a query's own hint
/// wins over it. With the session at zero a read of the building index is
/// refused at once; the same read with a generous hint waits and is served.
#[test]
fn a_query_hint_bounds_the_wait_over_the_session() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open db");
    // Enough vectors that the build is still running when the reads arrive.
    for start in (0..20_000).step_by(2_000) {
        create_items(&mut db, start..start + 2_000);
    }
    db.execute_cypher("SET vector_build_wait = '0ms'")
        .expect("session bound");
    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) \
         OPTIONS {m: 16, ef_construction: 100, metric: \"euclidean\", dimensions: 8}",
    )
    .expect("create vector index");
    let search = format!(
        "MATCH (n:Item) WITH *, vector_distance(n.embedding, [{}]) AS d \
         ORDER BY d ASC LIMIT 1 RETURN n.ext_id AS ext_id",
        spread_vector(7)
    );

    assert!(
        !db.index_builds().is_empty(),
        "the build finished before the reads, so they prove nothing"
    );
    let refused = db.execute_cypher(&search);
    assert!(
        refused.is_err(),
        "the session's zero bound refuses the building index: {refused:?}"
    );

    let served = db
        .execute_cypher(&format!("{search} /*+ vector_build_wait('2m') */"))
        .expect("the query's own bound waits for the build");
    assert_eq!(
        served
            .first()
            .and_then(|row| row.get("ext_id"))
            .and_then(|v| v.as_int()),
        Some(7)
    );
}

/// Writes that land WHILE the index is building must end up in it.
///
/// While a build runs the writer leaves the index alone: batching the vectors
/// through the build is much cheaper per vector than encoding and linking one
/// at a time. What makes that safe is the build's tap of applied writes, which
/// delivers everything that lands after the scan's snapshot and is folded in
/// until nothing is left, and only then hands maintenance back to the
/// writers. Without it those writes would simply be missing from the graph.
#[test]
fn writes_during_a_build_are_drained_into_the_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open db");

    db.execute_cypher(
        "UNWIND range(1, 3000) AS i CREATE (:Doc {tag: 'seed', embedding: [toFloat(i), 0.0, 0.0]})",
    )
    .expect("seed vectors");

    db.execute_cypher(
        "CREATE VECTOR INDEX doc_emb ON :Doc(embedding) OPTIONS {metric: \"cosine\", online_during_build: \"partial-recall\"}",
    )
    .expect("create vector index");

    // Written while the build is in flight, so only the tap can place them.
    db.execute_cypher(
        "UNWIND range(1, 200) AS i \
         CREATE (:Doc {tag: 'late', embedding: [0.0, toFloat(i), 0.0]})",
    )
    .expect("write during build");

    // Wait for the build to finish rather than sleeping a guess.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !db.index_builds().is_empty() {
        assert!(std::time::Instant::now() < deadline, "build never finished");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // Every late vector must be its own nearest neighbour. A vector the build
    // missed answers with something else entirely.
    for i in (1..=200).map(|i| i as f32) {
        let rows = db
            .execute_cypher(&format!(
                "MATCH (n:Doc) WITH n, vector_similarity(n.embedding, [0.0, {i:.1}, 0.0]) AS s \
                 ORDER BY s DESC LIMIT 1 RETURN n.tag AS tag"
            ))
            .expect("vector search");
        assert_eq!(
            rows.len(),
            1,
            "expected one nearest neighbour for the late vector {i}"
        );
        let tag = rows[0]
            .get("tag")
            .and_then(|v| v.as_str().map(String::from));
        assert_eq!(
            tag.as_deref(),
            Some("late"),
            "a vector written during the build was not folded into the index"
        );
    }
}
