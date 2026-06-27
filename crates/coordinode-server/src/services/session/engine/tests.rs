use super::*;
use coordinode_session::{SessionEvent, SessionManager, SessionOp};

/// Build a tempdir-backed database seeded with `n` `:Page {k}` nodes, returned
/// behind the shared lock the cursor engine expects.
fn seeded_db(n: i64) -> Arc<RwLock<Database>> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for k in 0..n {
        db.execute_cypher(&format!("CREATE (n:Page {{k: {k}}})"))
            .expect("seed");
    }
    // Leak the tempdir so the path stays alive for the test's duration; the OS
    // reclaims it on process exit (test-only).
    std::mem::forget(dir);
    Arc::new(RwLock::new(db))
}

/// Drain a cursor in `batch`-sized steps until it returns an empty batch,
/// returning every row in order.
fn drain(cursor: &mut dyn QueryCursor, batch: usize) -> Vec<Vec<Value>> {
    let mut all = Vec::new();
    loop {
        let rows = cursor.next_batch(batch).expect("next_batch");
        if rows.is_empty() {
            break;
        }
        all.extend(rows);
    }
    all
}

#[test]
fn keyset_cursor_drains_every_row_across_small_batches() {
    // A plain label scan routes to the keyset cursor; draining it in batches
    // smaller than the result must still visit every node exactly once, with
    // the column header established before the first batch.
    let engine = DatabaseCursorEngine::new(seeded_db(25));
    let mut cursor = engine
        .open_cursor("MATCH (n:Page) RETURN n.k", HashMap::new(), 0)
        .expect("open");

    assert_eq!(cursor.columns(), vec!["n.k".to_string()]);

    let rows = drain(cursor.as_mut(), 4);
    let mut seen: Vec<i64> = rows
        .iter()
        .map(|r| match r.first() {
            Some(Value::Int(v)) => *v,
            other => panic!("unexpected cell: {other:?}"),
        })
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..25).collect::<Vec<_>>(), "every node drained once");
}

#[test]
fn keyset_cursor_with_filter_still_drains_all_matches() {
    // A Filter above the scan keeps the plan keyset-eligible; the cursor must
    // refill across pages (filtered pages yield fewer rows) and return exactly
    // the matching nodes.
    let engine = DatabaseCursorEngine::new(seeded_db(40));
    let mut cursor = engine
        .open_cursor(
            "MATCH (n:Page) WHERE n.k >= 30 RETURN n.k",
            HashMap::new(),
            0,
        )
        .expect("open");

    let rows = drain(cursor.as_mut(), 3);
    let mut seen: Vec<i64> = rows
        .iter()
        .map(|r| match r.first() {
            Some(Value::Int(v)) => *v,
            other => panic!("unexpected cell: {other:?}"),
        })
        .collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        (30..40).collect::<Vec<_>>(),
        "only matches, all of them"
    );
}

#[test]
fn blocking_plan_routes_to_materialize_and_still_returns_all_rows() {
    // ORDER BY blocks keyset paging, so the engine materializes. The cursor
    // protocol is identical from the caller's side: drain still yields every
    // row, here in sorted order.
    let engine = DatabaseCursorEngine::new(seeded_db(10));
    let mut cursor = engine
        .open_cursor("MATCH (n:Page) RETURN n.k ORDER BY n.k", HashMap::new(), 0)
        .expect("open");

    let rows = drain(cursor.as_mut(), 4);
    let ordered: Vec<i64> = rows
        .iter()
        .map(|r| match r.first() {
            Some(Value::Int(v)) => *v,
            other => panic!("unexpected cell: {other:?}"),
        })
        .collect();
    assert_eq!(ordered, (0..10).collect::<Vec<_>>(), "sorted, complete");
}

#[test]
fn keyset_cursor_empty_label_yields_no_rows() {
    let engine = DatabaseCursorEngine::new(seeded_db(0));
    let mut cursor = engine
        .open_cursor("MATCH (n:Page) RETURN n.k", HashMap::new(), 0)
        .expect("open");
    assert!(cursor.next_batch(8).expect("batch").is_empty());
}

// ----- integration: real engine driven through the session core -----

/// Run one `Execute` op through the full session core (`SessionManager::run` →
/// `dispatch` → real `DatabaseCursorEngine` → keyset cursor paging) and collect
/// the events it streams back. This exercises the production composition the
/// gRPC binding wires up, minus the transport.
async fn run_execute(engine: DatabaseCursorEngine, query: &str) -> Vec<SessionEvent> {
    use tokio::sync::mpsc;

    let manager = SessionManager::new(Arc::new(engine));
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let handle = tokio::spawn(manager.open().run(op_rx, ev_tx));

    op_tx
        .send((
            1,
            SessionOp::Execute {
                query: query.to_string(),
                params: HashMap::new(),
                txid: 0,
                nonce: 0,
            },
        ))
        .await
        .expect("send");
    drop(op_tx);

    let mut events = Vec::new();
    while let Some((_request_id, event)) = ev_rx.recv().await {
        events.push(event);
    }
    handle.await.expect("session task");
    events
}

#[tokio::test]
async fn session_core_streams_keyset_cursor_open_rows_end() {
    // The real keyset cursor, driven through the session core, must stream a
    // CursorOpen (with the column header), one or more Rows batches covering
    // every node, then CursorEnd. The result spans many internal pages
    // (KEYSET_PAGE) but the protocol is identical to a small one.
    let events = run_execute(
        DatabaseCursorEngine::new(seeded_db(50)),
        "MATCH (n:Page) RETURN n.k",
    )
    .await;

    let mut iter = events.iter();
    match iter.next() {
        Some(SessionEvent::CursorOpen { columns }) => {
            assert_eq!(columns.as_slice(), ["n.k".to_string()]);
        }
        other => panic!("expected CursorOpen first, got {other:?}"),
    }

    let mut seen: Vec<i64> = Vec::new();
    let mut ended = false;
    for event in iter {
        match event {
            SessionEvent::Rows { rows } => {
                for row in rows {
                    match row.first() {
                        Some(Value::Int(v)) => seen.push(*v),
                        other => panic!("unexpected cell: {other:?}"),
                    }
                }
            }
            SessionEvent::CursorEnd { .. } => {
                ended = true;
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert!(ended, "stream must terminate with CursorEnd");
    seen.sort_unstable();
    assert_eq!(
        seen,
        (0..50).collect::<Vec<_>>(),
        "every node streamed once"
    );
}

#[tokio::test]
async fn session_core_streams_empty_keyset_result_without_rows() {
    let events = run_execute(
        DatabaseCursorEngine::new(seeded_db(0)),
        "MATCH (n:Page) RETURN n.k",
    )
    .await;
    assert!(
        matches!(
            events.as_slice(),
            [
                SessionEvent::CursorOpen { .. },
                SessionEvent::CursorEnd { .. }
            ]
        ),
        "empty result is open-then-end, no Rows: {events:?}"
    );
}

#[tokio::test]
async fn session_core_surfaces_a_query_error() {
    // A semantically invalid query fails at open_cursor; the session core must
    // surface a single Error event rather than a partial cursor stream.

    let events = run_execute(
        DatabaseCursorEngine::new(seeded_db(1)),
        "MATCH (n) RETURN undefined_var",
    )
    .await;
    assert!(
        matches!(events.as_slice(), [SessionEvent::Error { .. }]),
        "expected a single Error, got {events:?}"
    );
}
