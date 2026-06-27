//! SessionService gRPC integration tests.
//!
//! Exercises the multiplexed bidirectional `Session` stream against a real
//! `coordinode` binary: one stream carries many requests, each query runs as a
//! server-side cursor returning its real rows, the transaction sub-thread
//! interleaves with non-transactional requests, and concurrent queries stay
//! correlated. These drive the actual streaming handler and the
//! database-backed cursor engine end to end.
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration --test session
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use coordinode_integration::harness::CoordinodeProcess;
use coordinode_integration::proto::session::server_frame::Event;
use coordinode_integration::proto::session::session_service_client::SessionServiceClient;
use coordinode_integration::proto::session::{
    client_frame, Begin, ClientFrame, Commit, Execute, Rollback,
};

/// Begin an UNORDERED transaction (statements applied in arrival order; nonces
/// ignored). Used by the simple transaction tests that send nonce-0 statements.
fn begin(request_id: u64) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Begin(Begin {
            ordering: 2, // ORDERING_UNORDERED
            drain_timeout_ms: 0,
        })),
    }
}

/// Begin an ORDERED transaction (statements reassembled by nonce).
fn begin_ordered(request_id: u64) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Begin(Begin {
            ordering: 1, // ORDERING_ORDERED
            drain_timeout_ms: 0,
        })),
    }
}

fn execute(request_id: u64, query: &str) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Execute(Execute {
            query: query.to_string(),
            parameters: Default::default(),
            txid: 0,
            nonce: 0,
        })),
    }
}

fn commit(request_id: u64) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Commit(Commit {
            txid: 1,
            last_nonce: 0,
        })),
    }
}

/// An Execute frame bound to an open transaction `txid`.
fn execute_tx(request_id: u64, query: &str, txid: u64) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Execute(Execute {
            query: query.to_string(),
            parameters: Default::default(),
            txid,
            nonce: 0,
        })),
    }
}

/// An Execute frame bound to a transaction with an explicit ORDERED `nonce`.
fn execute_tx_nonce(request_id: u64, query: &str, txid: u64, nonce: u64) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Execute(Execute {
            query: query.to_string(),
            parameters: Default::default(),
            txid,
            nonce,
        })),
    }
}

/// A Commit frame carrying the expected final nonce of an ORDERED chain.
fn commit_tx_nonce(request_id: u64, txid: u64, last_nonce: u64) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Commit(Commit { txid, last_nonce })),
    }
}

/// A Commit frame for a specific transaction.
fn commit_tx(request_id: u64, txid: u64) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Commit(Commit {
            txid,
            last_nonce: 0,
        })),
    }
}

/// Open a session, send `frames`, and collect every returned `ServerFrame`
/// grouped by `request_id`. The server closes the response stream once the
/// client half-closes the request stream, so this drains to completion.
async fn run_session(
    proc: &CoordinodeProcess,
    frames: Vec<ClientFrame>,
) -> HashMap<u64, Vec<Event>> {
    let channel = tonic::transport::Endpoint::from_shared(proc.endpoint())
        .expect("valid endpoint")
        .connect()
        .await
        .expect("connect to session service");
    let mut client = SessionServiceClient::new(channel);

    let response = client
        .session(tokio_stream::iter(frames))
        .await
        .expect("open session");
    let mut inbound = response.into_inner();

    let mut by_id: HashMap<u64, Vec<Event>> = HashMap::new();
    while let Some(frame) = inbound.message().await.expect("server frame") {
        by_id
            .entry(frame.request_id)
            .or_default()
            .push(frame.event.expect("frame event"));
    }
    by_id
}

/// Total rows across the `Rows` frames of one request's event sequence.
fn row_count(events: &[Event]) -> usize {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Rows(b) => Some(b.rows.len()),
            _ => None,
        })
        .sum()
}

#[tokio::test]
async fn session_runs_a_real_query_and_pages_its_rows() {
    let proc = CoordinodeProcess::start().await;
    let by_id = run_session(&proc, vec![execute(1, "UNWIND [1, 2, 3] AS n RETURN n")]).await;
    let events = by_id.get(&1).map(Vec::as_slice).unwrap_or_default();
    match events {
        [Event::CursorOpen(open), rest @ ..] => {
            // Real column header from the RETURN clause, not an echo.
            assert_eq!(open.columns, vec!["n".to_string()]);
            assert_eq!(
                row_count(rest),
                3,
                "UNWIND of three values yields three rows"
            );
            assert!(
                matches!(rest.last(), Some(Event::CursorEnd(_))),
                "the cursor closes with CursorEnd"
            );
        }
        other => panic!("request 1 got {other:?}"),
    }
}

#[tokio::test]
async fn session_pages_a_label_scan_through_the_keyset_cursor() {
    // A `MATCH (n:Label) RETURN ...` scan is keyset-eligible: the server routes
    // it to the keyset cursor and pages it by storage key. Drive it end to end
    // through the real binary + gRPC stream and assert every seeded node comes
    // back exactly once, with the real column header and a terminating
    // CursorEnd. Multi-page refill across the page boundary is covered
    // deterministically by the engine-level tests; this proves the keyset path
    // is wired correctly through the transport.
    let proc = CoordinodeProcess::start().await;

    // Seed all nodes in a single statement (one write), drained to completion
    // before the scan: the scan must observe a committed dataset, and a second
    // session guarantees the write happened-before the read.
    run_session(
        &proc,
        vec![execute(
            1,
            "UNWIND range(0, 149) AS k CREATE (n:Scan {k: k})",
        )],
    )
    .await;

    // Scan in a fresh session: keyset-eligible, paged across the gRPC stream.
    let by_id = run_session(&proc, vec![execute(1, "MATCH (n:Scan) RETURN n.k AS k")]).await;
    let events = by_id.get(&1).map(Vec::as_slice).unwrap_or_default();
    match events {
        [Event::CursorOpen(open), rest @ ..] => {
            assert_eq!(open.columns, vec!["k".to_string()]);
            assert_eq!(row_count(rest), 150, "every seeded node is streamed once");
            assert!(
                matches!(rest.last(), Some(Event::CursorEnd(_))),
                "the keyset cursor closes with CursorEnd"
            );
        }
        other => panic!("scan request got {other:?}"),
    }
}

/// Drive a controlled session: Begin, run `write` inside the transaction, then
/// commit (or roll back). Returns once the transaction resolves. Sequences each
/// step on the response so Begin strictly precedes the in-transaction write.
async fn drive_transaction(proc: &CoordinodeProcess, write: &str, do_commit: bool) {
    use tokio_stream::wrappers::ReceiverStream;
    let channel = tonic::transport::Endpoint::from_shared(proc.endpoint())
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    let mut client = SessionServiceClient::new(channel);
    let (tx, rx) = tokio::sync::mpsc::channel::<ClientFrame>(8);
    let mut inbound = client
        .session(ReceiverStream::new(rx))
        .await
        .expect("open session")
        .into_inner();

    // Begin → handle.
    tx.send(begin(1)).await.expect("send begin");
    let txid = loop {
        let f = inbound.message().await.expect("frame").expect("not end");
        if f.request_id == 1 {
            match f.event {
                Some(Event::Begun(b)) => break b.txid,
                other => panic!("expected Begun, got {other:?}"),
            }
        }
    };

    // Write inside the transaction; wait for its cursor to finish.
    tx.send(execute_tx(2, write, txid))
        .await
        .expect("send write");
    loop {
        let f = inbound.message().await.expect("frame").expect("not end");
        if f.request_id == 2 {
            match f.event {
                Some(Event::CursorEnd(_)) => break,
                Some(Event::Error(e)) => panic!("in-txn write errored: {}", e.message),
                _ => {}
            }
        }
    }

    // Resolve.
    if do_commit {
        tx.send(commit_tx(3, txid)).await.expect("send commit");
        loop {
            let f = inbound.message().await.expect("frame").expect("not end");
            if f.request_id == 3 {
                match f.event {
                    Some(Event::Committed(_)) => break,
                    Some(Event::Error(e)) => panic!("commit errored: {}", e.message),
                    _ => {}
                }
            }
        }
    } else {
        let rollback = ClientFrame {
            request_id: 3,
            op: Some(client_frame::Op::Rollback(Rollback { txid })),
        };
        tx.send(rollback).await.expect("send rollback");
    }
    drop(tx);
    // Drain to end so the server finishes the transaction task.
    while inbound.message().await.expect("frame").is_some() {}
}

#[tokio::test]
async fn in_session_transaction_commit_persists_writes() {
    // A write inside a committed interactive transaction must be durable: a
    // fresh session sees it. Proves the real interactive transaction lifecycle
    // runs through the session core (begin, buffered write, commit).
    let proc = CoordinodeProcess::start().await;
    drive_transaction(&proc, "CREATE (n:Tx {k: 1})", true).await;

    let by_id = run_session(&proc, vec![execute(1, "MATCH (n:Tx) RETURN n.k AS k")]).await;
    let events = by_id.get(&1).map(Vec::as_slice).unwrap_or_default();
    match events {
        [Event::CursorOpen(_), rest @ ..] => {
            assert_eq!(row_count(rest), 1, "committed write is visible afterwards");
        }
        other => panic!("verify scan got {other:?}"),
    }
}

#[tokio::test]
async fn in_session_transaction_rollback_discards_writes() {
    // A write inside a rolled-back transaction must leave no trace.
    let proc = CoordinodeProcess::start().await;
    drive_transaction(&proc, "CREATE (n:Rb {k: 1})", false).await;

    let by_id = run_session(&proc, vec![execute(1, "MATCH (n:Rb) RETURN n.k AS k")]).await;
    let events = by_id.get(&1).map(Vec::as_slice).unwrap_or_default();
    match events {
        [Event::CursorOpen(_), rest @ ..] => {
            assert_eq!(row_count(rest), 0, "rolled-back write left no node");
        }
        // No rows at all (open+end) is equally valid.
        [] => {}
        other => panic!("verify scan got {other:?}"),
    }
}

#[tokio::test]
async fn in_session_transaction_aborts_on_a_failed_statement() {
    // First-failure rollback end to end: a valid write then a broken statement
    // inside one transaction. The broken statement errors, the transaction is
    // dead, the commit is rejected, and the earlier write never persists.
    use tokio_stream::wrappers::ReceiverStream;
    let proc = CoordinodeProcess::start().await;
    let channel = tonic::transport::Endpoint::from_shared(proc.endpoint())
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    let mut client = SessionServiceClient::new(channel);
    let (tx, rx) = tokio::sync::mpsc::channel::<ClientFrame>(8);
    let mut inbound = client
        .session(ReceiverStream::new(rx))
        .await
        .expect("open session")
        .into_inner();

    // Await the terminal event (CursorEnd / Committed / Error) of one request.
    async fn await_terminal(
        inbound: &mut tonic::Streaming<coordinode_integration::proto::session::ServerFrame>,
        want: u64,
    ) -> Event {
        loop {
            let f = inbound.message().await.expect("frame").expect("not end");
            if f.request_id == want {
                match f.event {
                    Some(e @ (Event::CursorEnd(_) | Event::Committed(_) | Event::Error(_))) => {
                        return e
                    }
                    _ => continue,
                }
            }
        }
    }

    tx.send(begin(1)).await.expect("begin");
    let txid = loop {
        let f = inbound.message().await.expect("frame").expect("not end");
        if let (1, Some(Event::Begun(b))) = (f.request_id, f.event) {
            break b.txid;
        }
    };

    // Valid write inside the transaction.
    tx.send(execute_tx(2, "CREATE (n:Ab {k: 1})", txid))
        .await
        .expect("write");
    assert!(matches!(
        await_terminal(&mut inbound, 2).await,
        Event::CursorEnd(_)
    ));

    // Broken statement: errors and aborts the transaction.
    tx.send(execute_tx(3, "THIS IS NOT CYPHER", txid))
        .await
        .expect("bad");
    assert!(matches!(
        await_terminal(&mut inbound, 3).await,
        Event::Error(_)
    ));

    // Commit is rejected because the transaction is aborted.
    tx.send(commit_tx(4, txid)).await.expect("commit");
    assert!(
        matches!(await_terminal(&mut inbound, 4).await, Event::Error(_)),
        "commit of an aborted transaction must error"
    );
    drop(tx);
    while inbound.message().await.expect("frame").is_some() {}

    // The earlier write never became durable.
    let by_id = run_session(&proc, vec![execute(1, "MATCH (n:Ab) RETURN n.k AS k")]).await;
    let events = by_id.get(&1).map(Vec::as_slice).unwrap_or_default();
    match events {
        [Event::CursorOpen(_), rest @ ..] => {
            assert_eq!(row_count(rest), 0, "aborted transaction left no node");
        }
        [] => {}
        other => panic!("verify scan got {other:?}"),
    }
}

#[tokio::test]
async fn ordered_transaction_reassembles_out_of_order_statements() {
    // An ORDERED transaction whose statements arrive out of nonce order must
    // still apply them in nonce order, commit cleanly, and persist every write.
    use tokio_stream::wrappers::ReceiverStream;
    let proc = CoordinodeProcess::start().await;
    let channel = tonic::transport::Endpoint::from_shared(proc.endpoint())
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    let mut client = SessionServiceClient::new(channel);
    let (tx, rx) = tokio::sync::mpsc::channel::<ClientFrame>(8);
    let mut inbound = client
        .session(ReceiverStream::new(rx))
        .await
        .expect("open session")
        .into_inner();

    tx.send(begin_ordered(1)).await.expect("begin");
    let txid = loop {
        let f = inbound.message().await.expect("frame").expect("not end");
        if let (1, Some(Event::Begun(b))) = (f.request_id, f.event) {
            break b.txid;
        }
    };

    // Send nonce 2 before nonce 1.
    tx.send(execute_tx_nonce(2, "CREATE (n:Ord {k: 2})", txid, 2))
        .await
        .expect("n2");
    tx.send(execute_tx_nonce(3, "CREATE (n:Ord {k: 1})", txid, 1))
        .await
        .expect("n1");
    tx.send(commit_tx_nonce(4, txid, 2)).await.expect("commit");

    // Commit must succeed (no abort, no drain timeout): reaching past this loop
    // means a Committed for request 4 arrived.
    loop {
        let f = inbound.message().await.expect("frame").expect("not end");
        if f.request_id == 4 {
            match f.event {
                Some(Event::Committed(_)) => break,
                Some(Event::Error(e)) => panic!("ordered commit errored: {}", e.message),
                _ => {}
            }
        }
    }
    drop(tx);
    while inbound.message().await.expect("frame").is_some() {}

    // Both writes persisted.
    let by_id = run_session(&proc, vec![execute(1, "MATCH (n:Ord) RETURN n.k AS k")]).await;
    let events = by_id.get(&1).map(Vec::as_slice).unwrap_or_default();
    match events {
        [Event::CursorOpen(_), rest @ ..] => {
            assert_eq!(row_count(rest), 2, "both ordered writes committed");
        }
        other => panic!("verify scan got {other:?}"),
    }
}

#[tokio::test]
async fn show_transactions_lists_an_open_transaction_with_a_countdown() {
    // Open a transaction on a session, then run SHOW TRANSACTIONS on the SAME
    // session and confirm the live registry surfaces it with an auto-abort
    // countdown. A controlled request stream sequences Begin strictly before
    // the SHOW so the read cannot race the registration.
    use tokio_stream::wrappers::ReceiverStream;

    let proc = CoordinodeProcess::start().await;
    let channel = tonic::transport::Endpoint::from_shared(proc.endpoint())
        .expect("valid endpoint")
        .connect()
        .await
        .expect("connect");
    let mut client = SessionServiceClient::new(channel);

    let (tx, rx) = tokio::sync::mpsc::channel::<ClientFrame>(8);
    let response = client
        .session(ReceiverStream::new(rx))
        .await
        .expect("open session");
    let mut inbound = response.into_inner();

    // Begin and wait for the handle before issuing the SHOW.
    tx.send(begin(1)).await.expect("send begin");
    let txid = loop {
        let frame = inbound.message().await.expect("frame").expect("not end");
        if frame.request_id == 1 {
            match frame.event {
                Some(Event::Begun(b)) => break b.txid,
                other => panic!("expected Begun, got {other:?}"),
            }
        }
    };
    assert_ne!(txid, 0, "transaction handle is non-zero");

    // SHOW TRANSACTIONS on the same session: the open transaction must appear.
    tx.send(execute(2, "SHOW TRANSACTIONS"))
        .await
        .expect("send show");
    let mut columns: Vec<String> = Vec::new();
    let mut rows = Vec::new();
    loop {
        let frame = inbound.message().await.expect("frame").expect("not end");
        if frame.request_id != 2 {
            continue;
        }
        match frame.event {
            Some(Event::CursorOpen(o)) => columns = o.columns,
            Some(Event::Rows(b)) => rows.extend(b.rows),
            Some(Event::CursorEnd(_)) => break,
            Some(Event::Error(e)) => panic!("SHOW TRANSACTIONS errored: {}", e.message),
            _ => {}
        }
    }
    drop(tx);

    assert_eq!(rows.len(), 1, "exactly one open transaction is listed");
    // Columns are addressed by name (row column order is not positional).
    use coordinode_integration::proto::common::property_value::Value as Pv;
    let col = |name: &str| {
        columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("SHOW TRANSACTIONS missing column {name}: {columns:?}"))
    };
    let cells = &rows[0].values;
    assert!(
        matches!(&cells[col("txid")].value, Some(Pv::IntValue(t)) if *t == txid as i64),
        "the listed transaction is the one we opened"
    );
    match &cells[col("auto_abort_in_ms")].value {
        Some(Pv::IntValue(auto_abort)) => assert!(
            *auto_abort > 0,
            "an idle transaction still has time left before auto-abort"
        ),
        other => panic!("auto_abort_in_ms not an int: {other:?}"),
    }
}

#[tokio::test]
async fn session_runs_concurrent_writes_without_deadlocking_the_runtime() {
    // Many auto-commit writes in one stream dispatch concurrently, each driving
    // a Raft commit. If those synchronous commits ran on the async worker
    // threads, a burst would block every worker inside a commit with no worker
    // left to drive the apply loop the commits wait on, deadlocking the runtime.
    // They run on the blocking pool instead, so the burst completes: every write
    // opens a cursor and terminates with CursorEnd.
    let proc = CoordinodeProcess::start().await;

    let writes: Vec<ClientFrame> = (1..=40)
        .map(|i| execute(i, &format!("CREATE (n:Conc {{k: {i}}})")))
        .collect();
    let by_id = run_session(&proc, writes).await;

    assert_eq!(by_id.len(), 40, "every concurrent write is answered");
    for rid in 1..=40u64 {
        match by_id.get(&rid).map(Vec::as_slice) {
            Some([Event::CursorOpen(_), rest @ ..]) => assert!(
                matches!(rest.last(), Some(Event::CursorEnd(_))),
                "write {rid} did not terminate with CursorEnd: {rest:?}"
            ),
            other => panic!("write {rid} did not complete cleanly: {other:?}"),
        }
    }

    // The data is durably committed: a follow-up scan sees all 40 nodes.
    let scan = run_session(&proc, vec![execute(1, "MATCH (n:Conc) RETURN n.k AS k")]).await;
    let events = scan.get(&1).map(Vec::as_slice).unwrap_or_default();
    match events {
        [Event::CursorOpen(_), rest @ ..] => {
            assert_eq!(row_count(rest), 40, "all concurrent writes committed");
        }
        other => panic!("scan got {other:?}"),
    }
}

#[tokio::test]
async fn session_interleaves_a_transaction_with_independent_queries() {
    let proc = CoordinodeProcess::start().await;
    let by_id = run_session(
        &proc,
        vec![
            begin(1),
            execute(2, "RETURN 1 AS a"),
            execute(3, "RETURN 2 AS b"),
            commit(4),
        ],
    )
    .await;

    // Begin → a single Begun carrying a non-zero transaction handle.
    match by_id.get(&1).map(Vec::as_slice) {
        Some([Event::Begun(b)]) => assert_ne!(b.txid, 0),
        other => panic!("request 1 (begin) got {other:?}"),
    }
    // Each non-transactional Execute → its own correlated cursor with its own
    // real column header and one row.
    for (rid, col) in [(2u64, "a"), (3u64, "b")] {
        match by_id.get(&rid).map(Vec::as_slice) {
            Some([Event::CursorOpen(open), rest @ ..]) => {
                assert_eq!(open.columns, vec![col.to_string()]);
                assert_eq!(row_count(rest), 1);
                assert!(matches!(rest.last(), Some(Event::CursorEnd(_))));
            }
            other => panic!("request {rid} (execute) got {other:?}"),
        }
    }
    // Commit → a single Committed, independent of the cursor responses above.
    match by_id.get(&4).map(Vec::as_slice) {
        Some([Event::Committed(_)]) => {}
        other => panic!("request 4 (commit) got {other:?}"),
    }
}

#[tokio::test]
async fn session_correlates_a_burst_of_concurrent_queries() {
    let proc = CoordinodeProcess::start().await;
    let frames: Vec<ClientFrame> = (1..=20)
        .map(|i| execute(i, "UNWIND [1, 2] AS n RETURN n"))
        .collect();
    let by_id = run_session(&proc, frames).await;

    assert_eq!(by_id.len(), 20, "every request id is answered exactly once");
    for rid in 1..=20u64 {
        match by_id.get(&rid).map(Vec::as_slice) {
            Some([Event::CursorOpen(open), rest @ ..]) => {
                assert_eq!(open.columns, vec!["n".to_string()]);
                assert_eq!(row_count(rest), 2, "responses are not crossed");
                assert!(matches!(rest.last(), Some(Event::CursorEnd(_))));
            }
            other => panic!("request {rid} got {other:?}"),
        }
    }
}
