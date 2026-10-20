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
    Begin, ClientFrame, Commit, Configure, Execute, Rollback, client_frame,
};

/// A transaction ID as a client draws it: a random UUID.
fn draw_txid() -> Vec<u8> {
    uuid::Uuid::new_v4().as_bytes().to_vec()
}

/// Begin a transaction under `txid` with the session's settings. Nothing
/// answers it; the statements sent after it run once it is open, in the
/// order they were sent.
fn begin(request_id: u64, txid: &[u8]) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Begin(Begin {
            txid: txid.to_vec(),
            operation: None,
        })),
    }
}

fn execute(request_id: u64, query: &str) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Execute(Execute {
            query: query.to_string(),
            parameters: Default::default(),
            ..Default::default()
        })),
    }
}

/// An Execute frame bound to the open transaction `txid`.
fn execute_tx(request_id: u64, query: &str, txid: &[u8]) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Execute(Execute {
            query: query.to_string(),
            parameters: Default::default(),
            txid: txid.to_vec(),
            ..Default::default()
        })),
    }
}

/// A Commit frame for a specific transaction.
fn commit_tx(request_id: u64, txid: &[u8]) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Commit(Commit {
            txid: txid.to_vec(),
            operation: None,
            expect: Vec::new(),
        })),
    }
}

/// A transaction ID as SHOW TRANSACTIONS renders it: a hyphenated UUID.
fn uuid_text(txid: &[u8]) -> String {
    uuid::Uuid::from_slice(txid).expect("a UUID").to_string()
}

/// Open a session stream on `proc` that the test feeds frame by frame.
async fn open_stream(
    proc: &CoordinodeProcess,
) -> (
    tokio::sync::mpsc::Sender<ClientFrame>,
    tonic::Streaming<coordinode_integration::proto::session::ServerFrame>,
) {
    use tokio_stream::wrappers::ReceiverStream;
    let channel = tonic::transport::Endpoint::from_shared(proc.endpoint())
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    let mut client = SessionServiceClient::new(channel);
    let (tx, rx) = tokio::sync::mpsc::channel::<ClientFrame>(8);
    let inbound = client
        .session(ReceiverStream::new(rx))
        .await
        .expect("open session")
        .into_inner();
    (tx, inbound)
}

/// The terminal event (CursorEnd / Committed / Error) of request `want`.
async fn await_terminal(
    inbound: &mut tonic::Streaming<coordinode_integration::proto::session::ServerFrame>,
    want: u64,
) -> Event {
    loop {
        let f = inbound.message().await.expect("frame").expect("not end");
        if f.request_id == want {
            match f.event {
                Some(e @ (Event::CursorEnd(_) | Event::Committed(_) | Event::Error(_))) => {
                    return e;
                }
                _ => continue,
            }
        }
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
        // Request id zero is the server speaking on its own: a connection
        // status the client never asked for. It correlates to no request, so
        // it is dropped here rather than counted as an answer to one. A real
        // client reads it for what it is, which is what
        // `session_reports_connection_status` covers.
        if frame.request_id == 0 {
            continue;
        }
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
    let (tx, mut inbound) = open_stream(proc).await;

    // Begin under the client's ID; the write after it waits for it.
    let txid = draw_txid();
    tx.send(begin(1, &txid)).await.expect("send begin");

    // Write inside the transaction; wait for its cursor to finish.
    tx.send(execute_tx(2, write, &txid))
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
        tx.send(commit_tx(3, &txid)).await.expect("send commit");
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
    let proc = CoordinodeProcess::start().await;
    let (tx, mut inbound) = open_stream(&proc).await;

    let txid = draw_txid();
    tx.send(begin(1, &txid)).await.expect("begin");

    // Valid write inside the transaction.
    tx.send(execute_tx(2, "CREATE (n:Ab {k: 1})", &txid))
        .await
        .expect("write");
    assert!(matches!(
        await_terminal(&mut inbound, 2).await,
        Event::CursorEnd(_)
    ));

    // Broken statement: errors and aborts the transaction.
    tx.send(execute_tx(3, "THIS IS NOT CYPHER", &txid))
        .await
        .expect("bad");
    assert!(matches!(
        await_terminal(&mut inbound, 3).await,
        Event::Error(_)
    ));

    // Commit is rejected because the transaction is aborted.
    tx.send(commit_tx(4, &txid)).await.expect("commit");
    match await_terminal(&mut inbound, 4).await {
        Event::Error(e) => assert_eq!(
            e.code,
            tonic::Code::Aborted as u32,
            "the commit is told the transaction was aborted: {}",
            e.message
        ),
        other => panic!("commit of an aborted transaction must error, got {other:?}"),
    }
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
async fn a_pipelined_transaction_runs_in_the_order_it_was_sent() {
    // A transaction's statements and commit sent back to back, without
    // waiting for answers, run in the order the stream delivered them: the
    // second statement sees the first one's write, and the commit lands both.
    let proc = CoordinodeProcess::start().await;
    let (tx, mut inbound) = open_stream(&proc).await;

    let txid = draw_txid();
    tx.send(begin(1, &txid)).await.expect("begin");
    tx.send(execute_tx(2, "CREATE (n:Ord {k: 1})", &txid))
        .await
        .expect("first");
    tx.send(execute_tx(3, "MATCH (n:Ord) SET n.k = n.k * 10", &txid))
        .await
        .expect("second");
    tx.send(commit_tx(4, &txid)).await.expect("commit");

    match await_terminal(&mut inbound, 4).await {
        Event::Committed(_) => {}
        other => panic!("the pipelined commit lands, got {other:?}"),
    }
    drop(tx);
    while inbound.message().await.expect("frame").is_some() {}

    use coordinode_integration::proto::common::property_value::Value as Pv;
    let by_id = run_session(&proc, vec![execute(1, "MATCH (n:Ord) RETURN n.k AS k")]).await;
    let events = by_id.get(&1).map(Vec::as_slice).unwrap_or_default();
    let values: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::Rows(b) => Some(b.rows.iter().map(|r| r.values[0].value.clone())),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(
        values,
        vec![Some(Pv::IntValue(10))],
        "the second statement ran after the first"
    );
}

#[tokio::test]
async fn show_transactions_lists_an_open_transaction_with_a_countdown() {
    // Open a transaction on a session, then run SHOW TRANSACTIONS on the SAME
    // session and confirm the live registry surfaces it with an auto-abort
    // countdown. A statement in the transaction, answered before the SHOW is
    // sent, proves it open, so the read cannot race the registration.
    let proc = CoordinodeProcess::start().await;
    let (tx, mut inbound) = open_stream(&proc).await;

    let txid = draw_txid();
    tx.send(begin(1, &txid)).await.expect("send begin");
    tx.send(execute_tx(2, "RETURN 1 AS one", &txid))
        .await
        .expect("send a statement in it");
    assert!(matches!(
        await_terminal(&mut inbound, 2).await,
        Event::CursorEnd(_)
    ));

    // SHOW TRANSACTIONS on the same session: the open transaction must appear.
    tx.send(execute(3, "SHOW TRANSACTIONS"))
        .await
        .expect("send show");
    let mut columns: Vec<String> = Vec::new();
    let mut rows = Vec::new();
    loop {
        let frame = inbound.message().await.expect("frame").expect("not end");
        if frame.request_id != 3 {
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
        matches!(&cells[col("txid")].value, Some(Pv::StringValue(t)) if *t == uuid_text(&txid)),
        "the listed transaction is the one we opened"
    );
    match &cells[col("expires_in_ms")].value {
        Some(Pv::IntValue(expires)) => assert!(
            *expires > 0,
            "an idle transaction still has time left before it expires"
        ),
        other => panic!("expires_in_ms not an int: {other:?}"),
    }
    assert!(
        matches!(&cells[col("timeout_outcome")].value, Some(Pv::StringValue(o)) if o == "rollback"),
        "the server's default outcome"
    );
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
    let (tx, mut inbound) = open_stream(&proc).await;
    let txid = draw_txid();
    for frame in [
        begin(1, &txid),
        execute(2, "RETURN 1 AS a"),
        execute_tx(3, "CREATE (:Interleaved {k: 1})", &txid),
        execute(4, "RETURN 2 AS b"),
        commit_tx(5, &txid),
    ] {
        tx.send(frame).await.expect("send");
    }
    drop(tx);
    let mut by_id: HashMap<u64, Vec<Event>> = HashMap::new();
    while let Some(frame) = inbound.message().await.expect("server frame") {
        if frame.request_id != 0 {
            by_id
                .entry(frame.request_id)
                .or_default()
                .push(frame.event.expect("frame event"));
        }
    }

    // Each non-transactional Execute → its own correlated cursor with its own
    // real column header and one row.
    for (rid, col) in [(2u64, "a"), (4u64, "b")] {
        match by_id.get(&rid).map(Vec::as_slice) {
            Some([Event::CursorOpen(open), rest @ ..]) => {
                assert_eq!(open.columns, vec![col.to_string()]);
                assert_eq!(row_count(rest), 1);
                assert!(matches!(rest.last(), Some(Event::CursorEnd(_))));
            }
            other => panic!("request {rid} (execute) got {other:?}"),
        }
    }
    // The transaction's statement ran before its commit, which was sent right
    // behind it without waiting for its answer.
    assert!(
        matches!(
            by_id.get(&3).and_then(|events| events.last()),
            Some(Event::CursorEnd(_))
        ),
        "{:?}",
        by_id.get(&3)
    );
    match by_id.get(&5).map(Vec::as_slice) {
        Some([Event::Committed(_)]) => {}
        other => panic!("request 5 (commit) got {other:?}"),
    }
    let scan = run_session(
        &proc,
        vec![execute(1, "MATCH (n:Interleaved) RETURN n.k AS k")],
    )
    .await;
    assert_eq!(
        row_count(scan.get(&1).map(Vec::as_slice).unwrap_or_default()),
        1,
        "the commit included the statement sent before it"
    );
}

/// A transaction outlives the stream that began it: the first stream closes
/// without committing, a second one continues the transaction by its ID,
/// sees its write and commits it.
#[tokio::test]
async fn a_transaction_continues_on_another_stream() {
    let proc = CoordinodeProcess::start().await;

    let (first, mut first_in) = open_stream(&proc).await;
    let txid = draw_txid();
    first.send(begin(1, &txid)).await.expect("begin");
    first
        .send(execute_tx(2, "CREATE (:Reattached {k: 1})", &txid))
        .await
        .expect("write");
    assert!(matches!(
        await_terminal(&mut first_in, 2).await,
        Event::CursorEnd(_)
    ));
    drop(first);
    while first_in.message().await.expect("frame").is_some() {}

    let (second, mut second_in) = open_stream(&proc).await;
    second
        .send(execute_tx(
            1,
            "MATCH (n:Reattached) RETURN count(n) AS c",
            &txid,
        ))
        .await
        .expect("read");
    assert!(matches!(
        await_terminal(&mut second_in, 1).await,
        Event::CursorEnd(_)
    ));
    second.send(commit_tx(2, &txid)).await.expect("commit");
    match await_terminal(&mut second_in, 2).await {
        Event::Committed(_) => {}
        other => panic!("the second stream commits it, got {other:?}"),
    }
    drop(second);
    while second_in.message().await.expect("frame").is_some() {}

    let scan = run_session(
        &proc,
        vec![execute(1, "MATCH (n:Reattached) RETURN n.k AS k")],
    )
    .await;
    assert_eq!(
        row_count(scan.get(&1).map(Vec::as_slice).unwrap_or_default()),
        1
    );
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

/// A Configure is answered with the connection's status, and a change to one
/// setting is confirmed by what is in effect rather than by what was sent.
///
/// This is what a client uses to know whether its writes can go through at
/// all: the status says whether the node can serve them and which node leads.
/// Against a single-node server the answer is always yes, which is the point
/// of checking it here, since a client cannot tell "standalone" from "in a
/// cluster" and should not have to.
#[tokio::test]
async fn session_reports_connection_status_and_confirms_settings() {
    let proc = CoordinodeProcess::start().await;

    let configure = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Configure(Configure {
            // Ask for MAJORITY reads by default on this connection.
            read_concern: Some(coordinode_integration::proto::replication::ReadConcern {
                level: 2,
                after_index: 0,
                at_timestamp: 0,
            }),
            ..Default::default()
        })),
    };
    let by_id = run_session(&proc, vec![configure]).await;

    match by_id.get(&1).map(Vec::as_slice) {
        Some([Event::ConnectionStatus(status)]) => {
            assert!(
                status.writable,
                "a single node serves its own writes; a client must be told so"
            );
            assert!(status.connected);
            assert_eq!(
                status
                    .settings
                    .as_ref()
                    .and_then(|s| s.read_concern.as_ref())
                    .map(|rc| rc.level),
                Some(2),
                "the answer must report the setting now in effect"
            );
        }
        other => panic!("request 1 (configure) got {other:?}"),
    }
}
