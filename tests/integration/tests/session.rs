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
use coordinode_integration::proto::session::{client_frame, Begin, ClientFrame, Commit, Execute};

fn begin(request_id: u64) -> ClientFrame {
    ClientFrame {
        request_id,
        op: Some(client_frame::Op::Begin(Begin {
            ordering: 0,
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
