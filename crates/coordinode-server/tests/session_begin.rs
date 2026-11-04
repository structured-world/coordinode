//! A transaction's start waiting for its causal bound is a command like any
//! other: it holds no thread while it waits, on the member that leads as on
//! one that passes it to the leader, and a Cancel naming it, or its client
//! going away, ends it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// Each test binary uses part of the shared members.
#[allow(dead_code)]
#[path = "support/members.rs"]
mod members;

use std::time::Duration;

use coordinode_server::proto::replication::{ReadConcern, ReadConcernLevel};
use coordinode_server::proto::session::{
    Begin, Cancel, ClientFrame, Commit, Configure, Execute, ServerFrame, client_frame,
    server_frame::Event, session_service_client::SessionServiceClient,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use members::leader_and_follower;

/// One session stream of a client.
struct Session {
    send: mpsc::Sender<ClientFrame>,
    frames: tonic::Streaming<ServerFrame>,
}

impl Session {
    async fn open(port: u16) -> Self {
        let mut client = SessionServiceClient::connect(format!("http://127.0.0.1:{port}"))
            .await
            .unwrap();
        let (send, outgoing) = mpsc::channel(16);
        let frames = client
            .session(ReceiverStream::new(outgoing))
            .await
            .unwrap()
            .into_inner();
        Self { send, frames }
    }

    async fn send(&self, request_id: u64, op: client_frame::Op) {
        self.send
            .send(ClientFrame {
                request_id,
                op: Some(op),
            })
            .await
            .unwrap();
    }

    /// The next event answering `request_id`, within `within`.
    async fn answer(&mut self, request_id: u64, within: Duration) -> Option<Event> {
        tokio::time::timeout(within, async {
            loop {
                let frame = self.frames.message().await.unwrap().expect("a frame");
                if frame.request_id == request_id {
                    return frame.event.expect("an event");
                }
            }
        })
        .await
        .ok()
    }
}

/// A bound far past any commit the group makes here, with a minute to wait.
fn unreachable_bound() -> client_frame::Op {
    client_frame::Op::Configure(Configure {
        read_concern: Some(ReadConcern {
            level: ReadConcernLevel::Majority as i32,
            after_ts: u64::MAX / 2,
            wait_ms: 60_000,
            ..ReadConcern::default()
        }),
        ..Configure::default()
    })
}

fn statement(query: &str, txid: Vec<u8>) -> client_frame::Op {
    client_frame::Op::Execute(Execute {
        query: query.to_owned(),
        txid,
        ..Execute::default()
    })
}

/// Run `scenario` on a group of two whose members share one blocking thread:
/// a command that held it while waiting would hold back every other.
fn on_one_blocking_thread<F, Fut>(scenario: F)
where
    F: FnOnce(u16, u16) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.block_on(async {
            let ((_leader, leader_port), (_follower, follower_port)) = leader_and_follower().await;
            tokio::time::timeout(
                Duration::from_secs(120),
                scenario(leader_port, follower_port),
            )
            .await
            .expect("TIMED OUT");
        });
    }));
    // A command still holding the blocking thread would hold the runtime's
    // end for as long as it waits.
    runtime.shutdown_timeout(Duration::from_secs(5));
    if let Err(failed) = ran {
        std::panic::resume_unwind(failed);
    }
}

/// A start that waits holds no thread, and a Cancel naming it ends it: a
/// begin whose transaction opens with its commit, on the leader and on a
/// follower that passes it to the leader.
fn a_waiting_start_holds_no_thread_and_ends_with_its_cancel(on_follower: bool) {
    on_one_blocking_thread(|leader, follower| async move {
        let port = if on_follower { follower } else { leader };
        let mut waiting = Session::open(port).await;
        let mut other = Session::open(port).await;
        let txid = vec![0x42; 16];

        waiting.send(1, unreachable_bound()).await;
        waiting
            .answer(1, Duration::from_secs(10))
            .await
            .expect("configured");
        waiting
            .send(
                2,
                client_frame::Op::Begin(Begin {
                    txid: txid.clone(),
                    ..Begin::default()
                }),
            )
            .await;
        waiting
            .send(
                3,
                client_frame::Op::Commit(Commit {
                    txid: txid.clone(),
                    ..Commit::default()
                }),
            )
            .await;
        // Let the start begin its wait before the other session needs the
        // one blocking thread.
        tokio::time::sleep(Duration::from_millis(500)).await;

        other
            .send(1, statement("RETURN 1 AS one", Vec::new()))
            .await;
        let answered = other.answer(1, Duration::from_secs(20)).await;
        assert!(
            matches!(answered, Some(Event::CursorOpen(_))),
            "another session's statement is held behind the waiting start: {answered:?}"
        );

        waiting
            .send(
                4,
                client_frame::Op::Cancel(Cancel {
                    target_request_id: 2,
                }),
            )
            .await;
        let ended = waiting.answer(2, Duration::from_secs(20)).await;
        match ended {
            Some(Event::Error(error)) => assert_eq!(
                error.code,
                tonic::Code::Cancelled as u32,
                "the cancel ends the start: {error:?}"
            ),
            other => panic!("the cancel ends the start, got {other:?}"),
        }
    });
}

#[test]
fn a_waiting_start_on_the_leader_ends_with_its_cancel() {
    a_waiting_start_holds_no_thread_and_ends_with_its_cancel(false);
}

#[test]
fn a_waiting_start_on_a_follower_holds_no_thread_and_ends_with_its_cancel() {
    a_waiting_start_holds_no_thread_and_ends_with_its_cancel(true);
}

/// A transaction's first statement carries its start: waiting for the
/// start's bound it holds no thread, on a follower as on the leader, and a
/// Cancel naming the statement ends it.
fn a_first_statement_waiting_for_its_start_holds_no_thread(on_follower: bool) {
    on_one_blocking_thread(|leader, follower| async move {
        let port = if on_follower { follower } else { leader };
        let mut waiting = Session::open(port).await;
        let mut other = Session::open(port).await;
        let txid = vec![0x43; 16];

        waiting.send(1, unreachable_bound()).await;
        waiting
            .answer(1, Duration::from_secs(10))
            .await
            .expect("configured");
        waiting
            .send(
                2,
                client_frame::Op::Begin(Begin {
                    txid: txid.clone(),
                    ..Begin::default()
                }),
            )
            .await;
        waiting.send(3, statement("RETURN 1 AS one", txid)).await;
        tokio::time::sleep(Duration::from_millis(500)).await;

        other
            .send(1, statement("RETURN 1 AS one", Vec::new()))
            .await;
        let answered = other.answer(1, Duration::from_secs(20)).await;
        assert!(
            matches!(answered, Some(Event::CursorOpen(_))),
            "another session's statement is held behind the waiting start: {answered:?}"
        );

        waiting
            .send(
                4,
                client_frame::Op::Cancel(Cancel {
                    target_request_id: 3,
                }),
            )
            .await;
        let ended = waiting.answer(3, Duration::from_secs(20)).await;
        match ended {
            Some(Event::Error(error)) => assert_eq!(
                error.code,
                tonic::Code::Cancelled as u32,
                "the cancel ends the statement: {error:?}"
            ),
            other => panic!("the cancel ends the statement, got {other:?}"),
        }
    });
}

#[test]
fn a_first_statement_on_the_leader_waits_for_its_start_without_a_thread() {
    a_first_statement_waiting_for_its_start_holds_no_thread(false);
}

#[test]
fn a_first_statement_on_a_follower_waits_for_its_start_without_a_thread() {
    a_first_statement_waiting_for_its_start_holds_no_thread(true);
}
