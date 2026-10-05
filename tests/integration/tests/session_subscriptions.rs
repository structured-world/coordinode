//! Change-stream subscriptions carried by the bidirectional Session.
//!
//! Against a real `coordinode` binary: a subscription opened on the session is
//! delivered in batches within the credit its client grants, a subscription
//! without credit sends nothing and leaves the session's queries alone,
//! acknowledging and cancelling ride the same stream, and the registration
//! outlives the session.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use coordinode_integration::harness::CoordinodeProcess;
use coordinode_integration::proto::replication::{
    AcknowledgeSubscriptionRequest, CancelSubscriptionRequest, ConsumerRetention, StrictRetention,
    SubscribeRequest, consumer_retention,
};
use coordinode_integration::proto::session::server_frame::Event;
use coordinode_integration::proto::session::session_service_client::SessionServiceClient;
use coordinode_integration::proto::session::{
    Cancel, ClientFrame, Credit, Execute, ServerFrame, Subscribe, client_frame,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Streaming;
use tonic_types::StatusExt;

/// One open session: frames go out on `tx`, come back on `inbound`.
struct Session {
    tx: mpsc::Sender<ClientFrame>,
    inbound: Streaming<ServerFrame>,
}

impl Session {
    async fn open(proc: &CoordinodeProcess) -> Self {
        let channel = tonic::transport::Endpoint::from_shared(proc.endpoint())
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let (tx, rx) = mpsc::channel::<ClientFrame>(32);
        let inbound = SessionServiceClient::new(channel)
            .session(ReceiverStream::new(rx))
            .await
            .expect("open session")
            .into_inner();
        Self { tx, inbound }
    }

    async fn send(&self, request_id: u64, op: client_frame::Op) {
        self.tx
            .send(ClientFrame {
                request_id,
                op: Some(op),
            })
            .await
            .expect("send frame");
    }

    /// The next frame, or `None` when none arrives within `within`.
    async fn next(&mut self, within: Duration) -> Option<ServerFrame> {
        tokio::time::timeout(within, self.inbound.message())
            .await
            .ok()
            .map(|frame| frame.expect("frame").expect("session open"))
    }

    /// Frames until one for `request_id` arrives; the others are handed to
    /// `other`.
    async fn answer(&mut self, request_id: u64, mut other: impl FnMut(ServerFrame)) -> Event {
        loop {
            let frame = self
                .next(Duration::from_secs(10))
                .await
                .expect("an answer within 10 s");
            if frame.request_id == request_id {
                return frame.event.expect("an event");
            }
            other(frame);
        }
    }

    /// Run `query` to its end, collecting subscription batches that arrive
    /// meanwhile into `batches`.
    async fn run(&mut self, request_id: u64, query: &str, batches: &mut Vec<ServerFrame>) {
        self.send(
            request_id,
            client_frame::Op::Execute(Execute {
                query: query.to_string(),
                parameters: Default::default(),
                txid: 0,
                nonce: 0,
            }),
        )
        .await;
        loop {
            let frame = self
                .next(Duration::from_secs(10))
                .await
                .expect("the query answers within 10 s");
            if frame.request_id != request_id {
                batches.push(frame);
                continue;
            }
            match frame.event {
                Some(Event::CursorEnd(_)) => return,
                Some(Event::Error(e)) => panic!("{query}: {}", e.message),
                _ => {}
            }
        }
    }
}

fn subscribe(consumer_id: &str, credit: u32) -> client_frame::Op {
    client_frame::Op::Subscribe(Subscribe {
        request: Some(SubscribeRequest {
            resume_token: None,
            filters: None,
            consumer_id: consumer_id.to_string(),
            incarnation: 0,
            retention: Some(ConsumerRetention {
                policy: Some(consumer_retention::Policy::Strict(StrictRetention {})),
            }),
        }),
        credit,
    })
}

/// Events in the change-event batches among `frames` for `request_id`.
fn events_for(frames: &[ServerFrame], request_id: u64) -> usize {
    frames
        .iter()
        .filter(|f| f.request_id == request_id)
        .map(|f| match &f.event {
            Some(Event::ChangeEvents(batch)) => batch.events.len(),
            _ => 0,
        })
        .sum()
}

/// Whether the last change-event batch for `request_id` among `frames` said
/// applied entries were still waiting past it.
fn last_more(frames: &[ServerFrame], request_id: u64) -> bool {
    frames
        .iter()
        .rev()
        .filter(|f| f.request_id == request_id)
        .find_map(|f| match &f.event {
            Some(Event::ChangeEvents(batch)) => Some(batch.more),
            _ => None,
        })
        .expect("a batch")
}

/// A subscription is sent exactly as many events as its client granted, in
/// batches; more credit releases more. A batch cut short by the credit says
/// more is waiting, and the one that reaches everything applied says not. A
/// subscription with no credit left sends nothing and does not hold up the
/// session's queries.
#[tokio::test(flavor = "multi_thread")]
async fn a_subscription_is_sent_what_its_credit_allows() {
    let proc = CoordinodeProcess::start().await;
    let mut s = Session::open(&proc).await;
    let mut frames = Vec::new();
    for i in 0..10 {
        s.run(100 + i, &format!("CREATE (:Item {{n: {i}}})"), &mut frames)
            .await;
    }

    s.send(1, subscribe("credited", 3)).await;
    match s.answer(1, |f| frames.push(f)).await {
        Event::Subscribed(sub) => assert_eq!(sub.incarnation, 1),
        other => panic!("expected Subscribed, got {other:?}"),
    }
    // Everything the credit allows arrives; then nothing more.
    while let Some(frame) = s.next(Duration::from_millis(500)).await {
        frames.push(frame);
    }
    assert_eq!(events_for(&frames, 1), 3, "the initial credit");
    assert!(last_more(&frames, 1), "the log holds more than 3 entries");

    // Out of credit: a query on the same session still answers at once.
    s.run(200, "RETURN 1 AS one", &mut frames).await;

    s.send(
        2,
        client_frame::Op::Credit(Credit {
            target_request_id: 1,
            events: 4,
        }),
    )
    .await;
    while let Some(frame) = s.next(Duration::from_millis(500)).await {
        frames.push(frame);
    }
    assert_eq!(
        events_for(&frames, 1),
        7,
        "the initial and the granted credit"
    );

    // Enough credit to drain the log: the last batch reaches the end.
    s.send(
        3,
        client_frame::Op::Credit(Credit {
            target_request_id: 1,
            events: 10_000,
        }),
    )
    .await;
    while let Some(frame) = s.next(Duration::from_millis(500)).await {
        frames.push(frame);
    }
    assert!(events_for(&frames, 1) > 7, "the rest of the log");
    assert!(!last_more(&frames, 1), "drained to what was applied");
}

/// Acknowledging and cancelling ride the session: an acknowledgement is
/// answered, and cancelling the registration ends the open subscription with
/// CONSUMER_TERMINATED carried in the canonical status.
#[tokio::test(flavor = "multi_thread")]
async fn acknowledging_and_cancelling_ride_the_session() {
    let proc = CoordinodeProcess::start().await;
    let mut s = Session::open(&proc).await;
    let mut frames = Vec::new();
    s.run(100, "CREATE (:Item {n: 1})", &mut frames).await;

    s.send(1, subscribe("acker", 1_000)).await;
    let incarnation = match s.answer(1, |f| frames.push(f)).await {
        Event::Subscribed(sub) => sub.incarnation,
        other => panic!("expected Subscribed, got {other:?}"),
    };
    let position = loop {
        let frame = s.next(Duration::from_secs(10)).await.expect("events");
        if let (1, Some(Event::ChangeEvents(batch))) = (frame.request_id, &frame.event) {
            break batch
                .events
                .last()
                .and_then(|e| e.position)
                .expect("a position");
        }
    };

    s.send(
        2,
        client_frame::Op::Acknowledge(AcknowledgeSubscriptionRequest {
            consumer_id: "acker".to_string(),
            incarnation,
            position: Some(position),
        }),
    )
    .await;
    assert!(matches!(s.answer(2, |_| {}).await, Event::Acknowledged(_)));

    s.send(
        3,
        client_frame::Op::CancelSubscription(CancelSubscriptionRequest {
            consumer_id: "acker".to_string(),
            incarnation,
        }),
    )
    .await;
    assert!(matches!(
        s.answer(3, |_| {}).await,
        Event::SubscriptionCancelled(_)
    ));
    match s.answer(1, |_| {}).await {
        Event::Error(e) => {
            let status = e.status.expect("the canonical status");
            let status = tonic::Status::with_details(
                tonic::Code::from_i32(status.code),
                status.message.clone(),
                prost::Message::encode_to_vec(&status).into(),
            );
            let info = status.get_details_error_info().expect("ErrorInfo");
            assert_eq!(info.reason, "CONSUMER_TERMINATED");
        }
        other => panic!("expected the subscription to end, got {other:?}"),
    }
}

/// The driver's subscription drains a backlog larger than its window by
/// granting credit as batches are taken, and a subscription resumed after an
/// acknowledgement starts right past it.
#[tokio::test(flavor = "multi_thread")]
async fn the_driver_drains_and_resumes_after_its_acknowledgement() {
    use coordinode_client::{CoordinodeClient, Retention, SubscribeOptions};

    let proc = CoordinodeProcess::start().await;
    let mut client = CoordinodeClient::connect(proc.endpoint())
        .await
        .expect("connect");
    for i in 0..20 {
        client
            .execute_cypher(format!("CREATE (:Item {{n: {i}}})"))
            .await
            .expect("write");
    }
    let node_inserts = |events: &[coordinode_client::ChangeEvent]| {
        events
            .iter()
            .flat_map(|e| &e.ops)
            .filter(|op| {
                op.kind == coordinode_client::ChangeKind::Insert
                    && op.key.len() == 16
                    && op.key.starts_with(b"node:")
            })
            .count()
    };

    let mut sub = client
        .subscribe(SubscribeOptions::register("driver", Retention::Strict).window(4))
        .await
        .expect("subscribe");
    let incarnation = sub.incarnation();
    let mut seen = Vec::new();
    while node_inserts(&seen) < 20 {
        let batch = tokio::time::timeout(Duration::from_secs(10), sub.next())
            .await
            .expect("the backlog drains")
            .expect("a batch")
            .expect("session open");
        assert!(batch.len() <= 4, "a batch within the window");
        seen.extend(batch);
    }
    let last = seen.last().expect("events").position;
    sub.acknowledge(last).await.expect("acknowledge");
    drop(sub);

    client
        .execute_cypher("CREATE (:Item {n: 20})")
        .await
        .expect("write after");
    let mut resumed = client
        .subscribe(SubscribeOptions::resume("driver", incarnation))
        .await
        .expect("resume");
    let batch = tokio::time::timeout(Duration::from_secs(10), resumed.next())
        .await
        .expect("the new write arrives")
        .expect("a batch")
        .expect("session open");
    assert!(
        batch
            .iter()
            .all(|e| e.log_index > seen.last().expect("seen").log_index),
        "resumed past the acknowledged position"
    );
    resumed.cancel_registration().await.expect("cancel");
}

/// Cancelling a subscription's request ends only its delivery: the
/// registration stays, and a later session resumes its incarnation.
#[tokio::test(flavor = "multi_thread")]
async fn the_registration_outlives_the_subscription() {
    let proc = CoordinodeProcess::start().await;
    let mut s = Session::open(&proc).await;
    s.send(1, subscribe("survivor", 0)).await;
    let incarnation = match s.answer(1, |_| {}).await {
        Event::Subscribed(sub) => sub.incarnation,
        other => panic!("expected Subscribed, got {other:?}"),
    };
    s.send(
        2,
        client_frame::Op::Cancel(Cancel {
            target_request_id: 1,
        }),
    )
    .await;
    drop(s);

    let mut again = Session::open(&proc).await;
    let mut resume = subscribe("survivor", 1);
    if let client_frame::Op::Subscribe(sub) = &mut resume {
        if let Some(request) = sub.request.as_mut() {
            request.incarnation = incarnation;
        }
    }
    again.send(1, resume).await;
    match again.answer(1, |_| {}).await {
        Event::Subscribed(sub) => assert_eq!(sub.incarnation, incarnation),
        other => panic!("expected the incarnation resumed, got {other:?}"),
    }
}
