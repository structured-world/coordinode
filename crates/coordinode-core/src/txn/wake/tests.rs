use super::*;
use std::sync::Arc;
use std::time::Instant;

/// A notification wakes a parked loop, and a loop with nothing to wait for
/// stays parked rather than returning on its own.
#[test]
fn a_notify_wakes_the_parked_loop_and_nothing_else_does() {
    let wake = Arc::new(Wake::default());
    let (tx, rx) = std::sync::mpsc::channel();
    let looped = Arc::clone(&wake);
    let handle = std::thread::spawn(move || {
        looped.bind();
        tx.send(()).expect("bound");
        looped.wait(None);
        tx.send(()).expect("woke");
    });
    rx.recv().expect("the loop bound itself");
    assert!(
        rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "the loop returned without a notification"
    );
    wake.notify();
    rx.recv_timeout(Duration::from_secs(5))
        .expect("the notification woke the loop");
    handle.join().expect("join");
}

/// A notification that lands before the loop parks is not slept through.
#[test]
fn a_notify_before_the_wait_returns_it_at_once() {
    let wake = Wake::default();
    wake.bind();
    wake.notify();
    let started = Instant::now();
    wake.wait(Some(Duration::from_secs(30)));
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// An interrupt wakes a loop even when a notification is already pending.
#[test]
fn an_interrupt_wakes_the_loop_through_a_pending_notification() {
    let wake = Arc::new(Wake::default());
    let (tx, rx) = std::sync::mpsc::channel();
    let looped = Arc::clone(&wake);
    let handle = std::thread::spawn(move || {
        looped.bind();
        tx.send(()).expect("bound");
        // A long sleep of the loop's own choosing.
        std::thread::park_timeout(Duration::from_secs(60));
        tx.send(()).expect("woke");
    });
    rx.recv().expect("the loop bound itself");
    wake.notify();
    wake.interrupt();
    rx.recv_timeout(Duration::from_secs(5))
        .expect("the interrupt woke the loop");
    handle.join().expect("join");
}

/// A deadline bounds the sleep when there is a reason for one.
#[test]
fn a_timeout_bounds_the_wait() {
    let wake = Wake::default();
    wake.bind();
    let started = Instant::now();
    wake.wait(Some(Duration::from_millis(50)));
    assert!(started.elapsed() < Duration::from_secs(5));
}
