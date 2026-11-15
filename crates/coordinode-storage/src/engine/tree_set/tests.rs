use super::*;

/// The set alone is granted only once no operation holds it, and an
/// operation arriving while it is waited for waits behind it.
#[test]
fn the_set_alone_waits_for_holders_and_keeps_new_ones_out() {
    let gate = Arc::new(TreeSetGate::default());
    let held = gate.hold();
    assert!(
        gate.exclusive(Duration::from_millis(20)).is_none(),
        "refused while an operation holds the set"
    );

    let (waiting, waited) = std::sync::mpsc::channel();
    let capture = {
        let gate = Arc::clone(&gate);
        std::thread::spawn(move || {
            waiting.send(()).expect("tell");
            gate.exclusive(Duration::from_secs(10)).map(drop).is_some()
        })
    };
    waited.recv().expect("the capture waits");
    // Give the capture time to register as waiting.
    std::thread::sleep(Duration::from_millis(50));
    let late = {
        let gate = Arc::clone(&gate);
        std::thread::spawn(move || {
            let _hold = gate.hold();
        })
    };
    std::thread::sleep(Duration::from_millis(50));
    assert!(!late.is_finished(), "a new holder waits behind the capture");
    drop(held);
    assert!(
        capture.join().expect("capture"),
        "granted once nobody holds"
    );
    late.join()
        .expect("the new holder goes on after the capture");
}

/// A capture refused past its timeout lets the operations it kept waiting go
/// on.
#[test]
fn a_refused_capture_lets_waiting_holders_go_on() {
    let gate = Arc::new(TreeSetGate::default());
    let held = gate.hold();
    let capture = {
        let gate = Arc::clone(&gate);
        std::thread::spawn(move || gate.exclusive(Duration::from_millis(100)).is_some())
    };
    std::thread::sleep(Duration::from_millis(30));
    let late = {
        let gate = Arc::clone(&gate);
        std::thread::spawn(move || {
            let _hold = gate.hold();
        })
    };
    assert!(
        !capture.join().expect("capture"),
        "refused past its timeout"
    );
    late.join().expect("the waiting holder goes on");
    drop(held);
}
