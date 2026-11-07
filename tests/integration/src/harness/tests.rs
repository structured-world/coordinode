use super::*;

/// A standalone server held `delay_ms` before it opens its store (a test
/// fault of the server), started without waiting for it.
fn held_start(delay_ms: u64) -> CoordinodeProcess {
    let data_dir = tempfile::TempDir::new().expect("tempdir");
    let (port, ops_port, rest_port) = (free_port(), free_port(), free_port());
    let mut launch = standalone_launch(
        port,
        ops_port,
        rest_port,
        data_dir.path().to_path_buf(),
        None,
    );
    launch.set_env("COORDINODE_TEST_STARTUP_DELAY_MS", &delay_ms.to_string());
    CoordinodeProcess {
        child: launch.spawn(),
        port,
        ops_port,
        rest_port,
        data_dir: Some(data_dir),
        config: None,
        identity: None,
        deployment: None,
        member_settings: String::new(),
        env: Vec::new(),
        launch,
    }
}

/// Wait until `proc` has bound its ops port: it accepts a connection.
async fn await_bound(proc: &CoordinodeProcess) {
    let addr = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, proc.ops_port));
    let deadline = Instant::now() + Duration::from_secs(60);
    while std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_err() {
        assert!(
            Instant::now() < deadline,
            "the server never bound its ops port"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// A start slower than its wait: the first attempt's timeout is reported
/// with where it stood (ports bound, store not open), the one diagnostic
/// start of the same command is ready later, and its traced phases show where
/// the time went. The diagnostic success does not replace the first fact.
#[tokio::test]
async fn a_slow_start_is_reported_with_its_diagnostic_start() {
    let mut proc = held_start(6_000);
    await_bound(&proc).await;
    let Err(StartFailure::NotReady(first)) = proc.wait_until_ready(Duration::from_secs(1)).await
    else {
        panic!("a start held 6 s after binding is not ready 1 s after binding");
    };
    assert_eq!(
        first.last,
        ReadyProbe::NoAnswer,
        "held after binding: the ports answer nothing yet"
    );
    let report = proc.diagnose(first, Duration::from_secs(30)).await;
    let DiagnosticOutcome::Ready(after) = report.outcome else {
        panic!("the diagnostic start is ready within 30 s: {report}");
    };
    assert!(
        after >= Duration::from_secs(6),
        "ready only after the hold: {after:?}"
    );
    for phase in [
        "listeners bound",
        "store opening",
        "consensus open",
        "serving",
    ] {
        assert!(
            report.output.contains(phase),
            "phase {phase:?} traced: {report}"
        );
    }
    let text = report.to_string();
    assert!(
        text.contains("was not ready within") && text.contains("ready after"),
        "both attempts reported: {text}"
    );
    assert!(
        has_exited(&mut proc.child),
        "the diagnostic start is stopped"
    );
}

/// A start that hangs: the diagnostic start hangs where the first did, the
/// report says so with the last phase reached, and both processes are gone.
#[tokio::test]
async fn a_hung_start_is_reported_with_where_it_hangs() {
    let mut proc = held_start(600_000);
    let first_pid = proc.pid();
    await_bound(&proc).await;
    let Err(StartFailure::NotReady(first)) = proc.wait_until_ready(Duration::from_secs(1)).await
    else {
        panic!("a hung start is not ready within 1 s");
    };
    assert_eq!(first.last, ReadyProbe::NoAnswer);
    // Long enough for the diagnostic start to bind, far shorter than the hold.
    let report = proc.diagnose(first, Duration::from_secs(10)).await;
    let DiagnosticOutcome::NotReady(again) = report.outcome else {
        panic!("the diagnostic start hangs too: {report}");
    };
    assert_eq!(again.last, ReadyProbe::NoAnswer);
    assert_ne!(again.pid, first_pid, "a new process ran the diagnosis");
    assert!(
        report.output.contains("listeners bound") && !report.output.contains("store opening"),
        "the trace stops before the store: {report}"
    );
    assert!(
        report.to_string().contains("not ready within"),
        "the report says the diagnostic start hung too: {report}"
    );
    assert!(
        has_exited(&mut proc.child),
        "the diagnostic start is stopped"
    );
}

/// A server that exits during startup is noticed and replaced by one on a
/// fresh port, instead of the test waiting out the start timeout or talking
/// to whatever answers on the old port later.
#[tokio::test]
async fn a_server_that_exits_during_startup_is_restarted_on_another_port() {
    let data_dir = tempfile::TempDir::new().expect("tempdir");
    let attempts = std::sync::atomic::AtomicU32::new(0);
    let first_port = std::sync::atomic::AtomicU32::new(0);

    let proc =
        CoordinodeProcess::spawn_on_free_port(data_dir, |port, ops_port, rest_port, data| {
            if attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
                first_port.store(u32::from(port), std::sync::atomic::Ordering::Relaxed);
                // A start that fails before binding: an unknown mode.
                let failing = Launch::new(
                    ["serve", "--mode", "no-such-mode", "--ops-addr", "[::1]:0"]
                        .map(Into::into)
                        .to_vec(),
                );
                return (failing, None);
            }
            (
                standalone_launch(port, ops_port, rest_port, data, None),
                None,
            )
        })
        .await;

    assert_eq!(attempts.load(std::sync::atomic::Ordering::Relaxed), 2);
    assert_ne!(
        u32::from(proc.port),
        first_port.load(std::sync::atomic::Ordering::Relaxed),
        "the retry must move to a fresh port"
    );

    // The process the harness returned is the one serving its port.
    let mut client = proc.cypher_client().await;
    client
        .execute_cypher(crate::proto::query::ExecuteCypherRequest {
            query: "RETURN 1".to_string(),
            parameters: Default::default(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
            ..Default::default()
        })
        .await
        .expect("the returned server answers");
}
