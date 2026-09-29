use super::*;

/// A server that exits during startup is noticed and replaced by one on a
/// fresh port, instead of the test waiting out the start timeout or talking
/// to whatever answers on the old port later.
#[tokio::test]
async fn a_server_that_exits_during_startup_is_restarted_on_another_port() {
    let data_dir = tempfile::TempDir::new().expect("tempdir");
    let attempts = std::sync::atomic::AtomicU32::new(0);
    let first_port = std::sync::atomic::AtomicU32::new(0);

    let proc = CoordinodeProcess::spawn_on_free_port(data_dir, |port, ops_port, data| {
        if attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            first_port.store(u32::from(port), std::sync::atomic::Ordering::Relaxed);
            // A start that fails before binding: an unknown mode.
            return Command::new(binary_path())
                .args(["serve", "--mode", "no-such-mode", "--ops-addr", "[::1]:0"])
                .spawn()
                .expect("spawn a failing start");
        }
        spawn_binary(port, ops_port, data)
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
            transaction_id: 0,
        })
        .await
        .expect("the returned server answers");
}
