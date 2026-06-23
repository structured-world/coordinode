use super::*;

/// Builder defaults: source tracking disabled, empty app fields.
#[test]
fn builder_defaults() {
    let builder = CoordinodeClient::builder("http://localhost:7080");
    // Build into config directly for inspection
    let config = ClientConfig {
        endpoint: "http://localhost:7080".into(),
        debug_source_tracking: builder.debug_source_tracking,
        app_name: builder.app_name.clone(),
        app_version: builder.app_version.clone(),
    };
    assert!(
        !config.debug_source_tracking,
        "tracking disabled by default"
    );
    assert!(config.app_name.is_empty());
    assert!(config.app_version.is_empty());
}

/// Builder properly stores configured values.
#[test]
fn builder_custom() {
    let builder = CoordinodeClient::builder("http://prod:7080")
        .debug_source_tracking(true)
        .app_name("my-svc")
        .app_version("v2.0");

    assert!(builder.debug_source_tracking);
    assert_eq!(builder.app_name, "my-svc");
    assert_eq!(builder.app_version, "v2.0");
    assert_eq!(builder.endpoint, "http://prod:7080");
}

/// source_tracking_enabled reflects the config.
///
/// We test through the config struct directly (no live server needed).
#[test]
fn source_tracking_flag_reflects_config() {
    let config_on = ClientConfig {
        endpoint: "http://localhost:7080".into(),
        debug_source_tracking: true,
        app_name: String::new(),
        app_version: String::new(),
    };
    let config_off = ClientConfig {
        debug_source_tracking: false,
        ..config_on.clone()
    };

    // Build a minimal client without connecting (just check the flag)
    // We can't call connect() without a server, so test via config fields.
    assert!(config_on.debug_source_tracking);
    assert!(!config_off.debug_source_tracking);
}

/// Location is captured from the real call site when tracking is on.
///
/// Validates #[track_caller] semantics: the location should point to
/// THIS test file, not to lib.rs internals.
#[test]
fn track_caller_captures_call_site() {
    // Simulate what the execute methods do:
    // Mark this function itself as the "caller" by calling Location::caller()
    // at the point where execute_cypher would call it.
    let tracking_enabled = true;

    #[track_caller]
    fn simulate_execute(enabled: bool) -> Option<&'static Location<'static>> {
        // Must call Location::caller() directly here — passing it as a
        // function pointer to bool::then() loses track_caller propagation
        // because the call goes through FnOnce in function.rs.
        if enabled {
            Some(Location::caller())
        } else {
            None
        }
    }

    let loc = simulate_execute(tracking_enabled);
    assert!(loc.is_some());
    let loc = loc.unwrap();

    // The location must point to this test file (not to the simulate_execute body).
    assert!(
        loc.file().contains("tests.rs"),
        "expected call site in this test file, got: {}",
        loc.file()
    );
    assert!(loc.line() > 0);
}

/// When tracking is disabled, no location is captured.
#[test]
fn track_caller_disabled() {
    #[track_caller]
    fn simulate_execute(enabled: bool) -> Option<&'static Location<'static>> {
        if enabled {
            Some(Location::caller())
        } else {
            None
        }
    }

    let loc = simulate_execute(false);
    assert!(loc.is_none());
}

/// invalid endpoint URI returns ClientError::InvalidEndpoint, not a panic.
#[tokio::test]
async fn connect_invalid_endpoint_returns_error() {
    let result = CoordinodeClient::connect("not a valid uri !!!").await;
    assert!(
        matches!(result, Err(ClientError::InvalidEndpoint(_))),
        "expected InvalidEndpoint error"
    );
}

// ── CausalToken tests ─────────────────────────────────────────────────────

/// CausalToken wraps the log index and exposes it via as_u64.
#[test]
fn causal_token_as_u64() {
    let token = CausalToken(42);
    assert_eq!(token.as_u64(), 42);
}

/// CausalToken::from(u64) round-trips through as_u64.
#[test]
fn causal_token_from_u64() {
    let token = CausalToken::from(100);
    assert_eq!(token.as_u64(), 100);
}

/// Zero token (standalone mode) is a valid value.
#[test]
fn causal_token_zero() {
    let token = CausalToken::from(0);
    assert_eq!(token.as_u64(), 0);
}

/// CausalToken ordering: higher index = happens-after.
#[test]
fn causal_token_ordering() {
    let earlier = CausalToken(10);
    let later = CausalToken(20);
    assert!(later > earlier);
    assert!(earlier < later);
    assert_eq!(earlier, CausalToken(10));
}

/// CausalToken is Copy: can be used multiple times without moving.
#[test]
fn causal_token_is_copy() {
    let token = CausalToken(7);
    // If CausalToken were not Copy, the second line would fail to compile.
    let _a = token;
    let _b = token;
    assert_eq!(_a, _b);
}
