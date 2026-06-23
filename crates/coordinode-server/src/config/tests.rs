use super::*;

#[test]
fn full_mode_parses() {
    assert_eq!(ServeMode::parse("full").unwrap(), ServeMode::Full);
}

#[test]
fn compute_mode_rejected_with_ee_message() {
    let err = ServeMode::parse("compute").unwrap_err();
    assert!(err.contains("coordinode-ee"), "expected EE mention: {err}");
}

#[test]
fn storage_mode_rejected_with_ee_message() {
    let err = ServeMode::parse("storage").unwrap_err();
    assert!(err.contains("coordinode-ee"), "expected EE mention: {err}");
}

#[test]
fn unknown_mode_rejected() {
    let err = ServeMode::parse("sharded").unwrap_err();
    assert!(err.contains("unknown"), "expected 'unknown' in: {err}");
}

#[test]
fn default_is_full() {
    assert_eq!(ServeMode::default(), ServeMode::Full);
}
