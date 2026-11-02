use super::*;

#[test]
fn default_is_current() {
    assert_eq!(ReadConsistencyMode::default(), ReadConsistencyMode::Current);
}

#[test]
fn parse_case_insensitive() {
    for (input, expected) in [
        ("current", ReadConsistencyMode::Current),
        ("CURRENT", ReadConsistencyMode::Current),
        ("Snapshot", ReadConsistencyMode::Snapshot),
        ("EXACT", ReadConsistencyMode::Exact),
        ("exact", ReadConsistencyMode::Exact),
    ] {
        assert_eq!(ReadConsistencyMode::from_str_opt(input), Some(expected));
    }
}

#[test]
fn parse_rejects_unknown() {
    assert_eq!(ReadConsistencyMode::from_str_opt(""), None);
    assert_eq!(ReadConsistencyMode::from_str_opt("strict"), None);
    assert_eq!(ReadConsistencyMode::from_str_opt("none"), None);
}

#[test]
fn roundtrip_as_str_parse() {
    for mode in [
        ReadConsistencyMode::Current,
        ReadConsistencyMode::Snapshot,
        ReadConsistencyMode::Exact,
    ] {
        assert_eq!(ReadConsistencyMode::from_str_opt(mode.as_str()), Some(mode));
    }
}

#[test]
fn requires_aligned_view_semantics() {
    assert!(!ReadConsistencyMode::Current.requires_aligned_view());
    assert!(ReadConsistencyMode::Snapshot.requires_aligned_view());
    assert!(ReadConsistencyMode::Exact.requires_aligned_view());
}

#[test]
fn display_matches_as_str() {
    assert_eq!(format!("{}", ReadConsistencyMode::Snapshot), "snapshot");
    assert_eq!(format!("{}", ReadConsistencyMode::Current), "current");
    assert_eq!(format!("{}", ReadConsistencyMode::Exact), "exact");
}
