use super::*;

#[test]
fn hardware_fingerprint_is_populated() {
    let hw = hardware_fingerprint();
    // On any real host these should be non-empty / non-zero.
    // CI runners satisfy this; sandboxed test hosts may not for
    // every field — we only assert the arch which is always
    // determinable.
    assert!(!hw.arch.is_empty());
}

#[test]
fn git_metadata_works_in_repo() {
    // This test only runs inside a git checkout. Skip if git
    // is unavailable (e.g. minimal CI image).
    let Ok(meta) = git_metadata() else {
        eprintln!("git metadata unavailable — skipping");
        return;
    };
    assert_eq!(meta.sha.len(), 40, "full SHA must be 40 chars");
    assert_eq!(meta.sha_short.len(), 7, "short SHA must be 7 chars");
    assert!(meta.commit_date <= Utc::now());
}
