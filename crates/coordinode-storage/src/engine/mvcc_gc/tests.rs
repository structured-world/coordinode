use super::*;

/// `Verdict` carries no `PartialEq`; compare by variant.
fn keeps(verdict: &Verdict) -> bool {
    matches!(verdict, Verdict::Keep)
}

fn removes(verdict: &Verdict) -> bool {
    matches!(verdict, Verdict::Remove)
}

/// Factory produces filters with the registered name.
#[test]
fn factory_name() {
    let watermark = Arc::new(AtomicU64::new(0));
    let factory = SeqnoRetentionFilterFactory::new(watermark);
    assert_eq!(factory.name(), "coordinode.seqno_retention");
}

/// Every version above the watermark is kept, whatever the key's history.
#[test]
fn keeps_every_version_inside_the_window() {
    let mut filter = SeqnoRetentionFilter::new(100);
    assert!(keeps(&filter.decide(b"k", 300)));
    assert!(keeps(&filter.decide(b"k", 200)));
    assert!(keeps(&filter.decide(b"k", 101)));
}

/// Expired versions shadowed by a kept (live) version are removed: reads
/// that reach this output are all above its install seqno and see the live
/// version; older history is served by the retained earlier tree versions.
#[test]
fn removes_expired_versions_under_a_live_one() {
    let mut filter = SeqnoRetentionFilter::new(100);
    assert!(keeps(&filter.decide(b"k", 300)), "live");
    assert!(removes(&filter.decide(b"k", 90)));
    assert!(removes(&filter.decide(b"k", 50)));
}

/// A version exactly at the watermark is expired.
#[test]
fn version_at_the_watermark_is_expired() {
    let mut filter = SeqnoRetentionFilter::new(100);
    assert!(keeps(&filter.decide(b"k", 101)));
    assert!(removes(&filter.decide(b"k", 100)));
}

/// A key not written since the window opened keeps its newest version
/// only (no data loss for cold keys).
#[test]
fn cold_key_keeps_its_newest_version() {
    let mut filter = SeqnoRetentionFilter::new(100);
    assert!(keeps(&filter.decide(b"cold", 40)));
    assert!(removes(&filter.decide(b"cold", 30)));
}

/// Per-key state resets at every key boundary: the second key's expired
/// version is its own newest, not "shadowed" by the first key's.
#[test]
fn state_resets_per_key() {
    let mut filter = SeqnoRetentionFilter::new(100);
    assert!(keeps(&filter.decide(b"a", 50)));
    assert!(removes(&filter.decide(b"a", 40)));
    assert!(keeps(&filter.decide(b"b", 50)));
    assert!(removes(&filter.decide(b"b", 40)));
    assert!(keeps(&filter.decide(b"c", 500)));
    assert!(removes(&filter.decide(b"c", 50)));
}

/// Watermark 0 (engine still opening) keeps everything: no seqno is `<= 0`
/// except a zeroed bottommost one, which is then the newest kept.
#[test]
fn zero_watermark_keeps_everything() {
    let mut filter = SeqnoRetentionFilter::new(0);
    assert!(keeps(&filter.decide(b"k", 3)));
    assert!(keeps(&filter.decide(b"k", 2)));
    assert!(keeps(&filter.decide(b"k", 1)));
}
