use super::*;

#[test]
fn default_is_majority() {
    let wc = WriteConcern::default();
    assert_eq!(wc.level, WriteConcernLevel::Majority);
    assert!(!wc.journal);
    assert_eq!(wc.timeout_ms, 0);
}

#[test]
fn constructors() {
    assert_eq!(WriteConcern::w0().level, WriteConcernLevel::W0);
    assert_eq!(WriteConcern::memory().level, WriteConcernLevel::Memory);
    assert_eq!(WriteConcern::cache().level, WriteConcernLevel::Cache);
    assert_eq!(WriteConcern::w1().level, WriteConcernLevel::W1);
    assert_eq!(WriteConcern::majority().level, WriteConcernLevel::Majority);

    let mj = WriteConcern::majority_journaled(5000);
    assert_eq!(mj.level, WriteConcernLevel::Majority);
    assert!(mj.journal);
    assert_eq!(mj.timeout_ms, 5000);
}

#[test]
fn effective_level_journal_upgrade() {
    // j:true + W0 → W1
    let wc = WriteConcern {
        level: WriteConcernLevel::W0,
        journal: true,
        timeout_ms: 0,
    };
    assert_eq!(wc.effective_level(), WriteConcernLevel::W1);

    // j:true + Memory → W1 (contradictory: can't be in-memory AND journaled)
    let wc_mem = WriteConcern {
        level: WriteConcernLevel::Memory,
        journal: true,
        timeout_ms: 0,
    };
    assert_eq!(wc_mem.effective_level(), WriteConcernLevel::W1);

    // j:true + Cache → Cache (NVMe survives process crash, j:true is redundant)
    let wc_cache = WriteConcern {
        level: WriteConcernLevel::Cache,
        journal: true,
        timeout_ms: 0,
    };
    assert_eq!(wc_cache.effective_level(), WriteConcernLevel::Cache);

    // j:true + W1 → W1 (no change)
    let wc2 = WriteConcern {
        level: WriteConcernLevel::W1,
        journal: true,
        timeout_ms: 0,
    };
    assert_eq!(wc2.effective_level(), WriteConcernLevel::W1);

    // j:false + W0 → W0 (no upgrade)
    assert_eq!(WriteConcern::w0().effective_level(), WriteConcernLevel::W0);
}

#[test]
fn requires_majority() {
    assert!(!WriteConcernLevel::W0.requires_majority());
    assert!(!WriteConcernLevel::Memory.requires_majority());
    assert!(!WriteConcernLevel::Cache.requires_majority());
    assert!(!WriteConcernLevel::W1.requires_majority());
    assert!(WriteConcernLevel::Majority.requires_majority());
}

#[test]
fn can_rollback() {
    assert!(WriteConcernLevel::W0.can_rollback());
    assert!(WriteConcernLevel::Memory.can_rollback());
    assert!(WriteConcernLevel::Cache.can_rollback());
    assert!(WriteConcernLevel::W1.can_rollback());
    assert!(!WriteConcernLevel::Majority.can_rollback());
}

#[test]
fn causal_safety() {
    assert!(!WriteConcernLevel::W0.is_causal_safe());
    assert!(!WriteConcernLevel::Memory.is_causal_safe());
    assert!(!WriteConcernLevel::Cache.is_causal_safe());
    assert!(!WriteConcernLevel::W1.is_causal_safe());
    assert!(WriteConcernLevel::Majority.is_causal_safe());
}

#[test]
fn is_volatile() {
    assert!(!WriteConcernLevel::W0.is_volatile());
    assert!(WriteConcernLevel::Memory.is_volatile());
    assert!(WriteConcernLevel::Cache.is_volatile());
    assert!(!WriteConcernLevel::W1.is_volatile());
    assert!(!WriteConcernLevel::Majority.is_volatile());
}

#[test]
fn validate_for_causal_session() {
    assert!(WriteConcern::majority()
        .validate_for_causal_session()
        .is_ok());
    assert!(WriteConcern::w0().validate_for_causal_session().is_err());
    assert!(WriteConcern::w1().validate_for_causal_session().is_err());
    assert!(WriteConcern::memory()
        .validate_for_causal_session()
        .is_err());
    assert!(WriteConcern::cache().validate_for_causal_session().is_err());

    // j:true + W0 upgrades to W1 → still not causal-safe
    let wc = WriteConcern {
        level: WriteConcernLevel::W0,
        journal: true,
        timeout_ms: 0,
    };
    assert!(wc.validate_for_causal_session().is_err());

    // j:true + Memory upgrades to W1 → still not causal-safe
    let wc_mem = WriteConcern {
        level: WriteConcernLevel::Memory,
        journal: true,
        timeout_ms: 0,
    };
    assert!(wc_mem.validate_for_causal_session().is_err());
}

#[test]
fn display() {
    assert_eq!(WriteConcernLevel::W0.to_string(), "w:0");
    assert_eq!(WriteConcernLevel::Memory.to_string(), "w:memory");
    assert_eq!(WriteConcernLevel::Cache.to_string(), "w:cache");
    assert_eq!(WriteConcernLevel::W1.to_string(), "w:1");
    assert_eq!(WriteConcernLevel::Majority.to_string(), "w:majority");
}

#[test]
fn validate_ok() {
    assert!(WriteConcern::w0().validate().is_ok());
    assert!(WriteConcern::memory().validate().is_ok());
    assert!(WriteConcern::cache().validate().is_ok());
    assert!(WriteConcern::w1().validate().is_ok());
    assert!(WriteConcern::majority().validate().is_ok());
    assert!(WriteConcern::majority_journaled(5000).validate().is_ok());
}
