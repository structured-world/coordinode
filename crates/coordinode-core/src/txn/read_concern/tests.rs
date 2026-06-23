use super::*;

#[test]
fn default_is_local() {
    let rc = ReadConcern::default();
    assert_eq!(rc.level, ReadConcernLevel::Local);
    assert!(rc.after_index.is_none());
    assert!(rc.at_timestamp.is_none());
}

#[test]
fn constructors_set_level() {
    assert_eq!(ReadConcern::local().level, ReadConcernLevel::Local);
    assert_eq!(ReadConcern::majority().level, ReadConcernLevel::Majority);
    assert_eq!(
        ReadConcern::linearizable().level,
        ReadConcernLevel::Linearizable
    );
    assert_eq!(
        ReadConcern::snapshot_at(100).level,
        ReadConcernLevel::Snapshot
    );
    assert_eq!(ReadConcern::snapshot_at(100).at_timestamp, Some(100));
}

#[test]
fn requires_leader() {
    assert!(!ReadConcernLevel::Local.requires_leader());
    assert!(!ReadConcernLevel::Majority.requires_leader());
    assert!(ReadConcernLevel::Linearizable.requires_leader());
    assert!(!ReadConcernLevel::Snapshot.requires_leader());
}

#[test]
fn is_durable() {
    assert!(!ReadConcernLevel::Local.is_durable());
    assert!(ReadConcernLevel::Majority.is_durable());
    assert!(ReadConcernLevel::Linearizable.is_durable());
    assert!(ReadConcernLevel::Snapshot.is_durable());
}

#[test]
fn validate_ok() {
    assert!(ReadConcern::local().validate().is_ok());
    assert!(ReadConcern::majority().validate().is_ok());
    assert!(ReadConcern::linearizable().validate().is_ok());
    assert!(ReadConcern::snapshot_at(100).validate().is_ok());
}

#[test]
fn validate_mutual_exclusion() {
    let rc = ReadConcern {
        level: ReadConcernLevel::Snapshot,
        after_index: Some(50),
        at_timestamp: Some(100),
    };
    assert!(rc.validate().is_err());
}

#[test]
fn validate_linearizable_no_after() {
    let rc = ReadConcern {
        level: ReadConcernLevel::Linearizable,
        after_index: Some(50),
        at_timestamp: None,
    };
    assert!(rc.validate().is_err());
}

#[test]
fn validate_at_timestamp_only_with_snapshot() {
    let rc = ReadConcern {
        level: ReadConcernLevel::Majority,
        after_index: None,
        at_timestamp: Some(100),
    };
    assert!(rc.validate().is_err());
}

#[test]
fn display() {
    assert_eq!(ReadConcernLevel::Local.to_string(), "local");
    assert_eq!(ReadConcernLevel::Majority.to_string(), "majority");
    assert_eq!(ReadConcernLevel::Linearizable.to_string(), "linearizable");
    assert_eq!(ReadConcernLevel::Snapshot.to_string(), "snapshot");
}
