use super::*;

fn position(group: u64, index: u64) -> HistoryPosition {
    HistoryPosition {
        history: HistoryId {
            deployment: crate::group::DeploymentId([7; 16]),
            group: crate::group::GroupId(group),
        },
        index,
    }
}

#[test]
fn default_is_local() {
    let rc = ReadConcern::default();
    assert_eq!(rc.level, ReadConcernLevel::Local);
    assert!(rc.after_ts.is_none());
    assert!(rc.after_position.is_none());
    assert!(rc.at_timestamp.is_none());
    assert!(rc.wait.is_none(), "an unnamed wait waits for nothing");
    assert!(!rc.is_causal());
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

/// Either causal bound makes the read causal, and a causal majority read
/// is valid.
#[test]
fn a_timestamp_or_a_position_is_a_causal_bound() {
    let by_ts = ReadConcern {
        level: ReadConcernLevel::Majority,
        after_ts: Some(50),
        ..Default::default()
    };
    let by_position = ReadConcern {
        level: ReadConcernLevel::Majority,
        after_position: Some(position(0, 9)),
        ..Default::default()
    };
    assert!(by_ts.is_causal());
    assert!(by_position.is_causal());
    assert!(by_ts.validate().is_ok());
    assert!(by_position.validate().is_ok());
}

#[test]
fn validate_mutual_exclusion() {
    for rc in [
        ReadConcern {
            level: ReadConcernLevel::Snapshot,
            after_ts: Some(50),
            at_timestamp: Some(100),
            ..Default::default()
        },
        ReadConcern {
            level: ReadConcernLevel::Snapshot,
            after_position: Some(position(0, 3)),
            at_timestamp: Some(100),
            ..Default::default()
        },
    ] {
        assert_eq!(
            rc.validate(),
            Err("a causal bound and at_timestamp are mutually exclusive")
        );
    }
}

#[test]
fn validate_linearizable_no_after() {
    let rc = ReadConcern {
        level: ReadConcernLevel::Linearizable,
        after_ts: Some(50),
        ..Default::default()
    };
    assert_eq!(
        rc.validate(),
        Err("afterClusterTime incompatible with linearizable")
    );
}

#[test]
fn validate_at_timestamp_only_with_snapshot() {
    let rc = ReadConcern {
        level: ReadConcernLevel::Majority,
        at_timestamp: Some(100),
        ..Default::default()
    };
    assert_eq!(
        rc.validate(),
        Err("at_timestamp only valid with snapshot read concern")
    );
}

/// Positions of different groups or deployments are different histories.
#[test]
fn histories_differ_by_deployment_and_group() {
    let a = position(0, 5).history;
    let other_group = position(1, 5).history;
    let other_deployment = HistoryId {
        deployment: crate::group::DeploymentId([8; 16]),
        ..a
    };
    assert_ne!(a, other_group);
    assert_ne!(a, other_deployment);
}

#[test]
fn display() {
    assert_eq!(ReadConcernLevel::Local.to_string(), "local");
    assert_eq!(ReadConcernLevel::Majority.to_string(), "majority");
    assert_eq!(ReadConcernLevel::Linearizable.to_string(), "linearizable");
    assert_eq!(ReadConcernLevel::Snapshot.to_string(), "snapshot");
}
