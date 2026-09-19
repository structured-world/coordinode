use super::*;

/// Leaving both axes unset gives the safest concern: a majority, fsynced.
#[test]
fn default_is_majority_journaled() {
    let wc = WriteConcern::default();
    assert_eq!(wc.w, WriteAck::Majority);
    assert_eq!(wc.journal, Journal::Journal);
    assert_eq!(wc.timeout_ms, 0);
}

/// Each constructor sets exactly the axis it names and leaves the other at
/// its default, so `memory()` is "w:1, in RAM" and never "in RAM on a
/// majority".
#[test]
fn constructors_set_one_axis_each() {
    assert_eq!(WriteConcern::w0().w, WriteAck::Acks(0));
    assert_eq!(WriteConcern::w0().journal, Journal::Journal);
    assert_eq!(WriteConcern::w1().w, WriteAck::Acks(1));
    assert_eq!(WriteConcern::acks(3).w, WriteAck::Acks(3));
    assert_eq!(WriteConcern::acks(3).journal, Journal::Journal);
    assert_eq!(WriteConcern::majority(), WriteConcern::default());

    let mem = WriteConcern::memory();
    assert_eq!((mem.w, mem.journal), (WriteAck::Acks(1), Journal::Memory));
    let cache = WriteConcern::cache();
    assert_eq!(
        (cache.w, cache.journal),
        (WriteAck::Acks(1), Journal::Cache)
    );

    let mj = WriteConcern::majority_with_timeout(5000);
    assert_eq!(mj.w, WriteAck::Majority);
    assert_eq!(mj.journal, Journal::Journal);
    assert_eq!(mj.timeout_ms, 5000);
}

/// A number is a count of members: `w:3` is three members whatever the
/// group's size, and `Majority` is the only rule that is not a count.
#[test]
fn a_number_is_a_count_and_never_a_mode() {
    assert_ne!(WriteAck::Acks(3), WriteAck::Majority);
    assert!(WriteAck::Acks(0).is_fire_and_forget());
    assert!(!WriteAck::Acks(1).is_fire_and_forget());
    assert!(!WriteAck::Acks(1).needs_replication());
    assert!(WriteAck::Acks(2).needs_replication());
    assert!(WriteAck::Majority.needs_replication());
}

/// The volatile path is a property of `j` with a caller that waits: `w:0`
/// takes the ordinary log path even with `j:memory`, because nobody waits
/// for the overlay.
#[test]
fn volatile_is_journal_below_journal_with_someone_waiting() {
    assert!(WriteConcern::memory().is_volatile());
    assert!(WriteConcern::cache().is_volatile());
    assert!(!WriteConcern::w1().is_volatile());
    assert!(!WriteConcern::majority().is_volatile());
    let w0_memory = WriteConcern {
        w: WriteAck::Acks(0),
        journal: Journal::Memory,
        timeout_ms: 0,
    };
    assert!(!w0_memory.is_volatile());
}

/// Only a fsynced majority is safe for a causal session: anything less can be
/// lost after the caller has learned its position.
#[test]
fn causal_safety_needs_majority_and_journal() {
    assert!(WriteConcern::majority().is_causal_safe());
    assert!(!WriteConcern::w1().is_causal_safe());
    assert!(!WriteConcern::w0().is_causal_safe());
    assert!(!WriteConcern::memory().is_causal_safe());
    assert!(!WriteConcern::acks(5).is_causal_safe());
    let majority_in_memory = WriteConcern {
        w: WriteAck::Majority,
        journal: Journal::Memory,
        timeout_ms: 0,
    };
    assert!(!majority_in_memory.is_causal_safe());
    assert!(majority_in_memory.can_rollback());
    assert!(!WriteConcern::majority().can_rollback());
    assert!(
        WriteConcern::majority()
            .validate_for_causal_session()
            .is_ok()
    );
    assert!(WriteConcern::w1().validate_for_causal_session().is_err());
    assert!(
        WriteConcern::memory()
            .validate_for_causal_session()
            .is_err()
    );
}

/// `w` above the group's member count is refused, not clamped: a caller who
/// asked for four copies in a group of three must learn that, not get three.
#[test]
fn acks_above_membership_are_refused_not_clamped() {
    assert_eq!(
        WriteConcern::acks(4).validate(Some(3)),
        Err(WriteConcernError::TooManyAcks {
            requested: 4,
            members: 3,
        })
    );
    assert!(WriteConcern::acks(3).validate(Some(3)).is_ok());
    assert!(WriteConcern::acks(4).validate(None).is_ok());
    assert!(WriteConcern::majority().validate(Some(1)).is_ok());
}

/// A volatile journal level with more than one acknowledging member is
/// refused explicitly, never quietly served as `journal`.
#[test]
fn volatile_journal_is_honoured_for_the_leader_only() {
    let majority_in_memory = WriteConcern {
        w: WriteAck::Majority,
        journal: Journal::Memory,
        timeout_ms: 0,
    };
    assert_eq!(
        majority_in_memory.validate(Some(3)),
        Err(WriteConcernError::VolatileReplication(majority_in_memory))
    );
    let two_in_cache = WriteConcern {
        w: WriteAck::Acks(2),
        journal: Journal::Cache,
        timeout_ms: 0,
    };
    assert!(two_in_cache.validate(None).is_err());
    assert!(WriteConcern::memory().validate(Some(3)).is_ok());
    let w0_memory = WriteConcern {
        w: WriteAck::Acks(0),
        journal: Journal::Memory,
        timeout_ms: 0,
    };
    assert!(w0_memory.validate(Some(3)).is_ok());
}

#[test]
fn display_names_both_axes() {
    assert_eq!(WriteConcern::w0().to_string(), "w:0,j:journal");
    assert_eq!(WriteConcern::memory().to_string(), "w:1,j:memory");
    assert_eq!(WriteConcern::cache().to_string(), "w:1,j:cache");
    assert_eq!(WriteConcern::acks(3).to_string(), "w:3,j:journal");
    assert_eq!(WriteConcern::majority().to_string(), "w:majority,j:journal");
    assert_eq!(
        WriteConcern::majority_with_timeout(500).to_string(),
        "w:majority,j:journal,wtimeout:500"
    );
}
