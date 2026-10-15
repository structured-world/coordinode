use super::*;

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// Both named profiles pass the checks a custom timing has to.
#[test]
fn the_named_profiles_are_valid_timings() {
    for profile in [ElectionTiming::RACK, ElectionTiming::PERSONAL_FLEET] {
        assert_eq!(
            ElectionTiming::new(
                profile.heartbeat(),
                profile.election_min(),
                profile.election_max()
            ),
            Ok(profile)
        );
    }
}

/// A rack fails over within a second; a personal fleet waits tens of seconds.
#[test]
fn the_profiles_differ_by_the_scale_they_are_for() {
    assert!(ElectionTiming::RACK.election_max() < Duration::from_secs(1));
    assert!(ElectionTiming::PERSONAL_FLEET.election_min() >= Duration::from_secs(10));
    assert_eq!(ElectionTiming::default(), ElectionTiming::RACK);
}

/// A heartbeat that cannot arrive twice within the shortest timeout is
/// refused: followers would stand against a healthy leader.
#[test]
fn a_heartbeat_too_slow_for_the_election_window_is_refused() {
    assert_eq!(
        ElectionTiming::new(ms(200), ms(300), ms(600)),
        Err(TimingError::HeartbeatTooSlow {
            heartbeat: ms(200),
            election_min: ms(300),
        })
    );
    // Exactly half is allowed.
    assert!(ElectionTiming::new(ms(150), ms(300), ms(600)).is_ok());
}

/// An empty or inverted window would have every member stand together.
#[test]
fn an_empty_election_window_is_refused() {
    for (min, max) in [(600, 600), (600, 300)] {
        assert_eq!(
            ElectionTiming::new(ms(100), ms(min), ms(max)),
            Err(TimingError::EmptyElectionWindow {
                election_min: ms(min),
                election_max: ms(max),
            })
        );
    }
}

/// Zero, and values the millisecond counter cannot hold, are out of range.
#[test]
fn zero_and_unrepresentable_values_are_refused() {
    assert_eq!(
        ElectionTiming::new(Duration::ZERO, ms(300), ms(600)),
        Err(TimingError::OutOfRange(Duration::ZERO))
    );
    let huge = Duration::from_millis(u64::MAX / 2);
    assert_eq!(
        ElectionTiming::new(ms(100), ms(300), huge),
        Err(TimingError::OutOfRange(huge)),
        "three times the longest timeout must still fit"
    );
}

/// A member that stands last waits past every other member's whole window,
/// keeping its heartbeat.
#[test]
fn a_member_that_stands_last_waits_past_every_other_window() {
    for profile in [ElectionTiming::RACK, ElectionTiming::PERSONAL_FLEET] {
        let last = profile.for_member(ElectionPreference::Last);
        assert!(last.election_min() > profile.election_max());
        assert!(last.election_min() < last.election_max());
        assert_eq!(last.heartbeat(), profile.heartbeat());
        assert_eq!(profile.for_member(ElectionPreference::Ordinary), profile);
    }
}
