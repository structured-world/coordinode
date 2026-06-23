use super::*;
use crate::state::PieceBitfield;

fn candidate(id: u64) -> SourceCandidate {
    SourceCandidate {
        node: NodeId(id),
        utilization: 0.0,
        bandwidth_to_target: 1.0,
        same_rack: false,
        tit_for_tat: 1.0,
        freshness: Freshness::Verified,
    }
}

#[test]
fn score_rewards_idle_fast_local_fresh_generous() {
    let base = candidate(0);
    let busy = SourceCandidate {
        utilization: 0.9,
        ..base
    };
    let local = SourceCandidate {
        same_rack: true,
        ..base
    };
    let in_flight = SourceCandidate {
        freshness: Freshness::InFlight,
        ..base
    };
    let generous = SourceCandidate {
        tit_for_tat: 3.0,
        ..base
    };
    assert!(base.score() > busy.score(), "idle beats busy");
    assert!(
        (local.score() - base.score() * 10.0).abs() < 1e-9,
        "same-rack 10x"
    );
    assert!(in_flight.score() < base.score(), "verified beats in-flight");
    assert!(generous.score() > base.score(), "uploader rewarded");
    // A saturated source contributes zero, not a negative score.
    let saturated = SourceCandidate {
        utilization: 1.5,
        ..base
    };
    assert_eq!(saturated.score(), 0.0);
}

#[test]
fn select_source_picks_highest_score_holder() {
    // 1 piece. Three peers hold it; node 2 is same-rack (10x) → wins.
    let mut st = SwarmState::new(1);
    for id in [1u64, 2, 3] {
        st.set_peer_bitfield(NodeId(id), PieceBitfield::full(1));
    }
    let cands = [
        candidate(1),
        SourceCandidate {
            same_rack: true,
            ..candidate(2)
        },
        candidate(3),
    ];
    assert_eq!(select_source(&st, 0, &cands), Some(NodeId(2)));
}

#[test]
fn select_source_skips_non_holders_and_ties_low_id() {
    let mut st = SwarmState::new(1);
    // Only nodes 2 and 3 hold the piece; node 1 (best stats) does NOT.
    st.set_peer_bitfield(NodeId(2), PieceBitfield::full(1));
    st.set_peer_bitfield(NodeId(3), PieceBitfield::full(1));
    let cands = [candidate(1), candidate(2), candidate(3)];
    // 1 is skipped (no copy); 2 and 3 tie → lowest id 2.
    assert_eq!(select_source(&st, 0, &cands), Some(NodeId(2)));
}

#[test]
fn select_source_none_when_no_holder() {
    let st = SwarmState::new(2);
    assert_eq!(select_source(&st, 0, &[candidate(1)]), None);
}
