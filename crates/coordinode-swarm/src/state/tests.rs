use super::*;

#[test]
fn bitfield_set_has_count_complete() {
    let mut bf = PieceBitfield::new(130); // spans 3 words
    assert_eq!(bf.len(), 130);
    assert!(!bf.has(0));
    bf.set(0);
    bf.set(65);
    bf.set(129);
    assert!(bf.has(0) && bf.has(65) && bf.has(129));
    assert!(!bf.has(1));
    assert_eq!(bf.count_set(), 3);
    assert!(!bf.is_complete());
    bf.set(200); // out of range → ignored
    assert_eq!(bf.count_set(), 3);

    assert!(PieceBitfield::full(130).is_complete());
}

#[test]
fn select_next_piece_is_rarest_first() {
    // 4 pieces. Source S has all; A has piece 0; B has pieces 0,1.
    // For a fresh node N: availability is p0=3, p1=2, p2=1, p3=1.
    // Rarest = p2 (tie p2/p3 at 1 → lowest index 2).
    let mut st = SwarmState::new(4);
    st.set_peer_bitfield(NodeId(0), PieceBitfield::full(4)); // source
    let mut a = PieceBitfield::new(4);
    a.set(0);
    st.set_peer_bitfield(NodeId(1), a);
    let mut b = PieceBitfield::new(4);
    b.set(0);
    b.set(1);
    st.set_peer_bitfield(NodeId(2), b);

    assert_eq!(st.availability(0), 3);
    assert_eq!(st.availability(2), 1);
    assert_eq!(
        st.select_next_piece(NodeId(99)),
        Some(2),
        "rarest, lowest-index tie"
    );
}

#[test]
fn select_skips_already_held_and_in_flight() {
    let mut st = SwarmState::new(3);
    st.set_peer_bitfield(NodeId(0), PieceBitfield::full(3)); // source has all
                                                             // N already holds piece 0.
    st.mark_piece(NodeId(1), 0);
    // Piece 1 is in flight to N.
    st.mark_in_flight(1, NodeId(0), NodeId(1));
    // → only piece 2 is selectable for N.
    assert_eq!(st.select_next_piece(NodeId(1)), Some(2));
    // Completing the in-flight does not change that 1 is now (about to be) held;
    // clearing it makes 1 selectable again (rarest tie → lowest index 1).
    st.complete_in_flight(1);
    assert_eq!(st.select_next_piece(NodeId(1)), Some(1));
}

#[test]
fn select_returns_none_when_complete() {
    let mut st = SwarmState::new(2);
    st.set_peer_bitfield(NodeId(0), PieceBitfield::full(2));
    st.set_peer_bitfield(NodeId(1), PieceBitfield::full(2));
    assert_eq!(st.select_next_piece(NodeId(1)), None);
}

#[test]
fn mark_piece_creates_bitfield_and_updates_availability() {
    let mut st = SwarmState::new(4);
    assert_eq!(st.availability(2), 0);
    st.mark_piece(NodeId(5), 2);
    assert_eq!(st.availability(2), 1);
    assert!(st.peer_bitfields.get(&NodeId(5)).expect("created").has(2));
}
