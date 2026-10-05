use super::*;

/// Numbers move forward one by one and are never handed out twice.
#[test]
fn allocation_moves_forward() {
    let mut alloc = IdentityAllocator::default();
    assert_eq!(alloc.allocate_index(), Ok(IndexId::from_raw(0)));
    assert_eq!(alloc.allocate_index(), Ok(IndexId::from_raw(1)));
    assert_eq!(alloc.allocate_generation(), Ok(GenerationId::from_raw(0)));
    assert_eq!(alloc.allocate_generation(), Ok(GenerationId::from_raw(1)));
    assert_eq!(
        alloc,
        IdentityAllocator {
            next_index: 2,
            next_generation: 2
        }
    );
}

/// The last numbers are refused rather than wrapped: wrapping would hand
/// out a number an existing or dropped object holds.
#[test]
fn exhaustion_is_an_error_not_a_wrap() {
    let mut alloc = IdentityAllocator {
        next_index: u64::MAX,
        next_generation: u64::MAX,
    };
    assert_eq!(alloc.allocate_index(), Err(IdentityExhausted("index")));
    assert_eq!(
        alloc.allocate_generation(),
        Err(IdentityExhausted("generation"))
    );
    assert_eq!(
        alloc.next_index,
        u64::MAX,
        "a refused allocation takes nothing"
    );
}

/// The identities serialize as their bare numbers.
#[test]
fn identities_serialize_as_numbers() {
    let bytes = rmp_serde::to_vec(&GenerationId::from_raw(7)).expect("encode");
    assert_eq!(rmp_serde::from_slice::<u64>(&bytes).expect("decode"), 7);
}
