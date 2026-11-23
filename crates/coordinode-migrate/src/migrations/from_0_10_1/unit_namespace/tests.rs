use super::*;
use crate::test_support::{named_frame, unnamed};

/// A unit 0.10.1 wrote, once named, is byte for byte the frame this build
/// writes for the same operations; a frame that reads already is left.
#[test]
fn a_named_unit_is_the_frame_this_build_writes() {
    let current = named_frame(7, 107);
    let old = unnamed(&current);
    assert!(
        decode_unit(&old).is_err(),
        "0.10.1's unit does not read now"
    );
    assert_eq!(name_unit(&old).expect("named"), Some(current.clone()));
    assert_eq!(name_unit(&current).expect("read"), None);
}

/// A frame that reads neither now nor once named is refused, not passed.
#[test]
fn a_frame_that_does_not_read_is_refused() {
    let mut damaged = unnamed(&named_frame(7, 107));
    let last = damaged.len() - 1;
    damaged[last] ^= 0xff;
    assert!(name_unit(&damaged).is_err());
}
