//! The protection a volatile commit keeps from its acceptance until its
//! drain hands it to the log.

use std::sync::Arc;

use coordinode_core::txn::drain::DrainHold;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};

use crate::engine::claims::OwnedReservation;
use crate::engine::pending::Staged;

/// A volatile commit's registered keys and invariant claims, and the clock
/// its drain takes a timestamp from. Fields drop in order: the keys, then
/// the claims, both after the entry is handed to the log.
#[derive(Debug)]
pub(crate) struct VolatileHold {
    pub(crate) staged: Staged,
    /// Held for its release on drop.
    pub(crate) _claims: Option<OwnedReservation>,
    pub(crate) oracle: Arc<TimestampOracle>,
}

impl DrainHold for VolatileHold {
    fn land(&mut self) -> Option<Timestamp> {
        let oracle = &self.oracle;
        self.staged
            .admit(|| oracle.next().as_raw())
            .map(Timestamp::from_raw)
    }
}
