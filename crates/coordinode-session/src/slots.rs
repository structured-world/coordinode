//! The member's slots for running commands, whichever way they arrive: sent
//! by this member's own clients on a session or a single call, or routed here
//! by another member. A command holds one of the member's slots, and one of
//! its session's when it has a session, from before it runs until it ends,
//! waits for a causal bound or a transaction's turn included. A command past
//! either is refused before anything of it runs, so the caller can retry it.
//!
//! Slots are taken without waiting: a session's reader that waited for one
//! would hold back the cancellations and acknowledgements that free one.

use std::num::NonZeroUsize;
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// A number of slots a member or a session can have: from one to
/// [`SlotCount::MAX`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotCount(NonZeroUsize);

impl SlotCount {
    /// The most slots there can be.
    pub const MAX: usize = Semaphore::MAX_PERMITS;

    /// `n` slots; `None` for none, or for more than [`Self::MAX`].
    pub const fn new(n: usize) -> Option<Self> {
        if n > Self::MAX {
            return None;
        }
        match NonZeroUsize::new(n) {
            Some(n) => Some(Self(n)),
            None => None,
        }
    }

    /// How many.
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// The slots one member runs commands in, shared by every way commands reach
/// it. Cheap to clone: the member's slots are shared.
#[derive(Debug, Clone)]
pub struct CommandSlots {
    member: Arc<Semaphore>,
    session_commands: SlotCount,
}

impl CommandSlots {
    /// Slots for `commands` commands at once on the member, at most
    /// `session_commands` of one session.
    pub fn new(commands: SlotCount, session_commands: SlotCount) -> Self {
        Self {
            member: Arc::new(Semaphore::new(commands.get())),
            session_commands,
        }
    }

    /// The slots of one new session: its own, within the member's.
    pub fn session(&self) -> SessionSlots {
        SessionSlots {
            member: Arc::clone(&self.member),
            session: Arc::new(Semaphore::new(self.session_commands.get())),
        }
    }

    /// A slot for a command that has no session (a single call), when the
    /// member has one free.
    pub fn take(&self) -> Option<CommandSlot> {
        Some(CommandSlot {
            _member: Arc::clone(&self.member).try_acquire_owned().ok()?,
            _session: None,
        })
    }

    /// The member's slots free now.
    pub fn available(&self) -> usize {
        self.member.available_permits()
    }
}

/// The slots of one session: at most its own number of commands at once, and
/// each also one of the member's.
#[derive(Debug, Clone)]
pub struct SessionSlots {
    member: Arc<Semaphore>,
    session: Arc<Semaphore>,
}

impl SessionSlots {
    /// A slot for one more command of the session, when the member and the
    /// session both have one free.
    pub fn take(&self) -> Option<CommandSlot> {
        // The session's first: a session at its own bound takes nothing of
        // the member's on the way to being refused.
        let session = Arc::clone(&self.session).try_acquire_owned().ok()?;
        let member = Arc::clone(&self.member).try_acquire_owned().ok()?;
        Some(CommandSlot {
            _member: member,
            _session: Some(session),
        })
    }
}

/// A slot a command holds until it ends; dropping it frees it.
#[derive(Debug)]
pub struct CommandSlot {
    _member: OwnedSemaphorePermit,
    _session: Option<OwnedSemaphorePermit>,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
