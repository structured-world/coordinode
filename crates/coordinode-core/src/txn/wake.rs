//! A wakeup a background loop sleeps on until something happens.
//!
//! A loop that polls a condition on a timer wakes the machine for nothing
//! whenever the condition has not changed, which on an idle database is
//! always. A [`Wake`] lets the code that changes the condition say so: the
//! loop parks until notified, or until a deadline it has a reason for.
//!
//! Notifying is one relaxed load once a wakeup is already pending, so a hot
//! path can notify on every event without contending on anything.

// no-std: replace the thread park with a caller-provided parker (an executor
// waker or a platform futex); the pending-flag protocol stays the same.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::Thread;
use std::time::Duration;

/// One loop's wakeup. The loop binds itself with [`Wake::bind`]; anyone holding
/// the wake calls [`Wake::notify`].
#[derive(Debug, Default)]
pub struct Wake {
    pending: AtomicBool,
    thread: OnceLock<Thread>,
}

impl Wake {
    /// Bind the calling thread as the one this wake unparks. Called once, at
    /// the top of the loop's thread.
    pub fn bind(&self) {
        // A second bind would mean two loops sharing one wake, which would lose
        // wakeups for one of them; the first binding stands.
        let _ = self.thread.set(std::thread::current());
    }

    /// Say that the condition the loop waits on may have changed.
    pub fn notify(&self) {
        if self.pending.load(Ordering::Relaxed) {
            return;
        }
        if !self.pending.swap(true, Ordering::AcqRel) {
            if let Some(thread) = self.thread.get() {
                thread.unpark();
            }
        }
    }

    /// Wake the loop whether or not a wakeup is already pending, for a stop
    /// that must not wait out a sleep the loop chose itself.
    pub fn interrupt(&self) {
        self.pending.store(true, Ordering::Release);
        if let Some(thread) = self.thread.get() {
            thread.unpark();
        }
    }

    /// Sleep until notified, or until `timeout` passes when one is given.
    ///
    /// A notification that arrived since the last wait returns at once: the
    /// unpark token outlives the gap between the loop's check and its park, so
    /// no event that happens in that gap is slept through.
    pub fn wait(&self, timeout: Option<Duration>) {
        if self.pending.swap(false, Ordering::AcqRel) {
            return;
        }
        match timeout {
            Some(timeout) => std::thread::park_timeout(timeout),
            None => std::thread::park(),
        }
        self.pending.store(false, Ordering::Release);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
