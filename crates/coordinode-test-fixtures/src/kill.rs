//! Ending a crash-test child process the way a kill ends a server.

/// The signal a crash-test child dies of on Unix: SIGKILL, so no destructor,
/// atexit or signal handler runs, and the OS files no crash report for it as
/// it does for an abort.
pub const CHILD_KILL_SIGNAL: i32 = 9;

/// SIGKILL to this process on Unix; an abort elsewhere, or if the signal
/// could not be sent. The calling thread does nothing more after sending it.
pub fn die_killed() -> ! {
    #[cfg(unix)]
    {
        use rustix::process::{Signal, getpid, kill_process};
        match kill_process(getpid(), Signal::KILL) {
            // The kill may land after the call returns (macOS delivers it to
            // a threaded process asynchronously): wait for it rather than
            // abort first.
            Ok(()) => loop {
                std::thread::park();
            },
            Err(e) => eprintln!("SIGKILL to itself failed ({e}); aborting instead"),
        }
    }
    std::process::abort()
}
