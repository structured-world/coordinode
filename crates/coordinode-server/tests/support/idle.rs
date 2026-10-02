//! How often an idle server wakes up.
//!
//! Starts the `coordinode` binary on an empty store, waits until it serves,
//! lets the work of the start itself settle, then counts the context switches
//! of every thread of the process over a window in which nothing is asked of
//! it. A background loop that polls on a timer shows up here as wakeups per
//! second; one that waits for its event shows up as nothing.

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// What one idle window measured.
#[derive(Debug)]
pub struct IdleSample {
    /// Context switches of the whole process over the window.
    pub switches: u64,
    /// Length of the window.
    pub window: Duration,
    /// Per-thread switches, by thread name, busiest first, where the
    /// platform names them.
    pub threads: Vec<(String, u64)>,
}

impl IdleSample {
    /// Wakeups per second over the window.
    pub fn per_sec(&self) -> f64 {
        self.switches as f64 / self.window.as_secs_f64()
    }
}

/// A running server, killed when dropped.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start `binary` on an empty store under `data`, wait until it is ready,
/// wait `settle` more, and count its context switches over `window`.
pub fn measure(binary: &Path, data: &Path, settle: Duration, window: Duration) -> IdleSample {
    let [grpc, rest, ops, pg] = coordinode_test_fixtures::alloc_ports::<4>();
    let child = Command::new(binary)
        .arg("serve")
        .arg("--data")
        .arg(data)
        .arg("--addr")
        .arg(format!("127.0.0.1:{grpc}"))
        .arg("--rest-addr")
        .arg(format!("127.0.0.1:{rest}"))
        .arg("--ops-addr")
        .arg(format!("127.0.0.1:{ops}"))
        .arg("--pg-addr")
        .arg(format!("127.0.0.1:{pg}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start coordinode serve");
    let server = Server(child);
    let pid = server.0.id();

    let until = Instant::now() + Duration::from_secs(60);
    while !ready(ops) {
        assert!(Instant::now() < until, "the server did not become ready");
        std::thread::sleep(Duration::from_millis(100));
    }
    std::thread::sleep(settle);

    let before = switches(pid);
    let started = Instant::now();
    std::thread::sleep(window);
    let after = switches(pid);
    let window = started.elapsed();
    drop(server);

    let total = |s: &[(String, u64)]| s.iter().map(|(_, n)| n).sum::<u64>();
    let mut threads: Vec<(String, u64)> = after
        .iter()
        .map(|(name, n)| {
            let was = before
                .iter()
                .filter(|(b, _)| b == name)
                .map(|(_, n)| *n)
                .sum::<u64>();
            (name.clone(), n.saturating_sub(was))
        })
        .filter(|(_, n)| *n > 0)
        .collect();
    threads.sort_by_key(|t| std::cmp::Reverse(t.1));
    IdleSample {
        // A thread that exited inside the window takes its count with it;
        // that only lowers the figure by switches that were not idle work.
        switches: total(&after).saturating_sub(total(&before)),
        window,
        threads,
    }
}

/// Whether the ops endpoint on `port` answers `/ready` with 200.
fn ready(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream
        .write_all(b"GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut reply = String::new();
    let _ = stream.read_to_string(&mut reply);
    reply.starts_with("HTTP/1.1 200")
}

/// Context switches per thread of process `pid`, voluntary and involuntary
/// together, keyed by thread name (threads of one name are summed).
#[cfg(target_os = "linux")]
fn switches(pid: u32) -> Vec<(String, u64)> {
    let mut by_name: Vec<(String, u64)> = Vec::new();
    let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
        return by_name;
    };
    for task in tasks.flatten() {
        let dir = task.path();
        let name = std::fs::read_to_string(dir.join("comm"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let Ok(status) = std::fs::read_to_string(dir.join("status")) else {
            continue;
        };
        let count: u64 = status
            .lines()
            .filter(|l| {
                l.starts_with("voluntary_ctxt_switches")
                    || l.starts_with("nonvoluntary_ctxt_switches")
            })
            .filter_map(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .sum();
        match by_name.iter_mut().find(|(n, _)| *n == name) {
            Some((_, n)) => *n += count,
            None => by_name.push((name, count)),
        }
    }
    by_name
}

/// Context switches of process `pid`. macOS reports the process total only,
/// through `top`'s CSW column (the kernel counter `proc_pidinfo` reads).
#[cfg(target_os = "macos")]
fn switches(pid: u32) -> Vec<(String, u64)> {
    let out = Command::new("top")
        .args(["-l", "1", "-pid", &pid.to_string(), "-stats", "pid,csw"])
        .output()
        .expect("run top");
    let text = String::from_utf8_lossy(&out.stdout);
    let pid = pid.to_string();
    let csw = text
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next()? == pid).then(|| fields.next())?
        })
        .next()
        .expect("top lists the server's context switches");
    // Large counts carry a `+` suffix when top rounds them.
    let csw: u64 = csw
        .trim_end_matches('+')
        .parse()
        .expect("the CSW column is a count");
    vec![("process".to_string(), csw)]
}
