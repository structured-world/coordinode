//! A read of the Raft log's segments waits on the disk without holding the
//! log's writer or a worker of the runtime: the leader reads whole sealed
//! segments to catch a member up, and that read held the manager its own
//! appends are written through, and the runtime thread it ran on, for as
//! long as the disk took.

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use lsm_tree::fs::{Fs, FsDirEntry, FsFile, FsMetadata, FsOpenOptions, StdFs};
use parking_lot::{Condvar, Mutex as PlMutex};

use super::*;

/// How long a step that should go on at once is given.
const PROMPTLY: Duration = Duration::from_secs(5);

/// Holds every read of a log segment while shut, counting the reads held.
#[derive(Default)]
struct Gate {
    /// Whether it is shut, and how many reads it held so far.
    state: PlMutex<(bool, usize)>,
    changed: Condvar,
}

impl Gate {
    fn shut(&self) {
        self.state.lock().0 = true;
    }

    fn open(&self) {
        self.state.lock().0 = false;
        self.changed.notify_all();
    }

    /// Wait here while shut.
    fn pass(&self) {
        let mut state = self.state.lock();
        if state.0 {
            state.1 += 1;
            self.changed.notify_all();
            while state.0 {
                self.changed.wait(&mut state);
            }
        }
    }

    /// Wait until a read is held, for at most `PROMPTLY`.
    fn wait_held(&self) -> bool {
        let mut state = self.state.lock();
        let deadline = std::time::Instant::now() + PROMPTLY;
        while state.1 == 0 {
            if self.changed.wait_until(&mut state, deadline).timed_out() {
                return state.1 > 0;
            }
        }
        true
    }
}

/// The standard file system, with every read of a log segment opened for
/// reading only held at `gate`: a disk that answers those reads late.
struct HeldReads {
    gate: Arc<Gate>,
}

impl Fs for HeldReads {
    fn open(&self, path: &Path, opts: &FsOpenOptions) -> lsm_tree::io::Result<Box<dyn FsFile>> {
        let file = StdFs.open(path, opts)?;
        let segment = path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("oplog-"));
        if segment && opts.read && !opts.write {
            return Ok(Box::new(HeldFile {
                inner: file,
                gate: Arc::clone(&self.gate),
            }));
        }
        Ok(file)
    }

    fn create_dir_all(&self, path: &Path) -> lsm_tree::io::Result<()> {
        StdFs.create_dir_all(path)
    }

    fn read_dir(&self, path: &Path) -> lsm_tree::io::Result<Vec<FsDirEntry>> {
        StdFs.read_dir(path)
    }

    fn remove_file(&self, path: &Path) -> lsm_tree::io::Result<()> {
        StdFs.remove_file(path)
    }

    fn remove_dir_all(&self, path: &Path) -> lsm_tree::io::Result<()> {
        StdFs.remove_dir_all(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> lsm_tree::io::Result<()> {
        StdFs.rename(from, to)
    }

    fn metadata(&self, path: &Path) -> lsm_tree::io::Result<FsMetadata> {
        StdFs.metadata(path)
    }

    fn sync_directory(&self, path: &Path) -> lsm_tree::io::Result<()> {
        StdFs.sync_directory(path)
    }

    fn exists(&self, path: &Path) -> lsm_tree::io::Result<bool> {
        StdFs.exists(path)
    }
}

/// A segment whose reads wait at the gate.
struct HeldFile {
    inner: Box<dyn FsFile>,
    gate: Arc<Gate>,
}

impl std::io::Read for HeldFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.gate.pass();
        self.inner.read(buf)
    }
}

impl std::io::Write for HeldFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl std::io::Seek for HeldFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl FsFile for HeldFile {
    fn sync_all(&self) -> lsm_tree::io::Result<()> {
        self.inner.sync_all()
    }

    fn sync_data(&self) -> lsm_tree::io::Result<()> {
        self.inner.sync_data()
    }

    fn metadata(&self) -> lsm_tree::io::Result<FsMetadata> {
        self.inner.metadata()
    }

    fn set_len(&self, size: u64) -> lsm_tree::io::Result<()> {
        self.inner.set_len(size)
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> lsm_tree::io::Result<usize> {
        self.gate.pass();
        self.inner.read_at(buf, offset)
    }

    fn lock_exclusive(&self) -> lsm_tree::io::Result<()> {
        self.inner.lock_exclusive()
    }
}

/// While a read of the log's segments waits on the disk, the runtime's only
/// worker runs other tasks and the log takes an append and makes it
/// durable; once the disk answers, the read returns what the segments hold.
#[test]
fn a_segment_read_holds_neither_the_writer_nor_a_worker() {
    let gate = Arc::new(Gate::default());
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
    .with_fs(Arc::new(HeldReads {
        gate: Arc::clone(&gate),
    }));
    // One worker: a read that waits on it holds the whole runtime.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("runtime");
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    runtime.block_on(async {
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open log");
        append_durable(
            &mut store,
            vec![make_entry(1, 1, "a"), make_entry(2, 1, "b")],
        )
        .await;
    });
    // Reopened, the log holds no entry in memory: a read goes to the segments.
    let mut store = LogStore::open(Arc::clone(&engine)).expect("reopen log");

    gate.shut();
    let mut reader = runtime.block_on(store.get_log_reader());
    let (read_tx, read_rx) = mpsc::channel();
    runtime.spawn(async move {
        read_tx
            .send(reader.try_get_log_entries(1..3).await)
            .expect("send the read");
    });
    assert!(gate.wait_held(), "the read reaches the segments");

    let (ran_tx, ran_rx) = mpsc::channel();
    runtime.spawn(async move { ran_tx.send(()).expect("send") });
    let worker_free = ran_rx.recv_timeout(PROMPTLY).is_ok();

    let (durable_tx, durable_rx) = mpsc::channel();
    runtime.spawn(async move {
        let (tx, rx) = <TypeConfig as openraft::type_config::TypeConfigExt>::oneshot();
        store
            .append(vec![make_entry(3, 1, "c")], IOFlushed::signal(tx))
            .await
            .expect("append");
        durable_tx
            .send(rx.await.expect("answered"))
            .expect("send durability");
    });
    let appended = durable_rx.recv_timeout(PROMPTLY);

    // Let the read finish before any verdict, so a failure leaves no thread
    // waiting at the gate.
    gate.open();
    let read = read_rx.recv_timeout(PROMPTLY);
    assert!(worker_free, "the read held the runtime's only worker");
    assert!(
        matches!(appended, Ok(Ok(()))),
        "the log took no append while its segments were read: {appended:?}"
    );
    let indexes: Vec<u64> = read
        .expect("the read returns once the disk answers")
        .expect("read")
        .iter()
        .map(|e| e.log_id.index)
        .collect();
    assert_eq!(indexes, [1, 2]);
}
