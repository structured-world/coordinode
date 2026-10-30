use super::*;

const INC: Incarnation = [7u8; 16];
const STD: lsm_tree::fs::StdFs = lsm_tree::fs::StdFs;

/// Write `payload` as a process that knows nothing of the copy.
fn write_fresh(dir: &Path, payload: &[u8]) -> StorageResult<()> {
    write_copy(
        &STD,
        dir,
        &mut CopyWriter::default(),
        &INC,
        payload,
        &Footprints::default(),
    )
}

fn written(payload: &[u8]) -> (tempfile::TempDir, Vec<u8>) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_fresh(dir.path(), payload).expect("write");
    let bytes = std::fs::read(slot_path(dir.path(), 0)).expect("read back");
    (dir, bytes)
}

fn held(payload: &[u8]) -> PromiseCopy {
    PromiseCopy::Held {
        incarnation: INC,
        payload: payload.to_vec(),
    }
}

/// A written copy reads back as the same payload and incarnation.
#[test]
fn a_written_copy_reads_back_whole() {
    let (dir, _) = written(b"vote");
    assert_eq!(read_copy(&STD, dir.path()).expect("read"), held(b"vote"));
}

/// No file is no copy, not an error.
#[test]
fn a_missing_copy_is_absent() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_eq!(
        read_copy(&STD, dir.path()).expect("read"),
        PromiseCopy::Absent
    );
}

/// Each new record is the copy, whichever slot it went to, over many writes
/// of one process and of processes that each write once.
#[test]
fn the_newest_record_is_the_copy() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut writer = CopyWriter::default();
    for i in 0..10u8 {
        let payload = [b'0' + i];
        write_copy(
            &STD,
            dir.path(),
            &mut writer,
            &INC,
            &payload,
            &Footprints::default(),
        )
        .expect("write");
        assert_eq!(read_copy(&STD, dir.path()).expect("read"), held(&payload));
    }
    for i in 0..10u8 {
        let payload = [b'a' + i];
        write_fresh(dir.path(), &payload).expect("write");
        assert_eq!(read_copy(&STD, dir.path()).expect("read"), held(&payload));
    }
}

/// A process writes a new slot first, and once a write succeeded it removes
/// all but two: the files do not pile up across reopens.
#[test]
fn spent_slots_are_removed() {
    let dir = tempfile::tempdir().expect("tempdir");
    for i in 0..5u8 {
        write_fresh(dir.path(), &[i]).expect("write");
        let files = copy_files(&STD, dir.path()).expect("list");
        assert!(files.len() <= 2, "after write {i}: {files:?}");
    }
    assert_eq!(read_copy(&STD, dir.path()).expect("read"), held(&[4]));
}

/// Every single damaged byte and every truncation is detected: a damaged
/// record is never read as some other promise.
#[test]
fn any_damaged_byte_or_truncation_is_detected() {
    let (_dir, bytes) = written(b"a promise of some length");
    for at in 0..bytes.len() {
        let mut flipped = bytes.clone();
        flipped[at] ^= 0x40;
        assert!(
            matches!(decode(&flipped), Slot::Damaged(_)),
            "byte {at} flipped"
        );
    }
    for len in 0..bytes.len() {
        assert!(
            matches!(decode(&bytes[..len]), Slot::Damaged(_)),
            "cut at {len}"
        );
    }
}

/// A torn write of a slot leaves the record before it as the copy, and the
/// next write leaves the slot holding that record alone.
#[test]
fn a_torn_slot_leaves_the_previous_record() {
    let (dir, _) = written(b"previous");
    std::fs::write(slot_path(dir.path(), 1), b"torn half of a new rec").expect("a torn slot");
    assert_eq!(
        read_copy(&STD, dir.path()).expect("read"),
        held(b"previous")
    );
    write_fresh(dir.path(), b"next").expect("write");
    assert_eq!(read_copy(&STD, dir.path()).expect("read"), held(b"next"));
    assert_eq!(
        std::fs::read(slot_path(dir.path(), 0)).expect("slot 0"),
        written_bytes(b"previous", 0),
        "the slot holding the newest whole record was not touched"
    );
}

/// The bytes of a record of `payload` at `sequence`, written by one process.
fn written_bytes(payload: &[u8], sequence: u64) -> Vec<u8> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut writer = CopyWriter::default();
    for i in 0..=sequence {
        let filler = if i == sequence { payload } else { b"x" };
        write_copy(
            &STD,
            dir.path(),
            &mut writer,
            &INC,
            filler,
            &Footprints::default(),
        )
        .expect("write");
    }
    // One process alternates between its first two slots.
    let target = u32::try_from(sequence % 2).expect("slot");
    std::fs::read(slot_path(dir.path(), target)).expect("read back")
}

/// A crash at each step of a write (opening the slot, writing it, syncing
/// it, and for a slot created now, syncing its directory) leaves a whole
/// record: the previous one, or the new one once it is synced. Never a torn
/// record taken for a promise, and never none. Both when the write creates
/// a slot and when it writes over an older one.
#[test]
fn a_crash_at_any_step_of_a_write_leaves_a_whole_record() {
    let mut reached = BTreeSet::new();
    for before in 1..=2usize {
        for op in FAULT_OPS {
            for skip in 0.. {
                let bench = Bench::new();
                let mut writer = Writer::default();
                for n in 0..before {
                    assert_eq!(bench.attempt(&mut writer, n, None), Outcome::Written);
                }
                let step = match bench.attempt(&mut writer, before, Some((op, skip))) {
                    Outcome::Failed(step) => step,
                    Outcome::Missed => break,
                    Outcome::Written => unreachable!("an armed write is failed or missed"),
                };
                reached.insert(step);
                bench.power_cut();
                let left = bench.index();
                // Only a failed directory sync comes after the new record
                // was synced: everything before leaves the previous one.
                let allowed = if step == "sync directory" {
                    before - 1..=before
                } else {
                    before - 1..=before - 1
                };
                assert!(
                    left.is_some_and(|n| allowed.contains(&n)),
                    "{before} written, {op:?} #{skip} failed {step}: the copy is p{left:?}"
                );
            }
        }
    }
    assert_eq!(
        reached,
        BTreeSet::from(STEPS),
        "every step of a write failed once"
    );
}

type Writer = CopyWriter;

fn put(fs: &dyn Fs, dir: &Path, writer: &mut Writer, payload: &[u8]) -> StorageResult<()> {
    write_copy(fs, dir, writer, &INC, payload, &Footprints::default())
}

use lsm_tree::fs::{CrashFs, Fault, FaultFs, FaultInjector, FaultOp, FaultRule, MemFs};
use std::collections::BTreeSet;

/// The operations a write of the copy fails at, besides listing the
/// directory (a failure there writes nothing) and removing spent slots (a
/// failure there fails no write): see their own tests.
const FAULT_OPS: [FaultOp; 5] = [
    FaultOp::Open,
    FaultOp::Read,
    FaultOp::Write,
    FaultOp::SyncAll,
    FaultOp::SyncDirectory,
];

/// Every step a write of the copy can fail at, as its error names it.
const STEPS: [&str; 6] = [
    "open to read",
    "read",
    "create",
    "write",
    "sync",
    "sync directory",
];

/// A fault: the operation, and how many operations of that kind pass first.
type Point = (FaultOp, u64);

/// What one attempt to write came to.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// The write reported success with no fault armed.
    Written,
    /// The armed fault was reached, at this step.
    Failed(&'static str),
    /// The write reported success before reaching the armed fault: the
    /// attempt failed nothing and is no case.
    Missed,
}

/// What a failed file sync leaves behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SyncFailure {
    /// The bytes stay dirty: a later sync of the file makes them durable.
    Dirty,
    /// The pages are marked clean without reaching the disk, as ext4 and XFS
    /// do after a failed writeback: reads still return the new bytes, and a
    /// later sync of the file reports success and makes nothing durable
    /// until the file is written again (Rebello et al., "Can Applications
    /// Recover from fsync Failures?", USENIX ATC 2020, sections 3.3-3.4).
    CleanStale,
}

/// Files whose last sync failed under [`SyncFailure::CleanStale`] and that
/// were not written since.
type StalePaths = std::sync::Arc<std::sync::Mutex<BTreeSet<PathBuf>>>;

/// The filesystem of a [`Bench`]: fault injection over a power-cut model,
/// with failed file syncs treated as `failure` says, and files changed since
/// their last effective sync torn by a power cut (see [`Bench::crash`]).
struct ModelFs {
    inner: FaultFs<CrashFs>,
    failure: SyncFailure,
    stale: StalePaths,
    /// Files changed since their last effective sync.
    unsynced: StalePaths,
    /// Called before each removal: a point where a test can look at what a
    /// write holds before it lets go of its spent slots.
    before_remove: Option<Probe>,
}

/// What a test runs at a point inside a write.
type Probe = std::sync::Arc<dyn Fn() + Send + Sync>;

/// A file opened through a [`ModelFs`].
struct ModelFile {
    inner: Box<dyn lsm_tree::fs::FsFile>,
    path: PathBuf,
    failure: SyncFailure,
    stale: StalePaths,
    unsynced: StalePaths,
}

impl ModelFile {
    fn written(&self) {
        self.stale.lock().expect("stale paths").remove(&self.path);
        self.unsynced
            .lock()
            .expect("unsynced paths")
            .insert(self.path.clone());
    }
}

impl std::io::Read for ModelFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl std::io::Write for ModelFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        if n > 0 {
            self.written();
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl std::io::Seek for ModelFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl lsm_tree::fs::FsFile for ModelFile {
    fn sync_all(&self) -> lsm_tree::io::Result<()> {
        if self.stale.lock().expect("stale paths").contains(&self.path) {
            // Clean pages: nothing to write back, and no error to report.
            return Ok(());
        }
        let synced = self.inner.sync_all();
        if synced.is_ok() {
            self.unsynced
                .lock()
                .expect("unsynced paths")
                .remove(&self.path);
        } else if self.failure == SyncFailure::CleanStale {
            self.stale
                .lock()
                .expect("stale paths")
                .insert(self.path.clone());
        }
        synced
    }

    fn sync_data(&self) -> lsm_tree::io::Result<()> {
        self.sync_all()
    }

    fn metadata(&self) -> lsm_tree::io::Result<lsm_tree::fs::FsMetadata> {
        self.inner.metadata()
    }

    fn set_len(&self, size: u64) -> lsm_tree::io::Result<()> {
        self.inner.set_len(size)?;
        self.written();
        Ok(())
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> lsm_tree::io::Result<usize> {
        self.inner.read_at(buf, offset)
    }

    fn lock_exclusive(&self) -> lsm_tree::io::Result<()> {
        self.inner.lock_exclusive()
    }
}

impl Fs for ModelFs {
    fn open(
        &self,
        path: &Path,
        opts: &FsOpenOptions,
    ) -> lsm_tree::io::Result<Box<dyn lsm_tree::fs::FsFile>> {
        let inner = self.inner.open(path, opts)?;
        let file = ModelFile {
            inner,
            path: path.to_path_buf(),
            failure: self.failure,
            stale: std::sync::Arc::clone(&self.stale),
            unsynced: std::sync::Arc::clone(&self.unsynced),
        };
        // A truncation changes the file as a write does.
        if opts.truncate {
            file.written();
        }
        Ok(Box::new(file))
    }

    fn create_dir_all(&self, path: &Path) -> lsm_tree::io::Result<()> {
        self.inner.create_dir_all(path)
    }

    fn read_dir(&self, path: &Path) -> lsm_tree::io::Result<Vec<lsm_tree::fs::FsDirEntry>> {
        self.inner.read_dir(path)
    }

    fn remove_file(&self, path: &Path) -> lsm_tree::io::Result<()> {
        if let Some(probe) = &self.before_remove {
            probe();
        }
        self.inner.remove_file(path)?;
        self.stale.lock().expect("stale paths").remove(path);
        self.unsynced.lock().expect("unsynced paths").remove(path);
        Ok(())
    }

    fn remove_dir_all(&self, path: &Path) -> lsm_tree::io::Result<()> {
        self.inner.remove_dir_all(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> lsm_tree::io::Result<()> {
        self.inner.rename(from, to)
    }

    fn metadata(&self, path: &Path) -> lsm_tree::io::Result<lsm_tree::fs::FsMetadata> {
        self.inner.metadata(path)
    }

    fn sync_directory(&self, path: &Path) -> lsm_tree::io::Result<()> {
        self.inner.sync_directory(path)
    }

    fn exists(&self, path: &Path) -> lsm_tree::io::Result<bool> {
        self.inner.exists(path)
    }
}

/// An endpoint in memory under power-cut and fault injection: the cases run
/// in seconds, not in the minutes real syncs take.
struct Bench {
    crash: std::sync::Arc<CrashFs>,
    faults: std::sync::Arc<FaultInjector>,
    fs: ModelFs,
}

impl Bench {
    const DIR: &'static str = "/endpoint";

    fn new() -> Self {
        Self::with(SyncFailure::Dirty)
    }

    fn with(failure: SyncFailure) -> Self {
        let crash = std::sync::Arc::new(CrashFs::new(MemFs::new()));
        let faults = std::sync::Arc::new(FaultInjector::new());
        let fs = ModelFs {
            inner: FaultFs::with_injector(CrashFs::clone(&crash), std::sync::Arc::clone(&faults)),
            failure,
            stale: StalePaths::default(),
            unsynced: StalePaths::default(),
            before_remove: None,
        };
        fs.create_dir_all(Path::new(Self::DIR))
            .expect("the endpoint");
        fs.sync_directory(Path::new("/"))
            .expect("the endpoint's name");
        Self { crash, faults, fs }
    }

    /// Cut the power. Every file changed since its last effective sync and
    /// still named on the disk keeps only the first half of what it read
    /// before: some of its pages were written back, some were not. That is
    /// the worst a write in place can leave, worse than losing it whole.
    fn power_cut(&self) {
        let changed: Vec<PathBuf> = self
            .fs
            .unsynced
            .lock()
            .expect("unsynced paths")
            .iter()
            .cloned()
            .collect();
        let mut torn = Vec::new();
        for path in changed {
            if let Some(bytes) = read_file(&self.fs, &path).expect("read before the cut") {
                torn.push((path, bytes));
            }
        }
        self.crash.crash();
        self.fs.stale.lock().expect("stale paths").clear();
        self.fs.unsynced.lock().expect("unsynced paths").clear();
        for (path, bytes) in torn {
            if !self.fs.inner.exists(&path).expect("exists") {
                continue;
            }
            let mut file = self
                .fs
                .inner
                .open(&path, &FsOpenOptions::new().write(true).truncate(true))
                .expect("tear");
            file.write_all(&bytes[..bytes.len() / 2]).expect("tear");
            file.sync_all().expect("tear");
        }
    }

    /// Write `p<n>` through `writer`, failing the operation at `point`.
    fn attempt(&self, writer: &mut Writer, n: usize, point: Option<Point>) -> Outcome {
        if let Some((op, skip)) = point {
            self.faults.arm(
                FaultRule::new(op, Fault::Error(lsm_tree::io::ErrorKind::Other))
                    .skip(skip)
                    .once(),
            );
        }
        let done = put(
            &self.fs,
            Path::new(Self::DIR),
            writer,
            format!("p{n}").as_bytes(),
        );
        self.faults.clear();
        if point.is_none() {
            done.expect("a write with no fault armed succeeds");
            return Outcome::Written;
        }
        match done {
            Ok(()) => Outcome::Missed,
            Err(e) => Outcome::Failed(step_of(&e)),
        }
    }

    /// The number of the record the copy holds, `None` for none.
    fn index(&self) -> Option<usize> {
        match read_copy(&self.fs, Path::new(Self::DIR)).expect("read the copy") {
            PromiseCopy::Held { payload, .. } => std::str::from_utf8(&payload[1..])
                .ok()
                .and_then(|n| n.parse().ok()),
            PromiseCopy::Absent | PromiseCopy::Damaged(_) => None,
        }
    }
}

/// The step a failed write stopped at, as its error names it.
fn step_of(e: &StorageError) -> &'static str {
    let message = match e {
        StorageError::Io(message) => message.as_str(),
        _ => "",
    };
    // Longest first: "sync directory" starts like "sync".
    let mut steps = STEPS;
    steps.sort_by_key(|step| std::cmp::Reverse(step.len()));
    steps
        .into_iter()
        .find(|step| {
            message
                .strip_prefix(step)
                .is_some_and(|rest| rest.starts_with(' '))
        })
        .ok_or_else(|| e.to_string())
        .expect("a failed write is an I/O error naming its step")
}

/// One chain: writes that succeed, a write failed at `first`, a retry of it
/// (or none) after an optional reopen without a power loss, and a next write
/// after another optional reopen, then a power cut.
#[derive(Clone, Copy, Debug)]
struct Chain {
    failure: SyncFailure,
    before: usize,
    first: Point,
    reopen: bool,
    retry: Option<Option<Point>>,
    reopen_again: bool,
    next: Option<Point>,
}

/// Run `chain`, recording which step each fault failed at; `false` when a
/// fault it arms is never reached, so the chain is no case of its own.
fn run(chain: Chain, reached: &mut BTreeSet<(&'static str, &'static str)>) -> bool {
    let bench = Bench::with(chain.failure);
    let mut writer = Writer::default();
    let mut last_success = 0;
    let mut step =
        |writer: &mut Writer, n: usize, point: Option<Point>, at: &'static str| match bench
            .attempt(writer, n, point)
        {
            Outcome::Written => {
                last_success = n;
                true
            }
            Outcome::Failed(step) => {
                reached.insert((at, step));
                true
            }
            Outcome::Missed => false,
        };
    for n in 0..chain.before {
        assert!(step(&mut writer, n, None, "before"));
    }
    if !step(&mut writer, chain.before, Some(chain.first), "first") {
        return false;
    }
    if chain.reopen {
        writer = Writer::default();
    }
    if let Some(point) = chain.retry
        && !step(&mut writer, chain.before, point, "retry")
    {
        return false;
    }
    if chain.reopen_again {
        writer = Writer::default();
    }
    if !step(&mut writer, chain.before + 1, chain.next, "next") {
        return false;
    }
    bench.power_cut();
    let left = bench.index();
    assert!(
        left.is_some_and(|n| n >= last_success),
        "{chain:?}: the last success was p{last_success}, the copy is p{left:?}"
    );
    true
}

/// Every chain of a write failed at any reachable operation, its retry (in
/// the same process, after a reopen without a power loss, or none) failed at
/// any reachable operation or not at all, and the next write likewise,
/// ending in a power cut: the copy is never older than the last write that
/// reported success. A failure that leaves a slot name undurable must not let
/// a later write trust it, and no write after a failure may go over the only
/// durable record, whichever slots the earlier writes took. Every step of
/// every write in the chain is shown failed at least once. Both with a failed
/// sync leaving its bytes dirty and with it leaving them clean but not on
/// disk, where no later sync without a new write makes them durable.
#[test]
fn a_crash_after_a_failed_write_and_its_retry_keeps_the_last_success() {
    let mut reached = BTreeSet::new();
    for (failure, before) in [SyncFailure::Dirty, SyncFailure::CleanStale]
        .into_iter()
        .flat_map(|failure| (1..=3usize).map(move |before| (failure, before)))
    {
        for reopen in [false, true] {
            for reopen_again in [false, true] {
                for op1 in FAULT_OPS {
                    for k1 in 0.. {
                        let base = Chain {
                            failure,
                            before,
                            first: (op1, k1),
                            reopen,
                            retry: None,
                            reopen_again,
                            next: None,
                        };
                        if !run(base, &mut reached) {
                            break;
                        }
                        let mut retries = vec![None, Some(None)];
                        for op2 in FAULT_OPS {
                            for k2 in 0.. {
                                let retry = Some(Some((op2, k2)));
                                if !run(Chain { retry, ..base }, &mut reached) {
                                    break;
                                }
                                retries.push(retry);
                            }
                        }
                        for retry in retries {
                            assert!(run(Chain { retry, ..base }, &mut reached));
                            for op3 in FAULT_OPS {
                                for k3 in 0.. {
                                    let next = Some((op3, k3));
                                    if !run(
                                        Chain {
                                            retry,
                                            next,
                                            ..base
                                        },
                                        &mut reached,
                                    ) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for at in ["first", "retry", "next"] {
        let seen: BTreeSet<&str> = reached
            .iter()
            .filter(|(position, _)| *position == at)
            .map(|(_, step)| *step)
            .collect();
        assert_eq!(
            seen,
            BTreeSet::from(STEPS),
            "the {at} write failed at every step"
        );
    }
}

/// A directory that cannot be listed fails the write before anything is
/// written: no slot is read or written blind.
#[test]
fn a_copy_that_cannot_be_listed_is_not_written() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A file where the endpoint's directory should be: listing it fails.
    let not_a_directory = dir.path().join("endpoint");
    std::fs::write(&not_a_directory, b"a file").expect("a file in the way");
    let refused = write_copy(
        &STD,
        &not_a_directory,
        &mut CopyWriter::default(),
        &INC,
        b"vote",
        &Footprints::default(),
    );
    assert!(
        matches!(&refused, Err(StorageError::Io(message)) if message.starts_with("list ")),
        "{refused:?}"
    );
    assert_eq!(
        std::fs::read(&not_a_directory).expect("the file"),
        b"a file",
        "nothing was written"
    );
}

/// A spent slot that cannot be removed after a write stays: the write still
/// succeeds, the copy is the record it wrote, a power cut keeps it, and a
/// later write of the same process removes the slot.
#[test]
fn a_spent_slot_that_cannot_be_removed_stays_until_a_later_write() {
    let bench = Bench::new();
    let dir = Path::new(Bench::DIR);
    let files = || copy_files(&bench.fs, dir).expect("list").len();
    // Three slots: one process's record, a second process's write that
    // failed after creating its slot, and a third process's first write.
    assert_eq!(
        bench.attempt(&mut Writer::default(), 0, None),
        Outcome::Written
    );
    assert_eq!(
        bench.attempt(&mut Writer::default(), 1, Some((FaultOp::SyncAll, 0))),
        Outcome::Failed("sync")
    );
    assert_eq!(files(), 2);
    let mut writer = Writer::default();
    bench.faults.arm(FaultRule::new(
        FaultOp::RemoveFile,
        Fault::Error(lsm_tree::io::ErrorKind::Other),
    ));
    let written = put(&bench.fs, dir, &mut writer, b"p2");
    bench.faults.clear();
    written.expect("a failed removal fails no write");
    assert_eq!(files(), 3, "the removal was refused and the slot stayed");
    assert_eq!(bench.index(), Some(2));
    assert_eq!(bench.attempt(&mut writer, 3, None), Outcome::Written);
    assert_eq!(files(), 2, "the next write removed the spent slot");
    bench.power_cut();
    assert_eq!(bench.index(), Some(3));
}

/// The copy's high water: its slot files and the bytes charged for them,
/// sampled inside each write at the last point before spent slots go, and
/// after every write. A process that has proved nothing writes a new slot,
/// so each reopen whose first write fails leaves one more slot, and the
/// first success writes one more still before it removes all but two. The
/// peak is the slots the store held, plus the opens that failed in a row,
/// plus one: three on a plain restart of a store in its steady two. The
/// charge is the footprint's account in whole allocation units, not what a
/// native file system reports. A removal that fails holds the peak until a
/// later write of the same process.
#[test]
fn the_copy_peaks_one_slot_above_what_it_held_and_the_opens_that_failed() {
    use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
    for (initial_slots, failed_opens, removal_fails) in [
        (1, 0, false),
        (1, 1, false),
        (1, 3, false),
        (1, 3, true),
        (2, 0, false),
        (2, 0, true),
        (2, 1, false),
        (2, 3, false),
        (2, 3, true),
    ] {
        let case = format!(
            "{initial_slots} slots held, {failed_opens} failed opens, removal fails \
             {removal_fails}"
        );
        let root = tempfile::tempdir().expect("tempdir");
        let dir = root.path().to_path_buf();
        let footprints = Footprints::open(&[EndpointConfig::new(
            "default",
            &dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )
        .with_hard_limit_bytes(1 << 20)])
        .expect("footprints");
        let footprint = std::sync::Arc::clone(footprints.for_path(&dir).expect("capped"));
        let unit = footprint.allocation(1);
        let crash = std::sync::Arc::new(CrashFs::new(MemFs::new()));
        let faults = std::sync::Arc::new(FaultInjector::new());
        let high = std::sync::Arc::new(std::sync::Mutex::new((0usize, 0u64)));
        let observe: Probe = {
            let crash = CrashFs::clone(&crash);
            let footprint = std::sync::Arc::clone(&footprint);
            let high = std::sync::Arc::clone(&high);
            let dir = dir.clone();
            std::sync::Arc::new(move || {
                let files = copy_files(&crash, &dir).expect("list the slots").len();
                let mut high = high.lock().expect("high water");
                high.0 = high.0.max(files);
                high.1 = high.1.max(footprint.charged());
            })
        };
        let fs = ModelFs {
            inner: FaultFs::with_injector(CrashFs::clone(&crash), std::sync::Arc::clone(&faults)),
            failure: SyncFailure::Dirty,
            stale: StalePaths::default(),
            unsynced: StalePaths::default(),
            before_remove: Some(std::sync::Arc::clone(&observe)),
        };
        fs.create_dir_all(&dir).expect("the endpoint");
        let write = |writer: &mut CopyWriter, payload: &[u8]| {
            let written = write_copy(&fs, &dir, writer, &INC, payload, &footprints);
            observe();
            written
        };
        // One process writes the slots the store holds.
        let mut first = CopyWriter::default();
        for _ in 0..initial_slots {
            write(&mut first, b"p0").expect("an earlier copy");
        }
        assert_eq!(
            copy_files(&fs, &dir).expect("list").len(),
            initial_slots,
            "{case}: the slots held"
        );
        for _ in 0..failed_opens {
            faults.arm(
                FaultRule::new(
                    FaultOp::SyncAll,
                    Fault::Error(lsm_tree::io::ErrorKind::Other),
                )
                .once(),
            );
            let failed = write(&mut CopyWriter::default(), b"failed");
            faults.clear();
            assert!(failed.is_err(), "{case}: the open's write failed");
        }
        let mut writer = CopyWriter::default();
        if removal_fails {
            faults.arm(FaultRule::new(
                FaultOp::RemoveFile,
                Fault::Error(lsm_tree::io::ErrorKind::Other),
            ));
        }
        write(&mut writer, b"p1").expect("the first success");
        faults.clear();
        let peak = initial_slots + failed_opens + 1;
        assert_eq!(
            *high.lock().expect("high water"),
            (peak, peak as u64 * unit),
            "{case}: the high water"
        );
        let held = copy_files(&fs, &dir).expect("list").len();
        assert_eq!(
            held,
            if removal_fails { peak } else { peak.min(2) },
            "{case}: slots after the first success"
        );
        write(&mut writer, b"p2").expect("a later write");
        assert_eq!(
            copy_files(&fs, &dir).expect("list").len(),
            2,
            "{case}: settled"
        );
        assert_eq!(footprint.charged(), 2 * unit, "{case}: settled charge");
    }
}

/// An endpoint of 64 KiB whose recovery reserve is a quarter of it, the copy
/// already written once, so its first slot is charged.
fn capped_endpoint() -> (tempfile::TempDir, Footprints) {
    use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
    let dir = tempfile::tempdir().expect("tempdir");
    write_fresh(dir.path(), b"given").expect("first copy");
    let endpoint = EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(64 * 1024)
    .with_recovery_reserve_bytes(16 * 1024);
    let footprints = Footprints::open(&[endpoint]).expect("footprints");
    (dir, footprints)
}

/// Client writes that fill an endpoint to what they may take leave the
/// recovery reserve, and a new slot of the copy lands in it: a store full of
/// data still restarts and keeps its promise.
#[test]
fn a_copy_write_takes_the_room_client_writes_leave() {
    use crate::engine::footprint::Admit;
    let (dir, footprints) = capped_endpoint();
    let footprint = footprints.for_path(dir.path()).expect("capped");
    let client_room = footprint.client_ceiling() - footprint.committed();
    let _clients = footprint
        .admit(client_room, Admit::Client)
        .expect("client writes fill what they may");
    assert!(
        footprint.admit(1, Admit::Client).is_err(),
        "clients are full"
    );
    write_copy(
        &STD,
        dir.path(),
        &mut CopyWriter::default(),
        &INC,
        b"next",
        &footprints,
    )
    .expect("the copy lands in the recovery reserve");
    assert_eq!(read_copy(&STD, dir.path()).expect("read"), held(b"next"));
}

/// Room already promised (to admitted writes, and to the memtables'
/// flushes) is not the copy's: a write that would take it is refused before
/// anything is written, as every recovery record is, and the previous record
/// stays the copy.
#[test]
fn a_copy_write_never_takes_promised_room() {
    use crate::engine::footprint::Admit;
    let (dir, footprints) = capped_endpoint();
    let footprint = footprints.for_path(dir.path()).expect("capped");
    let all_room = footprint.limit() - footprint.committed();
    let _promised = footprint
        .admit(all_room, Admit::Recovery)
        .expect("the whole limit promised");
    let refused = write_copy(
        &STD,
        dir.path(),
        &mut CopyWriter::default(),
        &INC,
        b"next",
        &footprints,
    );
    assert!(
        matches!(refused, Err(StorageError::CapacityExhausted { .. })),
        "{refused:?}"
    );
    assert_eq!(read_copy(&STD, dir.path()).expect("read"), held(b"given"));
    assert_eq!(
        copy_files(&STD, dir.path()).expect("list").len(),
        1,
        "no slot was created"
    );
}

/// A write the endpoint has no room for is refused before anything is
/// written, and the copy stays whole: a full disk never costs a promise
/// already given.
#[test]
fn a_write_without_room_keeps_the_previous_copy() {
    use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
    let dir = tempfile::tempdir().expect("tempdir");
    write_fresh(dir.path(), b"given").expect("first copy");
    // Room for the slot in place, charged by allocation unit, and not for
    // the new slot a reopened process writes first.
    let unit = 4096;
    let endpoint = EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(unit + unit / 2);
    let footprints = Footprints::open(&[endpoint]).expect("footprints");
    let refused = write_copy(
        &STD,
        dir.path(),
        &mut CopyWriter::default(),
        &INC,
        b"a larger promise",
        &footprints,
    );
    assert!(
        matches!(refused, Err(StorageError::CapacityExhausted { .. })),
        "{refused:?}"
    );
    assert_eq!(read_copy(&STD, dir.path()).expect("read"), held(b"given"));
}

/// A marker's incarnation survives its text form; an empty marker (written
/// before markers carried one) has none, and anything else is refused.
#[test]
fn an_incarnation_survives_its_marker_text() {
    let incarnation: Incarnation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xff];
    let text = format_incarnation(&incarnation);
    assert_eq!(parse_incarnation(&text).expect("parse"), Some(incarnation));
    assert_eq!(parse_incarnation("").expect("empty"), None);
    assert!(parse_incarnation("not hex").is_err());
    assert!(parse_incarnation(&"z".repeat(32)).is_err());
    // Thirty-two bytes that are not thirty-two ASCII characters: refused,
    // not split inside a character.
    let damaged = format!("a\u{e9}{}", "a".repeat(29));
    assert_eq!(damaged.len(), 32);
    assert!(parse_incarnation(&damaged).is_err());
}
