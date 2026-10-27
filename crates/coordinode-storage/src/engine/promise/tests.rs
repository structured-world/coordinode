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
/// the second slot and when it overwrites the older one.
#[test]
fn a_crash_at_any_step_of_a_write_leaves_a_whole_record() {
    use lsm_tree::fs::{CrashFs, Fault, FaultFs, FaultInjector, FaultOp, FaultRule, StdFs};
    let cases: [(&[&[u8]], FaultOp, bool); 7] = [
        (&[b"previous"], FaultOp::Open, true),
        (&[b"previous"], FaultOp::Write, true),
        (&[b"previous"], FaultOp::SyncAll, true),
        (&[b"previous"], FaultOp::SyncDirectory, false),
        (&[b"oldest", b"previous"], FaultOp::Open, true),
        (&[b"oldest", b"previous"], FaultOp::Write, true),
        (&[b"oldest", b"previous"], FaultOp::SyncAll, true),
    ];
    for (before, op, previous_only) in cases {
        let dir = tempfile::tempdir().expect("tempdir");
        let crash = std::sync::Arc::new(CrashFs::new(StdFs));
        let faults = std::sync::Arc::new(FaultInjector::new());
        let fs = FaultFs::with_injector(CrashFs::clone(&crash), std::sync::Arc::clone(&faults));
        let mut writer = CopyWriter::default();
        for payload in before {
            put(&fs, dir.path(), &mut writer, payload).expect("before");
        }
        faults.arm(FaultRule::new(
            op,
            Fault::Error(lsm_tree::io::ErrorKind::Other),
        ));
        let cut = put(&fs, dir.path(), &mut writer, b"next");
        assert!(
            cut.is_err(),
            "{op:?} after {}: the write reported success",
            before.len()
        );
        faults.clear();
        crash.crash();
        let left = read_copy(&fs, dir.path()).expect("read after the crash");
        if previous_only {
            assert_eq!(left, held(b"previous"), "{op:?} after {}", before.len());
        } else {
            assert!(
                left == held(b"previous") || left == held(b"next"),
                "{op:?} after {}: {left:?}",
                before.len()
            );
        }
    }
}

type Writer = CopyWriter;

fn put(fs: &dyn Fs, dir: &Path, writer: &mut Writer, payload: &[u8]) -> StorageResult<()> {
    write_copy(fs, dir, writer, &INC, payload, &Footprints::default())
}

/// Chains of a failed write, its retry (in the same process, or after a
/// reopen without a power loss) and the next write (in the same process, or
/// after one more such reopen), ending in a power cut:
/// the copy is never older than the last write that reported success. A
/// failed directory sync must not leave a later write trusting a slot name
/// that may not be durable, and a retry after a failed sync must not write
/// over the only durable record, whichever slots the earlier writes took.
#[test]
fn a_crash_after_a_failed_write_and_its_retry_keeps_the_last_success() {
    use lsm_tree::fs::{CrashFs, Fault, FaultFs, FaultInjector, FaultOp, FaultRule, MemFs};
    let failing = [
        FaultOp::Open,
        FaultOp::Write,
        FaultOp::SyncAll,
        FaultOp::SyncDirectory,
    ];
    let any = [
        None,
        Some(FaultOp::Open),
        Some(FaultOp::Write),
        Some(FaultOp::SyncAll),
        Some(FaultOp::SyncDirectory),
    ];
    let index = |copy: &PromiseCopy| match copy {
        PromiseCopy::Held { payload, .. } => std::str::from_utf8(&payload[1..])
            .ok()
            .and_then(|n| n.parse::<usize>().ok()),
        PromiseCopy::Absent | PromiseCopy::Damaged(_) => None,
    };
    for before in 1..=3usize {
        for first in failing {
            for reopen in [false, true] {
                for retry in any.iter().map(|op| Some(*op)).chain([None]) {
                    for (next, reopen_again) in
                        any.iter().flat_map(|op| [(*op, false), (*op, true)])
                    {
                        let case = format!(
                            "{before} written, {first:?} fails, reopen {reopen}, retry \
                             {retry:?}, reopen again {reopen_again}, next {next:?}"
                        );
                        // In memory: the cases run in a second, not in the
                        // minute real syncs take.
                        let dir = Path::new("/endpoint");
                        let crash = std::sync::Arc::new(CrashFs::new(MemFs::new()));
                        let faults = std::sync::Arc::new(FaultInjector::new());
                        let fs = FaultFs::with_injector(
                            CrashFs::clone(&crash),
                            std::sync::Arc::clone(&faults),
                        );
                        fs.create_dir_all(dir).expect("the endpoint");
                        fs.sync_directory(Path::new("/"))
                            .expect("the endpoint's name");
                        let mut writer = Writer::default();
                        let mut last_success = 0;
                        let mut attempt = |writer: &mut Writer, n: usize, op: Option<FaultOp>| {
                            if let Some(op) = op {
                                faults.arm(FaultRule::new(
                                    op,
                                    Fault::Error(lsm_tree::io::ErrorKind::Other),
                                ));
                            }
                            let done = put(&fs, dir, writer, format!("p{n}").as_bytes());
                            faults.clear();
                            if done.is_ok() {
                                last_success = n;
                            }
                        };
                        for n in 0..before {
                            attempt(&mut writer, n, None);
                        }
                        attempt(&mut writer, before, Some(first));
                        if reopen {
                            writer = Writer::default();
                        }
                        if let Some(op) = retry {
                            attempt(&mut writer, before, op);
                        }
                        if reopen_again {
                            writer = Writer::default();
                        }
                        attempt(&mut writer, before + 1, next);
                        crash.crash();
                        let left = read_copy(&fs, dir).expect("read after the crash");
                        assert!(
                            index(&left).is_some_and(|n| n >= last_success),
                            "{case}: the last success was p{last_success}, the copy is {left:?}"
                        );
                    }
                }
            }
        }
    }
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
