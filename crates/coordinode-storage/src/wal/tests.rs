use coordinode_core::txn::proposal::{Mutation, PartitionId};
use tempfile::TempDir;

use super::*;

fn make_mutations(n: usize) -> Vec<Mutation> {
    (0..n)
        .map(|i| Mutation::Put {
            partition: PartitionId::Node,
            key: format!("key:{i}").into_bytes(),
            value: format!("val:{i}").into_bytes(),
        })
        .collect()
}

// ── open / append / replay ────────────────────────────────────────────────

/// Fresh open produces empty replay and a usable WAL.
#[test]
fn fresh_open_no_replay() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");
    let (mut wal, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
    assert!(
        records.is_empty(),
        "fresh WAL should have no replay records"
    );
    // Append one record to verify WAL is operational.
    let lsn = wal.append(&make_mutations(3)).unwrap();
    assert_eq!(lsn, 0, "first record gets lsn 0");
}

/// Records appended to the WAL survive a close+reopen.
#[test]
fn append_and_replay() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");

    let mutations_a = make_mutations(2);
    let mutations_b = vec![Mutation::Delete {
        partition: PartitionId::Schema,
        key: b"schema:user".to_vec(),
    }];

    // Write two records.
    {
        let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
        wal.append(&mutations_a).unwrap();
        wal.append(&mutations_b).unwrap();
    }

    // Reopen — both records must replay.
    let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
    assert_eq!(records.len(), 2, "two records must replay");
    assert_eq!(records[0].lsn, 0);
    assert_eq!(records[0].mutations, mutations_a);
    assert_eq!(records[1].lsn, 1);
    assert_eq!(records[1].mutations, mutations_b);
}

/// LSN is monotonically increasing across appends.
#[test]
fn lsn_monotone() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");
    let (mut wal, _) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
    let lsns: Vec<u64> = (0..5)
        .map(|_| wal.append(&make_mutations(1)).unwrap())
        .collect();
    assert_eq!(lsns, vec![0, 1, 2, 3, 4]);
}

/// CRC mismatch in the middle stops replay at the corrupt record.
#[test]
fn crc_mismatch_stops_replay() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");

    // Write two valid records.
    {
        let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
        wal.append(&make_mutations(1)).unwrap();
        wal.append(&make_mutations(1)).unwrap();
    }

    // Corrupt the payload of the first record by flipping bytes in the payload
    // region (after the 16-byte header).
    {
        let mut data = std::fs::read(&path).unwrap();
        // First record header is 16 bytes, payload starts at byte 16.
        if data.len() > 17 {
            data[16] ^= 0xFF; // corrupt first byte of first payload
        }
        std::fs::write(&path, &data).unwrap();
    }

    // Replay should stop at the first corrupt record → 0 valid records.
    let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
    assert_eq!(records.len(), 0, "replay must stop at CRC mismatch");
}

/// Truncated tail (crash mid-write) is handled gracefully.
#[test]
fn truncated_tail_stops_replay() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");

    // Write one complete + one partial record.
    {
        let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
        wal.append(&make_mutations(1)).unwrap();
    }

    // Append a partial header (fewer than 16 bytes) simulating crash mid-write.
    {
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap(); // partial header
    }

    let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
    // First complete record must replay; partial header is silently discarded.
    assert_eq!(
        records.len(),
        1,
        "complete record before corrupt tail must replay"
    );
}

// ── checkpoint (rotation) ─────────────────────────────────────────────────

/// After checkpoint, the WAL file is empty and replay produces no records.
#[test]
fn checkpoint_clears_wal() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");

    let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
    wal.append(&make_mutations(3)).unwrap();

    // Simulate post-persist checkpoint.
    wal.checkpoint().unwrap();

    // After checkpoint: .old is gone, fresh .wal exists, no records to replay.
    let old_path = StandaloneWal::old_path_for(&path);
    assert!(!old_path.exists(), ".old must be deleted after checkpoint");

    let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
    assert!(
        records.is_empty(),
        "checkpointed WAL has no records to replay"
    );
}

/// Recovery after simulated crash between rotation rename and delete (.old exists).
#[test]
fn recovery_with_stale_old_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");
    let old_path = StandaloneWal::old_path_for(&path);

    // Simulate a .old file left by a crashed rotation (data already in SST).
    std::fs::write(&old_path, b"stale data - already in SST").unwrap();
    // No active WAL.
    assert!(!path.exists());

    // open() must delete .old and start fresh.
    let (_, records) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
    assert!(records.is_empty(), ".old contents must not be replayed");
    assert!(!old_path.exists(), ".old must be deleted on open");
    assert!(path.exists(), "fresh WAL must be created");
}

// ── delete_file ───────────────────────────────────────────────────────────

/// delete_file removes the WAL file.
#[test]
fn delete_file_removes_wal() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");

    {
        let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
        wal.append(&make_mutations(1)).unwrap();
    }
    assert!(path.exists());
    StandaloneWal::delete_file(&path).unwrap();
    assert!(!path.exists());
}

/// delete_file is a no-op when the file does not exist.
#[test]
fn delete_file_missing_is_noop() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("nonexistent.wal");
    assert!(StandaloneWal::delete_file(&path).is_ok());
}

// ── NoSync policy ─────────────────────────────────────────────────────────

/// NoSync mode appends records without fsync — records still replay correctly.
#[test]
fn nosync_append_and_replay() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");

    {
        let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::NoSync).unwrap();
        for _ in 0..10 {
            wal.append(&make_mutations(2)).unwrap();
        }
    }

    let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
    assert_eq!(records.len(), 10);
}

// ── Merge mutations ───────────────────────────────────────────────────────

/// Merge mutations (used for Adj posting lists) survive WAL round-trip.
#[test]
fn merge_mutation_survives_wal() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("standalone.wal");

    let merge = Mutation::Merge {
        partition: PartitionId::Adj,
        key: b"adj:KNOWS:out:42".to_vec(),
        operand: vec![1, 2, 3, 4],
    };

    {
        let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
        wal.append(std::slice::from_ref(&merge)).unwrap();
    }

    let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].mutations, vec![merge]);
}
