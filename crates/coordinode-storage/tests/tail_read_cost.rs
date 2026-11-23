//! What a change-stream page costs its reader: the entries it returns, not
//! the rest of the segment. A reader far behind the end of a large segment
//! reads its pages one after another, and each must read from its position
//! up to the page's last entry, never the segment's whole remainder. A binary
//! of its own, since it replaces the global allocator.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};

use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};
use tracking_allocator::{
    AllocationGroupId, AllocationGroupToken, AllocationRegistry, AllocationTracker,
};

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

/// Bytes allocated inside the counted group (the page reads).
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct Count;

impl AllocationTracker for Count {
    fn allocated(&self, _addr: usize, size: usize, _wrapped: usize, group: AllocationGroupId) {
        if group != AllocationGroupId::ROOT {
            ALLOCATED.fetch_add(size, Ordering::Relaxed);
        }
    }

    fn deallocated(
        &self,
        _addr: usize,
        _size: usize,
        _wrapped: usize,
        _source: AllocationGroupId,
        _current: AllocationGroupId,
    ) {
    }
}

/// Entries in the segment, each with a value of `VALUE` bytes: about 4 MiB.
const ENTRIES: u64 = 4_000;
const VALUE: usize = 1_000;

fn entry(index: u64) -> OplogEntry {
    OplogEntry {
        ts: 1_000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: format!("node:k{index}").into_bytes(),
            value: vec![b'v'; VALUE],
        }],
        is_migration: false,
        pre_images: None,
    }
}

/// A page of a few entries allocates for those entries and a read buffer,
/// wherever in a large segment it starts, sealed or still being written: a
/// reader 4 MiB behind pays the same per page as one at the end.
#[test]
fn a_page_costs_its_entries_not_the_rest_of_the_segment() {
    AllocationRegistry::set_global_tracker(Count).expect("the only tracker");
    AllocationRegistry::enable_tracking();
    let mut counted = AllocationGroupToken::register().expect("a group");
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut log = OplogManager::open(dir.path(), 0, 64 << 20, 50_000, 7 * 24 * 3600).unwrap();
        for index in 0..ENTRIES {
            log.append(&entry(index)).unwrap();
        }
        log.flush().unwrap();
        if sealed {
            log.rotate().unwrap();
        }
        let mut tail = OplogTailer::new(&[dir.path().to_path_buf()], ResumeToken::from_start(0))
            .expect("a tailer");
        let filters = CdcFilters::default();
        let mut most = 0;
        let mut read = 0;
        while read < ENTRIES {
            let before = ALLOCATED.load(Ordering::Relaxed);
            let page = {
                let _in = counted.enter();
                tail.read_next(4, &filters, u64::MAX).unwrap()
            };
            most = most.max(ALLOCATED.load(Ordering::Relaxed) - before);
            assert!(!page.is_empty(), "the reader stopped at {read}");
            for (got, _) in &page {
                assert_eq!(got.index, read, "entries in order, none skipped");
                read += 1;
            }
        }
        // Four entries of about 1 KiB, their decoding, a read buffer and
        // the cursor: far below the 4 MiB the segment holds.
        assert!(
            most < 256 << 10,
            "a page of 4 entries allocated {most} bytes (sealed: {sealed})"
        );
    }
}
