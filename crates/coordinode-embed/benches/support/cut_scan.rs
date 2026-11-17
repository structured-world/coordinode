//! The scenario of the state cut scan benches: a view of small or of large
//! rows copied page by page, several times over.

use std::time::{Duration, Instant};

use coordinode_core::budget::QueryBudget;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::write_concern::Journal;
use coordinode_embed::Database;
use coordinode_embed::consumer::CutScope;
use coordinode_storage::engine::partition::Partition;

/// Rows copied per page.
const PAGE: usize = 1024;
/// Full copies of the view measured per case.
const PASSES: usize = 20;

/// One copy pass's result: rows copied and the time of each page.
pub struct Measured {
    pub rows: usize,
    // The allocation bench reports no times.
    #[allow(dead_code)]
    pub wall: Duration,
    #[allow(dead_code)]
    pub pages: Vec<Duration>,
}

/// Copy a view of `rows` rows holding `value`-byte values `PASSES` times,
/// running `around` around the measured copies only.
pub fn case(rows: usize, value: usize, around: &mut dyn FnMut(&mut dyn FnMut())) -> Measured {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    for chunk in (0..rows).collect::<Vec<_>>().chunks(1000) {
        let batch =
            coordinode_storage::engine::scope::store_unit(chunk.iter().map(|i| Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:{i:08}").into_bytes(),
                value: vec![(i % 251) as u8; value],
            }));
        let engine = db.engine();
        let ts = engine.oracle().expect("an embedded engine").next().as_raw();
        engine
            .commit_logged(&batch, ts, Journal::Journal)
            .expect("rows");
    }
    db.persist().expect("persist");
    let scope = CutScope::rows(Partition::Node, b"node:".to_vec()).expect("scope");
    let cut = db.state_cut(scope, Duration::from_secs(10)).expect("cut");
    let budget = QueryBudget::new(256 << 20);
    let mut pages = Vec::new();
    let mut wall = Duration::ZERO;
    around(&mut || {
        let start = Instant::now();
        for _ in 0..PASSES {
            let mut after: Option<Vec<u8>> = None;
            let mut copied = 0;
            loop {
                let t = Instant::now();
                let page = db
                    .state_cut_scan(&cut, after.as_deref(), PAGE, &budget)
                    .expect("page");
                let Some(last) = page.last_key() else { break };
                copied += page.rows().len();
                after = Some(last.to_vec());
                pages.push(t.elapsed());
            }
            assert_eq!(copied, rows, "every row once");
        }
        wall = start.elapsed();
    });
    pages.sort_unstable();
    Measured {
        rows: rows * PASSES,
        wall,
        pages,
    }
}

/// The cases both benches run: small rows, then large ones.
pub const CASES: [(usize, usize); 2] = [(100_000, 16), (20_000, 1024)];
