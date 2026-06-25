//! Oplog: append-only segmented write-ahead log.
//!
//! The oplog serves four functions simultaneously:
//! 1. **Raft consensus** — stores proposed entries from the Raft leader
//! 2. **Crash recovery** — replays committed entries on restart
//! 3. **CDC** — streams change events to external consumers
//! 4. **PITR** — enables point-in-time restore to any committed index
//!
//! ## File layout
//!
//! Segments live in `<data_dir>/oplog/<shard_id>/`:
//! ```text
//! oplog-00000000000000000000.bin   ← oldest
//! oplog-00000000000000050000.bin
//! oplog-00000000000000100000.bin   ← newest
//! ```
//!
//! See [`segment`] for the on-disk binary format of each file.

pub mod convert;
pub mod entry;
pub mod manager;
pub mod segment;
pub mod tailer;

pub use convert::{mutation_to_op, mutations_to_ops};
pub use entry::{OplogEntry, OplogOp, PreImage, ShardId};
pub use manager::OplogManager;
pub use segment::{SegmentReader, SegmentWriter, FOOTER_SIZE, HEADER_SIZE, MAGIC};
pub use tailer::{CdcFilters, OplogTailer, ResumeToken};
