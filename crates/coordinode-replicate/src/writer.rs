//! [`ReplicatedWriter`]: the server-facing Cypher write coordination point.

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_core::graph::types::Value;
use coordinode_embed::db::{CypherResult, StatementOptions};
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::advisor::source::SourceContext;
use parking_lot::RwLock;

/// Routes Cypher execution into the replicated write path and surfaces the
/// committed Raft index of any write.
///
/// The actual materialisation of a write (resolving `NOW()`, `RAND()`,
/// trigger results) happens inside the executor during query execution; the
/// deterministic write-set is then proposed through the [`Database`]'s
/// injected proposal pipeline (a `RaftProposalPipeline` in cluster /
/// standalone-single-node mode, a local pipeline in embedded mode). This
/// writer is the seam the gRPC layer calls so that:
///
/// 1. write coordination lives above the engine and below the server, and
/// 2. [`CypherResult::write_stats`]`.applied_index` carries the committed
///    index of *this* write — the faithful causal `operationTime` token —
///    instead of the caller having to sample the node's current applied
///    index (which is not this write's index; the operationTime inaccuracy).
///
/// It is also the home for the `SeqnoConsumerRegistry` checkpoint
/// hook: a committed write advances the per-shard floor
/// here, with no change to the executor or the consensus engine.
pub struct ReplicatedWriter {
    database: Arc<RwLock<Database>>,
}

impl ReplicatedWriter {
    /// Wrap the shared database handle.
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self { database }
    }

    /// Borrow the underlying database handle for read-only paths that do
    /// not flow through the write coordination point (EXPLAIN, stats).
    pub fn database(&self) -> &Arc<RwLock<Database>> {
        &self.database
    }

    /// Execute a Cypher statement under `options`, on the shared (parallel)
    /// path. The returned [`CypherResult`] carries
    /// `write_stats.applied_index` = the committed Raft index of the write
    /// (`None` for reads and for embedded / non-replicated mode).
    ///
    /// A session `SET` command is refused here: a server runs statements of
    /// many clients against one database, so a setting belongs to the client
    /// session that names it and is carried in `options`, never written into
    /// the database where every other client's statements would read it.
    pub fn execute(
        &self,
        query: &str,
        params: Option<HashMap<String, Value>>,
        source: Option<&SourceContext>,
        options: &StatementOptions,
    ) -> Result<CypherResult, DatabaseError> {
        self.database
            .read()
            .execute_cypher_shared_with(query, params, source, options)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
