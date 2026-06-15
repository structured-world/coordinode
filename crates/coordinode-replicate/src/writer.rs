//! [`ReplicatedWriter`]: the server-facing Cypher write coordination point.

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::read_concern::ReadConcern;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_embed::db::CypherResult;
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::advisor::source::SourceContext;
use parking_lot::RwLock;

/// Substring of the `execute_cypher_shared` rejection for session `SET`
/// commands. Matching it (rather than a typed error) keeps the embed API
/// surface unchanged; the shared path returns `Semantic(..)` and we retry
/// under exclusive access.
const SET_REQUIRES_EXCLUSIVE: &str = "SET commands require exclusive Database access";

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
/// It is also the home for the upcoming `SeqnoConsumerRegistry` checkpoint
/// hook (ADR-028, R137a): a committed write advances the per-shard floor
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

    /// Execute a Cypher statement.
    ///
    /// Runs on the shared (parallel) path first; only session `SET`
    /// commands, which mutate session config and need exclusive access,
    /// fall back to the `&mut` path. The returned [`CypherResult`] carries
    /// `write_stats.applied_index` = the committed Raft index of the write
    /// (`None` for reads and for embedded / non-replicated mode).
    pub fn execute(
        &self,
        query: &str,
        params: Option<HashMap<String, Value>>,
        source: Option<&SourceContext>,
        read_concern: Option<&ReadConcern>,
        write_concern: Option<&WriteConcern>,
    ) -> Result<CypherResult, DatabaseError> {
        let shared = {
            let db = self.database.read();
            db.execute_cypher_shared(query, params.clone(), source, read_concern, write_concern)
        };
        match shared {
            Ok(result) => Ok(result),
            Err(DatabaseError::Semantic(msg)) if msg.contains(SET_REQUIRES_EXCLUSIVE) => {
                let mut db = self.database.write();
                db.execute_cypher_full(
                    query,
                    params,
                    source,
                    read_concern.cloned(),
                    write_concern.cloned(),
                )
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn writer() -> (ReplicatedWriter, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(dir.path()).expect("open database");
        (ReplicatedWriter::new(Arc::new(RwLock::new(db))), dir)
    }

    /// A write through the embedded (non-replicated) path returns no
    /// committed index — `applied_index` is `None`, not a fabricated 0.
    #[test]
    fn embedded_write_has_no_committed_index() {
        let (w, _dir) = writer();
        let result = w
            .execute("CREATE (n:Probe {v: 1}) RETURN n", None, None, None, None)
            .expect("write should succeed");
        assert_eq!(result.write_stats.nodes_created, 1);
        assert_eq!(
            result.write_stats.applied_index, None,
            "embedded pipeline has no Raft log → no committed index"
        );
    }

    /// A read-only statement records no write and no committed index.
    #[test]
    fn read_only_has_no_committed_index() {
        let (w, _dir) = writer();
        w.execute("CREATE (n:Probe {v: 1})", None, None, None, None)
            .expect("seed write");
        let result = w
            .execute("MATCH (n:Probe) RETURN n.v", None, None, None, None)
            .expect("read should succeed");
        assert!(!result.write_stats.has_mutations());
        assert_eq!(result.write_stats.applied_index, None);
    }

    /// The shared→exclusive fallback routes a session `SET` command to the
    /// `&mut` path instead of surfacing the rejection.
    #[test]
    fn session_set_falls_back_to_exclusive() {
        let (w, _dir) = writer();
        let result = w.execute(
            "SET vector_consistency = 'snapshot'",
            None,
            None,
            None,
            None,
        );
        assert!(
            result.is_ok(),
            "session SET must route through exclusive access, got {:?}",
            result.err()
        );
    }
}
