//! R-SNAP1: unified `read_consistency` knob.
//!
//! `read_consistency` governs **cross-modality snapshot alignment** — whether
//! graph, vector, full-text, document, and time-series reads inside a single
//! query resolve against the same HLC timestamp. Orthogonal to `read_concern`
//! (durability visibility, see `read_concern.rs`).
//!
//! Default selection rule (per arch/core/transactions.md § Read Consistency):
//! - Query touches >1 modality → auto-promoted to `Snapshot`
//! - Single-modality query → stays `Current` (perf-first default)
//! - User hint `/*+ read_consistency('snapshot') */` or `/*+ read_consistency('exact') */`
//!   always overrides the auto-promotion.
//!
//! The per-modality `vector_consistency` flag remains as a narrower override:
//! if the user explicitly sets `vector_consistency`, the vector modality
//! follows that flag even when `read_consistency` says otherwise — all other
//! modalities still follow `read_consistency`.

/// Cross-modality read-consistency mode.
///
/// Maps 1:1 to the `read_consistency` session variable and the
/// `/*+ read_consistency('mode') */` per-query hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadConsistencyMode {
    /// Each modality reads its latest state independently. No `WaitForTs`,
    /// no HNSW post-filter, no FTS segment filter. Lowest latency.
    /// Default for single-modality reads (e.g. pure vector KNN, pure graph
    /// traversal).
    #[default]
    Current,

    /// All modalities resolve against the same HLC timestamp `T`. The shard
    /// blocks on `MaxAssignedWatermark::wait_for(T, timeout)` before
    /// dispatching; HNSW search is post-filtered, tantivy reader is
    /// segment-filtered by `commit_ts ≤ T`. Auto-selected by the planner
    /// when a query touches >1 modality; user-selectable for any query.
    Snapshot,

    /// `Snapshot` semantics plus HNSW bypassed — vector search runs a
    /// brute-force scan with MVCC filter (100% recall, 10-100× slower).
    /// FTS and graph behave as in `Snapshot`. Use for audit /
    /// correctness-critical workloads.
    Exact,
}

impl ReadConsistencyMode {
    /// Parse from the string form used in `SET read_consistency = '...'`
    /// and `/*+ read_consistency('...') */`. Case-insensitive.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "current" => Some(Self::Current),
            "snapshot" => Some(Self::Snapshot),
            "exact" => Some(Self::Exact),
            _ => None,
        }
    }

    /// Canonical string form for EXPLAIN output and error messages.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Snapshot => "snapshot",
            Self::Exact => "exact",
        }
    }

    /// Whether this mode requires the executor to allocate a snapshot
    /// timestamp and call `applied_watermark.wait_for(T, timeout)` before
    /// dispatching the read.
    pub fn requires_snapshot_wait(&self) -> bool {
        matches!(self, Self::Snapshot | Self::Exact)
    }
}

impl std::fmt::Display for ReadConsistencyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_current() {
        assert_eq!(ReadConsistencyMode::default(), ReadConsistencyMode::Current);
    }

    #[test]
    fn parse_case_insensitive() {
        for (input, expected) in [
            ("current", ReadConsistencyMode::Current),
            ("CURRENT", ReadConsistencyMode::Current),
            ("Snapshot", ReadConsistencyMode::Snapshot),
            ("EXACT", ReadConsistencyMode::Exact),
            ("exact", ReadConsistencyMode::Exact),
        ] {
            assert_eq!(ReadConsistencyMode::from_str_opt(input), Some(expected));
        }
    }

    #[test]
    fn parse_rejects_unknown() {
        assert_eq!(ReadConsistencyMode::from_str_opt(""), None);
        assert_eq!(ReadConsistencyMode::from_str_opt("strict"), None);
        assert_eq!(ReadConsistencyMode::from_str_opt("none"), None);
    }

    #[test]
    fn roundtrip_as_str_parse() {
        for mode in [
            ReadConsistencyMode::Current,
            ReadConsistencyMode::Snapshot,
            ReadConsistencyMode::Exact,
        ] {
            assert_eq!(ReadConsistencyMode::from_str_opt(mode.as_str()), Some(mode));
        }
    }

    #[test]
    fn requires_snapshot_wait_semantics() {
        assert!(!ReadConsistencyMode::Current.requires_snapshot_wait());
        assert!(ReadConsistencyMode::Snapshot.requires_snapshot_wait());
        assert!(ReadConsistencyMode::Exact.requires_snapshot_wait());
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(format!("{}", ReadConsistencyMode::Snapshot), "snapshot");
        assert_eq!(format!("{}", ReadConsistencyMode::Current), "current");
        assert_eq!(format!("{}", ReadConsistencyMode::Exact), "exact");
    }
}
