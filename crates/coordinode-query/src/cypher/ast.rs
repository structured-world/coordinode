//! Typed AST nodes for OpenCypher read and write operations.
//!
//! The AST is produced by the parser from a Cypher query string.
//! It represents the syntactic structure of the query before semantic
//! analysis (variable binding, label validation, type checking).

use std::collections::HashMap;

use coordinode_core::graph::types::Value;

/// A complete Cypher query composed of one or more clauses.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub clauses: Vec<Clause>,
    /// Per-query optimizer hints extracted from `/*+ key('value') */` comments.
    pub hints: Vec<QueryHint>,
}

impl Query {
    /// Returns `true` if this query contains any write (mutating) clauses.
    ///
    /// Write clauses: CREATE, MERGE, MERGE ALL, UPSERT, DELETE, SET, REMOVE,
    /// and DDL operations (CREATE INDEX / DROP INDEX / ALTER LABEL / etc.).
    ///
    /// Used by the CypherService handler to enforce write-concern validation
    /// in causal sessions: a write in a causal session requires
    /// `writeConcern >= majority` to avoid dangling `operationTime` references
    /// when the leader crashes before replicating.
    pub fn is_write(&self) -> bool {
        self.clauses.iter().any(|c| {
            matches!(
                c,
                Clause::Create(_)
                    | Clause::Merge(_)
                    | Clause::MergeMany(_)
                    | Clause::Upsert(_)
                    | Clause::Delete(_)
                    | Clause::Set(_, _)
                    | Clause::Remove(_)
                    | Clause::DetachDocument(_)
                    | Clause::AttachDocument(_)
                    | Clause::AlterLabel(_)
                    | Clause::CreateTextIndex(_)
                    | Clause::DropTextIndex(_)
                    | Clause::CreateEncryptedIndex(_)
                    | Clause::DropEncryptedIndex(_)
                    | Clause::CreateIndex(_)
                    | Clause::DropIndex(_)
                    | Clause::CreateVectorIndex(_)
                    | Clause::DropVectorIndex(_)
                    | Clause::CreateEdgeType(_)
                    | Clause::CreateNodeType(_)
            )
        })
    }
}

/// Per-query optimizer hint from `/*+ key('value') */` syntax.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryHint {
    /// Override vector consistency mode for this query only.
    /// Syntax: `/*+ vector_consistency('snapshot') */`
    ///
    /// Narrower than `ReadConsistency` — applies to the vector modality
    /// only. When both `ReadConsistency` and `VectorConsistency` are set,
    /// `VectorConsistency` wins for vector operators; other modalities
    /// follow `ReadConsistency`.
    VectorConsistency(coordinode_core::graph::types::VectorConsistencyMode),

    /// Override read consistency mode (cross-modality snapshot alignment)
    /// for this query only. Syntax: `/*+ read_consistency('snapshot') */`.
    /// An explicit hint always beats the planner's auto-promotion rule.
    ReadConsistency(coordinode_core::txn::read_consistency::ReadConsistencyMode),
}

/// A single clause in a Cypher query.
#[derive(Debug, Clone, PartialEq)]
pub enum Clause {
    // Read clauses
    Match(MatchClause),
    OptionalMatch(MatchClause),
    Where(Expr),
    Return(ReturnClause),
    With(WithClause),
    Unwind(UnwindClause),
    OrderBy(Vec<SortItem>),
    Skip(Expr),
    Limit(Expr),
    AsOfTimestamp(Expr),

    // Procedure call
    Call(CallClause),

    // DDL clauses
    AlterLabel(AlterLabelClause),
    CreateTextIndex(CreateTextIndexClause),
    DropTextIndex(DropTextIndexClause),
    CreateEncryptedIndex(CreateEncryptedIndexClause),
    DropEncryptedIndex(DropEncryptedIndexClause),
    CreateIndex(CreateIndexClause),
    DropIndex(DropIndexClause),
    CreateVectorIndex(CreateVectorIndexClause),
    DropVectorIndex(DropVectorIndexClause),
    CreateEdgeType(CreateEdgeTypeClause),
    /// `CREATE NODE TYPE <name> [TEMPORAL] [WITH (...)]` — bitemporal-capable
    /// label DDL (R172a per ADR-027). Mirror of `CreateEdgeType` for nodes.
    CreateNodeType(CreateNodeTypeClause),

    /// `CREATE TRIGGER … ON :Label CREATE|UPDATE|DELETE BEFORE|AFTER COMMIT EXECUTE … [ON ERROR …]`.
    CreateTrigger(CreateTriggerClause),
    /// `DROP TRIGGER <name>`.
    DropTrigger(DropTriggerClause),
    /// `SHOW TRIGGERS`.
    ShowTriggers,
    /// `ALTER TRIGGER <name> { DISABLE | ENABLE | SET EXECUTE … | SET ON ERROR … }`.
    AlterTrigger(AlterTriggerClause),

    // Write clauses
    Create(CreateClause),
    Merge(MergeClause),
    /// MERGE ALL: Cartesian-product relationship merge — all matching src × tgt pairs.
    MergeMany(MergeClause),
    Upsert(UpsertClause),
    Delete(DeleteClause),
    /// `DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n) [TRANSFER EDGES ...]`
    ///
    /// Promotes a nested DOCUMENT property to a new graph node + edge.
    DetachDocument(DetachDocumentClause),
    /// `ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address [TRANSFER EDGES ...] [ON CONFLICT REPLACE] [ON REMAINING FAIL]`
    ///
    /// Demotes a graph node back to a nested DOCUMENT property on another node.
    AttachDocument(AttachDocumentClause),
    /// `MERGE NODES (a, b) INTO <target> [ON CONFLICT ...] [TRANSFER EDGES FROM <src> TO <dst>] [ON DUPLICATE ...] [TRANSFER EDGE PROPERTIES]`
    ///
    /// Native node merge — collapses two nodes into one preserving properties
    /// and edges in a single MVCC transaction. Replaces APOC's
    /// `apoc.refactor.mergeNodes()` with a first-class Cypher operation.
    MergeNodes(MergeNodesClause),
    /// SET clause with optional ON VIOLATION SKIP modifier.
    ///
    /// Syntax: `SET n.prop = val [ON VIOLATION SKIP]`
    ///
    /// With `ViolationMode::Skip`, nodes that would violate schema constraints
    /// are silently skipped (not updated) rather than failing the entire query.
    /// The caller receives only the rows that were successfully updated.
    Set(Vec<SetItem>, ViolationMode),
    Remove(Vec<RemoveItem>),
}

/// How to handle schema constraint violations during a SET operation.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ViolationMode {
    /// Fail the entire query on the first violation (default, strict semantics).
    #[default]
    Fail,
    /// Skip nodes that would violate schema constraints; continue with others.
    /// Syntax: `SET ... ON VIOLATION SKIP`
    Skip,
}

/// ALTER LABEL clause: change schema mode for a label.
///
/// Syntax: `ALTER LABEL User SET SCHEMA STRICT`
#[derive(Debug, Clone, PartialEq)]
pub struct AlterLabelClause {
    /// Label name to alter.
    pub label: String,
    /// New schema mode.
    pub mode: String,
}

/// Per-field configuration for text index DDL.
#[derive(Debug, Clone, PartialEq)]
pub struct TextIndexFieldSpec {
    /// Property name to index.
    pub property: String,
    /// Analyzer name: language ("english", "russian"), "auto_detect", "none".
    pub analyzer: Option<String>,
}

/// CREATE TEXT INDEX clause.
///
/// Simple syntax:   `CREATE TEXT INDEX idx ON :Label(property) LANGUAGE "english"`
/// Extended syntax: `CREATE TEXT INDEX idx ON :Label { title: { analyzer: "english" }, body: { analyzer: "auto_detect" } } DEFAULT LANGUAGE "english" LANGUAGE OVERRIDE "lang"`
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTextIndexClause {
    /// Index name.
    pub name: String,
    /// Label to index.
    pub label: String,
    /// Fields to index with per-field configuration.
    pub fields: Vec<TextIndexFieldSpec>,
    /// Default language for fields without explicit analyzer (DEFAULT LANGUAGE clause).
    /// In simple syntax, the LANGUAGE value is used here.
    pub default_language: Option<String>,
    /// Node property name for per-node language override (LANGUAGE OVERRIDE clause).
    pub language_override: Option<String>,
}

/// DROP TEXT INDEX clause.
///
/// Syntax: `DROP TEXT INDEX idx_name`
#[derive(Debug, Clone, PartialEq)]
pub struct DropTextIndexClause {
    /// Index name to drop.
    pub name: String,
}

/// CREATE ENCRYPTED INDEX clause (SSE).
///
/// Syntax: `CREATE ENCRYPTED INDEX idx_name ON :Label(property)`
///
/// Creates a searchable symmetric encryption index for equality queries
/// on encrypted fields. The client provides search tokens (HMAC-SHA256);
/// the server matches tokens without seeing plaintext.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateEncryptedIndexClause {
    /// Index name.
    pub name: String,
    /// Label to index.
    pub label: String,
    /// Property name to index.
    pub property: String,
}

/// DROP ENCRYPTED INDEX clause.
///
/// Syntax: `DROP ENCRYPTED INDEX idx_name`
#[derive(Debug, Clone, PartialEq)]
pub struct DropEncryptedIndexClause {
    /// Index name to drop.
    pub name: String,
}

/// CREATE [UNIQUE] [SPARSE] INDEX clause (B-tree single-field index).
///
/// Syntax: `CREATE [UNIQUE] [SPARSE] INDEX idx_name ON :Label(prop) [WHERE pred]`
///
/// The optional WHERE clause restricts index membership to nodes matching
/// a simple predicate (partial index). Supported predicates:
/// - `prop = 'string'`
/// - `prop = 42`
/// - `prop = true/false`
/// - `prop IS NOT NULL`
#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexClause {
    /// Index name.
    pub name: String,
    /// Label to index.
    pub label: String,
    /// Property to index.
    pub property: String,
    /// Whether this index enforces uniqueness.
    pub unique: bool,
    /// Whether to skip null values (sparse index).
    pub sparse: bool,
    /// Optional WHERE predicate expression for partial index.
    pub filter_expr: Option<Expr>,
}

/// DROP INDEX clause.
///
/// Syntax: `DROP INDEX idx_name`
#[derive(Debug, Clone, PartialEq)]
pub struct DropIndexClause {
    /// Index name to drop.
    pub name: String,
}

/// CREATE VECTOR INDEX clause (HNSW).
///
/// Syntax: `CREATE VECTOR INDEX idx ON :Label(property) OPTIONS { m: 16, ef_construction: 200, metric: "cosine", dimensions: 128 }`
///
/// Creates an HNSW approximate nearest-neighbor index for vector similarity search.
/// After creation, `VectorTopK` plans use HnswScan instead of brute-force O(N) scan.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CreateVectorIndexClause {
    /// Index name.
    pub name: String,
    /// Label to index.
    pub label: String,
    /// Vector property to index.
    pub property: String,
    /// HNSW M parameter: number of bi-directional links per node (default: 16).
    pub m: Option<usize>,
    /// HNSW ef_construction: dynamic list size during build (default: 200).
    pub ef_construction: Option<usize>,
    /// Distance metric: "cosine", "euclidean", "dot" (default: "cosine").
    pub metric: Option<String>,
    /// Vector dimensionality (required unless inferrable from data).
    pub dimensions: Option<u32>,
    /// In-RAM quantization codec selector. Accepted values:
    ///
    /// - `"none"` (default — f32 originals stay in RAM)
    /// - `"sq8"`
    /// - `"rabitq"` (1-bit; default `bits=1`)
    /// - `"rabitq-1bit"` / `"rabitq-2bit"` / `"rabitq-3bit"` /
    ///   `"rabitq-4bit"` (Extended-RaBitQ R862 — higher bits trade
    ///   RAM for recall)
    ///
    /// Case-insensitive. Unrecognized values fall back to `"none"`.
    pub quantization: Option<String>,
    /// Reader behaviour while the index is in the Building state.
    /// Accepted values (case-insensitive): `"block"`, `"partial-recall"`
    /// / `"partial_recall"`, `"offline"`. Unrecognized values fall back
    /// to `"block"`.
    pub online_during_build: Option<String>,
    /// HNSW ef_search: default dynamic candidate-list size during search
    /// (default: 200). Raise for higher recall on adversarial / sparsely
    /// connected data, at a latency cost.
    pub ef_search: Option<usize>,
    /// Number of approximate candidates re-scored with exact f32 distance
    /// before returning top-k (default: 100).
    pub rerank_candidates: Option<usize>,
    /// Verbatim trailing clause after the engine-known syntax, captured by the
    /// `extension_tail` grammar rule. `None` for a plain `CREATE VECTOR INDEX`;
    /// `Some(raw)` when an extension layer's clause (parsed by its own handler)
    /// follows. The planner routes a `Some` to a `LogicalOp::Extension`.
    pub extension_tail: Option<String>,
}

/// CREATE EDGE TYPE clause.
///
/// Syntax: `CREATE EDGE TYPE <name> [TEMPORAL] [WITH ( <decl>, ... )]`
///
/// Declares an edge type schema. When `temporal = true`, every edge instance
/// must carry a `valid_from` timestamp at write time and the storage layer
/// stores per-version edgeprop entries keyed on `(src, tgt, valid_from)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateEdgeTypeClause {
    /// Edge type name (e.g., "WORKS_AT").
    pub name: String,
    /// Whether the edge type carries bitemporal `(valid_from, valid_to)` fields.
    pub temporal: bool,
    /// User-declared edge properties from the optional `WITH (...)` block.
    pub properties: Vec<EdgePropertyDecl>,
}

/// `CREATE NODE TYPE <name> [TEMPORAL] [WITH (...)]` — bitemporal-capable
/// label DDL (R172a per ADR-027). Mirror of `CreateEdgeTypeClause` for nodes.
///
/// When `temporal == true`, every node record of this label carries the
/// `(valid_from, valid_to)` valid-time interval plus the engine-assigned
/// `__ingestion_ts__` (HLC commit-ts), and multiple versions of the same
/// `node_id` coexist on per-version storage keys. The TEMPORAL flag is
/// immutable for the lifetime of the label — toggling on an existing
/// label is rejected at DDL time; the migration path is "create a new
/// label, copy data, drop old".
#[derive(Debug, Clone, PartialEq)]
pub struct CreateNodeTypeClause {
    /// Label name (e.g., "Person").
    pub name: String,
    /// Whether nodes of this label are bitemporal.
    pub temporal: bool,
    /// User-declared node properties from the optional `WITH (...)` block.
    /// Reuses `EdgePropertyDecl` — the property-declaration grammar is the
    /// same for nodes and edges.
    pub properties: Vec<EdgePropertyDecl>,
}

/// Single edge-type property declaration: `name : TYPE [NOT NULL]`.
///
/// `type_name` is the lexical type identifier (e.g., "STRING", "TIMESTAMP")
/// — resolved against `coordinode_core::schema::PropertyType` in semantic.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgePropertyDecl {
    pub name: String,
    pub type_name: String,
    pub not_null: bool,
}

/// DROP VECTOR INDEX clause.
///
/// Syntax: `DROP VECTOR INDEX idx_name`
#[derive(Debug, Clone, PartialEq)]
pub struct DropVectorIndexClause {
    /// Index name to drop.
    pub name: String,
}

/// `CREATE TRIGGER` — defines a reactive Cypher body that fires on graph
/// mutations.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTriggerClause {
    /// Unique trigger name (used as DROP / ALTER / SHOW handle).
    pub name: String,
    /// Target: `:Label` or `[:EdgeType]` — narrows which mutations fire the trigger.
    pub target: TriggerTarget,
    /// Event kinds that fire the trigger (CREATE / UPDATE / DELETE; multiple allowed via `|`).
    pub events: TriggerEvents,
    /// Synchronous (`BEFORE COMMIT`) or asynchronous (`AFTER COMMIT`).
    pub timing: TriggerTiming,
    /// Trigger body — captured as a raw Cypher source string at parse time;
    /// re-parsed and executed when the trigger fires. The grammar guarantees
    /// the source is at least one syntactically valid clause sequence.
    pub body_source: String,
    /// L1 cascade-depth limit per-trigger override (the trigger architecture).
    /// Counter is shared across all triggers in one originating mutation;
    /// this field overrides the cluster default (`triggers.max_cascade_depth`,
    /// 10). Parsed from `CASCADE_LIMIT n` or its deprecated alias `MAXDEPTH n`.
    pub cascade_limit: Option<u32>,
    /// L2 unique-trigger fanout limit per-trigger override (the trigger architecture).
    /// Counter is per-trigger within one cascade; this field overrides the
    /// cluster default (`triggers.max_cascade_fanout`, 100). Parsed from
    /// `CASCADE_FANOUT n`.
    pub cascade_fanout: Option<u32>,
    /// Error-handling policy. `None` here means "use default for the timing":
    /// `BEFORE COMMIT` → `Propagate`, `AFTER COMMIT` → `Retry { n: 3, backoff_ms: 1000 }`.
    pub on_error: Option<OnErrorPolicy>,
}

/// `DROP TRIGGER <name>`.
#[derive(Debug, Clone, PartialEq)]
pub struct DropTriggerClause {
    /// Trigger name to drop.
    pub name: String,
}

/// `ALTER TRIGGER <name> { DISABLE | ENABLE | SET EXECUTE … | SET ON ERROR … }`.
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTriggerClause {
    /// Target trigger name.
    pub name: String,
    /// What to change.
    pub action: AlterTriggerAction,
}

/// Concrete change applied by `ALTER TRIGGER`.
#[derive(Debug, Clone, PartialEq)]
pub enum AlterTriggerAction {
    /// Stop firing without forgetting the definition.
    Disable,
    /// Resume firing after a prior `DISABLE`.
    Enable,
    /// Replace the body. Body source captured verbatim at parse time.
    SetBody(String),
    /// Replace the on-error policy.
    SetOnError(OnErrorPolicy),
}

/// Trigger filter — which graph elements activate the trigger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerTarget {
    /// `:Label` — fires on node mutations of the given label.
    Label(String),
    /// `[:EdgeType]` — fires on edge mutations of the given type.
    EdgeType(String),
}

/// Event-kind bitset (a single trigger can subscribe to any subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TriggerEvents {
    pub on_create: bool,
    pub on_update: bool,
    pub on_delete: bool,
}

impl TriggerEvents {
    /// Returns `true` if at least one event kind is enabled.
    pub fn any(self) -> bool {
        self.on_create || self.on_update || self.on_delete
    }
}

/// Execution timing relative to the originating Raft proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerTiming {
    /// Leader-only, inline with the Raft proposal — can abort the transaction.
    BeforeCommit,
    /// Oplog-consumer driven, post-commit — at-least-once execution.
    AfterCommit,
}

/// Per-trigger error handling policy (the trigger architecture).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnErrorPolicy {
    /// `BEFORE COMMIT`: aborts the originating transaction.
    /// `AFTER COMMIT`: writes the failure to `trigger_failures` immediately, no retries.
    Propagate,
    /// Durable retry queue with exponential backoff. Falls through to dead-letter on exhaustion.
    Retry {
        /// Maximum retry attempts before dead-lettering.
        n: u32,
        /// Backoff base in milliseconds; doubles each attempt. Default 1000 if omitted in DDL.
        backoff_ms: u32,
    },
    /// Skip retries — first failure lands directly in `trigger_failures`.
    DeadLetter,
}

impl OnErrorPolicy {
    /// Default policy when a `CREATE TRIGGER` statement omits `ON ERROR`.
    /// Spec'd in the trigger architecture: BEFORE → Propagate; AFTER → Retry 3 with 1s backoff.
    pub fn default_for(timing: TriggerTiming) -> Self {
        match timing {
            TriggerTiming::BeforeCommit => Self::Propagate,
            TriggerTiming::AfterCommit => Self::Retry {
                n: 3,
                backoff_ms: 1000,
            },
        }
    }
}

/// MATCH clause: one or more patterns.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchClause {
    pub patterns: Vec<Pattern>,
    /// Optional WHERE attached directly to MATCH.
    pub where_clause: Option<Expr>,
}

/// A graph pattern: sequence of node and relationship elements.
///
/// Examples:
///   `(n:User)`
///   `(a)-[:KNOWS]->(b)`
///   `(a)-[:KNOWS*2..5]->(b)`
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
    /// Named-path variable from `p = <pattern>` in a MATCH clause, if any.
    /// Bound to a path value when the pattern is planned as a path-producing
    /// op (currently `shortestPath`). `None` for CREATE / MERGE /
    /// WHERE-predicate patterns, which never bind a path.
    pub path_variable: Option<String>,
    /// True when this pattern came from `shortestPath(<pattern>)`. Planned as
    /// a single-pair BFS that binds `path_variable` to the resulting path.
    pub shortest_path: bool,
}

/// An element in a graph pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternElement {
    Node(NodePattern),
    Relationship(RelationshipPattern),
}

/// A node pattern: `(variable:Label {prop: value})`.
#[derive(Debug, Clone, PartialEq)]
pub struct NodePattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Labels (zero or more).
    pub labels: Vec<String>,
    /// Inline property map.
    pub properties: Vec<(String, Expr)>,
}

/// A relationship pattern: `-[variable:TYPE {props}]->`.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipPattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Relationship types (disjunction — any of these types).
    pub rel_types: Vec<String>,
    /// Direction of the relationship.
    pub direction: Direction,
    /// Variable-length traversal bounds (e.g., *2..5).
    pub length: Option<LengthBound>,
    /// Inline property map.
    pub properties: Vec<(String, Expr)>,
}

/// Relationship direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `-->` or `-[]->`
    Outgoing,
    /// `<--` or `<-[]-`
    Incoming,
    /// `--` or `-[]-`
    Both,
}

/// Variable-length path bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthBound {
    /// Minimum hops (default 1).
    pub min: Option<u64>,
    /// Maximum hops (default unbounded, but capped at 10 by engine).
    pub max: Option<u64>,
}

/// RETURN clause.
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnClause {
    pub distinct: bool,
    pub items: Vec<ReturnItem>,
}

/// A single RETURN item: expression with optional alias.
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnItem {
    pub expr: Expr,
    pub alias: Option<String>,
}

/// WITH clause: projection between query parts.
#[derive(Debug, Clone, PartialEq)]
pub struct WithClause {
    pub distinct: bool,
    pub items: Vec<ReturnItem>,
    pub where_clause: Option<Expr>,
}

/// UNWIND clause: `UNWIND expr AS variable`.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwindClause {
    pub expr: Expr,
    pub variable: String,
}

/// ORDER BY item: expression + direction.
#[derive(Debug, Clone, PartialEq)]
pub struct SortItem {
    pub expr: Expr,
    pub ascending: bool,
}

/// Expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value.
    Literal(Value),

    /// Parameter: `$name`.
    Parameter(String),

    /// Variable reference: `n`.
    Variable(String),

    /// Property access: `n.name`.
    PropertyAccess { expr: Box<Expr>, property: String },

    /// Binary operation: `a + b`, `a AND b`, etc.
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },

    /// Unary operation: `NOT x`, `-x`.
    UnaryOp { op: UnaryOperator, expr: Box<Expr> },

    /// Function call: `count(*)`, `vector_distance(a, b)`.
    FunctionCall {
        name: String,
        args: Vec<Expr>,
        distinct: bool,
    },

    /// List literal: `[1, 2, 3]`.
    List(Vec<Expr>),

    /// Map literal: `{key: value}`.
    MapLiteral(Vec<(String, Expr)>),

    /// Map projection: `n { .name, .age, posts: collect(p { .title }) }`.
    /// Builds a nested JSON object from a variable's properties.
    /// `.prop` shorthand expands to `prop: n.prop`.
    MapProjection {
        expr: Box<Expr>,
        items: Vec<MapProjectionItem>,
    },

    /// `x IN list` or `x IN [1,2,3]`.
    In { expr: Box<Expr>, list: Box<Expr> },

    /// `x IS NULL` / `x IS NOT NULL`.
    IsNull { expr: Box<Expr>, negated: bool },

    /// `x STARTS WITH y`, `x ENDS WITH y`, `x CONTAINS y`.
    StringMatch {
        expr: Box<Expr>,
        op: StringOp,
        pattern: Box<Expr>,
    },

    /// `CASE WHEN ... THEN ... ELSE ... END`.
    Case {
        operand: Option<Box<Expr>>,
        when_clauses: Vec<(Expr, Expr)>,
        else_clause: Option<Box<Expr>>,
    },

    /// Pattern predicate: `(a)-[:R]->(b)` in expression context.
    /// Evaluates to `true` if the pattern matches at least one path, `false` otherwise.
    /// Used in WHERE clauses: `WHERE (a)-[:KNOWS]->(b)` or `WHERE NOT (a)-[:KNOWS]->(b)`.
    PatternPredicate(Pattern),

    /// Subscript / index access: `expr[index]`.
    ///
    /// For list values: `list[0]` → first element, `list[-1]` → last element.
    /// For map values: `map["key"]` → value at key.
    /// Out-of-bounds list access and missing map keys evaluate to `null`.
    Subscript { expr: Box<Expr>, index: Box<Expr> },

    /// Existential subquery: `EXISTS { MATCH … [WHERE …] }`. Evaluates to
    /// `true` when the inner MATCH produces at least one row, correlated with
    /// the outer-scope bindings. Routed through the same MATCH planner as a
    /// top-level query (no second pattern-matching implementation).
    ExistsSubquery(Box<MatchClause>),

    /// Pattern comprehension: `[(a)-[:R]->(b) WHERE pred | expr]`. Matches the
    /// inner pattern (correlated with the outer row), filters by the optional
    /// `pred`, and collects `map` evaluated per match into a list. Planned
    /// through the same MATCH planner as a top-level query.
    PatternComprehension {
        /// The path pattern to match.
        pattern: Box<Pattern>,
        /// Optional filter over the matched rows.
        where_clause: Option<Box<Expr>>,
        /// Projection evaluated per matching row, collected into the result list.
        map: Box<Expr>,
    },

    /// List comprehension: `[x IN list WHERE pred | expr]`. For each element
    /// bound to `var`, the optional `pred` filters and the optional `map`
    /// projects (defaulting to `var` itself). Produces a new list.
    ListComprehension {
        /// Per-element variable name.
        var: String,
        /// Source list.
        list: Box<Expr>,
        /// Optional filter predicate over `var`.
        pred: Option<Box<Expr>>,
        /// Optional projection expression over `var` (defaults to `var`).
        map: Option<Box<Expr>>,
    },

    /// List quantifier predicate: `all/any/none/single(x IN list WHERE pred)`.
    /// Binds each list element to `var` and evaluates `pred`; the `kind`
    /// determines how the per-element booleans combine into the result.
    ListPredicate {
        /// Which quantifier (all / any / none / single).
        kind: ListPredicateKind,
        /// Per-element variable name.
        var: String,
        /// List being tested.
        list: Box<Expr>,
        /// Predicate evaluated per element (references `var`).
        pred: Box<Expr>,
    },

    /// `reduce(acc = init, x IN list | expr)` — left fold over a list.
    /// `acc` is seeded with `init`, then for each element bound to `var` the
    /// `expr` (which references `acc` and `var`) produces the next accumulator.
    Reduce {
        /// Accumulator variable name.
        acc: String,
        /// Initial accumulator value.
        init: Box<Expr>,
        /// Per-element variable name.
        var: String,
        /// List being folded.
        list: Box<Expr>,
        /// Step expression evaluated per element to update the accumulator.
        expr: Box<Expr>,
    },

    /// Star expression for `count(*)` or `RETURN *`.
    Star,
}

impl Expr {
    /// Replace all `Parameter(name)` nodes with `Literal(value)` from the params map.
    /// Unknown parameters (not in the map) remain as `Parameter` (will eval to Null).
    pub fn substitute_params(&mut self, params: &HashMap<String, Value>) {
        match self {
            Expr::Parameter(name) => {
                if let Some(value) = params.get(name.as_str()) {
                    *self = Expr::Literal(value.clone());
                }
            }
            Expr::PropertyAccess { expr, .. } => expr.substitute_params(params),
            Expr::BinaryOp { left, right, .. } => {
                left.substitute_params(params);
                right.substitute_params(params);
            }
            Expr::UnaryOp { expr, .. } => expr.substitute_params(params),
            Expr::FunctionCall { args, .. } => {
                for arg in args {
                    arg.substitute_params(params);
                }
            }
            Expr::List(items) => {
                for item in items {
                    item.substitute_params(params);
                }
            }
            Expr::MapLiteral(entries) => {
                for (_, v) in entries {
                    v.substitute_params(params);
                }
            }
            Expr::MapProjection { expr, items } => {
                expr.substitute_params(params);
                for item in items {
                    if let MapProjectionItem::Computed(_, ref mut value) = item {
                        value.substitute_params(params);
                    }
                }
            }
            Expr::In { expr, list } => {
                expr.substitute_params(params);
                list.substitute_params(params);
            }
            Expr::IsNull { expr, .. } => expr.substitute_params(params),
            Expr::StringMatch { expr, pattern, .. } => {
                expr.substitute_params(params);
                pattern.substitute_params(params);
            }
            Expr::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    op.substitute_params(params);
                }
                for (cond, result) in when_clauses {
                    cond.substitute_params(params);
                    result.substitute_params(params);
                }
                if let Some(el) = else_clause {
                    el.substitute_params(params);
                }
            }
            Expr::PatternPredicate(pattern) => {
                for elem in &mut pattern.elements {
                    if let PatternElement::Node(node) = elem {
                        for (_, v) in &mut node.properties {
                            v.substitute_params(params);
                        }
                    } else if let PatternElement::Relationship(rel) = elem {
                        for (_, v) in &mut rel.properties {
                            v.substitute_params(params);
                        }
                    }
                }
            }
            Expr::Subscript { expr, index } => {
                expr.substitute_params(params);
                index.substitute_params(params);
            }
            Expr::Reduce {
                init, list, expr, ..
            } => {
                init.substitute_params(params);
                list.substitute_params(params);
                expr.substitute_params(params);
            }
            Expr::ListPredicate { list, pred, .. } => {
                list.substitute_params(params);
                pred.substitute_params(params);
            }
            Expr::ListComprehension {
                list, pred, map, ..
            } => {
                list.substitute_params(params);
                if let Some(p) = pred {
                    p.substitute_params(params);
                }
                if let Some(m) = map {
                    m.substitute_params(params);
                }
            }
            Expr::PatternComprehension {
                pattern,
                where_clause,
                map,
            } => {
                for elem in &mut pattern.elements {
                    match elem {
                        PatternElement::Node(node) => {
                            for (_, v) in &mut node.properties {
                                v.substitute_params(params);
                            }
                        }
                        PatternElement::Relationship(rel) => {
                            for (_, v) in &mut rel.properties {
                                v.substitute_params(params);
                            }
                        }
                    }
                }
                if let Some(w) = where_clause {
                    w.substitute_params(params);
                }
                map.substitute_params(params);
            }
            Expr::ExistsSubquery(mc) => {
                for pattern in &mut mc.patterns {
                    for elem in &mut pattern.elements {
                        match elem {
                            PatternElement::Node(node) => {
                                for (_, v) in &mut node.properties {
                                    v.substitute_params(params);
                                }
                            }
                            PatternElement::Relationship(rel) => {
                                for (_, v) in &mut rel.properties {
                                    v.substitute_params(params);
                                }
                            }
                        }
                    }
                }
                if let Some(w) = &mut mc.where_clause {
                    w.substitute_params(params);
                }
            }
            Expr::Literal(_) | Expr::Variable(_) | Expr::Star => {}
        }
    }
}

/// An item in a map projection expression.
#[derive(Debug, Clone, PartialEq)]
pub enum MapProjectionItem {
    /// Property shorthand: `.name` → includes `name: expr.name`.
    Property(String),
    /// Computed/aliased entry: `alias: expression`.
    Computed(String, Expr),
}

// --- Procedure call type ---

/// CALL clause: invoke a named procedure.
///
/// Example: `CALL db.advisor.suggestions() YIELD id, severity, kind`
#[derive(Debug, Clone, PartialEq)]
pub struct CallClause {
    /// Dotted procedure name (e.g., `db.advisor.suggestions`).
    pub procedure: String,
    /// Positional arguments.
    pub args: Vec<Expr>,
    /// YIELD items: columns to select from the procedure output.
    /// If empty, all columns are returned.
    pub yield_items: Vec<YieldItem>,
}

/// A single item in a YIELD clause.
#[derive(Debug, Clone, PartialEq)]
pub struct YieldItem {
    /// Column name from the procedure output.
    pub name: String,
    /// Optional alias (AS ...).
    pub alias: Option<String>,
}

// --- Write clause types ---

/// CREATE clause: create one or more patterns.
///
/// Example: `CREATE (n:User {name: "Alice"})-[:KNOWS]->(m:User {name: "Bob"})`
#[derive(Debug, Clone, PartialEq)]
pub struct CreateClause {
    pub patterns: Vec<Pattern>,
}

/// MERGE clause: match or create a pattern, with optional actions.
///
/// Example: `MERGE (n:User {email: $email}) ON CREATE SET n.created = now()`
#[derive(Debug, Clone, PartialEq)]
pub struct MergeClause {
    pub pattern: Pattern,
    pub on_match: Vec<SetItem>,
    pub on_create: Vec<SetItem>,
}

/// UPSERT MATCH clause (CoordiNode extension): atomic upsert.
///
/// Example:
/// ```cypher
/// UPSERT MATCH (u:User {email: "alice@example.com"})
/// ON MATCH SET u.login_count = u.login_count + 1
/// ON CREATE CREATE (u:User {email: "alice@example.com", login_count: 1})
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UpsertClause {
    pub pattern: Pattern,
    pub on_match: Vec<SetItem>,
    pub on_create: Vec<Pattern>,
}

/// DELETE clause: delete nodes or relationships.
///
/// Example: `DELETE n` or `DETACH DELETE n` (also deletes connected edges).
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteClause {
    pub detach: bool,
    pub exprs: Vec<Expr>,
}

/// `MERGE NODES (a, b) INTO <target>` clause — see R180 / `arch/compatibility/native-procedures.md`.
///
/// Collapses two bound node variables into one within a single MVCC transaction.
/// The non-surviving node is DETACH DELETEd after property merge and edge transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeNodesClause {
    /// First node variable in `MERGE NODES (a, b)` — must be bound by preceding MATCH.
    pub source_a: String,
    /// Second node variable in `MERGE NODES (a, b)` — must be bound by preceding MATCH.
    pub source_b: String,
    /// `INTO <target>` — surviving node. Validated by parser to be one of `source_a` / `source_b`.
    pub target: String,
    /// Property merge strategy. Default: `KeepFirst` (target node wins).
    pub conflict: MergeNodesConflictStrategy,
    /// `TRANSFER EDGES FROM <src> TO <dst>` clause if present.
    /// Parser validates `dst == target` and `src` is the non-target source.
    /// When `None`, edges remain on the non-surviving node and are removed by the
    /// DETACH DELETE that finalises the merge (i.e., dropped).
    pub transfer_edges: Option<TransferEdgesEndpoints>,
    /// `ON DUPLICATE …` strategy for parallel edges discovered during transfer.
    /// Parser rejects this when `transfer_edges` is `None`.
    /// Default when transfer is present: `KeepBoth` (preserve parallel edges).
    pub duplicate: MergeNodesDuplicateStrategy,
    /// `TRANSFER EDGE PROPERTIES` — copy edge facets from non-surviving to surviving.
    ///
    /// Per arch/compatibility/native-procedures.md: "Edge properties are
    /// transferred by default." The clause is therefore a redundant
    /// readability marker — its absence does NOT mean "drop edge properties".
    /// The flag is preserved on the AST so future extensions can carry
    /// alternative policies without re-encoding the grammar.
    pub transfer_edge_properties: bool,
}

/// Endpoints of `TRANSFER EDGES FROM <src> TO <dst>` inside `MERGE NODES`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferEdgesEndpoints {
    /// Non-surviving source — edges are read from this node.
    pub src: String,
    /// Surviving destination — must equal `MergeNodesClause::target`.
    pub dst: String,
}

/// Property conflict resolution strategy for `MERGE NODES`.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MergeNodesConflictStrategy {
    /// `KEEP FIRST` (default) — surviving node's properties win on collision.
    /// Equivalent to `COALESCE(target.prop, source.prop)` but cheaper: the
    /// target's value is taken verbatim, source is only read for properties
    /// absent on the target.
    #[default]
    KeepFirst,
    /// `KEEP LAST` — non-surviving node's properties overwrite the surviving's.
    KeepLast,
    /// `COALESCE` — non-null values from the non-surviving fill nulls on the
    /// surviving. Differs from `KeepFirst` only when target has explicit-null
    /// property values: COALESCE replaces them, KeepFirst preserves them.
    Coalesce,
    /// `SET <expressions>` — per-property expressions. Each set item is
    /// evaluated against a row binding `a` → surviving-node, `b` → non-surviving-node.
    SetExpressions(Vec<SetItem>),
}

/// Duplicate-edge resolution strategy for `MERGE NODES TRANSFER EDGES`.
///
/// "Duplicate" means: target↔x already exists AND non-surviving↔x also exists
/// with the same edge type and direction.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MergeNodesDuplicateStrategy {
    /// `KEEP BOTH` (default) — both edges preserved as parallel edges.
    /// Posting list contains two entries for x.
    #[default]
    KeepBoth,
    /// `MERGE PROPERTIES` — single edge, edge facets merged via COALESCE
    /// (non-null source fills null target). Source edge is then removed.
    MergeProperties,
    /// `KEEP TARGET` — keep target↔x as-is, discard non-surviving↔x.
    /// Edge facets on the non-surviving edge are discarded.
    KeepTarget,
}

/// DETACH DOCUMENT clause: promote a nested DOCUMENT property to a graph node.
///
/// Example:
/// ```cypher
/// DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n)
///   TRANSFER EDGES ON n TO a WHERE type(r) IN ['SHIPS_TO', 'LIVES_AT']
/// ```
///
/// The new node receives the document's top-level keys as properties (shallow
/// promotion). Any nested maps remain as DOCUMENT properties on the new node.
#[derive(Debug, Clone, PartialEq)]
pub struct DetachDocumentClause {
    /// Source node variable (e.g. `n` in `n.address`). Must resolve to a bound
    /// row column produced by a preceding MATCH.
    pub source_variable: String,
    /// Property path on the source node (e.g. `["address"]` or `["meta", "shipping"]`).
    /// Must be non-empty.
    pub property_path: Vec<String>,
    /// Target node variable from the AS pattern (e.g. `a` in `(a:Address)`).
    pub target_variable: String,
    /// Labels applied to the new target node.
    pub target_labels: Vec<String>,
    /// Edge type connecting the target node to the source. If `None`, derive
    /// `HAS_<UPPER_SNAKE(property_path.last())>` at plan time.
    pub edge_type: Option<String>,
    /// Edge direction from the perspective of `source_variable`. `Outgoing`
    /// means `(source)-[:TYPE]->(target)`; `Incoming` means `(source)<-[:TYPE]-(target)`;
    /// the canonical form in the arch doc is `(a:Address)-[:HAS_ADDRESS]->(n)`
    /// which stores as `Incoming` from `n`'s perspective.
    pub edge_direction: EdgeFromSource,
    /// Optional variable name bound to the new edge (not yet used by executor).
    pub edge_variable: Option<String>,
    /// Optional TRANSFER EDGES clause.
    pub transfer: Option<TransferEdgesSpec>,
}

/// Direction of the edge created by DETACH DOCUMENT relative to the source node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeFromSource {
    /// `(source)-[:TYPE]->(target)` — source is the edge's source.
    Outgoing,
    /// `(source)<-[:TYPE]-(target)` — source is the edge's target (canonical form).
    Incoming,
}

/// `TRANSFER EDGES ON <node> TO <target> WHERE <predicate>` clause.
///
/// Re-points edges on `node_variable` to `target_variable` when the predicate
/// (typically `type(r) IN [...]`) matches.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferEdgesSpec {
    /// The node whose edges are being re-pointed (usually the source of DETACH).
    pub node_variable: String,
    /// The node that will receive the re-pointed edges (usually the new target).
    pub target_variable: String,
    /// Predicate filtering which edges to transfer. The edge variable in the
    /// predicate is always named `r`.
    pub predicate: Expr,
}

/// ATTACH DOCUMENT clause: demote a graph node to a nested DOCUMENT property.
///
/// Example:
/// ```cypher
/// ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address
///   TRANSFER EDGES ON a TO u WHERE type(r) = 'SHIPS_TO'
///   ON CONFLICT REPLACE
///   ON REMAINING FAIL
/// ```
///
/// The `source_variable` node's properties are read and written to the
/// `target_variable`'s property at `target_property_path` as a DOCUMENT.
/// The pattern's edge (a→u) is deleted. Any TRANSFER-matching edges on the
/// source are re-pointed onto the target before the source node is deleted.
#[derive(Debug, Clone, PartialEq)]
pub struct AttachDocumentClause {
    /// Source node variable (the node being demoted). From pattern `(a:L)-...->(u)`.
    pub source_variable: String,
    /// Optional labels on the source node (used to filter the match).
    pub source_labels: Vec<String>,
    /// Target node variable (the node receiving the new DOCUMENT property).
    pub target_variable: String,
    /// Optional labels on the target node.
    pub target_labels: Vec<String>,
    /// The edge type between source and target.
    pub edge_type: String,
    /// Direction of the edge relative to the source. `Outgoing` ⇒ `(a)-[:T]->(u)`;
    /// `Incoming` ⇒ `(a)<-[:T]-(u)`.
    pub edge_direction: EdgeFromSource,
    /// Optional edge variable (for downstream reference; not bound into row output).
    pub edge_variable: Option<String>,
    /// Target-path variable. Must equal `target_variable` per the arch spec
    /// (the nested property lives on the target node).
    pub target_property_variable: String,
    /// Property path where the promoted DOCUMENT is written (non-empty, e.g. `["address"]`).
    pub target_property_path: Vec<String>,
    /// Optional TRANSFER EDGES spec. Re-points edges before the source node is deleted.
    pub transfer: Option<TransferEdgesSpec>,
    /// If true, overwrite the target property when it already exists
    /// (`ON CONFLICT REPLACE`). Default false — existing target errors.
    pub on_conflict_replace: bool,
    /// If true, error when any untransferred edges remain on the source node
    /// (`ON REMAINING FAIL`). Default false — cascade-delete remaining edges.
    pub on_remaining_fail: bool,
}

/// A single SET operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SetItem {
    /// `n.prop = expr` — set a single property.
    Property {
        variable: String,
        property: String,
        expr: Expr,
    },
    /// `n.a.b.c = expr` — set a nested DOCUMENT path via merge operand.
    PropertyPath {
        variable: String,
        path: Vec<String>,
        expr: Expr,
    },
    /// `doc_push(n.tags, "new")` — document array/numeric mutation via merge operand.
    DocFunction {
        /// Function name: doc_push, doc_pull, doc_add_to_set, doc_inc.
        function: String,
        /// Variable name (e.g., "n").
        variable: String,
        /// Property path (e.g., ["tags"] or ["stats", "views"]).
        path: Vec<String>,
        /// Value expression to push/pull/add/increment.
        value_expr: Expr,
    },
    /// `n = {map}` — replace all properties with map.
    ReplaceProperties { variable: String, expr: Expr },
    /// `n += {map}` — merge properties from map.
    MergeProperties { variable: String, expr: Expr },
    /// `n:Label` — add a label.
    AddLabel { variable: String, label: String },
}

/// A single REMOVE operation.
#[derive(Debug, Clone, PartialEq)]
pub enum RemoveItem {
    /// `n.prop` — remove a property.
    Property { variable: String, property: String },
    /// `n.a.b.c` — remove a nested DOCUMENT path via merge operand.
    PropertyPath { variable: String, path: Vec<String> },
    /// `n:Label` — remove a label.
    Label { variable: String, label: String },
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Modulo,

    // Comparison
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,

    // Logical
    And,
    Or,
    Xor,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Not,
    Neg,
}

/// List quantifier kind for [`Expr::ListPredicate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListPredicateKind {
    /// `all(...)` — predicate holds for every element (vacuously true for `[]`).
    All,
    /// `any(...)` — predicate holds for at least one element (false for `[]`).
    Any,
    /// `none(...)` — predicate holds for no element (vacuously true for `[]`).
    None,
    /// `single(...)` — predicate holds for exactly one element.
    Single,
}

/// String matching operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringOp {
    StartsWith,
    EndsWith,
    Contains,
    /// `=~` regex match (whole-string, Neo4j semantics).
    Regex,
}

#[cfg(test)]
#[allow(clippy::panic, clippy::expect_used)]
mod tests;
