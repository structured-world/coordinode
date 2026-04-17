---
description: "CoordiNode Cypher function reference — scalar, aggregation, vector, full-text, spatial, and document functions with implementation status."
---

# Functions Reference

::: warning Functions that silently return null
Several standard OpenCypher scalar functions (`toInteger`, `toLower`, `length`, `abs`, and others) are recognized by the parser but not yet implemented — they return `null` without error. This table lists every function and its current status so you can avoid silent failures.
:::

## Scalar Functions

Functions that compute a single value from their arguments.

### Implemented ✅

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `coalesce` | `coalesce(expr, ...)` | first non-null | Any number of arguments |
| `toString` | `toString(x)` | String | Converts Int, Float, Bool, String |
| `size` | `size(x)` | Integer | String → **byte count** (UTF-8 bytes, not characters); Array → element count |
| `type` | `type(r)` | String | Relationship type (e.g. `"KNOWS"`) |
| `labels` | `labels(n)` | List\<String\> | Returns list with one label string. Subscript access `labels(n)[0]` is supported |
| `now` | `now()` | Timestamp | Current timestamp (microseconds since epoch) |


### Vector Functions ✅ 🔷

CoordiNode extensions. Operate on `Vector` or `Array` of Float/Int values.

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `vector_distance` | `vector_distance(a, b)` | Float | Euclidean (L2) distance |
| `vector_similarity` | `vector_similarity(a, b)` | Float | Cosine similarity. Range [0, 1] |
| `vector_dot` | `vector_dot(a, b)` | Float | Dot product |
| `vector_manhattan` | `vector_manhattan(a, b)` | Float | Manhattan (L1) distance. Only accepts `Vector` values — not `Array` |

`vector_distance`, `vector_similarity`, and `vector_dot` accept both `Vector` and `Array` of Float/Int. `vector_manhattan` requires native `Vector` values; passing plain arrays returns `null`. Both arguments must have identical dimensionality.

```cypher
MATCH (p:Product)
WHERE vector_distance(p.embedding, $query) < 0.5
RETURN p.name, vector_similarity(p.embedding, $query) AS score
ORDER BY score DESC
LIMIT 10
```

### Full-Text Functions ✅ 🔷

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `text_match` | `text_match(field, query)` | Boolean | True if node passed text filter in WHERE |
| `text_score` | `text_score(field, query)` | Float | BM25 score from the WHERE filter pass |

Both functions work by reading metadata set during WHERE evaluation. They are only meaningful after a `WHERE text_match(...)` clause in the same query; using `text_score` on a field without a full-text index (or without a paired `text_match`) is a query-time error, not a silent zero.

`text_match(field, query)` itself also hard-fails when the `(Label, property)` has no full-text index — it does **not** silently pass every row through. Earlier versions returned all rows with a warning, which turned `WHERE text_match(...)` into an unintentional no-op filter; the current behaviour is `ExecutionError: text_match() requires a full-text index on (:Label, property); create one with CREATE TEXT INDEX idx_name ON :Label(property)`. Create the index before invoking the filter.

BM25 uses tantivy's default parameters (`k1 = 1.2`, `b = 0.75`) and is not runtime-configurable (see [ADR-020](../arch/decisions.md) for rationale).

```cypher
MATCH (doc:Document)
WHERE text_match(doc.body, "raft consensus")
RETURN doc.title, text_score(doc.body, "raft consensus") AS relevance
ORDER BY relevance DESC
```

### Hybrid Scoring Helper ✅ 🔷

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `hybrid_score` | `hybrid_score(node, query [, weights])` | Float | Opinionated blend of vector + text scores cached on the row |

`hybrid_score` is sugar for the arithmetic pattern `w_vec · (1 - vector_distance) + w_text · text_score`. It reads the vector score cached by `VectorFilter` (from `WHERE vector_distance(...) < X` / `vector_similarity(...) > X`) and the BM25 score cached by `TextFilter` (from `WHERE text_match(...)`), then blends them with default weights `{vector: 0.65, text: 0.35}` or an override map passed as the third argument.

Vector normalisation is automatic based on the metric used in `WHERE`:
- `vector_similarity` (cosine) → used raw (already bounded to `[-1, 1]` / `[0, 1]`)
- `vector_distance` (L2) → `1 - clamp(raw, 0, 1)` (lower distance ⇒ higher similarity)
- `vector_manhattan` (L1) → `1 / (1 + raw)` (asymptotic, always in `(0, 1]`)
- `vector_dot` → used raw (unbounded — verify your data is normalised)

```cypher
// Default blend — equivalent to (1 - vector_distance) * 0.65 + text_score * 0.35
MATCH (c:Chunk)
WHERE text_match(c.body, "rust consensus")
  AND vector_distance(c.embedding, $query_vec) < 0.6
RETURN c.text, hybrid_score(c, "rust consensus") AS score
ORDER BY score DESC
LIMIT 20
```

```cypher
// Vector-heavy blend — override via weights map
RETURN hybrid_score(c, $q, {vector: 0.9, text: 0.1}) AS score
```

```cypher
// Single-modality degenerate cases are supported — if only one filter ran,
// hybrid_score returns that side's normalised score alone (no blend).
MATCH (d:Doc) WHERE text_match(d.body, "rust")
RETURN hybrid_score(d, "rust") AS score    // text-only, returns text_score
```

`hybrid_score` requires at least one of `text_match(...)` or `vector_distance(...)` / `vector_similarity(...)` in `WHERE` against the same node — calling it on a plan with neither filter is a query-time error (no silent zeros).

### Document-Level Aggregate ✅ 🔷

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `doc_score` | `doc_score(doc, query [, α, β, γ])` or `doc_score(doc, query, {alpha, beta, gamma})` | Float | Document-level aggregate over `HAS_CHUNK` children |

`doc_score` computes `α·max_chunk + β·avg_chunk + γ·coverage` for a Document by traversing its outward `HAS_CHUNK` edges and scoring each Chunk's `embedding` property against the query vector with cosine similarity. Defaults: `α=0.5`, `β=0.3`, `γ=0.2`.

```cypher
// Rank documents by relevance to a query embedding
MATCH (d:Document)
RETURN d.title, doc_score(d, $query_vec) AS score
ORDER BY score DESC LIMIT 20
```

```cypher
// Override weights — pure max_chunk (peak relevance), ignore breadth and coverage
RETURN doc_score(d, $query_vec, 1.0, 0.0, 0.0) AS score
```

```cypher
// Override via map — equivalent, keyword form
RETURN doc_score(d, $query_vec, {alpha: 0.7, beta: 0.2, gamma: 0.1}) AS score
```

**Semantics:**
- `max_chunk` and `avg_chunk` are over chunks whose `embedding` property is a valid vector of the query's dimension. Chunks without an embedding are ignored in the aggregate.
- `coverage = matching_chunks / total_chunks` where `total_chunks` counts **all** `HAS_CHUNK` children (including those without an embedding) and `matching_chunks` counts only chunks with a non-negative cosine similarity to the query (topically aligned). Chunks with negative cosine — pointing away from the query — are not "matching".
- If the document has zero `HAS_CHUNK` children, `doc_score` returns `0.0` (not `null`, not an error).
- If none of the scored chunks are "matching" (all have `sim ≤ 0`), `coverage = 0` and only the max/avg terms contribute.

**Error conditions** (plan-time where possible):
- `doc_score(d)` — fewer than 2 arguments rejected.
- `doc_score(d, q, a, b)` — 4 arguments rejected (use 2, 3, or 5).
- `doc_score(d, q, a, b, c, d2)` — 6+ arguments rejected.
- `doc_score(42, q)` — first argument must be a bound variable, not a literal.
- `doc_score(d, q, {delta: 0.5})` — map keys other than `alpha`, `beta`, `gamma` rejected.
- `doc_score` inside a `WHERE` clause — rejected (it is a correlated aggregate, not a filter predicate).

### Reciprocal Rank Fusion ✅ 🔷

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `rrf_score` | `rrf_score([method_exprs...], {vector: ..., text: ...})` | Float | Reciprocal Rank Fusion over N scoring methods |

`rrf_score` fuses multiple scoring methods (vector similarity, BM25 text match, edge vector) into one ranking by summing `1 / (60 + rank_i)` across methods. Rank is positional over the full result set, so this is a materialising operator — the planner inserts a `RankFuse` stage that scores every row, sorts per method, assigns 1-based competition ranks (ties share a rank), and writes the fused score to the row. The constant `k = 60` is the IR standard from Cormack et al. 2009 and is deliberately non-tunable — freezing `k` is part of the stable API contract.

```cypher
// Rank-fuse vector + text on the same chunks
MATCH (c:Chunk)
RETURN c.text,
       rrf_score([c.embedding, c.body], {vector: $query_vec, text: "raft consensus"}) AS score
ORDER BY score DESC
LIMIT 20
```

```cypher
// Node vector + edge vector + text in a single RRF
MATCH (u:User {id: $me})-[r:RATED]->(m:Movie)
RETURN m.title,
       rrf_score([m.embedding, r.context_emb, m.summary],
                 {vector: $query_vec, text: $query_text}) AS score
ORDER BY score DESC LIMIT 10
```

**Signature rules:**

- First argument is a non-empty list of property expressions. Each expression must resolve to either a vector property (HNSW index on `(Label, property)`, or an edge vector property) or a text property (full-text index on `(Label, property)`).
- Second argument is a map literal with the keys the methods need: `vector` when any method is a vector expression, `text` when any method is a text expression. Map parameters (`$q`) are accepted and shape-checked at execution time.
- Any third argument (including attempts to override `k`) is rejected at plan time.

**When RRF beats weighted sum:** vector similarity lives in `[0, 1]` while BM25 scores are unbounded, so a naïve weighted sum is dominated by whichever method has larger raw scores. RRF uses only ranks, so scale mismatches across methods cancel out — a document that is top-K on every method wins regardless of raw-score magnitude.

**Error conditions** (plan-time where possible):
- `rrf_score([...], query, {k: N})` — 3+ arguments rejected.
- `rrf_score([], ...)` — empty method list rejected.
- `rrf_score(..., {unknown: ...})` — map keys other than `vector` / `text` rejected.
- `rrf_score` inside a `WHERE` clause — rejected (rank is not defined pre-materialisation).
- Method `var.prop` with no HNSW vector index and no full-text index and no vector values — rejected at execute time with a message naming the offending method.

### Encrypted Search Function ✅ 🔷

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `encrypted_match` | `encrypted_match(field, token)` | Boolean | True if node passed SSE filter |

```cypher
MATCH (p:Patient)
WHERE encrypted_match(p.ssn, $encrypted_token)
RETURN p.id, p.name
```

### Spatial Functions ✅ 🔷

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `point` | `point({latitude: f, longitude: f})` | Geo point | WGS84 coordinate |
| `point.distance` | `point.distance(p1, p2)` | Float (meters) | Haversine great-circle distance |

```cypher
MATCH (r:Restaurant)
WHERE point.distance(r.location, point({latitude: 40.7128, longitude: -74.0060})) < 2000
RETURN r.name
ORDER BY point.distance(r.location, point({latitude: 40.7128, longitude: -74.0060}))
```

---

### Not Yet Implemented 📋

These functions are recognized by the parser but return `null`. Using them produces no error — just a silent `null` result.

::: danger Silently return null
Do not use the following functions in production code until they are implemented. They parse successfully but always return `null`.
:::

**Type conversion:**

| Function | Expected behavior |
|----------|------------------|
| `toInteger(x)` | Convert String/Float to Integer |
| `toFloat(x)` | Convert String/Integer to Float |
| `toBoolean(x)` | Convert String to Boolean |

**String:**

| Function | Expected behavior |
|----------|------------------|
| `toLower(x)` | Lowercase string |
| `toUpper(x)` | Uppercase string |
| `trim(x)` | Strip leading/trailing whitespace |
| `ltrim(x)` | Strip leading whitespace |
| `rtrim(x)` | Strip trailing whitespace |
| `length(x)` | String length (char count) |
| `left(x, n)` | First n characters |
| `right(x, n)` | Last n characters |
| `substring(x, start, length)` | Substring |
| `replace(x, find, repl)` | String replacement |
| `split(x, delimiter)` | Split string to list |
| `reverse(x)` | Reverse string or list |

**Math:**

| Function | Expected behavior |
|----------|------------------|
| `abs(x)` | Absolute value |
| `ceil(x)` | Round up to integer |
| `floor(x)` | Round down to integer |
| `round(x)` | Round to nearest integer |
| `sign(x)` | -1, 0, or 1 |
| `sqrt(x)` | Square root |
| `exp(x)` | e^x |
| `log(x)` | Natural logarithm |
| `log10(x)` | Base-10 logarithm |
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometric |
| `rand()` | Random float [0.0, 1.0) |

**List:**

| Function | Expected behavior |
|----------|------------------|
| `head(list)` | First element |
| `tail(list)` | All elements except first |
| `last(list)` | Last element |
| `range(start, end)` | Integer list [start..end] |
| `range(start, end, step)` | Integer list with step |
| `reverse(list)` | Reverse list |

**Node / graph:**

| Function | Expected behavior |
|----------|------------------|
| `id(n)` | Internal node ID |
| `elementId(n)` | Element ID string |
| `properties(n)` | All properties as map |
| `keys(n)` | Property key list |
| `nodes(path)` | Nodes in a path |
| `relationships(path)` | Relationships in a path |
| `length(path)` | Hop count in a path |

---

## Aggregation Functions

Aggregation functions require a GROUP BY context (explicit or implicit via non-aggregated columns).

### Implemented ✅

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `count` | `count(*)` | Integer | All rows (ignores DISTINCT) |
| `count` | `count(x)` | Integer | Non-null values |
| `count` | `count(DISTINCT x)` | Integer | Distinct non-null values |
| `sum` | `sum(x)` | Int or Float | Null values skipped |
| `avg` | `avg(x)` | Float | Null values skipped |
| `min` | `min(x)` | same as x | Works on Int, Float, String, Timestamp |
| `max` | `max(x)` | same as x | Works on Int, Float, String, Timestamp |
| `collect` | `collect(x)` | List | Collects non-null values into array |
| `collect` | `collect(DISTINCT x)` | List | Distinct non-null values |
| `stDev` | `stDev(x)` | Float | Sample standard deviation |
| `stDevP` | `stDevP(x)` | Float | Population standard deviation |
| `percentileCont` | `percentileCont(x, p)` | Float | Interpolated percentile (linear interpolation). `p` in [0.0, 1.0] — literal or query parameter |
| `percentileDisc` | `percentileDisc(x, p)` | Float | Discrete percentile (nearest rank). `p` in [0.0, 1.0] — literal or query parameter |

Both functions accept a literal (`0.9`) or a named query parameter (`$p`) as the percentile argument. Out-of-range values are clamped to `[0.0, 1.0]`.

```cypher
// Literal percentile
MATCH (emp:Employee)
RETURN percentileCont(emp.salary, 0.9) AS p90_salary

// Query parameter — pass { p: 0.95 } from the driver
MATCH (emp:Employee)
RETURN percentileCont(emp.salary, $p) AS salary_percentile
```

```cypher
MATCH (dept:Department)<-[:WORKS_IN]-(emp:Employee)
RETURN dept.name,
       count(emp)                      AS headcount,
       avg(emp.salary)                 AS avg_salary,
       max(emp.salary)                 AS max_salary,
       stDev(emp.salary)               AS salary_spread,
       collect(emp.name)               AS team
ORDER BY headcount DESC
```
