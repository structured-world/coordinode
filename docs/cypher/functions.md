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

Both functions work by reading metadata set during WHERE evaluation. They are only meaningful after a `WHERE text_match(...)` clause in the same query.

```cypher
MATCH (doc:Document)
WHERE text_match(doc.body, "raft consensus")
RETURN doc.title, text_score(doc.body, "raft consensus") AS relevance
ORDER BY relevance DESC
```

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
| `percentileCont` | `percentileCont(x, p)` | Float | Interpolated percentile (linear interpolation). `p` must be in [0.0, 1.0] |
| `percentileDisc` | `percentileDisc(x, p)` | Float | Discrete percentile (nearest rank). `p` must be in [0.0, 1.0] |

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
