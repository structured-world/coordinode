# Embedded (Rust)

`coordinode-embed` lets you run CoordiNode in-process — no separate server, no network hop, no external dependency. Ideal for desktop applications, test harnesses, and edge deployments.

## Add the Dependency

```toml
[dependencies]
coordinode-embed = "0.3"
```

## Minimal Example

```rust
use coordinode_embed::Database;

fn main() -> anyhow::Result<()> {
    let mut db = Database::open("/tmp/my-graph")?;

    // Execute Cypher directly — no gRPC
    db.execute_cypher("CREATE (n:Person {name: 'Alice'}) RETURN n")?;

    let rows = db.execute_cypher("MATCH (n:Person) RETURN n.name")?;
    println!("{rows:?}");
    Ok(())
}
```

## API Reference

```rust
use coordinode_embed::{Database, DatabaseError};
use coordinode_core::graph::types::Value;
use std::collections::HashMap;

// Open (creates directory if absent)
let mut db = Database::open("/path/to/data")?;

// Execute Cypher — no parameters
let rows = db.execute_cypher("MATCH (n:Person) RETURN n.name")?;

// Execute with parameters
let mut params = HashMap::new();
params.insert("name".to_string(), Value::String("Alice".into()));
let rows = db.execute_cypher_with_params(
    "MATCH (n:Person {name: $name}) RETURN n",
    params,
)?;
```

## What Is and Isn't Available

The embedded API exposes the full query engine — graph, vector, full-text, time-series — with the same Cypher dialect as the network server. Only network-facing features (gRPC server, REST transcoding, Prometheus scrape endpoint) are absent.

## Testing with the Embedded Engine

Use `tempfile::tempdir()` for isolated per-test databases that clean up automatically on drop:

```rust
use coordinode_embed::Database;

#[test]
fn test_find_related_concepts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher(
        "CREATE (:Concept {name: 'ML'})-[:RELATED_TO]->(:Concept {name: 'AI'})"
    ).expect("create");

    let rows = db
        .execute_cypher("MATCH (a:Concept)-[:RELATED_TO]->(b) RETURN b.name")
        .expect("match");

    assert_eq!(rows.len(), 1);
}
```

Add `tempfile` to `[dev-dependencies]`:

```toml
[dev-dependencies]
tempfile = "3"
```

## Next Step

See [Quick Start](../QUICKSTART) for a full example using the Docker server, or [Data Model](./data-model) to understand the graph + vector model.
