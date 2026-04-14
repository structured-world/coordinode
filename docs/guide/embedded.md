# Embedded (Rust)

`coordinode-embed` lets you run CoordiNode in-process — no separate server, no network hop, no external dependency. Ideal for desktop applications, test harnesses, and edge deployments.

## Add the Dependency

```toml
[dependencies]
coordinode-embed = "0.3"
tokio = { version = "1", features = ["full"] }
```

## Minimal Example

```rust
use coordinode_embed::{CoordinodeEmbed, Config};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db = CoordinodeEmbed::open(Config::in_memory()).await?;

    // Execute Cypher directly — no gRPC
    let result = db
        .query("CREATE (n:Person {name: $name}) RETURN n")
        .param("name", "Alice")
        .run()
        .await?;

    println!("{result:?}");
    Ok(())
}
```

## Configuration

```rust
// In-memory (data lost on drop — good for tests)
let config = Config::in_memory();

// Persistent on-disk
let config = Config::builder()
    .data_dir("/var/lib/myapp/graph")
    .build();

let db = CoordinodeEmbed::open(config).await?;
```

## What Is and Isn't Available

The embedded API exposes the full query engine — graph, vector, full-text, time-series — with the same Cypher dialect as the network server. Only network-facing features (gRPC server, REST transcoding, Prometheus scrape endpoint) are absent.

## Testing with the Embedded Engine

`Config::in_memory()` is the recommended approach for integration tests. Each test gets an isolated, zero-cleanup instance:

```rust
#[tokio::test]
async fn test_find_related_concepts() -> anyhow::Result<()> {
    let db = CoordinodeEmbed::open(Config::in_memory()).await?;

    db.query("CREATE (:Concept {name: 'ML'})-[:RELATED_TO]->(:Concept {name: 'AI'})")
        .run().await?;

    let result = db
        .query("MATCH (a:Concept)-[:RELATED_TO]->(b) RETURN b.name")
        .run().await?;

    assert_eq!(result.rows[0]["b.name"], "AI");
    Ok(())
}
```

## Next Step

See [Quick Start](../QUICKSTART) for a full example using the Docker server, or [Data Model](./data-model) to understand the graph + vector model.
