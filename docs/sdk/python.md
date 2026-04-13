---
title: Python SDK
description: Official Python SDK for CoordiNode — connect to the gRPC API with sync and async clients.
---

# Python SDK

The official CoordiNode Python SDK provides sync and async gRPC clients.

```bash
pip install coordinode
```

## Quick Start

```python
from coordinode import CoordinodeClient

client = CoordinodeClient("localhost:7080")

# Execute a Cypher query
results = client.cypher("MATCH (n:Person) RETURN n.name LIMIT 10")
for row in results:
    print(row["n.name"])

# Vector search
hits = client.vector_search(
    label="Document",
    property="embedding",
    vector=[0.1, 0.2, ...],  # 384-dim
    top_k=5,
)
```

## Source Location Tracking

Enable `debug_source_tracking` in development to let the Query Advisor map slow-query suggestions to exact call sites in your application.

```python
from coordinode import CoordinodeClient

client = CoordinodeClient(
    "localhost:7080",
    debug_source_tracking=True,
    app_name="my-service",
    app_version="1.2.3",
)

# Every query call automatically attaches the call site to the request.
# The server will record:
#   x-source-file: "api/handlers/feed.py"
#   x-source-line: "47"
#   x-source-app:  "my-service"
results = client.cypher("MATCH (u:User)-[:FOLLOWS]->(f) RETURN f LIMIT 10")
```

Source tracking uses Python's `inspect.stack()` to read the immediate caller's file and line number. The overhead is negligible and is zero when `debug_source_tracking=False`.

> **Recommendation:** Enable in development and staging; disable in production.

## See Also

- [TypeScript SDK](/sdk/typescript) — Node.js gRPC client
- [LlamaIndex integration](/sdk/llama-index) — PropertyGraphIndex with CoordiNode backend
- [LangChain integration](/sdk/langchain) — GraphCypherQAChain
- [GitHub: structured-world/coordinode-python](https://github.com/structured-world/coordinode-python)
