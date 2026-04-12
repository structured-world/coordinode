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

## See Also

- [LlamaIndex integration](/sdk/llama-index) — PropertyGraphIndex with CoordiNode backend
- [LangChain integration](/sdk/langchain) — GraphCypherQAChain
- [GitHub: structured-world/coordinode-python](https://github.com/structured-world/coordinode-python)
