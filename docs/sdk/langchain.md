# LangChain Integration

CoordiNode integrates with LangChain via `langchain-coordinode`, a `GraphStore` implementation that supports `GraphCypherQAChain` and any LangChain component expecting a graph backend.

## Installation

```bash
# Start CoordiNode
docker compose -f sdk/docker-compose.yml up -d

# Install packages
pip install coordinode langchain-coordinode langchain-openai
```

## GraphCypherQAChain

The most common use case: ask natural language questions answered via generated Cypher.

```python
from langchain_coordinode import CoordinodeGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

graph = CoordinodeGraph("localhost:7080")
graph.refresh_schema()

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4o-mini"),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

result = chain.invoke({"query": "What concepts are related to machine learning?"})
print(result["result"])
```

## Direct Cypher Queries

```python
with CoordinodeGraph("localhost:7080") as graph:
    rows = graph.query(
        "MATCH (n:Concept)-[:RELATED_TO]->(m) "
        "WHERE n.name = $name RETURN m.name AS related",
        params={"name": "machine learning"},
    )
    for row in rows:
        print(row["related"])
```

## Hybrid Retrieval (Graph + Vector + Text)

CoordiNode supports hybrid queries mixing graph traversal with vector similarity and full-text search in a single Cypher statement:

```python
import numpy as np
from langchain_coordinode import CoordinodeGraph

graph = CoordinodeGraph("localhost:7080")

question_embedding = get_embedding("transformer attention mechanism")  # your embedding model

rows = graph.query(
    """
    MATCH (topic:Concept {name: "machine learning"})-[:RELATED_TO*1..3]->(related)
    MATCH (related)<-[:ABOUT]-(doc:Document)
    WHERE vector_distance(doc.embedding, $q_vec) < 0.4
      AND text_match(doc.body, $keywords)
    RETURN doc.title,
           vector_distance(doc.embedding, $q_vec) AS relevance,
           text_score(doc.body, $keywords)          AS text_rank
    ORDER BY relevance LIMIT 10
    """,
    params={
        "q_vec": question_embedding.tolist(),
        "keywords": "transformer attention mechanism",
    },
)
for row in rows:
    print(f"{row['doc.title']}  relevance={row['relevance']:.3f}")
```

## GraphRAG Pipeline

A minimal GraphRAG pipeline: ingest documents, store embeddings, query via hybrid retrieval.

```python
from langchain_coordinode import CoordinodeGraph
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
graph = CoordinodeGraph("localhost:7080")

def ingest(docs: list[dict]) -> None:
    """Store documents with embeddings."""
    for doc in docs:
        vec = embeddings.embed_query(doc["text"])
        graph.query(
            "MERGE (d:Document {id: $id}) "
            "SET d.title = $title, d.body = $text, d.embedding = $vec",
            params={"id": doc["id"], "title": doc["title"],
                    "text": doc["text"], "vec": vec},
        )

def search(question: str, top_k: int = 5) -> list[dict]:
    """Find relevant documents via vector similarity."""
    vec = embeddings.embed_query(question)
    return graph.query(
        "MATCH (d:Document) "
        "WHERE vector_distance(d.embedding, $vec) < 0.5 "
        "RETURN d.title AS title, d.body AS text "
        f"ORDER BY vector_distance(d.embedding, $vec) LIMIT {top_k}",
        params={"vec": vec},
    )
```

## Schema Inspection

```python
graph = CoordinodeGraph("localhost:7080")
print(graph.schema)           # human-readable text
print(graph.structured_schema)  # {"node_props": {...}, "rel_props": {...}, ...}
```
