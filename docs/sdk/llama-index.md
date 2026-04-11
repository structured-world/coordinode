# LlamaIndex Integration

CoordiNode integrates with LlamaIndex via `llama-index-graph-stores-coordinode`, a `PropertyGraphStore` implementation that plugs into `PropertyGraphIndex` and LlamaIndex Knowledge Graph workflows.

## Installation

```bash
# Start CoordiNode
docker compose -f sdk/docker-compose.yml up -d

# Install packages
pip install coordinode llama-index-graph-stores-coordinode llama-index-core llama-index-llms-openai
```

## PropertyGraphIndex

Index documents and query them through the knowledge graph:

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.graph_stores.coordinode import CoordinodePropertyGraphStore
from llama_index.llms.openai import OpenAI

documents = SimpleDirectoryReader("./data").load_data()

graph_store = CoordinodePropertyGraphStore("localhost:7080")

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    llm=OpenAI(model="gpt-4o-mini"),
    show_progress=True,
)

query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("What is the relationship between attention and transformers?")
print(response)
```

## Load Existing Graph

```python
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.coordinode import CoordinodePropertyGraphStore

graph_store = CoordinodePropertyGraphStore("localhost:7080")
index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)

retriever = index.as_retriever(
    include_text=True,
    similarity_top_k=5,
)
nodes = retriever.retrieve("transformer attention mechanism")
for node in nodes:
    print(node.get_content())
```

## Hybrid Retrieval (Graph + Vector)

`CoordinodePropertyGraphStore` implements `supports_vector_queries = True`. LlamaIndex will automatically route vector queries through CoordiNode's HNSW index:

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.graph_stores.coordinode import CoordinodePropertyGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding

graph_store = CoordinodePropertyGraphStore("localhost:7080")

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    embed_model=OpenAIEmbedding(),
)

retriever = VectorContextRetriever(
    index.property_graph_store,
    embed_model=OpenAIEmbedding(),
    include_text=True,
    similarity_top_k=5,
)
nodes = retriever.retrieve("transformers and attention")
```

## Direct Cypher

```python
with CoordinodePropertyGraphStore("localhost:7080") as store:
    rows = store.structured_query(
        "MATCH (e:Entity)-[r]->(t) "
        "WHERE e.name = $name "
        "RETURN e.name, type(r), t.name LIMIT 20",
        param_map={"name": "BERT"},
    )
    for row in rows:
        print(row)
```

## Manual Node + Relation Upsert

```python
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.graph_stores.coordinode import CoordinodePropertyGraphStore

store = CoordinodePropertyGraphStore("localhost:7080")

nodes = [
    EntityNode(name="BERT", label="Model", properties={"year": 2018}),
    EntityNode(name="Attention", label="Concept"),
]
store.upsert_nodes(nodes)

relations = [
    Relation(label="USES", source_id="BERT", target_id="Attention"),
]
store.upsert_relations(relations)
```
