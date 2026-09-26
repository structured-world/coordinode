// ============================================================================
// CoordiNode GraphRAG Query Examples (GQ-1 through GQ-10)
// Document-as-Graph pattern: segmented documents stored as graph of chunks,
// entities, and semantic relationships.
//
// What each query exercises:
//
//   GQ-1  Hybrid vector+BM25           vector similarity + full-text score
//   GQ-2  Context expansion            OPTIONAL MATCH over the chunk chain
//   GQ-3  Entity-aware RAG             collect + map projection
//   GQ-4  Cross-document via entities  multi-hop, count(DISTINCT)
//   GQ-5  Document-level scoring       grouping with max / avg / collect
//   GQ-6  Centroid two-stage           document-level then chunk-level vectors
//   GQ-7  Edge vector traversal        vector property on a relationship
//   GQ-8  Multi-hop KG RAG             variable-length path *1..N
//   GQ-9  Temporal RAG                 AS OF TIMESTAMP time-travel read
//   GQ-10 Full pipeline                all of the above in one query
// ============================================================================

// --- Schema ---

// Document node — metadata about the source document
// CREATE LABEL Document (
//   title STRING,
//   source STRING,
//   corpus STRING,
//   created_at TIMESTAMP,
//   chunk_count INT,
//   centroid_embedding VECTOR(384)   -- average of all chunk embeddings
// );

// Section node — structural hierarchy (chapters, sections, headings)
// CREATE LABEL Section (
//   title STRING,
//   level INT,          -- heading level (1=H1, 2=H2, etc.)
//   position INT
// );

// Chunk node — the fundamental RAG unit (paragraph, fixed-size, or semantic chunk)
// CREATE LABEL Chunk (
//   text STRING,
//   embedding VECTOR(384),
//   position INT,
//   token_count INT,
//   metadata DOCUMENT   -- nested schema-free: {page, source_line, language, ...}
// );

// Entity node — extracted named entities, concepts, terms
// CREATE LABEL Entity (
//   name STRING,
//   type STRING,         -- PERSON, ORG, CONCEPT, LOCATION, ...
//   embedding VECTOR(384),
//   description STRING
// );

// Edges:
//   (:Document)-[:HAS_SECTION]->(:Section)
//   (:Section)-[:HAS_CHUNK]->(:Chunk)
//   (:Document)-[:HAS_CHUNK]->(:Chunk)          -- direct link (flat docs)
//   (:Chunk)-[:NEXT_CHUNK]->(:Chunk)             -- sequential ordering
//   (:Chunk)-[:MENTIONS]->(:Entity)              -- entity extraction
//   (:Entity)-[:RELATED_TO]->(:Entity)           -- knowledge graph layer
//   (:Chunk)-[:SIMILAR_TO {similarity: FLOAT, embedding: VECTOR(384)}]->(:Chunk)
//     ^^ the relationship carries its own embedding


// ============================================================================
// GQ-1. BASIC: Hybrid vector + full-text retrieval
// Find chunks most relevant to a question using both semantic and keyword match.
// ============================================================================

MATCH (c:Chunk)
WHERE vector_similarity(c.embedding, $query_vec) > 0.6
  AND text_score(c.text, $search_text) > 0.3
WITH c,
  vector_similarity(c.embedding, $query_vec) AS vec_score,
  text_score(c.text, $search_text) AS bm25_score,
  0.6 * vector_similarity(c.embedding, $query_vec) +
  0.4 * text_score(c.text, $search_text) AS hybrid_score
ORDER BY hybrid_score DESC
LIMIT 20
RETURN c.text, c.position, hybrid_score, vec_score, bm25_score;


// ============================================================================
// GQ-2. CONTEXT EXPANSION: Get surrounding chunks for LLM context window
// After finding relevant chunks, expand ±2 neighbors for coherent context.
// ============================================================================

MATCH (c:Chunk)
WHERE vector_similarity(c.embedding, $query_vec) > 0.7
WITH c ORDER BY vector_similarity(c.embedding, $query_vec) DESC LIMIT 5

// Walk backward up to 2 chunks
OPTIONAL MATCH (prev2:Chunk)-[:NEXT_CHUNK]->(prev1:Chunk)-[:NEXT_CHUNK]->(c)
// Walk forward up to 2 chunks
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next1:Chunk)-[:NEXT_CHUNK]->(next2:Chunk)

RETURN
  prev2.text AS context_before_2,
  prev1.text AS context_before_1,
  c.text AS matched_chunk,
  next1.text AS context_after_1,
  next2.text AS context_after_2,
  vector_similarity(c.embedding, $query_vec) AS score;


// ============================================================================
// GQ-3. ENTITY-AWARE RAG: Find chunks + their extracted entities
// Enrich LLM context with structured knowledge from entity graph.
// ============================================================================

MATCH (c:Chunk)
WHERE vector_similarity(c.embedding, $query_vec) > 0.7
WITH c ORDER BY vector_similarity(c.embedding, $query_vec) DESC LIMIT 10

MATCH (c)-[:MENTIONS]->(e:Entity)
WITH c, collect(DISTINCT e { .name, .type }) AS entities,
  vector_similarity(c.embedding, $query_vec) AS score

RETURN c.text, entities, score
ORDER BY score DESC;


// ============================================================================
// GQ-4. CROSS-DOCUMENT: Find related chunks across documents via entity graph
// "What other documents discuss the same entities?"
// ============================================================================

// Start from a known chunk
MATCH (source:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)

// Find other chunks that mention the same entities
MATCH (e)<-[:MENTIONS]-(related:Chunk)
WHERE related.id <> source.id

// Score by entity overlap count + vector similarity
WITH related,
  count(DISTINCT e) AS shared_entities,
  vector_similarity(related.embedding, source.embedding) AS semantic_sim

// Get the parent document
MATCH (d:Document)-[:HAS_CHUNK]->(related)

RETURN d.title, related.text,
  shared_entities, semantic_sim,
  shared_entities * 0.3 + semantic_sim * 0.7 AS combined_score
ORDER BY combined_score DESC
LIMIT 10;


// ============================================================================
// GQ-5. DOCUMENT-LEVEL SCORING: Top documents by aggregate chunk relevance
// "Which documents are most relevant to my question?"
// ============================================================================

MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE vector_similarity(c.embedding, $query_vec) > 0.5
WITH d,
  max(vector_similarity(c.embedding, $query_vec)) AS best_chunk_score,
  avg(vector_similarity(c.embedding, $query_vec)) AS avg_chunk_score,
  count(c) AS matching_chunks,
  collect(c.text)[0..3] AS top_snippets
ORDER BY best_chunk_score * 0.7 + avg_chunk_score * 0.3 DESC
LIMIT 10
RETURN d.title, d.source, best_chunk_score, avg_chunk_score,
  matching_chunks, top_snippets;


// ============================================================================
// GQ-6. CENTROID SEARCH: Two-stage retrieval for large corpora
// Stage 1: find candidate documents by centroid embedding.
// Stage 2: drill into chunks of top documents.
// ============================================================================

// Stage 1: document-level vector search (one embedding per document)
MATCH (d:Document)
WHERE vector_similarity(d.centroid_embedding, $query_vec) > 0.5
WITH d ORDER BY vector_similarity(d.centroid_embedding, $query_vec) DESC LIMIT 50

// Stage 2: chunk-level search within top documents
MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WHERE vector_similarity(c.embedding, $query_vec) > 0.6
RETURN d.title, c.text, c.position,
  vector_similarity(c.embedding, $query_vec) AS chunk_score
ORDER BY chunk_score DESC
LIMIT 20;


// ============================================================================
// GQ-7. SEMANTIC EDGE TRAVERSAL: Navigate via vector-weighted edges
// Find chunks related through semantic similarity edges, ranking each edge
// by how close its own embedding is to the question.
// ============================================================================

MATCH (start:Chunk)
WHERE vector_similarity(start.embedding, $query_vec) > 0.8
WITH start ORDER BY vector_similarity(start.embedding, $query_vec) DESC LIMIT 3

// Follow semantic similarity edges, filtering by edge vector
MATCH (start)-[r:SIMILAR_TO]->(related:Chunk)
WHERE vector_similarity(r.embedding, $query_vec) > 0.6
  AND r.similarity > 0.7

RETURN start.text AS seed_chunk,
  related.text AS related_chunk,
  r.similarity AS edge_similarity,
  vector_similarity(r.embedding, $query_vec) AS edge_query_relevance
ORDER BY edge_query_relevance DESC;


// ============================================================================
// GQ-8. MULTI-HOP KNOWLEDGE GRAPH RAG
// Navigate entity relationships to build structured context for LLM.
// ============================================================================

// Find entities mentioned in relevant chunks
MATCH (c:Chunk)
WHERE vector_similarity(c.embedding, $query_vec) > 0.7
WITH c LIMIT 5

MATCH (c)-[:MENTIONS]->(e1:Entity)

// Expand 1-2 hops through entity relationships
MATCH path = (e1)-[:RELATED_TO*1..2]->(e2:Entity)

// Find chunks that mention these connected entities
MATCH (e2)<-[:MENTIONS]-(context:Chunk)
WHERE context.id <> c.id

RETURN DISTINCT
  e1.name AS seed_entity,
  e2.name AS connected_entity,
  type(last(relationships(path))) AS relation,
  context.text AS additional_context
LIMIT 30;


// ============================================================================
// GQ-9. TEMPORAL RAG: Search across document versions (time-travel)
// "What did the contract say about liability BEFORE the amendment?"
// A full-text index holds only the current state, so a time-travel read
// filters the snapshot directly instead of scoring through the index.
// The timestamp must fall inside the retention window (7 days by default).
// ============================================================================

MATCH (d:Document {title: "Service Agreement v2"})-[:HAS_CHUNK]->(c:Chunk)
WHERE toLower(c.text) CONTAINS 'liability'
RETURN c.text, c.position
ORDER BY c.position
AS OF TIMESTAMP '2026-03-15T10:00:00Z';


// ============================================================================
// GQ-10. FULL GRAPHRAG PIPELINE: The "killer query"
// Combines: graph traversal + vector similarity + BM25 + entity graph +
//           document metadata + nested document access + aggregation
// ============================================================================

// Step 1: Find relevant chunks (hybrid search)
MATCH (c:Chunk)
WHERE vector_similarity(c.embedding, $query_vec) > 0.6
WITH c,
  0.6 * vector_similarity(c.embedding, $query_vec) +
  0.4 * text_score(c.text, $search_text) AS hybrid_score
ORDER BY hybrid_score DESC LIMIT 30

// Step 2: Get parent document metadata
MATCH (d:Document)-[:HAS_CHUNK]->(c)

// Step 3: Get entities mentioned in these chunks
MATCH (c)-[:MENTIONS]->(e:Entity)

// Step 4: Get surrounding context (±1 chunk)
OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)

// Step 5: Aggregate per document
WITH d, c, hybrid_score, prev, next,
  collect(DISTINCT e { .name, .type }) AS entities

// Step 6: Build LLM-ready context
RETURN
  d.title AS document,
  d.source AS source,
  c.text AS chunk,
  prev.text AS before,
  next.text AS after,
  entities,
  hybrid_score,
  c.metadata.page AS page,
  c.metadata.section AS section
ORDER BY hybrid_score DESC;
