#!/usr/bin/env bash
# CoordiNode Quickstart — Seed Data Script
#
# Seeds a running CoordiNode instance with sample knowledge graph data:
#   4 Concept nodes, 4 Document nodes (with 384-dim embeddings),
#   8 edges (RELATED_TO + ABOUT).
#
# Usage:
#   docker compose up -d
#   ./examples/quickstart/seed.sh
#
# Requires: curl, python3

set -euo pipefail

BASE_URL="${COORDINODE_REST_URL:-http://localhost:7081}"
CYPHER_URL="${BASE_URL}/v1/query/cypher"
# Every data call names its namespace; a fresh server creates `default`.
NAMESPACE="${COORDINODE_NAMESPACE:-default}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_FILE="${SCRIPT_DIR}/seed-data.json"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: seed-data.json not found at $DATA_FILE"
    exit 1
fi

# Helper: execute a Cypher query via REST proxy
cypher() {
    local body="$1"
    local result
    result=$(curl -sf -X POST "$CYPHER_URL" \
        -H "Content-Type: application/json" \
        -H "X-CoordiNode-Namespace: $NAMESPACE" \
        -d "$body" 2>&1) || {
        echo "  FAILED"
        echo "  Response: $result"
        return 1
    }
    echo "$result"
}

echo "=== CoordiNode Quickstart Seed ==="
echo "Target: $BASE_URL"
echo ""

# Wait for health endpoint
echo -n "Waiting for CoordiNode... "
for i in $(seq 1 30); do
    if curl -sf "http://localhost:7084/health" >/dev/null 2>&1; then
        echo "ready!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "TIMEOUT — is CoordiNode running? (docker compose up -d)"
        exit 1
    fi
    sleep 1
done

# --- Insert nodes ---
echo ""
echo "--- Inserting concepts ---"

# Generate cypher queries with inline vectors using python3
python3 -c "
import json, sys

data = json.load(open('$DATA_FILE'))

for c in data['concepts']:
    vec_str = '[' + ','.join(str(v) for v in c['embedding']) + ']'
    # Escape for JSON string
    name_escaped = c['name'].replace('\"', '\\\\\"')
    query = f'CREATE (c:Concept {{name: \"{name_escaped}\", embedding: {vec_str}}})'
    payload = json.dumps({'query': query})
    print(payload)
    print(c['name'], file=sys.stderr)
" 2>/tmp/coordinode-seed-names.txt | while IFS= read -r payload; do
    cypher "$payload" >/dev/null
done
while IFS= read -r name; do
    echo "  Created concept: $name"
done < /tmp/coordinode-seed-names.txt

echo ""
echo "--- Inserting documents ---"

python3 -c "
import json, sys

data = json.load(open('$DATA_FILE'))

for d in data['documents']:
    vec_str = '[' + ','.join(str(v) for v in d['embedding']) + ']'
    title_escaped = d['title'].replace('\"', '\\\\\"')
    body_escaped = d['body'].replace('\"', '\\\\\"')
    query = f'CREATE (d:Document {{title: \"{title_escaped}\", body: \"{body_escaped}\", embedding: {vec_str}}})'
    payload = json.dumps({'query': query})
    print(payload)
    print(d['title'], file=sys.stderr)
" 2>/tmp/coordinode-seed-names.txt | while IFS= read -r payload; do
    cypher "$payload" >/dev/null
done
while IFS= read -r name; do
    echo "  Created document: $name"
done < /tmp/coordinode-seed-names.txt

# --- Edges ---
echo ""
echo "--- Creating relationships ---"

cypher '{"query": "MATCH (a:Concept {name: \"machine learning\"}), (b:Concept {name: \"deep learning\"}) CREATE (a)-[:RELATED_TO]->(b)"}' >/dev/null
cypher '{"query": "MATCH (a:Concept {name: \"machine learning\"}), (b:Concept {name: \"natural language processing\"}) CREATE (a)-[:RELATED_TO]->(b)"}' >/dev/null
cypher '{"query": "MATCH (a:Concept {name: \"deep learning\"}), (b:Concept {name: \"natural language processing\"}) CREATE (a)-[:RELATED_TO]->(b)"}' >/dev/null
cypher '{"query": "MATCH (a:Concept {name: \"deep learning\"}), (b:Concept {name: \"computer vision\"}) CREATE (a)-[:RELATED_TO]->(b)"}' >/dev/null
echo "  Created 4 RELATED_TO edges"

cypher '{"query": "MATCH (d:Document {title: \"Attention Is All You Need\"}), (c:Concept {name: \"natural language processing\"}) CREATE (d)-[:ABOUT]->(c)"}' >/dev/null
cypher '{"query": "MATCH (d:Document {title: \"Deep Residual Learning for Image Recognition\"}), (c:Concept {name: \"computer vision\"}) CREATE (d)-[:ABOUT]->(c)"}' >/dev/null
cypher '{"query": "MATCH (d:Document {title: \"BERT Pre-training of Deep Bidirectional Transformers\"}), (c:Concept {name: \"natural language processing\"}) CREATE (d)-[:ABOUT]->(c)"}' >/dev/null
cypher '{"query": "MATCH (d:Document {title: \"Language Models are Few-Shot Learners\"}), (c:Concept {name: \"deep learning\"}) CREATE (d)-[:ABOUT]->(c)"}' >/dev/null
echo "  Created 4 ABOUT edges"

# --- Verify ---
echo ""
echo "--- Verifying ---"

result=$(cypher '{"query": "MATCH (n) RETURN count(n) AS total_nodes"}')
echo "  $result"

echo ""
echo "=== Seed complete! ==="
echo ""
echo "Try the hybrid query:"
echo "  curl -s -X POST $CYPHER_URL \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -H 'X-CoordiNode-Namespace: $NAMESPACE' \\"
echo "    -d @examples/quickstart/hybrid-query.json | python3 -m json.tool"
