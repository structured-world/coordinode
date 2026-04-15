---
title: TypeScript / Node.js SDK
description: Official TypeScript SDK for CoordiNode — gRPC client with source location tracking for the Query Advisor.
---

# TypeScript / Node.js SDK

The official CoordiNode TypeScript SDK wraps the gRPC API with a typed client and optional source-location tracking that feeds the server's [EXPLAIN SUGGEST](/cypher/extensions#explain-suggest) query advisor.

```bash
npm install coordinode
# or
yarn add coordinode
```

## Quick Start

```typescript
import { CoordinodeClient } from "coordinode";

const client = new CoordinodeClient("localhost:7080");

// Execute a Cypher query
const rows = await client.executeCypher(
  "MATCH (n:Person) RETURN n.name LIMIT 10"
);
for (const row of rows) {
  console.log(row["n.name"]);
}

// With parameters
const result = await client.executeCypherWithParams(
  "MATCH (u:User {id: $id}) RETURN u",
  { id: 42 }
);
```

## Source Location Tracking

Enable `debugSourceTracking` in development to let the Query Advisor map slow-query suggestions to exact call sites in your application.

```typescript
import { CoordinodeClient } from "coordinode";

const client = new CoordinodeClient("localhost:7080", {
  debugSourceTracking: true,
  appName: "my-service",
  appVersion: process.env.npm_package_version,
});

// Every query call automatically attaches the call site to the request.
// The server will record:
//   x-source-file: "src/api/handlers/feed.ts"
//   x-source-line: "47"
//   x-source-app:  "my-service"
const rows = await client.executeCypher(
  "MATCH (u:User)-[:FOLLOWS]->(f) RETURN f LIMIT 10"
);
```

Source tracking uses Node.js's `Error.captureStackTrace` / `CallSite` API to read the immediate caller's file and line number. The overhead is < 5 µs per call and is zero when `debugSourceTracking` is `false`.

> **Recommendation:** Enable in development and staging; disable in production.

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `debugSourceTracking` | `boolean` | `false` | Attach caller `file:line` to every query request |
| `appName` | `string` | `""` | Application name included in source-tracking metadata |
| `appVersion` | `string` | `""` | Application version included in source-tracking metadata |
| `deadline` | `number` | `30000` | Default per-call deadline in milliseconds |

## See Also

- [Python SDK](/sdk/python) — sync and async clients
- [LlamaIndex integration](/sdk/llama-index) — PropertyGraphIndex with CoordiNode backend
- [LangChain integration](/sdk/langchain) — GraphCypherQAChain
- [GitHub: structured-world/coordinode-js](https://github.com/structured-world/coordinode-js)
