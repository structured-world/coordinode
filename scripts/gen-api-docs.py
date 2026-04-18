#!/usr/bin/env python3
"""
Generate gRPC API reference documentation from .proto files.

Parses all proto files in proto/coordinode/v1/ and generates
markdown pages in docs/api/ — one page per service, one for common
types shared across services, and an index page.

Run: python3 scripts/gen-api-docs.py
Integrated into: package.json "docs:build" script
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field as dc_field
from typing import Optional

ROOT = Path(__file__).parent.parent
PROTO_DIR = ROOT / "proto" / "coordinode" / "v1"
DOCS_API_DIR = ROOT / "docs" / "api"

GITHUB_PROTO_BASE = (
    "https://github.com/structured-world/coordinode-proto-ce/blob/main/coordinode/v1"
)

# Services to document with their display order and page slug
SERVICE_ORDER = [
    "CypherService",
    "VectorService",
    "TextService",
    "GraphService",
    "SchemaService",
    "BlobService",
    "HealthService",
    "ClusterService",
]

# Services that are internal (inter-node only) — get a warning banner
INTERNAL_SERVICES = {"RaftService"}

SERVICE_SLUG = {
    "CypherService": "cypher",
    "VectorService": "vector",
    "TextService": "text",
    "GraphService": "graph",
    "SchemaService": "schema",
    "BlobService": "blob",
    "HealthService": "health",
    "ClusterService": "cluster",
    "ChangeStreamService": "change-stream",
    "RaftService": "raft",
}

# Messages that belong to "common types" (shared across services)
COMMON_MESSAGES = {
    "PropertyValue",
    "Vector",
    "PropertyList",
    "PropertyMap",
    "NodeId",
    "EdgeId",
    "HlcTimestamp",
    "Node",
    "Edge",
    "PaginationRequest",
    "PaginationResponse",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EnumValue:
    name: str
    number: int
    comment: str = ""


@dataclass
class ProtoEnum:
    name: str
    comment: str = ""
    values: list = dc_field(default_factory=list)


@dataclass
class ProtoField:
    name: str
    type_name: str  # raw type string
    number: int
    comment: str = ""
    repeated: bool = False
    is_oneof: bool = False
    oneof_group: str = ""


@dataclass
class ProtoMessage:
    name: str
    comment: str = ""
    fields: list = dc_field(default_factory=list)
    oneofs: dict = dc_field(default_factory=dict)  # oneof_name -> [ProtoField]
    nested_enums: list = dc_field(default_factory=list)


@dataclass
class RpcMethod:
    name: str
    request: str
    response: str
    streaming_request: bool = False
    streaming_response: bool = False
    comment: str = ""
    http_method: str = ""
    http_path: str = ""


@dataclass
class ProtoService:
    name: str
    comment: str = ""
    methods: list = dc_field(default_factory=list)


@dataclass
class ParsedProto:
    """Parsed contents of a single .proto file."""

    path: Path
    package: str = ""
    services: list = dc_field(default_factory=list)
    messages: dict = dc_field(default_factory=dict)  # name -> ProtoMessage
    enums: dict = dc_field(default_factory=dict)  # name -> ProtoEnum


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


HTTP_METHODS = ("get", "post", "put", "delete", "patch")
COMMENT_RE = re.compile(r"^\s*//\s?(.*)")
PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+);")
SERVICE_RE = re.compile(r"^\s*service\s+(\w+)\s*\{")
RPC_RE = re.compile(
    r"^\s*rpc\s+(\w+)\s*\((stream\s+)?(\w+)\)\s*returns\s*\((stream\s+)?(\w+)\)"
)
MESSAGE_RE = re.compile(r"^\s*message\s+(\w+)\s*\{")
ENUM_RE = re.compile(r"^\s*enum\s+(\w+)\s*\{")
ONEOF_RE = re.compile(r"^\s*oneof\s+(\w+)\s*\{")
FIELD_RE = re.compile(
    r"^\s*(repeated\s+|optional\s+)?(.+?)\s+(\w+)\s*=\s*(\d+)\s*;"
)
ENUM_VALUE_RE = re.compile(r"^\s*(\w+)\s*=\s*(\d+)\s*;")
HTTP_LINE_RE = re.compile(
    r"^\s*(" + "|".join(HTTP_METHODS) + r')\s*:\s*"([^"]+)"'
)


def _collect_comments(lines: list[str], start: int) -> str:
    """Collect // comment lines immediately before position `start`."""
    comments: list[str] = []
    i = start - 1
    while i >= 0:
        m = COMMENT_RE.match(lines[i])
        if m:
            comments.append(m.group(1))
            i -= 1
        elif lines[i].strip() == "":
            break
        else:
            break
    return " ".join(reversed(comments)).strip()


def _brace_end(lines: list[str], start: int) -> int:
    """Return the index of the line with the closing brace of a block that
    opened at or after `start` (where the opening `{` is on line `start`)."""
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0:
            return i
    return len(lines) - 1


def _extract_http_annotation(lines: list[str], rpc_body_start: int) -> tuple[str, str]:
    """Extract HTTP method and path from the RPC body lines."""
    end = _brace_end(lines, rpc_body_start)
    for i in range(rpc_body_start, end + 1):
        m = HTTP_LINE_RE.match(lines[i])
        if m:
            return m.group(1).upper(), m.group(2)
    return "", ""


def _strip_package(type_name: str) -> str:
    """Strip proto package prefix from a type name (handles map<K,V> generics)."""
    type_name = type_name.strip()
    if type_name.startswith("map<"):
        inner = type_name[4:-1]  # strip "map<" and trailing ">"
        parts = inner.split(",", 1)
        if len(parts) == 2:
            k = _strip_package(parts[0].strip())
            v = _strip_package(parts[1].strip())
            return f"map<{k}, {v}>"
        return type_name
    if "." in type_name:
        return type_name.rsplit(".", 1)[-1]
    return type_name


def _parse_enum_block(lines: list[str], start: int) -> ProtoEnum:
    """Parse an enum block starting at `start` (the line with `enum ... {`)."""
    m = ENUM_RE.match(lines[start])
    name = m.group(1) if m else "Unknown"
    comment = _collect_comments(lines, start)
    enum = ProtoEnum(name=name, comment=comment)

    end = _brace_end(lines, start)
    pending_comment = ""
    for i in range(start + 1, end):
        line = lines[i]
        cm = COMMENT_RE.match(line)
        if cm:
            pending_comment = cm.group(1)
            continue
        vm = ENUM_VALUE_RE.match(line)
        if vm:
            enum.values.append(
                EnumValue(name=vm.group(1), number=int(vm.group(2)), comment=pending_comment)
            )
        pending_comment = ""
    return enum


def _parse_message_block(lines: list[str], start: int) -> ProtoMessage:
    """Parse a message block starting at `start`."""
    m = MESSAGE_RE.match(lines[start])
    name = m.group(1) if m else "Unknown"
    comment = _collect_comments(lines, start)
    msg = ProtoMessage(name=name, comment=comment)

    end = _brace_end(lines, start)
    i = start + 1
    pending_comment = ""
    current_oneof: Optional[str] = None
    oneof_depth = 0

    while i <= end:
        line = lines[i]

        # blank
        if not line.strip():
            pending_comment = ""
            i += 1
            continue

        # comment
        cm = COMMENT_RE.match(line)
        if cm:
            if pending_comment:
                pending_comment += " " + cm.group(1)
            else:
                pending_comment = cm.group(1)
            i += 1
            continue

        # closing brace of oneof or message
        if line.strip() == "}":
            if current_oneof and oneof_depth > 0:
                oneof_depth -= 1
                if oneof_depth == 0:
                    current_oneof = None
            i += 1
            continue

        # nested enum
        if ENUM_RE.match(line):
            nested_enum = _parse_enum_block(lines, i)
            msg.nested_enums.append(nested_enum)
            i = _brace_end(lines, i) + 1
            pending_comment = ""
            continue

        # nested message — skip (rare, not in our protos)
        if MESSAGE_RE.match(line):
            i = _brace_end(lines, i) + 1
            pending_comment = ""
            continue

        # oneof block
        om = ONEOF_RE.match(line)
        if om:
            current_oneof = om.group(1)
            oneof_depth = 1
            msg.oneofs[current_oneof] = []
            i += 1
            pending_comment = ""
            continue

        # opening brace continuation lines (e.g. oneof opened inline)
        if line.strip() == "{":
            if current_oneof:
                oneof_depth += 1
            i += 1
            continue

        # field
        fm = FIELD_RE.match(line)
        if fm:
            repeated = bool(fm.group(1) and "repeated" in fm.group(1))
            type_name = _strip_package(fm.group(2).strip())
            field_name = fm.group(3)
            field_num = int(fm.group(4))
            f = ProtoField(
                name=field_name,
                type_name=type_name,
                number=field_num,
                comment=pending_comment,
                repeated=repeated,
                is_oneof=current_oneof is not None,
                oneof_group=current_oneof or "",
            )
            if current_oneof:
                msg.oneofs[current_oneof].append(f)
            else:
                msg.fields.append(f)
        pending_comment = ""
        i += 1

    return msg


def parse_proto_file(path: Path) -> ParsedProto:
    """Parse a single .proto file into a ParsedProto."""
    lines = path.read_text(encoding="utf-8").splitlines()
    result = ParsedProto(path=path)

    # Extract package
    for line in lines:
        m = PACKAGE_RE.match(line)
        if m:
            result.package = m.group(1)
            break

    i = 0
    while i < len(lines):
        line = lines[i]

        # Service
        sm = SERVICE_RE.match(line)
        if sm:
            svc_name = sm.group(1)
            svc_comment = _collect_comments(lines, i)
            svc = ProtoService(name=svc_name, comment=svc_comment)
            svc_end = _brace_end(lines, i)

            j = i + 1
            rpc_pending_comment = ""
            while j < svc_end:
                sline = lines[j]
                cm = COMMENT_RE.match(sline)
                if cm:
                    if rpc_pending_comment:
                        rpc_pending_comment += " " + cm.group(1)
                    else:
                        rpc_pending_comment = cm.group(1)
                    j += 1
                    continue
                rm = RPC_RE.match(sline)
                if rm:
                    stream_req = bool(rm.group(2))
                    stream_resp = bool(rm.group(4))
                    rpc = RpcMethod(
                        name=rm.group(1),
                        request=_strip_package(rm.group(3)),
                        response=_strip_package(rm.group(5)),
                        streaming_request=stream_req,
                        streaming_response=stream_resp,
                        comment=rpc_pending_comment,
                    )
                    rpc_pending_comment = ""
                    # Check for HTTP annotation body
                    if "{" in sline:
                        rpc.http_method, rpc.http_path = _extract_http_annotation(lines, j)
                        j = _brace_end(lines, j) + 1
                    else:
                        j += 1
                    svc.methods.append(rpc)
                    continue
                if not sline.strip() or sline.strip() in ("{", "}"):
                    rpc_pending_comment = ""
                j += 1

            result.services.append(svc)
            i = svc_end + 1
            continue

        # Top-level message
        mm = MESSAGE_RE.match(line)
        if mm:
            msg = _parse_message_block(lines, i)
            result.messages[msg.name] = msg
            i = _brace_end(lines, i) + 1
            continue

        # Top-level enum
        em = ENUM_RE.match(line)
        if em:
            enum = _parse_enum_block(lines, i)
            result.enums[enum.name] = enum
            i = _brace_end(lines, i) + 1
            continue

        i += 1

    return result


# ---------------------------------------------------------------------------
# Markdown generation helpers
# ---------------------------------------------------------------------------


def _fmt_type(field: ProtoField) -> str:
    t = f"`{field.type_name}`"
    if field.repeated:
        t = f"`{field.type_name}[]`"
    return t


def _escape_md(text: str) -> str:
    """Escape characters that break VitePress/Vue parsing in table cells."""
    # Pipe breaks markdown tables
    text = text.replace("|", "\\|")
    # Angle brackets are parsed as Vue component tags by VitePress
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text


def _field_table(fields: list[ProtoField], messages: dict, link_prefix: str = "") -> str:
    if not fields:
        return "_No fields._\n"
    rows = ["| Field | Type | Description |", "|-------|------|-------------|"]
    for f in fields:
        name = f"`{f.name}`"
        typ = _fmt_type(f)
        desc = _escape_md(f.comment) if f.comment else "—"
        rows.append(f"| {name} | {typ} | {desc} |")
    return "\n".join(rows) + "\n"


def _enum_table(enum: ProtoEnum) -> str:
    rows = ["| Value | Number | Description |", "|-------|--------|-------------|"]
    for v in enum.values:
        desc = _escape_md(v.comment) if v.comment else "—"
        rows.append(f"| `{v.name}` | `{v.number}` | {desc} |")
    return "\n".join(rows) + "\n"


def _anchor(name: str) -> str:
    return name.lower().replace("_", "-")


def _message_section(msg: ProtoMessage, messages: dict, heading: str = "###") -> str:
    parts: list[str] = []
    anchor = _anchor(msg.name)
    parts.append(f"{heading} {msg.name} {{#{anchor}}}\n")
    if msg.comment:
        parts.append(f"{_escape_md(msg.comment)}\n")

    if msg.nested_enums:
        for ne in msg.nested_enums:
            parts.append(f"**`{ne.name}`** enum:\n")
            parts.append(_enum_table(ne))

    all_fields = list(msg.fields)
    for oneof_name, oneof_fields in msg.oneofs.items():
        all_fields.extend(oneof_fields)

    if all_fields:
        parts.append(_field_table(all_fields, messages))
    else:
        parts.append("_No fields._\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------


def _collect_message_names_for_service(
    svc: ProtoService, all_messages: dict
) -> list[str]:
    """Collect all message names referenced by a service (request/response types
    and their transitive dependencies within the same proto file set)."""
    seen: set[str] = set()
    queue: list[str] = []
    for m in svc.methods:
        queue.extend([m.request, m.response])

    def _visit(name: str) -> None:
        if name in seen or name not in all_messages:
            return
        seen.add(name)
        msg = all_messages[name]
        for f in msg.fields:
            _visit(f.type_name)
        for oneof_fields in msg.oneofs.values():
            for f in oneof_fields:
                _visit(f.type_name)

    for name in queue:
        _visit(name)

    return [n for n in seen if n not in COMMON_MESSAGES]


def generate_service_page(
    svc: ProtoService,
    all_messages: dict,
    all_enums: dict,
    proto_path_hint: str = "",
) -> str:
    """Generate a full markdown page for a single service."""
    slug = SERVICE_SLUG.get(svc.name, svc.name.lower())
    lines: list[str] = []

    # frontmatter — YAML: escape only double quotes (no HTML entities needed)
    lines.append("---")
    if svc.comment:
        safe_desc = svc.comment.replace('"', '\\"')
        lines.append(f'description: "{svc.name} — {safe_desc}"')
    lines.append("---\n")

    # Title
    lines.append(f"# {svc.name}\n")
    if svc.comment:
        lines.append(f"{_escape_md(svc.comment)}\n")
    if svc.name in INTERNAL_SERVICES:
        lines.append(
            "::: warning Internal API\n"
            "This service is used for **inter-node communication only**. "
            "It is not intended for external clients and may change without notice.\n"
            ":::\n"
        )
    if proto_path_hint:
        lines.append(
            f"::: tip Proto source\n"
            f"[{proto_path_hint}]({GITHUB_PROTO_BASE}/{proto_path_hint})\n"
            f":::\n"
        )

    # Check if any method has HTTP annotation (transcoding paths)
    has_http = any(m.http_method for m in svc.methods)
    if has_http and svc.name not in INTERNAL_SERVICES:
        lines.append(
            "::: info Transport\n"
            "All methods are available via:\n"
            "- **gRPC** on port **7080** — native high-performance API\n"
            "- **REST/JSON** on port **7081** — HTTP/JSON transcoding via embedded structured-proxy\n"
            ":::\n"
        )

    # CypherService: add query parameters reference
    if svc.name == "CypherService":
        lines.append(
            "## Query Parameters (REST)\n"
            "\n"
            "Named parameters are passed in the `parameters` map. "
            "Each value is a JSON object with a single key matching the `PropertyValue` proto field name:\n"
            "\n"
            "```json\n"
            "{\n"
            '  "query": "MATCH (n:Concept {name: $name}) RETURN n",\n'
            '  "parameters": {\n'
            '    "name":            { "stringValue": "machine learning" },\n'
            '    "limit":           { "intValue": "10" },\n'
            '    "threshold":       { "floatValue": 0.4 },\n'
            '    "active":          { "boolValue": true },\n'
            '    "question_vector": { "vectorValue": { "values": [0.1, 0.2, 0.3] } },\n'
            '    "tags":            { "listValue": { "values": [{"stringValue": "ml"}, {"stringValue": "ai"}] } }\n'
            "  }\n"
            "}\n"
            "```\n"
            "\n"
            "Available parameter types:\n"
            "\n"
            "| JSON key | Proto field | Example |\n"
            "|----------|-------------|---------|\n"
            '| `stringValue` | `string` | `{ "stringValue": "alice" }` |\n'
            '| `intValue` | `int64` | `{ "intValue": "42" }` |\n'
            '| `floatValue` | `double` | `{ "floatValue": 0.95 }` |\n'
            '| `boolValue` | `bool` | `{ "boolValue": true }` |\n'
            '| `vectorValue` | `Vector` | `{ "vectorValue": { "values": [0.1, 0.2] } }` |\n'
            '| `listValue` | `PropertyList` | `{ "listValue": { "values": [...] } }` |\n'
            '| `bytesValue` | `bytes` | `{ "bytesValue": "&lt;base64&gt;" }` |\n'
            "\n"
            "> **Vector parameters:** use `vectorValue`, not `listValue`. "
            "`vectorValue` maps to the dedicated `Vector` proto type and enables "
            "vector index pushdown in the query planner.\n"
        )

    # Methods
    if svc.methods:
        lines.append("## Methods\n")
        for rpc in svc.methods:
            lines.append(f"### {rpc.name}\n")
            if rpc.comment:
                lines.append(f"{_escape_md(rpc.comment)}\n")

            if rpc.http_method and rpc.http_path:
                lines.append(
                    f"**HTTP**: `{rpc.http_method} {rpc.http_path}`\n"
                )
            elif rpc.streaming_response and not rpc.http_path:
                lines.append("**Transport:** Server-streaming gRPC only (no HTTP transcoding)\n")

            # Request
            req_type = rpc.request
            stream_prefix = "stream " if rpc.streaming_request else ""
            lines.append(f"**Request:** `{stream_prefix}{req_type}`\n")
            if req_type in all_messages:
                msg = all_messages[req_type]
                all_fields = list(msg.fields)
                for oneof_fields in msg.oneofs.values():
                    all_fields.extend(oneof_fields)
                if all_fields:
                    lines.append(_field_table(all_fields, all_messages))
                else:
                    lines.append("_Empty request (no fields)._\n")
            else:
                lines.append(f"_See [common types](./common-types#{_anchor(req_type)})._\n")

            # Response
            resp_type = rpc.response
            stream_prefix = "stream " if rpc.streaming_response else ""
            lines.append(f"**Response:** `{stream_prefix}{resp_type}`\n")
            if resp_type in all_messages:
                msg = all_messages[resp_type]
                all_fields = list(msg.fields)
                for oneof_fields in msg.oneofs.values():
                    all_fields.extend(oneof_fields)
                if all_fields:
                    lines.append(_field_table(all_fields, all_messages))
                else:
                    lines.append("_Empty response (no fields)._\n")
            else:
                lines.append(
                    f"_See [common types](./common-types#{_anchor(resp_type)})._\n"
                )

    # Types section — messages specific to this service
    local_msgs = _collect_message_names_for_service(svc, all_messages)
    # Add all messages from this service's proto that aren't common
    if local_msgs:
        lines.append("## Types\n")
        for name in sorted(local_msgs):
            if name in all_messages:
                lines.append(_message_section(all_messages[name], all_messages))
                lines.append("")

    # Enums specific to this service's proto
    # (Filter by what's referenced in messages of this service)
    referenced_enums: set[str] = set()
    for name in local_msgs:
        if name in all_messages:
            msg = all_messages[name]
            all_fields = list(msg.fields)
            for oneof_fields in msg.oneofs.values():
                all_fields.extend(oneof_fields)
            for f in all_fields:
                if f.type_name in all_enums:
                    referenced_enums.add(f.type_name)
            for ne in msg.nested_enums:
                pass  # already rendered inline

    # Also check RPC request/response messages for enums
    for rpc in svc.methods:
        for msg_name in (rpc.request, rpc.response):
            if msg_name in all_messages:
                msg = all_messages[msg_name]
                all_fields = list(msg.fields)
                for oneof_fields in msg.oneofs.values():
                    all_fields.extend(oneof_fields)
                for f in all_fields:
                    if f.type_name in all_enums:
                        referenced_enums.add(f.type_name)

    if referenced_enums:
        lines.append("## Enums\n")
        for ename in sorted(referenced_enums):
            enum = all_enums[ename]
            anchor = _anchor(ename)
            lines.append(f"### {ename} {{#{anchor}}}\n")
            if enum.comment:
                lines.append(f"{_escape_md(enum.comment)}\n")
            lines.append(_enum_table(enum))
            lines.append("")

    return "\n".join(lines)


PROPERTY_VALUE_JSON_NOTE = """\
**JSON encoding (REST API)**

`PropertyValue` is a `oneof` — in JSON, encode it as an object with exactly one key
matching the camelCase proto field name:

```json
{ "stringValue": "hello" }
{ "intValue": "42" }
{ "floatValue": 0.95 }
{ "boolValue": true }
{ "bytesValue": "aGVsbG8=" }
{ "vectorValue": { "values": [0.1, 0.2, 0.3] } }
{ "listValue": { "values": [{"stringValue": "a"}, {"stringValue": "b"}] } }
{ "mapValue": { "entries": { "key": {"intValue": "1"} } } }
```

> **Vector parameters:** use `vectorValue`, not `listValue`. `vectorValue` maps to the
> dedicated `Vector` proto type and enables vector index pushdown in the query planner.
"""


def generate_common_types_page(all_messages: dict, all_enums: dict) -> str:
    """Generate the common types reference page."""
    lines: list[str] = []
    lines.append("---")
    lines.append('description: "Common types shared across all CoordiNode gRPC services"')
    lines.append("---\n")
    lines.append("# Common Types\n")
    lines.append(
        "Types shared across multiple services. "
        "Each service page links here for shared message definitions.\n"
    )

    lines.append("## Messages\n")
    for name in sorted(COMMON_MESSAGES):
        if name in all_messages:
            lines.append(_message_section(all_messages[name], all_messages))
            if name == "PropertyValue":
                lines.append(PROPERTY_VALUE_JSON_NOTE)
            lines.append("")

    # Common enums (those in types.proto / shared files)
    common_enums = {k: v for k, v in all_enums.items() if k not in all_messages}
    if common_enums:
        lines.append("## Enums\n")
        for name, enum in sorted(common_enums.items()):
            if name in ("ServingStatus", "TraversalDirection"):
                # These are service-specific enums included in common for reference
                anchor = _anchor(name)
                lines.append(f"### {name} {{#{anchor}}}\n")
                if enum.comment:
                    lines.append(f"{enum.comment}\n")
                lines.append(_enum_table(enum))
                lines.append("")

    return "\n".join(lines)


def generate_index_page(services: list[ProtoService]) -> str:
    """Generate docs/api/index.md."""
    lines: list[str] = []
    lines.append("---")
    lines.append('description: "CoordiNode API Reference — gRPC and REST"')
    lines.append("---\n")
    lines.append("# API Reference\n")
    lines.append(
        "CoordiNode exposes services via **gRPC on port 7080** and "
        "**REST/JSON on port 7081** (HTTP/JSON transcoding via embedded structured-proxy). "
        "Bolt protocol (7082) and WebSocket subscriptions (7083) are planned for a future release.\n"
    )
    lines.append("## Services\n")
    lines.append("| Service | Description | Proto |")
    lines.append("|---------|-------------|-------|")

    for svc in services:
        slug = SERVICE_SLUG.get(svc.name, svc.name.lower())
        desc = svc.comment or "—"
        lines.append(f"| [{svc.name}](./{slug}) | {desc} | — |")

    lines.append("")
    lines.append("## Common Types\n")
    lines.append(
        "Messages and enums shared across services: "
        "[Common Types](./common-types)\n"
    )
    lines.append("## Ports\n")
    lines.append("| Port | Protocol | Status | Purpose |")
    lines.append("|------|----------|--------|---------|")
    lines.append("| 7080 | gRPC (HTTP/2) | **Active** | Native gRPC API, inter-node communication |")
    lines.append("| 7081 | HTTP/1.1 + JSON | **Active** | REST/JSON transcoding via structured-proxy |")
    lines.append("| 7082 | Bolt | Planned | Neo4j wire protocol compatibility |")
    lines.append("| 7083 | WebSocket | Planned | Subscriptions, live queries |")
    lines.append("| 7084 | HTTP | **Active** | Prometheus `/metrics`, `/health`, `/ready` |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not PROTO_DIR.exists():
        print(f"ERROR: proto dir not found: {PROTO_DIR}", file=sys.stderr)
        sys.exit(1)

    DOCS_API_DIR.mkdir(parents=True, exist_ok=True)

    # Parse all proto files
    all_messages: dict[str, ProtoMessage] = {}
    all_enums: dict[str, ProtoEnum] = {}
    all_services: list[ProtoService] = []
    service_proto_path: dict[str, str] = {}  # svc name -> relative proto path

    proto_files = sorted(PROTO_DIR.rglob("*.proto"))
    # Skip google vendored files
    proto_files = [p for p in proto_files if "google" not in p.parts]

    for proto_file in proto_files:
        parsed = parse_proto_file(proto_file)
        all_messages.update(parsed.messages)
        all_enums.update(parsed.enums)
        rel = proto_file.relative_to(PROTO_DIR)
        for svc in parsed.services:
            all_services.append(svc)
            service_proto_path[svc.name] = str(rel)

    # Order services
    ordered_services: list[ProtoService] = []
    svc_by_name = {s.name: s for s in all_services}
    for name in SERVICE_ORDER:
        if name in svc_by_name:
            ordered_services.append(svc_by_name[name])
    # Append any services not in SERVICE_ORDER
    for svc in all_services:
        if svc.name not in SERVICE_ORDER:
            ordered_services.append(svc)

    # Generate service pages
    print("Generating API reference pages:")
    for svc in ordered_services:
        slug = SERVICE_SLUG.get(svc.name, svc.name.lower())
        proto_hint = service_proto_path.get(svc.name, "")
        page = generate_service_page(svc, all_messages, all_enums, proto_hint)
        out_path = DOCS_API_DIR / f"{slug}.md"
        out_path.write_text(page, encoding="utf-8")
        print(f"  wrote docs/api/{slug}.md")

    # Generate common types page
    common_page = generate_common_types_page(all_messages, all_enums)
    common_path = DOCS_API_DIR / "common-types.md"
    common_path.write_text(common_page, encoding="utf-8")
    print(f"  wrote docs/api/common-types.md")

    # Generate index
    index_page = generate_index_page(ordered_services)
    index_path = DOCS_API_DIR / "index.md"
    index_path.write_text(index_page, encoding="utf-8")
    print(f"  wrote docs/api/index.md")

    print(
        f"\nDone: {len(ordered_services)} services, "
        f"{len(all_messages)} messages, {len(all_enums)} enums."
    )


if __name__ == "__main__":
    main()
