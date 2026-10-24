"""Dependency, license and attribution evidence for one resolved consumer graph.

Standard-library only: it runs on the metadata Cargo resolved for an actual
consumer, bound to its feature set, target triple and lockfile, so the evidence
describes what that consumer builds rather than the source workspace.
"""
import json
import pathlib
import tomllib

# Files a package ships whose text a redistributor must carry along.
ATTRIBUTION_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT", "UNLICENSE")


def reachable(metadata):
    """Ids of the packages the root builds: normal and build edges, never dev."""
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    root = metadata["resolve"]["root"]
    seen, pending = set(), [root]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for dep in nodes[current]["deps"]:
            kinds = {kind["kind"] for kind in dep["dep_kinds"]}
            # `None` is a normal edge; dev edges never reach a consumer's build.
            if kinds - {"dev"}:
                pending.append(dep["pkg"])
    seen.discard(root)
    return seen


def lock_checksums(lock_text):
    """Registry checksums by (name, version) from a Cargo.lock."""
    lock = tomllib.loads(lock_text)
    return {(p["name"], p["version"]): p["checksum"]
            for p in lock.get("package", []) if "checksum" in p}


def attribution_files(manifest_path):
    """License and notice texts at the root of a package's source."""
    root = pathlib.Path(manifest_path).parent
    return sorted(entry.name for entry in root.iterdir()
                  if entry.is_file() and entry.name.upper().startswith(ATTRIBUTION_PREFIXES))


def inventory(metadata, lock_text, context):
    """One entry per package the consumer builds, refusing any without a license.

    `context` names what the graph was resolved for (toolchain, features,
    target triple, source) and is carried into the evidence unchanged.
    """
    checksums = lock_checksums(lock_text)
    packages = {p["id"]: p for p in metadata["packages"]}
    entries, unlicensed = [], []
    for package_id in sorted(reachable(metadata)):
        package = packages[package_id]
        license_expr = package.get("license")
        license_file = package.get("license_file")
        if not license_expr and not license_file:
            unlicensed.append(f"{package['name']} {package['version']}")
        entries.append({
            "name": package["name"],
            "version": package["version"],
            "source": package.get("source"),
            "license": license_expr,
            "license_file": license_file,
            "checksum": checksums.get((package["name"], package["version"])),
            "attribution_files": attribution_files(package["manifest_path"]),
        })
    if unlicensed:
        raise ValueError(f"packages without a license declaration: {unlicensed}")
    return {"context": context, "packages": entries}


def cyclonedx(evidence):
    """The inventory as a CycloneDX 1.5 SBOM."""
    components = []
    for entry in evidence["packages"]:
        component = {
            "type": "library",
            "name": entry["name"],
            "version": entry["version"],
            "purl": f"pkg:cargo/{entry['name']}@{entry['version']}",
        }
        if entry["license"]:
            component["licenses"] = [{"expression": entry["license"]}]
        if entry["checksum"]:
            component["hashes"] = [{"alg": "SHA-256", "content": entry["checksum"]}]
        components.append(component)
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"properties": [
            {"name": f"coordinode:{key}", "value": str(value)}
            for key, value in sorted(evidence["context"].items())
        ]},
        "components": components,
    }


def write_evidence(directory, stem, metadata, lock_text, context):
    """Write `<stem>-licenses.json` and `<stem>-sbom.cdx.json`; return the inventory."""
    evidence = inventory(metadata, lock_text, context)
    directory = pathlib.Path(directory)
    (directory / f"{stem}-licenses.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (directory / f"{stem}-sbom.cdx.json").write_text(json.dumps(cyclonedx(evidence), indent=2) + "\n")
    return evidence
