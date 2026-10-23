"""Prepare registry archives in a dedicated release checkout, never an editor checkout.

Runtime Rust sources remain unchanged. Only package manifests, the two build
scripts and distributable metadata are prepared. Generated protobuf stays out
of Git. The caller must test the source before preparation and the resulting
archives afterwards; release PR generation uses a separate, clean checkout.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tomllib

import tomlkit


def command(*args, root):
    return subprocess.check_output(args, cwd=root, text=True)


def library_closure(metadata, entrypoint):
    """Traverse every target and optional dependency, but never dev-only edges."""
    members = set(metadata["workspace_members"])
    packages = {p["id"]: p for p in metadata["packages"]}
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    start = next(p["id"] for p in packages.values() if p["name"] == entrypoint)
    visited = set()
    pending = [start]
    while pending:
        item = pending.pop()
        if item in visited:
            continue
        visited.add(item)
        for dep in nodes[item]["deps"]:
            if any(k["kind"] != "dev" for k in dep["dep_kinds"]):
                pending.append(dep["pkg"])
    internal = {packages[p]["name"] for p in visited if p in members}
    external = [packages[p] for p in visited if p not in members]
    return internal, external


def prepare_manifest(document, version, names, codegen):
    """Describe only delivered library targets, preserving runtime features."""
    document = tomlkit.parse(tomlkit.dumps(document))
    package = document["package"]
    package["include"] = ["/src/**", "/build.rs", "/bindings/**", "/LICENSE",
                          "/NOTICES", "/COPYRIGHT", "/README.md", "/Cargo.toml"]
    for switch in ("autobins", "autoexamples", "autotests", "autobenches"):
        package[switch] = False
    for section in ("dev-dependencies", "bench", "test", "example", "bin"):
        document.pop(section, None)
    for target in document.get("target", {}).values():
        target.pop("dev-dependencies", None)
    # A normalized registry manifest must never pull private fixtures into
    # dependency resolution. The complete source suite was run before this.
    lib = document.setdefault("lib", tomlkit.table())
    lib["test"] = False
    lib["bench"] = False
    if codegen:
        document.pop("build-dependencies", None)
    for dependencies in [document.get("dependencies", {}), document.get("build-dependencies", {})]:
        for name, dep in dependencies.items():
            if name == "coordinode-workspace-hack":
                raise ValueError("hakari dependency was not removed")
            if name in names and "path" in dep:
                dep["version"] = "=" + version
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=pathlib.Path, required=True,
                        help="dedicated disposable release checkout")
    args = parser.parse_args()
    root = args.checkout.resolve()
    config = tomllib.loads((root / ".config/registry-packages.toml").read_text())
    names = set(config["packages"])
    if len(names) != len(config["packages"]):
        raise ValueError("duplicate package in release inventory")
    version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    source = command("git", "rev-parse", "HEAD", root=root).strip()
    if command("git", "status", "--porcelain", root=root).strip():
        raise ValueError("source checkout must be clean before package preparation")
    proto = command("git", "rev-parse", "HEAD", root=root / "proto").strip()
    if command("git", "status", "--porcelain", root=root / "proto").strip():
        raise ValueError("proto checkout must match its recorded revision")
    # Use hakari's supported publication path. Do not ship a dummy hack crate.
    subprocess.run(["cargo", "hakari", "remove-deps", "--yes"], cwd=root, check=True)
    metadata = json.loads(command("cargo", "metadata", "--all-features", "--format-version=1", root=root))
    internal, external = library_closure(metadata, config["entrypoint"])
    if internal != names:
        raise ValueError(f"release closure mismatch: missing={internal - names}, extra={names - internal}")
    git_dependencies = [p["name"] for p in external if (p.get("source") or "").startswith("git+")]
    if git_dependencies:
        raise ValueError(f"registry closure still depends on Git: {git_dependencies}")
    manifests = {p["name"]: pathlib.Path(p["manifest_path"]) for p in metadata["packages"] if p["id"] in metadata["workspace_members"]}
    by_id = {p["id"]: p["name"] for p in metadata["packages"]}
    # Build before replacing either build script. Cargo reports OUT_DIR even
    # when its fingerprint is fresh, so no filesystem guessing is required.
    build = subprocess.Popen(["cargo", "build", "-p", "coordinode-raft", "-p", "coordinode-replicate",
                              "--message-format=json"], cwd=root, stdout=subprocess.PIPE, text=True)
    outputs = {}
    for line in build.stdout:
        event = json.loads(line)
        if event.get("reason") == "build-script-executed":
            name = by_id.get(event["package_id"])
            if name in config["bindings"]:
                outputs[name] = pathlib.Path(event["out_dir"])
        elif event.get("reason") == "compiler-message":
            print(event["message"].get("rendered", ""), end="")
    if build.wait() != 0:
        raise RuntimeError("source binding build failed")
    if command("git", "rev-parse", "HEAD", root=root / "proto").strip() != proto:
        raise ValueError("proto revision changed during generation")
    if command("git", "status", "--porcelain", root=root / "proto").strip():
        raise ValueError("proto checkout changed during generation")
    report = {"version": version, "source_commit": source, "proto_commit": proto, "packages": {},
              "source_ci_commit": os.environ.get("SOURCE_CI_COMMIT"),
              "dependency_licenses": sorted([
                  {"name": p["name"], "version": p["version"], "license": p.get("license"), "source": p.get("source")}
                  for p in external], key=lambda p: (p["name"], p["version"]))}
    for name in config["packages"]:
        manifest = manifests[name]
        folder = manifest.parent
        if name in config["bindings"]:
            binding_dir = folder / "bindings"
            binding_dir.mkdir(exist_ok=False)
            files = config["bindings"][name]
            for filename in files:
                shutil.copyfile(outputs[name] / filename, binding_dir / filename)
            (binding_dir / "files.txt").write_text("\n".join(files) + "\n")
            shutil.copyfile(root / "scripts/release/prepared-build.rs", folder / "build.rs")
        for filename in ("LICENSE", "NOTICES", "COPYRIGHT"):
            shutil.copyfile(root / filename, folder / filename)
        doc = prepare_manifest(tomlkit.parse(manifest.read_text()), version, names, name in config["bindings"])
        manifest.write_text(tomlkit.dumps(doc))
        report["packages"][name] = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (root / "target").mkdir(exist_ok=True)
    (root / "target/registry-preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {len(names)} libraries at {version} from {source}; proto {proto}")


if __name__ == "__main__":
    main()
