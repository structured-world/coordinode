"""Build and exercise registry packages outside their source workspace.

Run after prepare.py in the dedicated release checkout. Cargo constructs the
index for the unpublished archives; cargo-local-registry mirrors their external
dependencies. The consumer uses source replacement, never path/git overrides.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile
import tempfile
import tomllib

from dependency_policy import check_native_stack_helpers
from supply_chain import write_evidence


def host_triple(prefix, cwd, env):
    """The target triple the toolchain builds for by default."""
    info = subprocess.check_output([*prefix, "rustc", "-vV"], cwd=cwd, env=env, text=True)
    return next(line.split(": ", 1)[1] for line in info.splitlines() if line.startswith("host: "))


def inspect_archive(archive, name, version, names, bindings):
    """Reject incomplete archives and unresolved source-workspace dependencies."""
    prefix = f"{name}-{version}/"
    with tarfile.open(archive) as package:
        files = {}
        for member in package.getmembers():
            path = pathlib.PurePosixPath(member.name)
            if not member.name.startswith(prefix) or ".." in path.parts or path.is_absolute():
                raise ValueError(f"invalid archive path: {member.name}")
            if not member.isfile():
                raise ValueError(f"unexpected non-file archive entry: {member.name}")
            relative = member.name[len(prefix):]
            if relative in files:
                raise ValueError(f"duplicate archive entry: {relative}")
            files[relative] = member
        required = {"Cargo.toml", "src/lib.rs", "LICENSE", "NOTICES", "COPYRIGHT", "README.md"}
        required.update(f"bindings/{b}" for b in bindings)
        if bindings:
            required.update({"bindings/files.txt", "build.rs"})
        if missing := required - files.keys():
            raise ValueError(f"{name}: missing archive files: {sorted(missing)}")
        manifest = tomllib.loads(package.extractfile(files["Cargo.toml"]).read().decode())
        if manifest["package"]["name"] != name or manifest["package"]["version"] != version:
            raise ValueError(f"{name}: archive identity mismatch")
        if manifest["package"].get("license") == "MIT OR Apache-2.0":
            for license_file in ("LICENSE-MIT", "LICENSE-APACHE"):
                if license_file not in files:
                    raise ValueError(f"{name}: missing {license_file}")
        scopes = [manifest, *manifest.get("target", {}).values()]
        for scope in scopes:
            if scope.get("dev-dependencies"):
                raise ValueError(f"{name}: source-only development dependencies remain")
            for kind in ("dependencies", "build-dependencies"):
                for alias, dependency in scope.get(kind, {}).items():
                    if any(k in dependency for k in ("path", "git", "workspace")):
                        raise ValueError(f"{name}: unresolved source dependency {alias}")
                    actual = dependency.get("package", alias)
                    if actual == "coordinode-workspace-hack":
                        raise ValueError("workspace feature-unification helper leaked into archive")
                    if actual in names and dependency.get("version") != "=" + version:
                        raise ValueError(f"{name}: {actual} is not bound to the coordinated version")
        if bindings:
            listing = package.extractfile(files["bindings/files.txt"]).read().decode().splitlines()
            if listing != bindings:
                raise ValueError(f"{name}: generated binding inventory differs")
    return {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": len(files)}


QUICKSTART = "examples/embedded-quickstart"


def admits(requirement, version):
    """Whether a plain Cargo requirement (`0.9`, `1.2.3`: caret) admits `version`."""
    wanted = [int(part) for part in requirement.split(".")]
    have = [int(part) for part in version.split("-")[0].split(".")]
    # Caret: the leftmost non-zero component of the requirement is fixed.
    fixed = next((i for i, part in enumerate(wanted) if part != 0), len(wanted) - 1) + 1
    return have[:fixed] == wanted[:fixed] and have[:len(wanted)] >= wanted


def check_quickstart_requirements(root, names, version):
    """The published example must build against the version being qualified."""
    manifest = tomllib.loads((root / QUICKSTART / "Cargo.toml").read_text())
    for name, requirement in manifest["dependencies"].items():
        if name not in names:
            continue
        if not isinstance(requirement, str) or not admits(requirement, version):
            raise ValueError(
                f"{QUICKSTART}/Cargo.toml requires {name} {requirement!r}, which does not "
                f"admit {version}: update the example's requirement with the release")


def run_quickstart(root, network, prefix, env, workdir):
    """Build the example project as published and run it twice on one directory.

    The second run checks the documented repeat-run behaviour: it adds its own
    rows to the database the first run left and verifies them.
    """
    project = workdir / "embedded-quickstart"
    (project / "src").mkdir(parents=True)
    for relative in ("Cargo.toml", "src/main.rs"):
        (project / relative).write_text((root / QUICKSTART / relative).read_text())
    if (workdir / ".cargo").exists():
        (project / ".cargo").mkdir()
        (project / ".cargo/config.toml").write_text(
            (workdir / ".cargo/config.toml").read_text())
    data = workdir / "quickstart-data"
    for _ in range(2):
        subprocess.run([*prefix, "cargo", "run", *network, "--", str(data)],
                       cwd=project, env=env, check=True)


def consumer_manifest(version):
    return (
        '[package]\nname="coordinode-registry-consumer"\nversion="0.0.0"\nedition="2024"\n'
        '[features]\nreplication=["coordinode-embed/replication"]\n'
        '[dependencies]\n'
        f'coordinode-embed="={version}"\ncoordinode-core="={version}"\n'
        f'coordinode-storage="={version}"\n')


def mirror_fresh_dependencies(root, names, version, registry, toolchains):
    """Discover allowed external versions, then mirror them for archive tests.

    Path patches expose only the unpublished archives to dependency resolution.
    They are never used to build or qualify the consumer.
    """
    with tempfile.TemporaryDirectory(prefix="coordinode-dependency-discovery-") as temporary:
        discovery = pathlib.Path(temporary)
        packages = discovery / "packages"
        packages.mkdir()
        for name in names:
            archive = root / f"target/package/{name}-{version}.crate"
            with tarfile.open(archive) as package:
                package.extractall(packages, filter="data")
        patches = '[patch.crates-io]\n' + ''.join(
            f'{name}={{path={json.dumps(str(packages / f"{name}-{version}"))}}}\n'
            for name in names)
        env = dict(os.environ, CARGO_HOME=str(discovery / "cargo-home"),
                   CARGO_TARGET_DIR=str(discovery / "target"))
        for index, toolchain in enumerate(toolchains):
            consumer = discovery / str(index)
            (consumer / "src").mkdir(parents=True)
            (consumer / "src/main.rs").write_text('fn main() {}\n')
            (consumer / "Cargo.toml").write_text(consumer_manifest(version) + patches)
            prefix = ["rustup", "run", toolchain] if toolchain else []
            graph = subprocess.check_output(
                [*prefix, "cargo", "metadata", "--all-features", "--format-version=1"],
                cwd=consumer, env=env, text=True)
            (root / "target/fresh-dependency-metadata.json").write_text(graph)
            check_native_stack_helpers(json.loads(graph)['packages'])
            subprocess.run(["cargo-local-registry", "--sync", str(consumer / "Cargo.lock"),
                            "--no-delete", str(registry)], cwd=consumer, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=pathlib.Path, required=True)
    parser.add_argument("--consumer-toolchain", action="append",
                        help="rustup toolchain for the external consumer; repeat to check a matrix")
    parser.add_argument("--source", choices=("prepared", "crates-io"), default="prepared",
                        help="exercise unpublished archives or the actual published registry packages")
    args = parser.parse_args()
    root = args.checkout.resolve()
    inventory = tomllib.loads((root / ".config/registry-packages.toml").read_text())
    report_path = root / "target/registry-preparation.json"
    report = json.loads(report_path.read_text())
    names = inventory["packages"]
    version = report["version"]
    check_quickstart_requirements(root, set(names), version)
    toolchains = args.consumer_toolchain or [None]
    if args.source == "prepared":
        package_args = ["cargo", "package", "--allow-dirty", "--no-verify"]
        for name in names:
            package_args.extend(["-p", name])
        subprocess.run(package_args, cwd=root, check=True)
        for name in names:
            archive = root / f"target/package/{name}-{version}.crate"
            report["packages"][name]["archive"] = inspect_archive(
                archive, name, version, set(names), inventory["bindings"].get(name, []))
        registry = root / "target/package/tmp-registry"
        packaged_lock = (root / "Cargo.lock").read_bytes()
        # --no-delete preserves the unpublished package entries made by Cargo.
        # Use the installed executable directly: this tool's CLI does not consume
        # Cargo's extra `local-registry` argument in its current release.
        subprocess.run(["cargo-local-registry", "--sync", str(root / "Cargo.lock"),
                        "--no-delete", str(registry)], cwd=root, check=True)
        if (root / "Cargo.lock").read_bytes() != packaged_lock:
            raise RuntimeError("registry mirroring changed Cargo.lock; regenerate and requalify archives")
        mirror_fresh_dependencies(root, names, version, registry, toolchains)
    with tempfile.TemporaryDirectory(prefix="coordinode-package-consumer-") as temporary:
        consumer = pathlib.Path(temporary)
        (consumer / "src").mkdir()
        (consumer / ".cargo").mkdir()
        (consumer / "src/main.rs").write_text((root / "scripts/release/consumer.rs").read_text())
        (consumer / "Cargo.toml").write_text(consumer_manifest(version))
        if args.source == "prepared":
            (consumer / ".cargo/config.toml").write_text(
                '[source.crates-io]\nreplace-with="prepared"\n[source.prepared]\n'
                f'local-registry={json.dumps(str(registry))}\n')
        # Do not inherit a source-workspace output directory into the fixture.
        env = dict(os.environ, CARGO_TARGET_DIR=str(consumer / "target"))
        if args.source == "crates-io":
            # A fresh Cargo home excludes local registry/path overrides and
            # cached unpublished archives from the real consumer qualification.
            env["CARGO_HOME"] = str(consumer / "cargo-home")
        network = ["--offline"] if args.source == "prepared" else []
        qualified = []
        for index, toolchain in enumerate(toolchains):
            prefix = ["rustup", "run", toolchain] if toolchain else []
            # Each toolchain resolves its own fresh consumer graph. Never let
            # an earlier toolchain's lock hide a newer allowed resolution.
            (consumer / "Cargo.lock").unlink(missing_ok=True)
            graph = subprocess.check_output(
                [*prefix, "cargo", "metadata", *network, "--all-features", "--format-version=1"],
                cwd=consumer, env=env, text=True)
            (root / f"target/{args.source}-consumer-metadata.json").write_text(graph)
            check_native_stack_helpers(json.loads(graph)['packages'])
            host = host_triple(prefix, consumer, env)
            evidence = {}
            for feature in (None, "replication"):
                features = ["--features", feature] if feature else []
                name = feature or "standalone"
                # The graph this feature set builds for this target, with the
                # lock it resolved: the license and SBOM evidence describe it.
                graph = json.loads(subprocess.check_output(
                    [*prefix, "cargo", "metadata", *network, *features,
                     "--filter-platform", host, "--format-version=1"],
                    cwd=consumer, env=env, text=True))
                stem = f"{args.source}-{toolchain or 'current'}-{name}"
                written = write_evidence(
                    root / "target", stem, graph, (consumer / "Cargo.lock").read_text(),
                    {"source": args.source, "version": version,
                     "toolchain": toolchain or "current", "features": name, "target": host})
                evidence[name] = len(written["packages"])
                data = consumer / f"run-{index}-{name}"
                data.mkdir()
                command = [*prefix, "cargo", "run", *network, *features]
                subprocess.run([*command, "--", str(data)], cwd=consumer, env=env, check=True)
            quickstart = consumer / f"quickstart-{index}"
            quickstart.mkdir()
            if args.source == "prepared":
                (quickstart / ".cargo").mkdir()
                (quickstart / ".cargo/config.toml").write_text(
                    (consumer / ".cargo/config.toml").read_text())
            run_quickstart(root, network, prefix, env, quickstart)
            qualified.append({"toolchain": toolchain or "current", "target": host,
                              "standalone": "passed", "replication_build": "passed",
                              "quickstart_example": "passed",
                              "license_inventory_packages": evidence})
    report[f"{args.source}_consumer"] = qualified
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.source} external consumer passed; consumer build/data cleaned")


if __name__ == "__main__":
    main()
