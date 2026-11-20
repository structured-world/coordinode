// Prepare registry archives in a dedicated release checkout, never an editor checkout.
//
// Runtime Rust sources remain unchanged. Only package manifests, the two build
// scripts and distributable metadata are prepared. Generated protobuf stays out
// of Git. The caller must test the source before preparation and the resulting
// archives afterwards; release PR generation uses a separate, clean checkout.
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { parse, stringify } from "smol-toml";
import { checkNativeStackHelpers } from "./dependency_policy.ts";

type Table = Record<string, unknown>;

export type Command = (args: string[], root: string) => string;

const runCommand: Command = (args, root) =>
  execFileSync(args[0], args.slice(1), { cwd: root, encoding: "utf8", maxBuffer: 1 << 30 });

interface MetadataPackage {
  id: string;
  name: string;
  version?: string;
  source?: string | null;
  license?: string | null;
  manifest_path?: string;
}
interface CargoMetadata {
  packages: MetadataPackage[];
  workspace_members: string[];
  resolve: { nodes: { id: string; deps: { pkg: string; dep_kinds: { kind: string | null }[] }[] }[] };
}

/** Traverse every target and optional dependency, but never dev-only edges. */
export function libraryClosure(metadata: CargoMetadata, entrypoint: string): [Set<string>, MetadataPackage[]] {
  const members = new Set(metadata.workspace_members);
  const packages = new Map(metadata.packages.map((p) => [p.id, p]));
  const nodes = new Map(metadata.resolve.nodes.map((n) => [n.id, n]));
  const start = metadata.packages.find((p) => p.name === entrypoint)?.id as string;
  const visited = new Set<string>();
  const pending = [start];
  while (pending.length > 0) {
    const item = pending.pop() as string;
    if (visited.has(item)) continue;
    visited.add(item);
    for (const dep of nodes.get(item)?.deps ?? []) {
      if (dep.dep_kinds.some((k) => k.kind !== "dev")) pending.push(dep.pkg);
    }
  }
  const internal = new Set([...visited].filter((p) => members.has(p)).map((p) => packages.get(p)?.name as string));
  const external = [...visited].filter((p) => !members.has(p)).map((p) => packages.get(p) as MetadataPackage);
  return [internal, external];
}

/** Describe only delivered library targets, preserving runtime features. */
export function prepareManifest(document: Table, version: string, names: Set<string>, codegen: boolean): Table {
  const doc = structuredClone(document);
  const pkg = doc.package as Table;
  // `assets/` holds the data files sources embed, such as the CJK dictionary.
  pkg.include = ["/src/**", "/build.rs", "/bindings/**", "/assets/**", "/LICENSE", "/LICENSE-*", "/NOTICES",
    "/COPYRIGHT", "/README.md", "/Cargo.toml"];
  for (const toggle of ["autobins", "autoexamples", "autotests", "autobenches"]) pkg[toggle] = false;
  for (const section of ["dev-dependencies", "bench", "test", "example", "bin"]) delete doc[section];
  for (const target of Object.values((doc.target as Table | undefined) ?? {})) {
    delete (target as Table)["dev-dependencies"];
  }
  // A normalized registry manifest must never pull private fixtures into
  // dependency resolution. The complete source suite was run before this.
  const lib = (doc.lib ??= {}) as Table;
  lib.test = false;
  lib.bench = false;
  if (codegen) {
    // The prepared build script reads shipped bindings and needs no build
    // dependency. A feature that enabled one must stop naming it, or the
    // manifest refers to a dependency it no longer has.
    const removed = new Set(Object.keys((doc["build-dependencies"] as Table | undefined) ?? {}).map((n) => `dep:${n}`));
    delete doc["build-dependencies"];
    const features = (doc.features as Record<string, string[]> | undefined) ?? {};
    for (const [name, enables] of Object.entries(features)) {
      features[name] = enables.filter((item) => !removed.has(item));
    }
  }
  for (const section of ["dependencies", "build-dependencies"]) {
    for (const [name, dep] of Object.entries((doc[section] as Table | undefined) ?? {})) {
      if (name === "coordinode-workspace-hack") throw new Error("hakari dependency was not removed");
      if (names.has(name) && typeof dep === "object" && dep !== null && "path" in dep) {
        (dep as Table).version = `=${version}`;
      }
    }
  }
  return doc;
}

export type Build = (root: string) => { status: number | null; stdout: string };

const runBuild: Build = (root) => {
  const build = spawnSync(
    "cargo",
    ["build", "-p", "coordinode-raft", "-p", "coordinode-replicate", "--message-format=json"],
    { cwd: root, encoding: "utf8", maxBuffer: 1 << 30, stdio: ["ignore", "pipe", "inherit"] },
  );
  return { status: build.status, stdout: build.stdout };
};

export function main(
  argv: string[] = process.argv.slice(2),
  command: Command = runCommand,
  run: (args: string[], root: string) => void = (args, root) => {
    execFileSync(args[0], args.slice(1), { cwd: root, stdio: "inherit" });
  },
  build: Build = runBuild,
): void {
  const { values } = parseArgs({ args: argv, options: { checkout: { type: "string" } } });
  if (values.checkout === undefined) throw new Error("--checkout is required (a dedicated disposable release checkout)");
  const root = resolve(values.checkout);
  const config = parse(readFileSync(join(root, ".config/registry-packages.toml"), "utf8")) as {
    entrypoint: string;
    packages: string[];
    bindings: Record<string, string[]>;
  };
  const names = new Set(config.packages);
  if (names.size !== config.packages.length) throw new Error("duplicate package in release inventory");
  const version = (parse(readFileSync(join(root, "Cargo.toml"), "utf8")) as { workspace: { package: { version: string } } })
    .workspace.package.version;
  const source = command(["git", "rev-parse", "HEAD"], root).trim();
  if (command(["git", "status", "--porcelain"], root).trim()) {
    throw new Error("source checkout must be clean before package preparation");
  }
  const proto = command(["git", "rev-parse", "HEAD"], join(root, "proto")).trim();
  if (command(["git", "status", "--porcelain"], join(root, "proto")).trim()) {
    throw new Error("proto checkout must match its recorded revision");
  }
  // Use hakari's supported publication path. Do not ship a dummy hack crate.
  run(["cargo", "hakari", "remove-deps", "--yes"], root);
  const metadata = JSON.parse(command(["cargo", "metadata", "--all-features", "--format-version=1"], root)) as CargoMetadata;
  const [internal, external] = libraryClosure(metadata, config.entrypoint);
  checkNativeStackHelpers(external);
  const missing = [...names].filter((n) => !internal.has(n));
  const extra = [...internal].filter((n) => !names.has(n));
  if (missing.length > 0 || extra.length > 0) {
    throw new Error(`release closure mismatch: missing=${JSON.stringify(missing)}, extra=${JSON.stringify(extra)}`);
  }
  const gitDependencies = external.filter((p) => (p.source ?? "").startsWith("git+")).map((p) => p.name);
  if (gitDependencies.length > 0) {
    throw new Error(`registry closure still depends on Git: ${JSON.stringify(gitDependencies)}`);
  }
  const workspace = new Set(metadata.workspace_members);
  const manifests = new Map(
    metadata.packages.filter((p) => workspace.has(p.id)).map((p) => [p.name, p.manifest_path as string]),
  );
  const byId = new Map(metadata.packages.map((p) => [p.id, p.name]));
  // Build before replacing either build script. Cargo reports OUT_DIR even when
  // its fingerprint is fresh, so no filesystem guessing is required.
  const built = build(root);
  const outputs = new Map<string, string>();
  for (const line of built.stdout.split("\n")) {
    if (!line.trim()) continue;
    const event = JSON.parse(line) as {
      reason: string;
      package_id?: string;
      out_dir?: string;
      message?: { rendered?: string };
    };
    if (event.reason === "build-script-executed") {
      const name = byId.get(event.package_id as string);
      if (name !== undefined && name in config.bindings) outputs.set(name, event.out_dir as string);
    } else if (event.reason === "compiler-message") {
      process.stdout.write(event.message?.rendered ?? "");
    }
  }
  if (built.status !== 0) throw new Error("source binding build failed");
  if (command(["git", "rev-parse", "HEAD"], join(root, "proto")).trim() !== proto) {
    throw new Error("proto revision changed during generation");
  }
  if (command(["git", "status", "--porcelain"], join(root, "proto")).trim()) {
    throw new Error("proto checkout changed during generation");
  }
  const report = {
    version,
    source_commit: source,
    proto_commit: proto,
    packages: {} as Record<string, { manifest_sha256: string }>,
    source_ci_commit: process.env.SOURCE_CI_COMMIT ?? null,
    dependency_licenses: external
      .map((p) => ({ name: p.name, version: p.version, license: p.license ?? null, source: p.source ?? null }))
      .sort((a, b) => {
        const [x, y] = [`${a.name}\0${a.version}`, `${b.name}\0${b.version}`];
        return x < y ? -1 : x > y ? 1 : 0;
      }),
  };
  for (const name of config.packages) {
    const manifest = manifests.get(name) as string;
    const folder = dirname(manifest);
    if (name in config.bindings) {
      const bindingDir = join(folder, "bindings");
      mkdirSync(bindingDir);
      const files = config.bindings[name];
      for (const filename of files) copyFileSync(join(outputs.get(name) as string, filename), join(bindingDir, filename));
      writeFileSync(join(bindingDir, "files.txt"), files.join("\n") + "\n");
      copyFileSync(join(root, "scripts/release/prepared-build.rs"), join(folder, "build.rs"));
    }
    for (const filename of ["LICENSE", "NOTICES", "COPYRIGHT"]) {
      if (!existsSync(join(folder, filename))) copyFileSync(join(root, filename), join(folder, filename));
    }
    const doc = prepareManifest(parse(readFileSync(manifest, "utf8")) as Table, version, names, name in config.bindings);
    writeFileSync(manifest, stringify(doc));
    report.packages[name] = { manifest_sha256: createHash("sha256").update(readFileSync(manifest)).digest("hex") };
  }
  mkdirSync(join(root, "target"), { recursive: true });
  writeFileSync(join(root, "target/registry-preparation.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(`Prepared ${names.size} libraries at ${version} from ${source}; proto ${proto}`);
}

if (import.meta.main) main();
