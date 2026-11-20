// Build and exercise registry packages outside their source workspace.
//
// Run after prepare.ts in the dedicated release checkout. Cargo constructs the
// index for the unpublished archives; cargo-local-registry mirrors their
// external dependencies. The consumer uses source replacement, never path/git
// overrides.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, posix, resolve } from "node:path";
import { parseArgs } from "node:util";
import { parse } from "smol-toml";
import { checkNativeStackHelpers } from "./dependency_policy.ts";
import { writeEvidence, type Metadata } from "./supply_chain.ts";
import { readTarGz } from "./tar.ts";

type Table = Record<string, unknown>;

export type Run = (args: string[], options: { cwd: string; env?: NodeJS.ProcessEnv }) => void;
export type Output = (args: string[], options: { cwd: string; env?: NodeJS.ProcessEnv }) => string;

const runCommand: Run = (args, { cwd, env }) => {
  execFileSync(args[0], args.slice(1), { cwd, env, stdio: "inherit" });
};
const outputCommand: Output = (args, { cwd, env }) =>
  execFileSync(args[0], args.slice(1), { cwd, env, encoding: "utf8", maxBuffer: 1 << 30, stdio: ["ignore", "pipe", "inherit"] });

// A file a source embeds by literal path; computed paths (OUT_DIR) do not match.
const EMBEDDED_FILE = /\binclude(?:_bytes|_str)?!\(\s*"([^"]+)"\s*\)/g;

/** The target triple the toolchain builds for by default. */
function hostTriple(prefix: string[], cwd: string, env: NodeJS.ProcessEnv, output: Output): string {
  const info = output([...prefix, "rustc", "-vV"], { cwd, env });
  const line = info.split("\n").find((l) => l.startsWith("host: ")) as string;
  return line.slice("host: ".length).trim();
}

/** Reject incomplete archives and unresolved source-workspace dependencies. */
export function inspectArchive(
  archive: string,
  name: string,
  version: string,
  names: Set<string>,
  bindings: string[],
): { sha256: string; files: number } {
  const bytes = readFileSync(archive);
  const prefix = `${name}-${version}/`;
  const files = new Map<string, Buffer>();
  for (const member of readTarGz(bytes)) {
    const parts = member.name.split("/");
    if (!member.name.startsWith(prefix) || parts.includes("..") || member.name.startsWith("/")) {
      throw new Error(`invalid archive path: ${member.name}`);
    }
    if (member.type !== "0") throw new Error(`unexpected non-file archive entry: ${member.name}`);
    const relative = member.name.slice(prefix.length);
    if (files.has(relative)) throw new Error(`duplicate archive entry: ${relative}`);
    files.set(relative, member.data);
  }
  const required = new Set(["Cargo.toml", "src/lib.rs", "LICENSE", "NOTICES", "COPYRIGHT", "README.md"]);
  for (const b of bindings) required.add(`bindings/${b}`);
  if (bindings.length > 0) {
    required.add("bindings/files.txt");
    required.add("build.rs");
  }
  const missing = [...required].filter((f) => !files.has(f)).sort();
  if (missing.length > 0) throw new Error(`${name}: missing archive files: ${JSON.stringify(missing)}`);
  // A file a source embeds by literal path must ship: it may sit behind a
  // feature no consumer build enables. Computed paths (OUT_DIR) name none.
  for (const [relative, data] of files) {
    if (!relative.endsWith(".rs")) continue;
    for (const match of data.toString("utf8").matchAll(EMBEDDED_FILE)) {
      const target = posix.normalize(posix.join(posix.dirname(relative), match[1]));
      // A path out of the package can never ship: an unconditional one fails
      // the consumer build, the rest are source-workspace doctests of shared
      // documentation.
      if (target.startsWith("../")) continue;
      if (!files.has(target)) throw new Error(`${name}: ${relative} embeds ${target}, absent from the archive`);
    }
  }
  const manifest = parse((files.get("Cargo.toml") as Buffer).toString("utf8")) as Table & { package: Table };
  if (manifest.package.name !== name || manifest.package.version !== version) {
    throw new Error(`${name}: archive identity mismatch`);
  }
  if (manifest.package.license === "MIT OR Apache-2.0") {
    for (const licenseFile of ["LICENSE-MIT", "LICENSE-APACHE"]) {
      if (!files.has(licenseFile)) throw new Error(`${name}: missing ${licenseFile}`);
    }
  }
  const scopes = [manifest, ...Object.values((manifest.target as Table | undefined) ?? {})] as Table[];
  for (const scope of scopes) {
    const dev = scope["dev-dependencies"] as Table | undefined;
    if (dev !== undefined && Object.keys(dev).length > 0) {
      throw new Error(`${name}: source-only development dependencies remain`);
    }
    for (const kind of ["dependencies", "build-dependencies"]) {
      for (const [alias, dependency] of Object.entries((scope[kind] as Table | undefined) ?? {})) {
        const dep = (typeof dependency === "object" && dependency !== null ? dependency : { version: dependency }) as Table;
        if (["path", "git", "workspace"].some((k) => k in dep)) {
          throw new Error(`${name}: unresolved source dependency ${alias}`);
        }
        const actual = (dep.package as string | undefined) ?? alias;
        if (actual === "coordinode-workspace-hack") {
          throw new Error("workspace feature-unification helper leaked into archive");
        }
        if (names.has(actual) && dep.version !== `=${version}`) {
          throw new Error(`${name}: ${actual} is not bound to the coordinated version`);
        }
      }
    }
  }
  if (bindings.length > 0) {
    const listing = (files.get("bindings/files.txt") as Buffer).toString("utf8").split("\n");
    if (listing[listing.length - 1] === "") listing.pop();
    if (JSON.stringify(listing) !== JSON.stringify(bindings)) {
      throw new Error(`${name}: generated binding inventory differs`);
    }
  }
  return { sha256: createHash("sha256").update(bytes).digest("hex"), files: files.size };
}

export const QUICKSTART = "examples/embedded-quickstart";

/** Whether a plain Cargo requirement (`0.9`, `1.2.3`: caret) admits `version`. */
export function admits(requirement: string, version: string): boolean {
  const wanted = requirement.split(".").map(Number);
  const have = version.split("-")[0].split(".").map(Number);
  // Caret: the leftmost non-zero component of the requirement is fixed.
  const firstNonZero = wanted.findIndex((part) => part !== 0);
  const fixed = (firstNonZero === -1 ? wanted.length - 1 : firstNonZero) + 1;
  for (let i = 0; i < fixed; i++) if (have[i] !== wanted[i]) return false;
  for (let i = 0; i < wanted.length; i++) {
    if ((have[i] ?? 0) !== wanted[i]) return (have[i] ?? 0) > wanted[i];
  }
  return true;
}

/** The published example must build against the version being qualified. */
export function checkQuickstartRequirements(root: string, names: Set<string>, version: string): void {
  const manifest = parse(readFileSync(join(root, QUICKSTART, "Cargo.toml"), "utf8")) as {
    dependencies: Record<string, unknown>;
  };
  for (const [name, requirement] of Object.entries(manifest.dependencies)) {
    if (!names.has(name)) continue;
    if (typeof requirement !== "string" || !admits(requirement, version)) {
      throw new Error(
        `${QUICKSTART}/Cargo.toml requires ${name} ${JSON.stringify(requirement)}, which does not ` +
          `admit ${version}: update the example's requirement with the release`,
      );
    }
  }
}

/**
 * Build the example project as published and run it twice on one directory.
 * The second run checks the documented repeat-run behaviour: it adds its own
 * rows to the database the first run left and verifies them.
 */
function runQuickstart(root: string, network: string[], prefix: string[], env: NodeJS.ProcessEnv, workdir: string, run: Run): void {
  const project = join(workdir, "embedded-quickstart");
  mkdirSync(join(project, "src"), { recursive: true });
  for (const relative of ["Cargo.toml", "src/main.rs"]) {
    writeFileSync(join(project, relative), readFileSync(join(root, QUICKSTART, relative)));
  }
  if (existsSync(join(workdir, ".cargo"))) {
    mkdirSync(join(project, ".cargo"));
    writeFileSync(join(project, ".cargo/config.toml"), readFileSync(join(workdir, ".cargo/config.toml")));
  }
  const data = join(workdir, "quickstart-data");
  for (let i = 0; i < 2; i++) run([...prefix, "cargo", "run", ...network, "--", data], { cwd: project, env });
}

function consumerManifest(version: string): string {
  return (
    '[package]\nname="coordinode-registry-consumer"\nversion="0.0.0"\nedition="2024"\n' +
    '[features]\nreplication=["coordinode-embed/replication"]\n' +
    "[dependencies]\n" +
    `coordinode-embed="=${version}"\ncoordinode-core="=${version}"\n` +
    `coordinode-storage="=${version}"\n`
  );
}

/** Unpack a crate archive into `directory`, refusing paths that leave it. */
function extractCrate(archive: string, directory: string): void {
  for (const member of readTarGz(readFileSync(archive))) {
    if (member.type !== "0") continue;
    const parts = member.name.split("/");
    if (parts.includes("..") || member.name.startsWith("/")) throw new Error(`invalid archive path: ${member.name}`);
    const target = join(directory, member.name);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, member.data);
  }
}

/**
 * Discover allowed external versions, then mirror them for archive tests.
 * Path patches expose only the unpublished archives to dependency resolution.
 * They are never used to build or qualify the consumer.
 */
function mirrorFreshDependencies(
  root: string,
  names: string[],
  version: string,
  registry: string,
  toolchains: (string | null)[],
  run: Run,
  output: Output,
): void {
  const discovery = mkdtempSync(join(tmpdir(), "coordinode-dependency-discovery-"));
  try {
    const packages = join(discovery, "packages");
    mkdirSync(packages);
    for (const name of names) extractCrate(join(root, `target/package/${name}-${version}.crate`), packages);
    const patches =
      "[patch.crates-io]\n" +
      names.map((name) => `${name}={path=${JSON.stringify(join(packages, `${name}-${version}`))}}\n`).join("");
    const env = { ...process.env, CARGO_HOME: join(discovery, "cargo-home"), CARGO_TARGET_DIR: join(discovery, "target") };
    toolchains.forEach((toolchain, index) => {
      const consumer = join(discovery, String(index));
      mkdirSync(join(consumer, "src"), { recursive: true });
      writeFileSync(join(consumer, "src/main.rs"), "fn main() {}\n");
      writeFileSync(join(consumer, "Cargo.toml"), consumerManifest(version) + patches);
      const prefix = toolchain ? ["rustup", "run", toolchain] : [];
      const graph = output([...prefix, "cargo", "metadata", "--all-features", "--format-version=1"], { cwd: consumer, env });
      writeFileSync(join(root, "target/fresh-dependency-metadata.json"), graph);
      checkNativeStackHelpers((JSON.parse(graph) as { packages: { name: string }[] }).packages);
      run(["cargo-local-registry", "--sync", join(consumer, "Cargo.lock"), "--no-delete", registry], { cwd: consumer, env });
    });
  } finally {
    rmSync(discovery, { recursive: true, force: true });
  }
}

interface Report {
  version: string;
  packages: Record<string, Table>;
  [key: string]: unknown;
}

export function main(argv: string[] = process.argv.slice(2), run: Run = runCommand, output: Output = outputCommand): void {
  const { values } = parseArgs({
    args: argv,
    options: {
      checkout: { type: "string" },
      "consumer-toolchain": { type: "string", multiple: true },
      source: { type: "string", default: "prepared" },
      help: { type: "boolean" },
    },
  });
  if (values.help) {
    console.log("usage: check_packages.ts --checkout DIR [--consumer-toolchain T]... [--source prepared|crates-io]");
    return;
  }
  if (values.checkout === undefined) throw new Error("--checkout is required");
  const source = values.source as string;
  if (source !== "prepared" && source !== "crates-io") throw new Error("--source must be prepared or crates-io");
  const root = resolve(values.checkout);
  const inventory = parse(readFileSync(join(root, ".config/registry-packages.toml"), "utf8")) as {
    packages: string[];
    bindings?: Record<string, string[]>;
  };
  const reportPath = join(root, "target/registry-preparation.json");
  const report = JSON.parse(readFileSync(reportPath, "utf8")) as Report;
  const names = inventory.packages;
  const version = report.version;
  checkQuickstartRequirements(root, new Set(names), version);
  const toolchains: (string | null)[] = values["consumer-toolchain"] ?? [null];
  const registry = join(root, "target/package/tmp-registry");
  if (source === "prepared") {
    run(["cargo", "package", "--allow-dirty", "--no-verify", ...names.flatMap((n) => ["-p", n])], { cwd: root });
    for (const name of names) {
      report.packages[name].archive = inspectArchive(
        join(root, `target/package/${name}-${version}.crate`),
        name,
        version,
        new Set(names),
        inventory.bindings?.[name] ?? [],
      );
    }
    const packagedLock = readFileSync(join(root, "Cargo.lock"));
    // --no-delete preserves the unpublished package entries made by Cargo.
    // Use the installed executable directly: this tool's CLI does not consume
    // Cargo's extra `local-registry` argument in its current release.
    run(["cargo-local-registry", "--sync", join(root, "Cargo.lock"), "--no-delete", registry], { cwd: root });
    if (!readFileSync(join(root, "Cargo.lock")).equals(packagedLock)) {
      throw new Error("registry mirroring changed Cargo.lock; regenerate and requalify archives");
    }
    mirrorFreshDependencies(root, names, version, registry, toolchains, run, output);
  }
  const consumer = mkdtempSync(join(tmpdir(), "coordinode-package-consumer-"));
  try {
    mkdirSync(join(consumer, "src"));
    mkdirSync(join(consumer, ".cargo"));
    writeFileSync(join(consumer, "src/main.rs"), readFileSync(join(root, "scripts/release/consumer.rs")));
    writeFileSync(join(consumer, "Cargo.toml"), consumerManifest(version));
    if (source === "prepared") {
      writeFileSync(
        join(consumer, ".cargo/config.toml"),
        `[source.crates-io]\nreplace-with="prepared"\n[source.prepared]\nlocal-registry=${JSON.stringify(registry)}\n`,
      );
    }
    // Do not inherit a source-workspace output directory into the fixture.
    const env: NodeJS.ProcessEnv = { ...process.env, CARGO_TARGET_DIR: join(consumer, "target") };
    // A fresh Cargo home excludes local registry/path overrides and cached
    // unpublished archives from the real consumer qualification.
    if (source === "crates-io") env.CARGO_HOME = join(consumer, "cargo-home");
    const network = source === "prepared" ? ["--offline"] : [];
    const qualified: Table[] = [];
    toolchains.forEach((toolchain, index) => {
      const prefix = toolchain ? ["rustup", "run", toolchain] : [];
      // Each toolchain resolves its own fresh consumer graph. Never let an
      // earlier toolchain's lock hide a newer allowed resolution.
      if (existsSync(join(consumer, "Cargo.lock"))) unlinkSync(join(consumer, "Cargo.lock"));
      const graph = output([...prefix, "cargo", "metadata", ...network, "--all-features", "--format-version=1"], {
        cwd: consumer,
        env,
      });
      writeFileSync(join(root, `target/${source}-consumer-metadata.json`), graph);
      checkNativeStackHelpers((JSON.parse(graph) as { packages: { name: string }[] }).packages);
      const host = hostTriple(prefix, consumer, env, output);
      const evidence: Record<string, number> = {};
      for (const feature of [null, "replication"]) {
        const features = feature ? ["--features", feature] : [];
        const name = feature ?? "standalone";
        // The graph this feature set builds for this target, with the lock it
        // resolved: the license and SBOM evidence describe it.
        const resolved = JSON.parse(
          output([...prefix, "cargo", "metadata", ...network, ...features, "--filter-platform", host, "--format-version=1"], {
            cwd: consumer,
            env,
          }),
        ) as Metadata;
        const stem = `${source}-${toolchain ?? "current"}-${name}`;
        const written = writeEvidence(join(root, "target"), stem, resolved, readFileSync(join(consumer, "Cargo.lock"), "utf8"), {
          source,
          version,
          toolchain: toolchain ?? "current",
          features: name,
          target: host,
        });
        evidence[name] = written.packages.length;
        const data = join(consumer, `run-${index}-${name}`);
        mkdirSync(data);
        run([...prefix, "cargo", "run", ...network, ...features, "--", data], { cwd: consumer, env });
      }
      const quickstart = join(consumer, `quickstart-${index}`);
      mkdirSync(quickstart);
      if (source === "prepared") {
        mkdirSync(join(quickstart, ".cargo"));
        writeFileSync(join(quickstart, ".cargo/config.toml"), readFileSync(join(consumer, ".cargo/config.toml")));
      }
      runQuickstart(root, network, prefix, env, quickstart, run);
      qualified.push({
        toolchain: toolchain ?? "current",
        target: host,
        standalone: "passed",
        replication_build: "passed",
        quickstart_example: "passed",
        license_inventory_packages: evidence,
      });
    });
    report[`${source}_consumer`] = qualified;
  } finally {
    rmSync(consumer, { recursive: true, force: true });
  }
  // The report preparation wrote in the named checkout, updated in place.
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  console.log(`${source} external consumer passed; consumer build/data cleaned`);
}

if (import.meta.main) main();
