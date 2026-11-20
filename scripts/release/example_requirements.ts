// Bring the builds outside the workspace to the release a PR prepares.
//
// release-plz versions the workspace and its lockfile only. Run after
// `release-plz release-pr --output json`, on the release PR's branch:
//
// - the example builds from registry packages as an application does, so each
//   of its requirements on a registry package becomes the caret requirement of
//   the release version;
// - a standalone workspace that builds the workspace crates by path (the
//   benchmarks, built `--locked`) has its lockfile record their new versions,
//   every other locked package kept as it was.
//
// The changes are committed and pushed to that branch, so the PR carries them.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { parse } from "smol-toml";

export const QUICKSTART = "examples/embedded-quickstart/Cargo.toml";
// Standalone workspaces with a committed lockfile that depend on workspace
// crates by path.
const PATH_LOCKED = ["benches/vector-ann"];
const REPOSITORY = "structured-world/coordinode";

/** The caret requirement admitting `version` and its compatible releases: `0.10.0` gives `0.10`. */
export function requirementFor(version: string): string {
  const [major, minor] = version.split("-")[0].split(".");
  return `${major}.${minor}`;
}

function escape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * `text` with every `[dependencies]` requirement on one of `names` set to the
 * release's caret requirement; other entries are kept byte for byte.
 */
export function updateManifest(text: string, names: Set<string>, version: string): string {
  const manifest = parse(text) as { dependencies?: Record<string, unknown> };
  const wanted = requirementFor(version);
  let updated = text;
  for (const [name, requirement] of Object.entries(manifest.dependencies ?? {})) {
    if (!names.has(name)) continue;
    if (typeof requirement !== "string") {
      throw new Error(`${QUICKSTART}: ${name} must be a plain version requirement`);
    }
    const line = new RegExp(`^(${escape(name)}\\s*=\\s*)"[^"]*"`, "gm");
    const count = [...updated.matchAll(line)].length;
    if (count !== 1) throw new Error(`${QUICKSTART}: ${name} is not one \`name = "version"\` line`);
    updated = updated.replace(line, `$1"${wanted}"`);
  }
  return updated;
}

/**
 * Bring the lockfile of the standalone `workspace` under `root` to the versions
 * of the path crates it builds; true when it changed. Resolving without
 * `--locked` rewrites only the entries that no longer match their manifests, so
 * registry packages keep their locked versions.
 */
export function refreshLock(root: string, workspace: string): boolean {
  const lock = join(root, workspace, "Cargo.lock");
  const before = readFileSync(lock);
  execFileSync("cargo", ["tree", "--quiet", "--depth", "0", "--manifest-path", join(root, workspace, "Cargo.toml")], {
    cwd: root,
    stdio: ["ignore", "ignore", "inherit"],
  });
  return !readFileSync(lock).equals(before);
}

/** The head branch of the release PR release-plz reported, or null. */
export function releaseBranch(prJson: string): string | null {
  const prs = (JSON.parse(prJson || "{}") as { prs?: { head_branch: string }[] }).prs ?? [];
  return prs.length > 0 ? prs[0].head_branch : null;
}

function git(root: string, ...args: string[]): void {
  execFileSync("git", args, { cwd: root, stdio: "inherit" });
}

function main(): void {
  const { values } = parseArgs({ options: { checkout: { type: "string" }, "pr-json": { type: "string" } } });
  if (values.checkout === undefined || values["pr-json"] === undefined) {
    throw new Error("--checkout and --pr-json are required");
  }
  const root = resolve(values.checkout);
  const branch = releaseBranch(readFileSync(values["pr-json"], "utf8"));
  if (branch === null) {
    console.log("no release PR: nothing to update");
    return;
  }
  const remote = `https://x-access-token:${process.env.GIT_TOKEN}@github.com/${REPOSITORY}.git`;
  git(root, "fetch", "--quiet", remote, `refs/heads/${branch}`);
  git(root, "checkout", "--quiet", "--detach", "FETCH_HEAD");
  const version = (parse(readFileSync(join(root, "Cargo.toml"), "utf8")) as { workspace: { package: { version: string } } })
    .workspace.package.version;
  const names = new Set(
    (parse(readFileSync(join(root, ".config/registry-packages.toml"), "utf8")) as { packages: string[] }).packages,
  );
  const changed: string[] = [];
  const path = join(root, QUICKSTART);
  const before = readFileSync(path, "utf8");
  const after = updateManifest(before, names, version);
  if (after !== before) {
    writeFileSync(path, after);
    changed.push(QUICKSTART);
  }
  for (const workspace of PATH_LOCKED) {
    if (refreshLock(root, workspace)) changed.push(`${workspace}/Cargo.lock`);
  }
  if (changed.length === 0) {
    console.log(`the builds outside the workspace already take ${version}`);
    return;
  }
  git(root, "add", ...changed);
  git(
    root,
    "-c",
    "user.name=sw-release-bot[bot]",
    "-c",
    "user.email=sw-release-bot[bot]@users.noreply.github.com",
    "commit",
    "--quiet",
    "-m",
    `chore(release): bring the builds outside the workspace to ${version}`,
  );
  git(root, "push", "--quiet", remote, `HEAD:refs/heads/${branch}`);
  console.log(`${changed.join(", ")} take ${version} on ${branch}`);
}

if (import.meta.main) main();
