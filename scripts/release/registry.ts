// Reconcile an immutable coordinated release against crates.io.
import { execFileSync, spawnSync } from "node:child_process";
import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { parse } from "smol-toml";

export type Read = (path: string) => Promise<unknown>;

/** Read public registry metadata without publication credentials. */
export async function request(path: string): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`https://crates.io/api/v1/${path}`, {
      headers: {
        "Content-Type": "application/json",
        "User-Agent": "coordinode-release (github.com/structured-world/coordinode)",
      },
      signal: AbortSignal.timeout(30_000),
    });
  } catch {
    throw new Error("crates.io unavailable; reconcile the original release before retrying");
  }
  if (response.status === 404 && path.startsWith("crates/")) return null;
  if (!response.ok) throw new Error(`crates.io GET returned HTTP ${response.status}`);
  return response.json();
}

interface Report {
  version: string;
  source_commit: string;
  packages: Record<string, { archive: { sha256: string } }>;
  registry_checksums_verified?: boolean;
}

/** Existing versions must be the exact qualified archives, including on retries. */
export async function checkVersions(report: Report, read: Read, allowMissing: boolean): Promise<void> {
  for (const [name, pkg] of Object.entries(report.packages)) {
    const data = (await read(`crates/${name}/${report.version}`)) as
      | { version: { checksum: string; yanked: boolean } }
      | null;
    if (data === null) {
      if (allowMissing) continue;
      throw new Error(`${name} ${report.version} is not published`);
    }
    if (data.version.checksum !== pkg.archive.sha256 || data.version.yanked) {
      throw new Error(`${name}: registry version does not match the qualified archive`);
    }
  }
}

/**
 * Every package already exists on crates.io: a trusted-publishing token cannot
 * create a crate, so a new one would stop the release half-published.
 */
export async function checkRegistered(report: Pick<Report, "packages">, read: Read): Promise<void> {
  const missing: string[] = [];
  for (const name of Object.keys(report.packages).sort()) {
    if ((await read(`crates/${name}`)) === null) missing.push(name);
  }
  if (missing.length > 0) {
    throw new Error(
      `${missing.join(", ")} not on crates.io: trusted publishing cannot create a crate; ` +
        "publish its first version from the prepared archive, add the trusted publisher, then rerun",
    );
  }
}

interface Pull {
  merged_at: string | null;
  base: { ref: string };
  head: { ref: string };
  merge_commit_sha: unknown;
}

/** Resolve a retry from the forge's approved merge, never a moving PR head. */
export function mergedReleaseRevision(pull: Pull): string {
  if (pull.merged_at === null || pull.base.ref !== "main" || !pull.head.ref.startsWith("release-plz-")) {
    throw new Error("retry requires a merged release PR targeting main");
  }
  const revision = pull.merge_commit_sha;
  if (typeof revision !== "string" || !/^[0-9a-f]{40}$/.test(revision)) {
    throw new Error("release PR has no valid merge commit identifier");
  }
  return revision;
}

/**
 * Match release-plz's commit selection before creating qualified archives.
 *
 * release-plz checks out an ancestor PR head for a merge commit. Squash merges
 * retain the checked-out source. Never transfer a source-suite result to
 * different files, even if both commits belong to the same release PR.
 */
export function publicationRevision(root: string, source: string, releaseHead: string): string {
  const git = (...args: string[]) => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
  if (git("rev-parse", "--is-shallow-repository") !== "false") {
    throw new Error("publication requires complete Git history");
  }
  const sourceTree = git("rev-parse", `${source}^{tree}`);
  const present = spawnSync("git", ["cat-file", "-e", `${releaseHead}^{commit}`], { cwd: root });
  // Complete source history cannot omit an ancestor: a deleted squashed PR
  // head is not a candidate for release-plz's ancestor checkout.
  if (present.status !== 0) return source;
  const ancestor = spawnSync("git", ["merge-base", "--is-ancestor", releaseHead, source], { cwd: root });
  if (ancestor.status === 1) return source;
  if (ancestor.status !== 0) throw new Error(`git merge-base exited with ${ancestor.status}`);
  if (git("rev-parse", `${releaseHead}^{tree}`) !== sourceTree) {
    throw new Error("release-plz would publish a different tree from the successful source suite");
  }
  return releaseHead;
}

function gh(path: string, ...extra: string[]): unknown {
  return JSON.parse(execFileSync("gh", ["api", ...extra, path], { encoding: "utf8" }));
}

function output(lines: string): void {
  appendFileSync(process.env.GITHUB_OUTPUT as string, lines);
}

async function main(): Promise<void> {
  const { positionals, values } = parseArgs({
    allowPositionals: true,
    options: { checkout: { type: "string" }, "release-pr": { type: "string" } },
  });
  const mode = positionals[0];
  if (!["source", "eligible", "preflight", "verify"].includes(mode) || positionals.length !== 1) {
    throw new Error("mode must be one of: source, eligible, preflight, verify");
  }
  if (values.checkout === undefined) throw new Error("--checkout is required");
  const releasePr = values["release-pr"] === undefined ? undefined : Number(values["release-pr"]);
  if (releasePr !== undefined && (mode !== "source" || !Number.isInteger(releasePr) || releasePr <= 0)) {
    throw new Error("--release-pr requires source mode and a positive PR number");
  }
  const root = resolve(values.checkout);
  if (process.env.GITHUB_REPOSITORY !== "structured-world/coordinode") {
    throw new Error("release requires the authoritative repository");
  }
  if (process.env.GITHUB_REF !== "refs/heads/main") throw new Error("release requires the main branch");
  let source = execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim();
  if (mode === "source") {
    if (releasePr !== undefined) {
      source = mergedReleaseRevision(gh(`repos/structured-world/coordinode/pulls/${releasePr}`) as Pull);
    }
    execFileSync("git", ["merge-base", "--is-ancestor", source, "origin/main"], { cwd: root });
    output(`commit=${source}\n`);
    console.log(`Selected source commit: ${source}`);
    return;
  }
  if (mode === "eligible") {
    // This is release-plz's documented release_always=false contract. Check it
    // before expensive preparation or checking an old server-only tag.
    const pulls = gh(`repos/structured-world/coordinode/commits/${source}/pulls`) as
      ({ number: number } & Pull)[];
    const release = pulls.find((p) => p.merged_at !== null && p.head.ref.startsWith("release-plz-"));
    let publication = source;
    if (release !== undefined) {
      const pages = gh(`repos/structured-world/coordinode/pulls/${release.number}/commits`, "--paginate", "--slurp") as
        { sha: string }[][];
      const commits = pages.flat();
      if (commits.length === 0) throw new Error("merged release PR has no recorded commits");
      publication = publicationRevision(root, source, commits[commits.length - 1].sha);
    }
    output(`publish=${release !== undefined}\npublication_commit=${publication}\nverified_commit=${source}\n`);
    console.log(`Merged release PR: ${release !== undefined ? "True" : "False"}`);
    return;
  }
  const reportPath = join(root, "target/registry-preparation.json");
  const report = JSON.parse(readFileSync(reportPath, "utf8")) as Report;
  const names = (parse(readFileSync(join(root, ".config/registry-packages.toml"), "utf8")) as { packages: string[] })
    .packages;
  const reported = Object.keys(report.packages);
  if (reported.length !== names.length || !names.every((n) => n in report.packages) || report.source_commit !== source) {
    throw new Error("qualification report does not describe this checkout and inventory");
  }
  if (mode === "preflight") {
    // An argv list with no shell, and the value starts with refs/tags/v, so
    // the version from our own report cannot become a git option.
    const tag = spawnSync("git", ["rev-parse", "--verify", `refs/tags/v${report.version}^{commit}`], {
      cwd: root,
      encoding: "utf8",
    });
    if (tag.status === 0 && tag.stdout.trim() !== source) {
      throw new Error("version belongs to another source revision; prepare a new release PR");
    }
    await checkRegistered(report, request);
    await checkVersions(report, request, true);
  } else {
    await checkVersions(report, request, false);
    report.registry_checksums_verified = true;
    // The report preparation wrote in the named checkout, updated in place.
    writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  }
  console.log(`${mode}: verified ${names.length} coordinated packages`);
}

if (import.meta.main) await main();
