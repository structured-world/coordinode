// Calculate library release baselines without confusing server tags with uploads.
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { parse, stringify } from "smol-toml";
import { type Read, request } from "./registry.ts";

interface PackageConfig {
  name: string;
  publish?: boolean;
  git_only?: boolean;
  semver_check?: boolean;
  [key: string]: unknown;
}
export interface ReleaseConfig {
  workspace: { publish?: boolean; [key: string]: unknown };
  package: PackageConfig[];
  [key: string]: unknown;
}

interface CrateResponse {
  crate?: { id?: string };
  versions?: { num?: unknown; yanked?: unknown }[];
}

/** Semantic-version precedence (semver.org, section 11). */
export function compareVersions(a: string, b: string): number {
  const parts = (v: string) => {
    const match = /^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+.*)?$/.exec(v);
    if (match === null) throw new Error(`${v}: not a semantic version`);
    return { core: match.slice(1, 4).map(Number), pre: match[4]?.split(".") ?? [] };
  };
  const x = parts(a);
  const y = parts(b);
  for (let i = 0; i < 3; i++) if (x.core[i] !== y.core[i]) return x.core[i] - y.core[i];
  if (x.pre.length === 0 || y.pre.length === 0) return y.pre.length - x.pre.length;
  for (let i = 0; i < Math.min(x.pre.length, y.pre.length); i++) {
    const [p, q] = [x.pre[i], y.pre[i]];
    if (p === q) continue;
    const [pn, qn] = [/^\d+$/.test(p), /^\d+$/.test(q)];
    if (pn && qn) return Number(p) - Number(q);
    if (pn !== qn) return pn ? -1 : 1;
    return p < q ? -1 : 1;
  }
  return x.pre.length - y.pre.length;
}

/** Use Git only where no published library API exists to compare against. */
export async function releasePrConfig(document: ReleaseConfig, names: string[], read: Read): Promise<ReleaseConfig> {
  const result = structuredClone(document);
  const defaults = result.workspace;
  const packages = result.package;
  const configured = new Set(packages.filter((p) => p.publish ?? defaults.publish ?? true).map((p) => p.name));
  if (
    names.length === 0 ||
    new Set(names).size !== names.length ||
    configured.size !== names.length ||
    !names.every((n) => configured.has(n))
  ) {
    throw new Error("publication configuration does not match the package inventory");
  }
  if (new Set(packages.map((p) => p.name)).size !== packages.length) {
    throw new Error("duplicate package configuration in release inventory");
  }
  const published = new Map<string, Map<string, boolean>>();
  for (const pkg of packages) {
    if (!configured.has(pkg.name)) continue;
    const data = (await read(`crates/${pkg.name}`)) as CrateResponse | null;
    if (data === null) {
      // This config calculates a PR only. Upload always uses the normal
      // checked-in config; Git-only packages cannot be published.
      pkg.git_only = true;
      pkg.publish = false;
      pkg.semver_check = false;
      continue;
    }
    if (
      typeof data !== "object" ||
      Array.isArray(data) ||
      typeof data.crate !== "object" ||
      data.crate?.id !== pkg.name ||
      !Array.isArray(data.versions) ||
      !data.versions.every((v) => typeof v?.num === "string" && typeof v.yanked === "boolean")
    ) {
      throw new Error(`${pkg.name}: unexpected registry response`);
    }
    published.set(pkg.name, new Map(data.versions.map((v) => [v.num as string, v.yanked as boolean])));
  }
  // The libraries share one version and require each other exactly, so the
  // published API of the newest coordinated release can only be built as a
  // whole. A yanked member leaves the others' baselines unresolvable: there is
  // no registry API to compare against, so this calculation skips the check.
  const newest = [...published.values()]
    .flatMap((versions) => [...versions.keys()])
    .reduce<string | null>((max, v) => (max === null || compareVersions(v, max) > 0 ? v : max), null);
  const yanked = [...published].filter(([, versions]) => newest !== null && versions.get(newest) === true).map(([n]) => n);
  if (yanked.length > 0) {
    console.log(
      `API comparison skipped: ${newest} of ${yanked.join(", ")} is yanked, so that release has no buildable baseline`,
    );
    for (const pkg of packages) if (published.has(pkg.name)) pkg.semver_check = false;
  }
  return result;
}

export async function main(argv: string[] = process.argv.slice(2), read: Read = request): Promise<void> {
  const { values } = parseArgs({ args: argv, options: { checkout: { type: "string" } } });
  if (values.checkout === undefined) throw new Error("--checkout is required");
  const root = resolve(values.checkout);
  const output = join(root, "target/release-pr-config.toml");
  if (spawnSync("git", ["check-ignore", "--quiet", output], { cwd: root }).status !== 0) {
    throw new Error("release-PR configuration output must be ignored");
  }
  const document = parse(readFileSync(join(root, ".release-plz.toml"), "utf8")) as unknown as ReleaseConfig;
  const names = (parse(readFileSync(join(root, ".config/registry-packages.toml"), "utf8")) as { packages: string[] })
    .packages;
  const result = await releasePrConfig(document, names, read);
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, stringify(result));
  console.log(`Release-PR configuration prepared for ${names.length} libraries`);
}

if (import.meta.main) await main();
