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
  for (const pkg of packages) {
    if (!configured.has(pkg.name)) continue;
    const data = (await read(`crates/${pkg.name}`)) as { crate?: { id?: string } } | null;
    if (data === null) {
      // This config calculates a PR only. Upload always uses the normal
      // checked-in config; Git-only packages cannot be published.
      pkg.git_only = true;
      pkg.publish = false;
      pkg.semver_check = false;
    } else if (typeof data !== "object" || Array.isArray(data) || typeof data.crate !== "object" || data.crate?.id !== pkg.name) {
      throw new Error(`${pkg.name}: unexpected registry response`);
    }
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
