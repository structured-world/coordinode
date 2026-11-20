// Dependency, license and attribution evidence for one resolved consumer graph.
//
// It runs on the metadata Cargo resolved for an actual consumer, bound to its
// feature set, target triple and lockfile, so the evidence describes what that
// consumer builds rather than the source workspace.
import { readdirSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { parse } from "smol-toml";

// Files a package ships whose text a redistributor must carry along.
const ATTRIBUTION_PREFIXES = ["LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT", "UNLICENSE"];

interface DepKind {
  kind: string | null;
}
interface Dep {
  pkg: string;
  dep_kinds: DepKind[];
}
interface Node {
  id: string;
  deps: Dep[];
}
export interface MetadataPackage {
  id: string;
  name: string;
  version: string;
  source?: string | null;
  license?: string | null;
  license_file?: string | null;
  manifest_path: string;
}
export interface Metadata {
  packages: MetadataPackage[];
  resolve: { root: string; nodes: Node[] };
}
export interface Entry {
  name: string;
  version: string;
  source: string | null;
  license: string | null;
  license_file: string | null;
  checksum: string | null;
  attribution_files: string[];
}
export interface Evidence {
  context: Record<string, string>;
  packages: Entry[];
}

/** Ids of the packages the root builds: normal and build edges, never dev. */
export function reachable(metadata: Metadata): Set<string> {
  const nodes = new Map(metadata.resolve.nodes.map((node) => [node.id, node]));
  const root = metadata.resolve.root;
  const seen = new Set<string>();
  const pending = [root];
  while (pending.length > 0) {
    const current = pending.pop() as string;
    if (seen.has(current)) continue;
    seen.add(current);
    for (const dep of nodes.get(current)?.deps ?? []) {
      // A `null` kind is a normal edge; dev edges never reach a consumer's build.
      if (dep.dep_kinds.some((kind) => kind.kind !== "dev")) pending.push(dep.pkg);
    }
  }
  seen.delete(root);
  return seen;
}

/** Registry checksums by `name version` from a Cargo.lock. */
export function lockChecksums(lockText: string): Map<string, string> {
  const lock = parse(lockText) as { package?: { name: string; version: string; checksum?: string }[] };
  const checksums = new Map<string, string>();
  for (const p of lock.package ?? []) {
    if (p.checksum !== undefined) checksums.set(`${p.name} ${p.version}`, p.checksum);
  }
  return checksums;
}

/** License and notice texts at the root of a package's source. */
export function attributionFiles(manifestPath: string): string[] {
  return readdirSync(dirname(manifestPath), { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .filter((name) => ATTRIBUTION_PREFIXES.some((prefix) => name.toUpperCase().startsWith(prefix)))
    .sort();
}

/**
 * One entry per package the consumer builds, refusing any without a license.
 * `context` names what the graph was resolved for (toolchain, features, target
 * triple, source) and is carried into the evidence unchanged.
 */
export function inventory(metadata: Metadata, lockText: string, context: Record<string, string>): Evidence {
  const checksums = lockChecksums(lockText);
  const packages = new Map(metadata.packages.map((p) => [p.id, p]));
  const entries: Entry[] = [];
  const unlicensed: string[] = [];
  for (const id of [...reachable(metadata)].sort()) {
    const p = packages.get(id) as MetadataPackage;
    const license = p.license ?? null;
    const licenseFile = p.license_file ?? null;
    if (!license && !licenseFile) unlicensed.push(`${p.name} ${p.version}`);
    entries.push({
      name: p.name,
      version: p.version,
      source: p.source ?? null,
      license,
      license_file: licenseFile,
      checksum: checksums.get(`${p.name} ${p.version}`) ?? null,
      attribution_files: attributionFiles(p.manifest_path),
    });
  }
  if (unlicensed.length > 0) {
    throw new Error(`packages without a license declaration: ${JSON.stringify(unlicensed)}`);
  }
  return { context, packages: entries };
}

/** The inventory as a CycloneDX 1.5 SBOM. */
export function cyclonedx(evidence: Evidence) {
  const components = evidence.packages.map((entry) => {
    const component: Record<string, unknown> = {
      type: "library",
      name: entry.name,
      version: entry.version,
      purl: `pkg:cargo/${entry.name}@${entry.version}`,
    };
    if (entry.license) component.licenses = [{ expression: entry.license }];
    if (entry.checksum) component.hashes = [{ alg: "SHA-256", content: entry.checksum }];
    return component;
  });
  return {
    bomFormat: "CycloneDX",
    specVersion: "1.5",
    version: 1,
    metadata: {
      properties: Object.keys(evidence.context)
        .sort()
        .map((key) => ({ name: `coordinode:${key}`, value: String(evidence.context[key]) })),
    },
    components,
  };
}

/** Write `<stem>-licenses.json` and `<stem>-sbom.cdx.json`; return the inventory. */
export function writeEvidence(
  directory: string,
  stem: string,
  metadata: Metadata,
  lockText: string,
  context: Record<string, string>,
): Evidence {
  const evidence = inventory(metadata, lockText, context);
  // The stem names files in `directory`; one carrying a separator (a
  // toolchain name given with a path in it) would write elsewhere.
  if (basename(stem) !== stem) throw new Error(`evidence stem ${JSON.stringify(stem)} is not a file name`);
  writeFileSync(join(directory, `${stem}-licenses.json`), JSON.stringify(evidence, null, 2) + "\n");
  writeFileSync(join(directory, `${stem}-sbom.cdx.json`), JSON.stringify(cyclonedx(evidence), null, 2) + "\n");
  return evidence;
}
