// The evidence lists what a consumer builds and redistributes.
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { inventory, type Metadata, writeEvidence } from "./supply_chain.ts";

function pkg(root: string, name: string, license: string | null = "MIT", files = ["LICENSE"]) {
  const directory = join(root, name);
  mkdirSync(directory);
  writeFileSync(join(directory, "Cargo.toml"), "");
  for (const file of files) writeFileSync(join(directory, file), "text");
  return {
    id: `${name} 1.0.0`,
    name,
    version: "1.0.0",
    source: "registry+https://github.com/rust-lang/crates.io-index",
    license,
    license_file: null,
    manifest_path: join(directory, "Cargo.toml"),
  };
}

const edge = (target: string, ...kinds: (string | null)[]) => ({
  pkg: `${target} 1.0.0`,
  dep_kinds: kinds.map((kind) => ({ kind })),
});

const LOCK = `
[[package]]
name = "normal"
version = "1.0.0"
checksum = "aa"

[[package]]
name = "built"
version = "1.0.0"
checksum = "bb"
`;

function graph(root: string, normalLicense: string | null = "MIT"): Metadata {
  return {
    packages: [
      pkg(root, "consumer"),
      pkg(root, "normal", normalLicense, ["LICENSE-MIT", "NOTICE", "README.md"]),
      pkg(root, "built"),
      pkg(root, "dev-only"),
    ],
    resolve: {
      root: "consumer 1.0.0",
      nodes: [
        { id: "consumer 1.0.0", deps: [edge("normal", null), edge("dev-only", "dev")] },
        { id: "normal 1.0.0", deps: [edge("built", "build")] },
        { id: "built 1.0.0", deps: [] },
        { id: "dev-only 1.0.0", deps: [] },
      ],
    },
  };
}

function withDirectory(body: (directory: string) => void): void {
  const directory = mkdtempSync(join(tmpdir(), "coordinode-release-test-"));
  try {
    body(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

describe("supply chain", () => {
  test("the inventory follows normal and build edges only", () => {
    // A dev-only dependency never reaches a consumer's build, so it is not
    // part of what the consumer redistributes.
    withDirectory((root) => {
      const evidence = inventory(graph(root), LOCK, { features: "standalone" });
      assert.deepEqual(
        evidence.packages.map((e) => e.name),
        ["built", "normal"],
      );
      const normal = evidence.packages[1];
      assert.equal(normal.checksum, "aa");
      assert.deepEqual(normal.attribution_files, ["LICENSE-MIT", "NOTICE"]);
      assert.deepEqual(evidence.context, { features: "standalone" });
    });
  });

  test("a package without a license is refused", () => {
    withDirectory((root) => {
      assert.throws(() => inventory(graph(root, null), LOCK, {}), /normal 1\.0\.0/);
    });
  });

  test("the SBOM names each component with its license and hash", () => {
    withDirectory((root) => {
      const out = join(root, "out");
      mkdirSync(out);
      writeEvidence(out, "stem", graph(root), LOCK, { target: "x" });
      const sbom = JSON.parse(readFileSync(join(out, "stem-sbom.cdx.json"), "utf8"));
      const licenses = JSON.parse(readFileSync(join(out, "stem-licenses.json"), "utf8"));
      assert.equal(sbom.bomFormat, "CycloneDX");
      assert.equal(sbom.specVersion, "1.5");
      const normal = sbom.components.find((c: { name: string }) => c.name === "normal");
      assert.equal(normal.purl, "pkg:cargo/normal@1.0.0");
      assert.deepEqual(normal.licenses, [{ expression: "MIT" }]);
      assert.deepEqual(normal.hashes, [{ alg: "SHA-256", content: "aa" }]);
      assert.deepEqual(sbom.metadata.properties, [{ name: "coordinode:target", value: "x" }]);
      assert.equal(licenses.packages.length, 2);
    });
  });
});
