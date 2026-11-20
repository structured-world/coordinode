// Archive gates reject source-only inputs before a consumer can hide them.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { admits, checkQuickstartRequirements, inspectArchive, main, QUICKSTART } from "./check_packages.ts";
import { writeTarGz } from "./tar.ts";

function withDirectory<T>(body: (directory: string) => T): T {
  const directory = mkdtempSync(join(tmpdir(), "coordinode-release-test-"));
  try {
    return body(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function baseFiles(manifest: string): Record<string, string> {
  const files: Record<string, string> = Object.fromEntries(
    ["src/lib.rs", "LICENSE", "NOTICES", "COPYRIGHT", "README.md"].map((n) => [n, "fixture"]),
  );
  files["Cargo.toml"] = manifest;
  return files;
}

function inspect(
  dependency = "",
  { omitted, extra = {}, bindings = [] }: { omitted?: string; extra?: Record<string, string>; bindings?: string[] } = {},
) {
  // The fixture is an actual gzip tar archive, not a mock of the reader. This
  // exercises normalization checks independently of Cargo's source workspace,
  // where path dependencies would otherwise hide mistakes.
  const files = baseFiles('[package]\nname="coordinode-embed"\nversion="1.0.0"\n' + dependency);
  if (omitted !== undefined) delete files[omitted];
  Object.assign(files, extra);
  return withDirectory((directory) => {
    const path = join(directory, "package.crate");
    const entries = Object.fromEntries(Object.entries(files).map(([n, v]) => [`coordinode-embed-1.0.0/${n}`, v]));
    writeFileSync(path, writeTarGz(entries));
    return inspectArchive(path, "coordinode-embed", "1.0.0", new Set(["coordinode-embed", "coordinode-core"]), bindings);
  });
}

describe("archive", () => {
  test("an exact registry edge is accepted", () => {
    const result = inspect('[dependencies.coordinode-core]\nversion="=1.0.0"\n');
    assert.equal(result.files, 6);
    assert.equal(result.sha256.length, 64);
  });

  test("a dual-licensed provider must ship both license texts", () => {
    // Package preparation must not replace the provider's upstream license
    // grant with the engine's license or omit either original text.
    assert.throws(() => inspect('license="MIT OR Apache-2.0"\n', { extra: { "LICENSE-MIT": "MIT" } }), /LICENSE-APACHE/);
    inspect('license="MIT OR Apache-2.0"\n', { extra: { "LICENSE-MIT": "MIT", "LICENSE-APACHE": "Apache" } });
  });

  test("a path edge cannot masquerade as a registry package", () => {
    assert.throws(() => inspect('[dependencies.coordinode-core]\nversion="=1.0.0"\npath="../core"\n'), /source dependency/);
  });

  test("target-specific dev dependencies are not shipped", () => {
    assert.throws(
      () => inspect("[target.'cfg(windows)'.dev-dependencies.fixture]\nversion=\"1\"\n"),
      /development dependencies/,
    );
  });

  test("a shared type dependency requires the same release", () => {
    assert.throws(() => inspect('[dependencies.coordinode-core]\nversion="1.0"\n'), /coordinated version/);
  });

  test("the attribution file must be in each archive", () => {
    assert.throws(() => inspect("", { omitted: "NOTICES" }), /NOTICES/);
  });

  test("the binding list cannot omit a generated input", () => {
    assert.throws(
      () =>
        inspect("", {
          extra: { "build.rs": "fn main() {}", "bindings/a.rs": "", "bindings/files.txt": "wrong.rs\n" },
          bindings: ["a.rs"],
        }),
      /inventory differs/,
    );
  });

  test("a file the sources embed must be in the archive", () => {
    // A source that embeds a data file compiles in the workspace but not from
    // an archive that left the file out, and only with the feature that
    // reaches that source: the consumer build can miss it.
    const tokenizer = 'const DICT: &[u8] = include_bytes!("../../assets/dict.zst");\n';
    assert.throws(() => inspect("", { extra: { "src/cjk/tokenizers.rs": tokenizer } }), /assets\/dict\.zst/);
    inspect("", { extra: { "src/cjk/tokenizers.rs": tokenizer, "assets/dict.zst": "z" } });
    // Shared documentation a source-workspace doctest embeds from outside the
    // package is not part of it.
    inspect("", { extra: { "src/guide.rs": '#[doc = include_str!("../../docs/guide.md")]\n' } });
    // A path built at compile time names no shipped file.
    inspect("", { extra: { "src/gen.rs": 'include!(concat!(env!("OUT_DIR"), "/a.rs"));\n' } });
  });

  test("archive paths cannot escape the package", () => {
    assert.throws(() => inspect("", { extra: { "../outside": "not allowed" } }), /invalid archive path/);
  });
});

describe("quickstart requirement", () => {
  test("caret requirements admit only compatible releases", () => {
    assert.equal(admits("0.9", "0.9.0"), true);
    assert.equal(admits("0.9", "0.9.4"), true);
    assert.equal(admits("0.9", "0.10.0"), false);
    assert.equal(admits("0.9.2", "0.9.1"), false);
    assert.equal(admits("1", "1.4.0"), true);
    assert.equal(admits("1.2", "2.0.0"), false);
  });

  test("an example left behind the release is refused", () => {
    // The published example is the documented consumer entry point: a release
    // its requirement does not admit would publish an example that builds
    // against the previous release, or not at all.
    withDirectory((root) => {
      mkdirSync(join(root, QUICKSTART), { recursive: true });
      writeFileSync(
        join(root, QUICKSTART, "Cargo.toml"),
        '[package]\nname="quickstart"\n[dependencies]\ncoordinode-embed="0.9"\ntempfile="3"\n',
      );
      checkQuickstartRequirements(root, new Set(["coordinode-embed"]), "0.9.3");
      assert.throws(
        () => checkQuickstartRequirements(root, new Set(["coordinode-embed"]), "0.10.0"),
        /does not admit 0\.10\.0/,
      );
    });
  });
});

describe("checker startup", () => {
  test("the registry checker starts without the preparation environment", () => {
    // Upload verification runs after the disposable preparation environment
    // is gone: the checker starts from its own directory alone.
    const result = spawnSync(process.execPath, [join(import.meta.dirname, "check_packages.ts"), "--help"], {
      encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /--source/);
  });
});

describe("fresh resolution", () => {
  test("a new native dependency is rejected before consumer compilation", () => {
    // The source lock can be pure Rust while the published version range
    // permits a new native build helper. Reject that fresh graph before
    // compiling it, and retain the graph explaining the refusal.
    withDirectory((root) => {
      mkdirSync(join(root, ".config"));
      writeFileSync(join(root, ".config/registry-packages.toml"), 'packages=["coordinode-embed"]\nbindings={}\n');
      mkdirSync(join(root, "target/package"), { recursive: true });
      writeFileSync(
        join(root, "target/registry-preparation.json"),
        JSON.stringify({ version: "1.0.0", packages: { "coordinode-embed": {} } }),
      );
      writeFileSync(join(root, "Cargo.lock"), "# Qualified source lock\n");
      mkdirSync(join(root, "scripts/release"), { recursive: true });
      writeFileSync(join(root, "scripts/release/consumer.rs"), "fn main() {}\n");
      mkdirSync(join(root, QUICKSTART, "src"), { recursive: true });
      writeFileSync(join(root, QUICKSTART, "Cargo.toml"), '[package]\nname="quickstart"\n[dependencies]\ncoordinode-embed="1.0"\n');
      writeFileSync(join(root, QUICKSTART, "src/main.rs"), "fn main() {}\n");
      const files = Object.fromEntries(
        Object.entries(baseFiles('[package]\nname="coordinode-embed"\nversion="1.0.0"\n')).map(([n, v]) => [
          `coordinode-embed-1.0.0/${n}`,
          v,
        ]),
      );
      writeFileSync(join(root, "target/package/coordinode-embed-1.0.0.crate"), writeTarGz(files));
      const graph = JSON.stringify({
        packages: [
          { name: "pest", version: "2.9.4" },
          { name: "psm", version: "0.1.24" },
        ],
      });
      for (const source of ["prepared", "crates-io"]) {
        const runs: string[][] = [];
        assert.throws(
          () =>
            main(
              ["--checkout", root, "--source", source],
              (args) => {
                runs.push(args);
              },
              () => graph,
            ),
          /native stack helper/,
        );
        const builds = runs.filter((args) => args.includes("cargo") && args.includes("run"));
        assert.deepEqual(builds, [], "reject the native graph before cargo run");
        const filename = source === "prepared" ? "fresh-dependency-metadata.json" : "crates-io-consumer-metadata.json";
        assert.deepEqual(JSON.parse(readFileSync(join(root, "target", filename), "utf8")), JSON.parse(graph));
      }
    });
  });
});
