// A server tag cannot establish a library's first registry baseline.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { parse } from "smol-toml";
import { main, type ReleaseConfig, releasePrConfig } from "./pr_config.ts";

const SOURCE = `[workspace]
publish = false
git_only = true
git_tag_name = "v{{ version }}"
[[package]]
name = "server"
version_group = "shared"
git_tag_enable = true
[[package]]
name = "first"
version_group = "shared"
publish = true
git_only = false
semver_check = true
[[package]]
name = "second"
version_group = "shared"
publish = true
git_only = false
semver_check = true
`;

// Plain objects: the TOML reader builds null-prototype tables.
const plain = <T>(value: T): T => JSON.parse(JSON.stringify(value));
const original = () => plain(parse(SOURCE)) as unknown as ReleaseConfig;
const names = ["first", "second"];

describe("release-PR configuration", () => {
  test("a first publication uses Git only for the calculation", async () => {
    // Keep the coordinated tag and version group, but never use this derived
    // configuration to perform the first upload.
    const source = original();
    const result = await releasePrConfig(source, names, async () => null);
    assert.deepEqual(source, original());
    assert.deepEqual(result.workspace, source.workspace);
    assert.deepEqual(result.package[0], source.package[0]);
    for (const pkg of result.package.slice(1)) {
      assert.equal(pkg.git_only, true);
      assert.equal(pkg.publish, false);
      assert.equal(pkg.semver_check, false);
      assert.equal(pkg.version_group, "shared");
    }
  });

  test("published packages keep the exact original configuration", async () => {
    // Never replace a published API baseline with a Git-only comparison.
    const result = await releasePrConfig(original(), names, async (path) => ({ crate: { id: path.split("/").pop() } }));
    assert.deepEqual(result, original());
  });

  test("a partial inventory preserves existing registry checks", async () => {
    const result = await releasePrConfig(original(), names, async (path) =>
      path === "crates/first" ? { crate: { id: "first" } } : null,
    );
    assert.deepEqual(result.package[1], original().package[1]);
    assert.equal(result.package[2].git_only, true);
    assert.equal(result.package[2].publish, false);
  });

  test("a registry failure is not an initial release", async () => {
    const source = original();
    await assert.rejects(
      releasePrConfig(source, names, async () => {
        throw new Error("registry unavailable");
      }),
      /registry unavailable/,
    );
    assert.deepEqual(source, original());
  });

  test("a malformed response is not an initial release", async () => {
    for (const response of [{}, [], { crate: { id: "another-package" } }]) {
      await assert.rejects(releasePrConfig(original(), names, async () => response), /registry response/);
    }
  });

  test("configuration and inventory must match before network work", async () => {
    for (const subset of [[], ["first"], ["first", "first"], ["first", "unknown"]]) {
      await assert.rejects(
        releasePrConfig(original(), subset, async () => assert.fail("unexpected request")),
        /inventory/,
      );
    }
  });

  test("the CLI writes an ignored configuration without changing source", async () => {
    // An ignored output keeps release-plz's clean-source check effective.
    const root = mkdtempSync(join(tmpdir(), "coordinode-release-test-"));
    try {
      execFileSync("git", ["init", "--quiet", root]);
      writeFileSync(join(root, ".gitignore"), "/target/\n");
      writeFileSync(join(root, ".release-plz.toml"), SOURCE);
      mkdirSync(join(root, ".config"));
      writeFileSync(join(root, ".config/registry-packages.toml"), 'packages = ["first", "second"]\n');
      const output = join(root, "target/release-pr-config.toml");
      // A missing first crate followed by a failed read of the second must not
      // emit a partial configuration that a later step can use.
      const replies: (null | Error)[] = [null, new Error("registry unavailable")];
      await assert.rejects(
        main(["--checkout", root], async () => {
          const reply = replies.shift();
          if (reply instanceof Error) throw reply;
          return reply ?? null;
        }),
        /registry unavailable/,
      );
      assert.equal(existsSync(output), false);
      assert.equal(readFileSync(join(root, ".release-plz.toml"), "utf8"), SOURCE);
      await main(["--checkout", root], async () => null);
      assert.ok(existsSync(output));
      execFileSync("git", ["check-ignore", "--quiet", output], { cwd: root });
      assert.equal(readFileSync(join(root, ".release-plz.toml"), "utf8"), SOURCE);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
