// The release PR carries the builds outside the workspace up to the release.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, test } from "node:test";
import { admits } from "./check_packages.ts";
import { refreshLock, releaseBranch, requirementFor, updateManifest } from "./example_requirements.ts";

const MANIFEST =
  '[package]\nname = "quickstart"\n\n[workspace]\n\n' +
  '[dependencies]\ncoordinode-embed = "0.9"\ncoordinode-core = "0.9"\ntempfile = "3"\n';
const NAMES = new Set(["coordinode-embed", "coordinode-core"]);

describe("requirement", () => {
  test("the requirement admits the release and no breaking one", () => {
    for (const [version, wanted] of [
      ["0.10.0", "0.10"],
      ["0.9.3", "0.9"],
      ["1.4.2", "1.4"],
      ["0.11.0-rc.1", "0.11"],
    ]) {
      assert.equal(requirementFor(version), wanted);
      assert.equal(admits(wanted, version), true);
    }
  });

  test("registry requirements move and nothing else changes", () => {
    // The example the release qualifies is the one in the PR: after the update
    // the release's own check admits the version.
    const updated = updateManifest(MANIFEST, NAMES, "0.10.0");
    assert.equal(updated, MANIFEST.replaceAll('"0.9"', '"0.10"'));
    assert.ok(updated.includes('tempfile = "3"'));
  });

  test("an example already on the release is left as it is", () => {
    const current = MANIFEST.replaceAll('"0.9"', '"0.10"');
    assert.equal(updateManifest(current, NAMES, "0.10.4"), current);
  });

  test("a requirement in table form is refused, not guessed", () => {
    const manifest = '[dependencies]\ncoordinode-embed = { version = "0.9" }\n';
    assert.throws(() => updateManifest(manifest, NAMES, "0.10.0"), /plain version requirement/);
  });
});

describe("path lock", () => {
  // A standalone workspace locked on a path crate builds `--locked` again once
  // the release bumped that crate.
  const write = (root: string, path: string, text: string) => {
    mkdirSync(dirname(join(root, path)), { recursive: true });
    writeFileSync(join(root, path), text);
  };
  const lockedBuild = (root: string) =>
    spawnSync(
      "cargo",
      ["tree", "--quiet", "--locked", "--offline", "--depth", "0", "--manifest-path", join(root, "bench/Cargo.toml")],
      { encoding: "utf8" },
    ).status;

  test("a bumped path crate is recorded in the lock", () => {
    const root = mkdtempSync(join(tmpdir(), "coordinode-release-test-"));
    try {
      const crate = (version: string) => `[package]\nname = "lib"\nversion = "${version}"\nedition = "2024"\n`;
      write(root, "lib/Cargo.toml", crate("0.1.0"));
      write(root, "lib/src/lib.rs", "");
      write(
        root,
        "bench/Cargo.toml",
        '[workspace]\n\n[package]\nname = "bench"\nversion = "0.1.0"\n' +
          'edition = "2024"\n\n[dependencies]\nlib = { path = "../lib" }\n',
      );
      write(root, "bench/src/main.rs", "fn main() {}\n");
      // The committed lock, on the version before the release.
      execFileSync("cargo", ["generate-lockfile", "--offline", "--manifest-path", join(root, "bench/Cargo.toml")]);
      assert.equal(lockedBuild(root), 0);
      assert.equal(refreshLock(root, "bench"), false, "a lock that matches stays");

      // The release bumps the crate: the committed lock no longer builds.
      write(root, "lib/Cargo.toml", crate("0.2.0"));
      assert.notEqual(lockedBuild(root), 0);

      assert.equal(refreshLock(root, "bench"), true);
      assert.ok(readFileSync(join(root, "bench/Cargo.lock"), "utf8").includes('name = "lib"\nversion = "0.2.0"'));
      assert.equal(lockedBuild(root), 0);
      assert.equal(refreshLock(root, "bench"), false, "nothing left to bring");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("release PR", () => {
  test("the branch comes from release-plz output", () => {
    const output = '{"prs":[{"head_branch":"release-plz-2026-10-25","number":80}]}';
    assert.equal(releaseBranch(output), "release-plz-2026-10-25");
  });

  test("no release PR means nothing to update", () => {
    assert.equal(releaseBranch('{"prs":[]}'), null);
    assert.equal(releaseBranch(""), null);
  });
});
