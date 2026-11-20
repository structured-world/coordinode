// A partial publication is resumable only with identical artifacts.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { checkRegistered, checkVersions, mergedReleaseRevision, publicationRevision } from "./registry.ts";

function withDirectory(body: (directory: string) => void): void {
  const directory = mkdtempSync(join(tmpdir(), "coordinode-release-test-"));
  try {
    body(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

describe("release retry", () => {
  const pull = () => ({
    merged_at: "2026-01-01T00:00:00Z" as string | null,
    base: { ref: "main" },
    head: { ref: "release-plz-test" },
    merge_commit_sha: "a1".repeat(20) as unknown,
  });

  test("a retry selects the recorded merge commit", () => {
    // A retry must retain the approved release instead of moving with main.
    assert.equal(mergedReleaseRevision(pull()), "a1".repeat(20));
  });

  test("a retry refuses unmerged or unrelated pull requests", () => {
    // Dispatch cannot turn an ordinary or unapproved PR into a release.
    for (const variant of [
      { ...pull(), merged_at: null },
      { ...pull(), base: { ref: "development" } },
      { ...pull(), head: { ref: "feature/test" } },
    ]) {
      assert.throws(() => mergedReleaseRevision(variant), /merged release PR/);
    }
  });

  test("a retry refuses a moving ref in place of the merge commit", () => {
    // Only the forge's immutable commit identifier can reach checkout.
    assert.throws(() => mergedReleaseRevision({ ...pull(), merge_commit_sha: "refs/heads/main" }), /commit identifier/);
  });
});

describe("publication revision", () => {
  // Qualify the commit release-plz packages, not an ancestor with another tree.
  const git = (root: string, ...args: string[]) =>
    execFileSync("git", ["-c", "user.name=release-test", "-c", "user.email=release-test@example.org", ...args], {
      cwd: root,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();

  test("an ancestor release head requires the same verified tree", () => {
    withDirectory((root) => {
      git(root, "init", "--quiet");
      writeFileSync(join(root, "input"), "qualified");
      git(root, "add", "input");
      git(root, "commit", "--quiet", "-m", "release input");
      const release = git(root, "rev-parse", "HEAD");
      git(root, "commit", "--quiet", "--allow-empty", "-m", "same tree");
      assert.equal(publicationRevision(root, git(root, "rev-parse", "HEAD"), release), release);
      // A changed merge result cannot borrow an ancestor's source-CI pass.
      writeFileSync(join(root, "input"), "different");
      git(root, "add", "input");
      git(root, "commit", "--quiet", "-m", "different tree");
      assert.throws(() => publicationRevision(root, git(root, "rev-parse", "HEAD"), release), /different tree/);
    });
  });

  test("a squashed release keeps the verified source", () => {
    withDirectory((root) => {
      git(root, "init", "--quiet");
      git(root, "commit", "--quiet", "--allow-empty", "-m", "squashed release");
      const source = git(root, "rev-parse", "HEAD");
      // The original PR commit is absent after squash and branch deletion.
      assert.equal(publicationRevision(root, source, "1".repeat(40)), source);
    });
  });
});

describe("registry", () => {
  const report = () => ({
    version: "1.0.0",
    source_commit: "",
    packages: {
      "coordinode-core": { archive: { sha256: "first" } },
      "coordinode-embed": { archive: { sha256: "second" } },
    },
  });

  test("a partial release can resume but is not complete", async () => {
    // A successful first upload does not certify the missing dependent.
    const read = async (path: string) => (path.includes("core/") ? { version: { checksum: "first", yanked: false } } : null);
    await checkVersions(report(), read, true);
    await assert.rejects(checkVersions(report(), read, false), /not published/);
  });

  test("a changed archive cannot continue a partial release", async () => {
    // Never silently accept a version uploaded from another source or lock.
    await assert.rejects(
      checkVersions(report(), async () => ({ version: { checksum: "other", yanked: false } }), true),
      /qualified archive/,
    );
  });

  test("a yanked archive is not a qualified delivery", async () => {
    await assert.rejects(
      checkVersions(report(), async () => ({ version: { checksum: "first", yanked: true } }), true),
      /qualified archive/,
    );
  });

  test("a crate absent from the registry stops before any upload", async () => {
    // A trusted-publishing token cannot create a crate, so a new package in
    // the set would stop the release half-published at its own upload.
    // Preflight names it before the first upload instead.
    const read = async (path: string) => (path === "crates/coordinode-embed" ? null : { crate: {} });
    await assert.rejects(checkRegistered(report(), read), /coordinode-embed\b.*first version/);
    await checkRegistered(report(), async () => ({ crate: {} }));
  });

  test("publisher provisioning is not a release operation", () => {
    // CI consumes an existing publisher and cannot register one.
    const result = spawnSync(process.execPath, [join(import.meta.dirname, "registry.ts"), "trust", "--checkout", "."], {
      encoding: "utf8",
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /mode must be one of/);
  });
});
