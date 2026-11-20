// Regression tests for optional closure and source-only dependency removal.
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { parse } from "smol-toml";
import { libraryClosure, main, prepareManifest } from "./prepare.ts";

type Table = Record<string, unknown>;

describe("preparation", () => {
  test("an optional target runtime is kept but dev cycles are not", () => {
    // Replication is opt-in, not outside the deliverable. An embed -> query ->
    // embed dev cycle and private test fixtures must not become releases.
    const packages = ["embed", "query", "raft", "fixtures"].map((n) => ({ id: n, name: n, source: null }));
    const edge = (n: string, k: string | null) => ({ pkg: n, dep_kinds: [{ kind: k }] });
    const metadata = {
      packages,
      workspace_members: packages.map((p) => p.id),
      resolve: {
        nodes: [
          { id: "embed", deps: [edge("query", null), edge("raft", null)] },
          { id: "query", deps: [edge("embed", "dev"), edge("fixtures", "dev")] },
          { id: "raft", deps: [] },
          { id: "fixtures", deps: [] },
        ],
      },
    };
    const [internal, external] = libraryClosure(metadata, "embed");
    assert.deepEqual(internal, new Set(["embed", "query", "raft"]));
    assert.deepEqual(external, []);
  });

  test("prepared codegen has no generator or test dependency", () => {
    // A registry consumer must not need the proto submodule or resolve
    // unpublished fixtures, including platform-specific dev dependencies.
    const original = parse(`[package]
name="coordinode-raft"
[dependencies]
coordinode-core={workspace=true}
[build-dependencies]
tonic-prost-build="0.14"
[dev-dependencies]
coordinode-test-fixtures={path="../fixtures"}
[target.'cfg(windows)'.dev-dependencies]
coordinode-test-fixtures={path="../fixtures"}
[[bench]]
name="source-only"
`) as Table;
    const result = prepareManifest(original, "1.0.0", new Set(["coordinode-core"]), true);
    assert.equal("build-dependencies" in result, false);
    assert.equal("dev-dependencies" in result, false);
    assert.equal("bench" in result, false);
    assert.equal("dev-dependencies" in ((result.target as Table)["cfg(windows)"] as Table), false);
    // Plain objects: the TOML reader builds null-prototype tables.
    assert.deepEqual(JSON.parse(JSON.stringify(result.dependencies)), JSON.parse(JSON.stringify(original.dependencies)));
    assert.equal("build-dependencies" in original, true);
    const include = (result.package as Table).include as string[];
    assert.ok(include.includes("/bindings/**"));
    assert.ok(include.includes("/LICENSE-*"));
    // Data files the sources embed (the CJK dictionary) ship with them.
    assert.ok(include.includes("/assets/**"));
  });

  test("a feature stops naming a removed build dependency", () => {
    // A feature that enabled the generator would name a dependency the
    // prepared manifest no longer has, and cargo refuses to parse it.
    const original = parse(`[package]
name="coordinode-replicate"
[features]
default=["transfer"]
transfer=["dep:tonic", "dep:tonic-prost-build"]
[dependencies]
tonic={version="0.14", optional=true}
[build-dependencies]
tonic-prost-build={version="0.14", optional=true}
`) as Table;
    const result = prepareManifest(original, "1.0.0", new Set(), true);
    assert.equal("build-dependencies" in result, false);
    assert.deepEqual((result.features as Table).transfer, ["dep:tonic"]);
    assert.deepEqual((result.features as Table).default, ["transfer"]);
  });

  test("the workspace hack cannot silently ship", () => {
    // Failing preparation is safer than publishing the feature-unification
    // helper and its benchmark/test dependency union to every consumer.
    const doc = parse('[package]\nname="x"\n[dependencies]\ncoordinode-workspace-hack={path="../hack"}\n') as Table;
    assert.throws(() => prepareManifest(doc, "1.0.0", new Set(["x"]), false), /hakari/);
  });

  // Changing proto contents without moving HEAD must invalidate generated
  // bindings. A dirty source checkout likewise cannot claim an exact source
  // revision. Simulate both around the real preparation entrypoint.
  function runChangedCheckout({ sourceDirty = false, nativeStackHelper }: { sourceDirty?: boolean; nativeStackHelper?: string } = {}) {
    const root = mkdtempSync(join(tmpdir(), "coordinode-release-test-"));
    try {
      mkdirSync(join(root, ".config"));
      mkdirSync(join(root, "proto"));
      const crate = join(root, "crates/coordinode-embed");
      mkdirSync(crate, { recursive: true });
      writeFileSync(join(crate, "Cargo.toml"), '[package]\nname="coordinode-embed"\nversion="1.0.0"\n');
      writeFileSync(join(root, "Cargo.toml"), '[workspace.package]\nversion="1.0.0"\n');
      writeFileSync(
        join(root, ".config/registry-packages.toml"),
        'entrypoint="coordinode-embed"\npackages=["coordinode-embed"]\n[bindings]\n',
      );
      for (const name of ["LICENSE", "NOTICES", "COPYRIGHT"]) writeFileSync(join(root, name), "fixture");
      const metadata = {
        packages: [{ id: "embed", name: "coordinode-embed", manifest_path: join(crate, "Cargo.toml") }] as Table[],
        workspace_members: ["embed"],
        resolve: { nodes: [{ id: "embed", deps: [] as Table[] }] as Table[] },
      };
      if (nativeStackHelper !== undefined) {
        metadata.packages.push({ id: "native-helper", name: nativeStackHelper, version: "1.0.0", source: "registry+fixture" });
        (metadata.resolve.nodes[0].deps as Table[]).push({ pkg: "native-helper", dep_kinds: [{ kind: null }] });
        metadata.resolve.nodes.push({ id: "native-helper", deps: [] });
      }
      const protoStatus = ["", " M replication.proto\n"];
      main(
        ["--checkout", root],
        (args, cwd) => {
          if (args.join(" ") === "git rev-parse HEAD") return "a".repeat(40);
          if (args.join(" ") === "git status --porcelain") {
            if (cwd.endsWith("proto")) return protoStatus.shift() as string;
            return sourceDirty ? " M src/lib.rs\n" : "";
          }
          if (args[0] === "cargo" && args[1] === "metadata") return JSON.stringify(metadata);
          throw new Error(`unexpected command ${args.join(" ")}`);
        },
        () => {},
        () => ({ status: 0, stdout: "" }),
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }

  test("proto edits during generation are rejected", () => {
    assert.throws(() => runChangedCheckout(), /proto checkout changed/);
  });

  test("editor changes cannot be published as the committed source", () => {
    assert.throws(() => runChangedCheckout({ sourceDirty: true }), /source checkout must be clean/);
  });

  test("native stack helpers cannot reenter the registry closure", () => {
    // Exercise release admission, not just a name-filter helper. Either
    // stack-switching package must stop preparation before building bindings.
    for (const name of ["stacker", "psm"]) {
      assert.throws(() => runChangedCheckout({ nativeStackHelper: name }), /native stack helper/);
    }
  });
});
