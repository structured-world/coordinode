"""Regression tests for optional closure and source-only dependency removal."""
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import tomlkit

from prepare import library_closure, main, prepare_manifest


class PreparationTest(unittest.TestCase):
    def test_optional_target_runtime_is_kept_but_dev_cycles_are_not(self):
        # Replication is opt-in, not outside the deliverable. An embed -> query
        # -> embed dev cycle and private test fixtures must not become releases.
        packages = [{"id": n, "name": n, "source": None} for n in ("embed", "query", "raft", "fixtures")]
        edge = lambda n, k: {"pkg": n, "dep_kinds": [{"kind": k, "target": None}]}
        metadata = {"packages": packages, "workspace_members": [p["id"] for p in packages],
                    "resolve": {"nodes": [
                        {"id": "embed", "deps": [edge("query", None), edge("raft", None)]},
                        {"id": "query", "deps": [edge("embed", "dev"), edge("fixtures", "dev")]},
                        {"id": "raft", "deps": []}, {"id": "fixtures", "deps": []}]}}
        internal, external = library_closure(metadata, "embed")
        self.assertEqual(internal, {"embed", "query", "raft"})
        self.assertEqual(external, [])

    def test_prepared_codegen_has_no_generator_or_test_dependency(self):
        # A registry consumer must not need the proto submodule or resolve
        # unpublished fixtures, including platform-specific dev dependencies.
        original = tomlkit.parse('''[package]
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
''')
        result = prepare_manifest(original, "1.0.0", {"coordinode-core"}, True)
        self.assertNotIn("build-dependencies", result)
        self.assertNotIn("dev-dependencies", result)
        self.assertNotIn("bench", result)
        self.assertNotIn("dev-dependencies", result["target"]["cfg(windows)"])
        self.assertEqual(result["dependencies"], original["dependencies"])
        self.assertIn("build-dependencies", original)
        self.assertIn("/bindings/**", result["package"]["include"])

    def test_workspace_hack_cannot_silently_ship(self):
        # Failing preparation is safer than publishing the feature-unification
        # helper and its benchmark/test dependency union to every consumer.
        doc = tomlkit.parse('[package]\nname="x"\n[dependencies]\ncoordinode-workspace-hack={path="../hack"}\n')
        with self.assertRaisesRegex(ValueError, "hakari"):
            prepare_manifest(doc, "1.0.0", {"x"}, False)

    def run_changed_checkout(self, *, source_dirty=False):
        # Changing proto contents without moving HEAD must invalidate generated
        # bindings. A dirty source checkout likewise cannot claim an exact
        # source revision. Simulate both around the real preparation entrypoint.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / '.config').mkdir()
            (root / 'proto').mkdir()
            crate = root / 'crates/coordinode-embed'
            crate.mkdir(parents=True)
            (crate / 'Cargo.toml').write_text('[package]\nname="coordinode-embed"\nversion="1.0.0"\n')
            (root / 'Cargo.toml').write_text('[workspace.package]\nversion="1.0.0"\n')
            (root / '.config/registry-packages.toml').write_text(
                'entrypoint="coordinode-embed"\npackages=["coordinode-embed"]\n[bindings]\n')
            for name in ('LICENSE', 'NOTICES', 'COPYRIGHT'):
                (root / name).write_text('fixture')
            metadata = {'packages': [{'id': 'embed', 'name': 'coordinode-embed',
                                     'manifest_path': str(crate / 'Cargo.toml')}],
                        'workspace_members': ['embed'],
                        'resolve': {'nodes': [{'id': 'embed', 'deps': []}]}}
            proto_status = iter(['', ' M replication.proto\n'])

            def output(*args, root):
                if args == ('git', 'rev-parse', 'HEAD'):
                    return 'a' * 40
                if args == ('git', 'status', '--porcelain'):
                    if root.name == 'proto':
                        return next(proto_status)
                    return ' M src/lib.rs\n' if source_dirty else ''
                if args[:2] == ('cargo', 'metadata'):
                    return json.dumps(metadata)
                raise AssertionError(args)

            with patch('sys.argv', ['prepare.py', '--checkout', str(root)]), \
                 patch('prepare.command', side_effect=output), \
                 patch('prepare.subprocess.run'), patch('prepare.subprocess.Popen') as build:
                build.return_value.stdout = []
                build.return_value.wait.return_value = 0
                main()

    def test_proto_edits_during_generation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'proto checkout changed'):
            self.run_changed_checkout()

    def test_editor_changes_cannot_be_published_as_the_committed_source(self):
        with self.assertRaisesRegex(ValueError, 'source checkout must be clean'):
            self.run_changed_checkout(source_dirty=True)


if __name__ == "__main__":
    unittest.main()
