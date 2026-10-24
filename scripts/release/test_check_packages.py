"""Archive gates reject source-only inputs before a consumer can hide them."""
import io
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import check_packages
from check_packages import inspect_archive


class ArchiveTest(unittest.TestCase):
    def inspect(self, dependency='', *, omitted=None, extra=None, bindings=()):
        # The fixture is an actual gzip tar archive, not a mock of tarfile.
        # This exercises normalization checks independently of Cargo's source
        # workspace, where path dependencies would otherwise hide mistakes.
        files = {name: b'fixture' for name in
                 ('src/lib.rs', 'LICENSE', 'NOTICES', 'COPYRIGHT', 'README.md')}
        files['Cargo.toml'] = (
            '[package]\nname="coordinode-embed"\nversion="1.0.0"\n' + dependency
        ).encode()
        if omitted is not None:
            del files[omitted]
        files.update(extra or {})
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / 'package.crate'
            with tarfile.open(path, 'w:gz') as archive:
                for name, value in files.items():
                    entry = tarfile.TarInfo('coordinode-embed-1.0.0/' + name)
                    entry.size = len(value)
                    archive.addfile(entry, io.BytesIO(value))
            return inspect_archive(path, 'coordinode-embed', '1.0.0',
                                   {'coordinode-embed', 'coordinode-core'}, list(bindings))

    def test_exact_registry_edge_is_accepted(self):
        result = self.inspect('[dependencies.coordinode-core]\nversion="=1.0.0"\n')
        self.assertEqual(result['files'], 6)
        self.assertEqual(len(result['sha256']), 64)

    def test_path_edge_cannot_masquerade_as_a_registry_package(self):
        with self.assertRaisesRegex(ValueError, 'source dependency'):
            self.inspect('[dependencies.coordinode-core]\nversion="=1.0.0"\npath="../core"\n')

    def test_target_specific_dev_dependencies_are_not_shipped(self):
        with self.assertRaisesRegex(ValueError, 'development dependencies'):
            self.inspect('[target.\'cfg(windows)\'.dev-dependencies.fixture]\nversion="1"\n')

    def test_shared_type_dependency_requires_the_same_release(self):
        with self.assertRaisesRegex(ValueError, 'coordinated version'):
            self.inspect('[dependencies.coordinode-core]\nversion="1.0"\n')

    def test_attribution_file_must_be_in_each_archive(self):
        with self.assertRaisesRegex(ValueError, 'NOTICES'):
            self.inspect(omitted='NOTICES')

    def test_binding_list_cannot_omit_a_generated_input(self):
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            self.inspect(extra={'build.rs': b'fn main() {}', 'bindings/a.rs': b'',
                                'bindings/files.txt': b'wrong.rs\n'}, bindings=['a.rs'])

    def test_archive_paths_cannot_escape_the_package(self):
        with self.assertRaisesRegex(ValueError, 'invalid archive path'):
            self.inspect(extra={'../outside': b'not allowed'})


class CheckerStartupTest(unittest.TestCase):
    def test_registry_checker_does_not_require_the_preparation_environment(self):
        # Upload verification runs after the disposable preparation environment
        # is gone. It must start with the standard library alone.
        script = pathlib.Path(__file__).with_name('check_packages.py')
        result = subprocess.run([sys.executable, '-B', '-S', str(script), '--help'],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--source', result.stdout)


class FreshResolutionTest(unittest.TestCase):
    def test_new_native_dependency_is_rejected_before_consumer_compilation(self):
        # The source lock can be pure Rust while the published version range
        # permits a new native build helper. Reject that fresh graph before
        # compiling it, and retain the graph explaining the refusal.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / '.config').mkdir()
            (root / '.config/registry-packages.toml').write_text(
                'packages=["coordinode-embed"]\nbindings={}\n')
            (root / 'target/package').mkdir(parents=True)
            (root / 'target/registry-preparation.json').write_text(json.dumps(
                {'version': '1.0.0', 'packages': {'coordinode-embed': {}}}))
            (root / 'Cargo.lock').write_text('# Qualified source lock\n')
            (root / 'scripts/release').mkdir(parents=True)
            (root / 'scripts/release/consumer.rs').write_text('fn main() {}\n')
            archive = root / 'target/package/coordinode-embed-1.0.0.crate'
            files = {name: b'fixture' for name in
                     ('src/lib.rs', 'LICENSE', 'NOTICES', 'COPYRIGHT', 'README.md')}
            files['Cargo.toml'] = b'[package]\nname="coordinode-embed"\nversion="1.0.0"\n'
            with tarfile.open(archive, 'w:gz') as package:
                for name, data in files.items():
                    member = tarfile.TarInfo('coordinode-embed-1.0.0/' + name)
                    member.size = len(data)
                    package.addfile(member, io.BytesIO(data))
            graph = json.dumps({'packages': [
                {'name': 'pest', 'version': '2.9.4'},
                {'name': 'psm', 'version': '0.1.24'},
            ]})
            for source in ('prepared', 'crates-io'):
                with self.subTest(source=source):
                    argv = ['check_packages.py', '--checkout', str(root), '--source', source]
                    with patch.object(sys, 'argv', argv), \
                            patch.object(check_packages.subprocess, 'run') as run, \
                            patch.object(check_packages.subprocess, 'check_output', return_value=graph):
                        with self.assertRaisesRegex(ValueError, 'native stack helper'):
                            check_packages.main()
                    builds = [call.args[0] for call in run.call_args_list
                              if 'cargo' in call.args[0] and 'run' in call.args[0]]
                    self.assertEqual(builds, [], 'reject the native graph before cargo run')
                    filename = ('fresh-dependency-metadata.json' if source == 'prepared'
                                else 'crates-io-consumer-metadata.json')
                    evidence = root / 'target' / filename
                    self.assertEqual(json.loads(evidence.read_text()), json.loads(graph))


if __name__ == '__main__':
    unittest.main()
