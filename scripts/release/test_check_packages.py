"""Archive gates reject source-only inputs before a consumer can hide them."""
import io
import pathlib
import tarfile
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
