"""A server tag cannot establish a library's first registry baseline."""
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import tomlkit

from pr_config import main, release_pr_config


SOURCE = '''[workspace]
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
'''


class ReleasePrConfigTest(unittest.TestCase):
    def setUp(self):
        self.original = tomlkit.parse(SOURCE)

    def test_first_publication_uses_git_only_for_calculation(self):
        # Keep the coordinated tag and version group, but never use this
        # derived configuration to perform the first upload.
        result = release_pr_config(self.original, ['first', 'second'], lambda _: None)
        self.assertEqual(tomlkit.dumps(self.original), SOURCE)
        self.assertEqual(result['workspace'], self.original['workspace'])
        self.assertEqual(result['package'][0], self.original['package'][0])
        for package in result['package'][1:]:
            self.assertTrue(package['git_only'])
            self.assertFalse(package['publish'])
            self.assertFalse(package['semver_check'])
            self.assertEqual(package['version_group'], 'shared')

    def test_published_packages_keep_the_exact_original_configuration(self):
        # Never replace a published API baseline with a Git-only comparison.
        result = release_pr_config(self.original, ['first', 'second'],
                                   lambda path: {'crate': {'id': path.split('/')[-1]}})
        self.assertEqual(tomlkit.dumps(result), SOURCE)

    def test_partial_inventory_preserves_existing_registry_checks(self):
        result = release_pr_config(self.original, ['first', 'second'],
                                   lambda path: {'crate': {'id': 'first'}} if path == 'crates/first' else None)
        self.assertEqual(result['package'][1], self.original['package'][1])
        self.assertTrue(result['package'][2]['git_only'])
        self.assertFalse(result['package'][2]['publish'])

    def test_registry_failure_is_not_an_initial_release(self):
        def read(_):
            raise RuntimeError('registry unavailable')
        with self.assertRaisesRegex(RuntimeError, 'registry unavailable'):
            release_pr_config(self.original, ['first', 'second'], read)
        self.assertEqual(tomlkit.dumps(self.original), SOURCE)

    def test_malformed_response_is_not_an_initial_release(self):
        for response in ({}, [], {'crate': {'id': 'another-package'}}):
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, 'registry response'):
                    release_pr_config(self.original, ['first', 'second'], lambda _: response)

    def test_configuration_and_inventory_must_match_before_network_work(self):
        for names in ([], ['first'], ['first', 'first'], ['first', 'unknown']):
            with self.subTest(names=names):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    release_pr_config(self.original, names, lambda _: self.fail('unexpected request'))

    def test_cli_writes_an_ignored_configuration_without_changing_source(self):
        # An ignored output keeps release-plz's clean-source check effective.
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            subprocess.run(['git', 'init', '--quiet', directory], check=True)
            (root / '.gitignore').write_text('/target/\n')
            (root / '.release-plz.toml').write_text(SOURCE)
            (root / '.config').mkdir()
            (root / '.config/registry-packages.toml').write_text('packages = ["first", "second"]\n')
            with patch.object(sys, 'argv', ['pr_config.py', '--checkout', directory]), \
                 patch('pr_config.request', return_value=None):
                main()
            output = root / 'target/release-pr-config.toml'
            self.assertTrue(output.is_file())
            subprocess.run(['git', 'check-ignore', '--quiet', str(output)], cwd=root, check=True)
            self.assertEqual((root / '.release-plz.toml').read_text(), SOURCE)


if __name__ == '__main__':
    unittest.main()
