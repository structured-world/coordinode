"""A partial publication is resumable only with identical artifacts."""
import unittest
import pathlib
import subprocess
import tempfile
import sys

import registry
from registry import check_versions


class ReleaseRetryTest(unittest.TestCase):
    def setUp(self):
        self.pull = {
            'merged_at': '2026-01-01T00:00:00Z',
            'base': {'ref': 'main'},
            'head': {'ref': 'release-plz-test'},
            'merge_commit_sha': 'a1' * 20,
        }

    def test_retry_selects_the_recorded_merge_commit(self):
        # A retry must retain the approved release instead of moving with main.
        self.assertEqual(registry.merged_release_revision(self.pull),
                         self.pull['merge_commit_sha'])

    def test_retry_refuses_unmerged_or_unrelated_pull_requests(self):
        # Dispatch cannot turn an ordinary or unapproved PR into a release.
        variants = [dict(self.pull, merged_at=None),
                    dict(self.pull, base={'ref': 'development'}),
                    dict(self.pull, head={'ref': 'feature/test'})]
        for pull in variants:
            with self.subTest(pull=pull), self.assertRaisesRegex(ValueError, 'merged release PR'):
                registry.merged_release_revision(pull)

    def test_retry_refuses_a_moving_ref_in_place_of_the_merge_commit(self):
        # Only the forge's immutable commit identifier can reach checkout.
        self.pull['merge_commit_sha'] = 'refs/heads/main'
        with self.assertRaisesRegex(ValueError, 'commit identifier'):
            registry.merged_release_revision(self.pull)


class PublicationRevisionTest(unittest.TestCase):
    """Qualify the commit release-plz packages, not an ancestor with another tree."""

    def test_ancestor_release_head_requires_the_same_verified_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)

            def git(*args):
                return subprocess.check_output(['git', '-c', 'user.name=release-test',
                                                '-c', 'user.email=release-test@example.org', *args],
                                               cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

            git('init', '--quiet')
            (root / 'input').write_text('qualified')
            git('add', 'input')
            git('commit', '--quiet', '-m', 'release input')
            release = git('rev-parse', 'HEAD')
            git('commit', '--quiet', '--allow-empty', '-m', 'same tree')
            source = git('rev-parse', 'HEAD')
            self.assertEqual(registry.publication_revision(root, source, release), release)
            # A changed merge result cannot borrow an ancestor's source-CI pass.
            (root / 'input').write_text('different')
            git('add', 'input')
            git('commit', '--quiet', '-m', 'different tree')
            with self.assertRaisesRegex(ValueError, 'different tree'):
                registry.publication_revision(root, git('rev-parse', 'HEAD'), release)

    def test_squashed_release_keeps_the_verified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            subprocess.run(['git', 'init', '--quiet', directory], check=True)
            subprocess.run(['git', '-c', 'user.name=release-test',
                            '-c', 'user.email=release-test@example.org', 'commit', '--quiet',
                            '--allow-empty', '-m', 'squashed release'], cwd=root, check=True)
            source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            # Original PR commit is absent after squash + branch deletion.
            self.assertEqual(registry.publication_revision(root, source, '1' * 40), source)


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.report = {"version": "1.0.0", "packages": {
            "coordinode-core": {"archive": {"sha256": "first"}},
            "coordinode-embed": {"archive": {"sha256": "second"}},
        }}

    def test_partial_release_can_resume_but_is_not_complete(self):
        # A successful first upload does not certify the missing dependent.
        def read(path):
            return {"version": {"checksum": "first", "yanked": False}} if 'core/' in path else None
        check_versions(self.report, read, allow_missing=True)
        with self.assertRaisesRegex(ValueError, 'not published'):
            check_versions(self.report, read, allow_missing=False)

    def test_changed_archive_cannot_continue_a_partial_release(self):
        # Never silently accept a version uploaded from another source/lock.
        with self.assertRaisesRegex(ValueError, 'qualified archive'):
            check_versions(self.report, lambda _: {"version": {"checksum": "other", "yanked": False}},
                           allow_missing=True)

    def test_yanked_archive_is_not_a_qualified_delivery(self):
        with self.assertRaisesRegex(ValueError, 'qualified archive'):
            check_versions(self.report, lambda _: {"version": {"checksum": "first", "yanked": True}},
                           allow_missing=True)

    def test_publisher_provisioning_is_not_a_release_operation(self):
        # CI consumes an existing publisher and cannot register one.
        result = subprocess.run([sys.executable, '-B', registry.__file__, 'trust',
                                 '--checkout', '.'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice: 'trust'", result.stderr)


if __name__ == '__main__':
    unittest.main()
