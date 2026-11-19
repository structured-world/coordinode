"""The release PR carries the builds outside the workspace up to the release."""
import pathlib
import subprocess
import tempfile
import unittest

import check_packages
from example_requirements import refresh_lock, release_branch, requirement_for, update_manifest

MANIFEST = (
    '[package]\nname = "quickstart"\n\n[workspace]\n\n'
    '[dependencies]\ncoordinode-embed = "0.9"\ncoordinode-core = "0.9"\ntempfile = "3"\n'
)
NAMES = {"coordinode-embed", "coordinode-core"}


class RequirementTest(unittest.TestCase):
    def test_the_requirement_admits_the_release_and_no_breaking_one(self):
        for version, wanted in (("0.10.0", "0.10"), ("0.9.3", "0.9"), ("1.4.2", "1.4"),
                                ("0.11.0-rc.1", "0.11")):
            with self.subTest(version=version):
                self.assertEqual(requirement_for(version), wanted)
                self.assertTrue(check_packages.admits(wanted, version))

    def test_registry_requirements_move_and_nothing_else_changes(self):
        # The example the release qualifies is the one in the PR: after the
        # update the release's own check admits the version.
        updated = update_manifest(MANIFEST, NAMES, "0.10.0")
        self.assertEqual(updated, MANIFEST.replace('"0.9"', '"0.10"'))
        self.assertIn('tempfile = "3"', updated)

    def test_an_example_already_on_the_release_is_left_as_it_is(self):
        current = MANIFEST.replace('"0.9"', '"0.10"')
        self.assertEqual(update_manifest(current, NAMES, "0.10.4"), current)

    def test_a_requirement_in_table_form_is_refused_not_guessed(self):
        manifest = '[dependencies]\ncoordinode-embed = { version = "0.9" }\n'
        with self.assertRaisesRegex(ValueError, "plain version requirement"):
            update_manifest(manifest, NAMES, "0.10.0")


class PathLockTest(unittest.TestCase):
    """A standalone workspace locked on a path crate builds `--locked` again
    once the release bumped that crate."""

    @staticmethod
    def write(root, path, text):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def locked_build(self, root):
        return subprocess.run(
            ["cargo", "tree", "--quiet", "--locked", "--offline", "--depth", "0",
             "--manifest-path", str(root / "bench/Cargo.toml")],
            capture_output=True, text=True)

    def test_a_bumped_path_crate_is_recorded_in_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            crate = '[package]\nname = "lib"\nversion = "{}"\nedition = "2024"\n'
            self.write(root, "lib/Cargo.toml", crate.format("0.1.0"))
            self.write(root, "lib/src/lib.rs", "")
            self.write(root, "bench/Cargo.toml",
                       '[workspace]\n\n[package]\nname = "bench"\nversion = "0.1.0"\n'
                       'edition = "2024"\n\n[dependencies]\nlib = { path = "../lib" }\n')
            self.write(root, "bench/src/main.rs", "fn main() {}\n")
            # The committed lock, on the version before the release.
            subprocess.run(["cargo", "generate-lockfile", "--offline",
                            "--manifest-path", str(root / "bench/Cargo.toml")], check=True)
            self.assertEqual(self.locked_build(root).returncode, 0)
            self.assertFalse(refresh_lock(root, "bench"), "a lock that matches stays")

            # The release bumps the crate: the committed lock no longer builds.
            self.write(root, "lib/Cargo.toml", crate.format("0.2.0"))
            self.assertNotEqual(self.locked_build(root).returncode, 0)

            self.assertTrue(refresh_lock(root, "bench"))
            self.assertIn('name = "lib"\nversion = "0.2.0"',
                          (root / "bench/Cargo.lock").read_text())
            self.assertEqual(self.locked_build(root).returncode, 0)
            self.assertFalse(refresh_lock(root, "bench"), "nothing left to bring")


class ReleasePrTest(unittest.TestCase):
    def test_the_branch_comes_from_release_plz_output(self):
        output = '{"prs":[{"head_branch":"release-plz-2026-10-25","number":80}]}'
        self.assertEqual(release_branch(output), "release-plz-2026-10-25")

    def test_no_release_pr_means_nothing_to_update(self):
        self.assertIsNone(release_branch('{"prs":[]}'))
        self.assertIsNone(release_branch(""))


if __name__ == "__main__":
    unittest.main()
