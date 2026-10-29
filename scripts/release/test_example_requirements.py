"""The release PR carries the example's requirements up to the release."""
import unittest

import check_packages
from example_requirements import release_branch, requirement_for, update_manifest

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


class ReleasePrTest(unittest.TestCase):
    def test_the_branch_comes_from_release_plz_output(self):
        output = '{"prs":[{"head_branch":"release-plz-2026-10-25","number":80}]}'
        self.assertEqual(release_branch(output), "release-plz-2026-10-25")

    def test_no_release_pr_means_nothing_to_update(self):
        self.assertIsNone(release_branch('{"prs":[]}'))
        self.assertIsNone(release_branch(""))


if __name__ == "__main__":
    unittest.main()
