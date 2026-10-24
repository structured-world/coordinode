import json
import pathlib
import tempfile
import unittest

import supply_chain


def package(root, name, license_expr="MIT", files=("LICENSE",)):
    directory = pathlib.Path(root) / name
    directory.mkdir()
    (directory / "Cargo.toml").write_text("")
    for file in files:
        (directory / file).write_text("text")
    return {
        "id": f"{name} 1.0.0",
        "name": name,
        "version": "1.0.0",
        "source": "registry+https://github.com/rust-lang/crates.io-index",
        "license": license_expr,
        "license_file": None,
        "manifest_path": str(directory / "Cargo.toml"),
    }


def edge(target, *kinds):
    return {"pkg": f"{target} 1.0.0", "dep_kinds": [{"kind": kind} for kind in kinds]}


LOCK = '''
[[package]]
name = "normal"
version = "1.0.0"
checksum = "aa"

[[package]]
name = "built"
version = "1.0.0"
checksum = "bb"
'''


class SupplyChainTest(unittest.TestCase):
    def graph(self, root, normal_license="MIT"):
        packages = [
            package(root, "consumer"),
            package(root, "normal", normal_license, ("LICENSE-MIT", "NOTICE", "README.md")),
            package(root, "built"),
            package(root, "dev-only"),
        ]
        nodes = [
            {"id": "consumer 1.0.0", "deps": [edge("normal", None), edge("dev-only", "dev")]},
            {"id": "normal 1.0.0", "deps": [edge("built", "build")]},
            {"id": "built 1.0.0", "deps": []},
            {"id": "dev-only 1.0.0", "deps": []},
        ]
        return {"packages": packages, "resolve": {"root": "consumer 1.0.0", "nodes": nodes}}

    def test_inventory_follows_normal_and_build_edges_only(self):
        # A dev-only dependency never reaches a consumer's build, so it is not
        # part of what the consumer redistributes.
        with tempfile.TemporaryDirectory() as root:
            evidence = supply_chain.inventory(self.graph(root), LOCK, {"features": "standalone"})
        names = [entry["name"] for entry in evidence["packages"]]
        self.assertEqual(names, ["built", "normal"])
        normal = evidence["packages"][1]
        self.assertEqual(normal["checksum"], "aa")
        self.assertEqual(normal["attribution_files"], ["LICENSE-MIT", "NOTICE"])
        self.assertEqual(evidence["context"], {"features": "standalone"})

    def test_a_package_without_a_license_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "normal 1.0.0"):
                supply_chain.inventory(self.graph(root, normal_license=None), LOCK, {})

    def test_sbom_names_each_component_with_its_license_and_hash(self):
        with tempfile.TemporaryDirectory() as root:
            out = pathlib.Path(root) / "out"
            out.mkdir()
            supply_chain.write_evidence(out, "stem", self.graph(root), LOCK, {"target": "x"})
            sbom = json.loads((out / "stem-sbom.cdx.json").read_text())
            licenses = json.loads((out / "stem-licenses.json").read_text())
        self.assertEqual(sbom["bomFormat"], "CycloneDX")
        self.assertEqual(sbom["specVersion"], "1.5")
        normal = next(c for c in sbom["components"] if c["name"] == "normal")
        self.assertEqual(normal["purl"], "pkg:cargo/normal@1.0.0")
        self.assertEqual(normal["licenses"], [{"expression": "MIT"}])
        self.assertEqual(normal["hashes"], [{"alg": "SHA-256", "content": "aa"}])
        self.assertIn({"name": "coordinode:target", "value": "x"}, sbom["metadata"]["properties"])
        self.assertEqual(len(licenses["packages"]), 2)


if __name__ == "__main__":
    unittest.main()
