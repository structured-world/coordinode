"""Bring the published example's requirements to the release a PR prepares.

The example builds from registry packages as an application does, so it is not
a workspace member and release-plz never updates its manifest. Run after
`release-plz release-pr --output json`: on the release PR's branch, each
requirement on a registry package becomes the caret requirement of the release
version, committed and pushed to that branch, so the PR carries it.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import tomllib

QUICKSTART = "examples/embedded-quickstart/Cargo.toml"
REPOSITORY = "structured-world/coordinode"


def requirement_for(version):
    """The caret requirement admitting `version` and its compatible releases:
    `0.10.0` gives `0.10`, `1.4.2` gives `1.4`."""
    major, minor = version.split("-")[0].split(".")[:2]
    return f"{major}.{minor}"


def update_manifest(text, names, version):
    """`text` with every `[dependencies]` requirement on one of `names` set to
    the release's caret requirement; other entries are kept byte for byte."""
    manifest = tomllib.loads(text)
    wanted = requirement_for(version)
    for name, requirement in manifest.get("dependencies", {}).items():
        if name not in names:
            continue
        if not isinstance(requirement, str):
            raise ValueError(f"{QUICKSTART}: {name} must be a plain version requirement")
        line = re.compile(rf'^({re.escape(name)}\s*=\s*)"[^"]*"', re.MULTILINE)
        text, count = line.subn(rf'\g<1>"{wanted}"', text)
        if count != 1:
            raise ValueError(f"{QUICKSTART}: {name} is not one `name = \"version\"` line")
    return text


def release_branch(pr_json):
    """The head branch of the release PR release-plz reported, or None."""
    prs = json.loads(pr_json or "{}").get("prs") or []
    return prs[0]["head_branch"] if prs else None


def git(root, *args, **kwargs):
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=pathlib.Path, required=True)
    parser.add_argument("--pr-json", type=pathlib.Path, required=True,
                        help="release-plz release-pr --output json")
    args = parser.parse_args()
    root = args.checkout.resolve()
    branch = release_branch(args.pr_json.read_text())
    if branch is None:
        print("no release PR: nothing to update")
        return
    token = os.environ["GIT_TOKEN"]
    remote = f"https://x-access-token:{token}@github.com/{REPOSITORY}.git"
    git(root, "fetch", "--quiet", remote, f"refs/heads/{branch}")
    git(root, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    names = set(tomllib.loads(
        (root / ".config/registry-packages.toml").read_text())["packages"])
    path = root / QUICKSTART
    before = path.read_text()
    after = update_manifest(before, names, version)
    if after == before:
        print(f"{QUICKSTART} already admits {version}")
        return
    path.write_text(after)
    git(root, "add", QUICKSTART)
    git(root, "-c", "user.name=sw-release-bot[bot]",
        "-c", "user.email=sw-release-bot[bot]@users.noreply.github.com",
        "commit", "--quiet", "-m",
        f"chore(examples): require the {requirement_for(version)} release")
    git(root, "push", "--quiet", remote, f"HEAD:refs/heads/{branch}")
    print(f"{QUICKSTART} requires {requirement_for(version)} on {branch}")


if __name__ == "__main__":
    main()
