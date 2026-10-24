"""Reconcile an immutable coordinated release against crates.io."""
import argparse
import json
import os
import pathlib
import re
import subprocess
import tomllib
import urllib.error
import urllib.request


def request(method, path):
    """Read public registry metadata without publication credentials."""
    headers = {"Content-Type": "application/json",
               "User-Agent": "coordinode-release (github.com/structured-world/coordinode)"}
    req = urllib.request.Request("https://crates.io/api/v1/" + path, method=method,
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404 and method == "GET" and path.startswith("crates/"):
            return None
        raise RuntimeError(f"crates.io {method} returned HTTP {error.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("crates.io unavailable; reconcile the original release before retrying") from None


def check_versions(report, read, *, allow_missing):
    """Existing versions must be the exact qualified archives, including on retries."""
    for name, package in report["packages"].items():
        data = read(f"crates/{name}/{report['version']}")
        if data is None:
            if allow_missing:
                continue
            raise ValueError(f"{name} {report['version']} is not published")
        version = data["version"]
        if version["checksum"] != package["archive"]["sha256"] or version["yanked"]:
            raise ValueError(f"{name}: registry version does not match the qualified archive")


def merged_release_revision(pull):
    """Resolve a retry from the forge's approved merge, never a moving PR head."""
    if (pull['merged_at'] is None or pull['base']['ref'] != 'main'
            or not pull['head']['ref'].startswith('release-plz-')):
        raise ValueError('retry requires a merged release PR targeting main')
    revision = pull['merge_commit_sha']
    if not isinstance(revision, str) or re.fullmatch(r'[0-9a-f]{40}', revision) is None:
        raise ValueError('release PR has no valid merge commit identifier')
    return revision


def publication_revision(root, source, release_head):
    """Match release-plz's commit selection before creating qualified archives.

    release-plz 0.3.174 checks out an ancestor PR head for a merge commit. Squash
    merges retain the checked-out source. Never transfer a source-suite result
    to different files, even if both commits belong to the same release PR.
    """
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()

    if git('rev-parse', '--is-shallow-repository') != 'false':
        raise ValueError('publication requires complete Git history')
    source_tree = git('rev-parse', f'{source}^{{tree}}')
    present = subprocess.run(['git', 'cat-file', '-e', f'{release_head}^{{commit}}'],
                             cwd=root, capture_output=True)
    if present.returncode != 0:
        # Complete source history cannot omit an ancestor: a deleted squashed
        # PR head is not a candidate for release-plz's ancestor checkout.
        return source
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', release_head, source], cwd=root)
    if ancestor.returncode == 1:
        return source
    ancestor.check_returncode()
    if git('rev-parse', f'{release_head}^{{tree}}') != source_tree:
        raise ValueError('release-plz would publish a different tree from the successful source suite')
    return release_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("source", "eligible", "preflight", "verify"))
    parser.add_argument("--checkout", type=pathlib.Path, required=True)
    parser.add_argument("--release-pr", type=int, help="merged release PR to retry")
    args = parser.parse_args()
    if args.release_pr is not None and (args.mode != 'source' or args.release_pr <= 0):
        parser.error('--release-pr requires source mode and a positive PR number')
    root = args.checkout.resolve()
    if os.environ.get("GITHUB_REPOSITORY") != "structured-world/coordinode":
        raise ValueError("release requires the authoritative repository")
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise ValueError("release requires the main branch")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.mode == 'source':
        if args.release_pr is not None:
            pull = json.loads(subprocess.check_output([
                'gh', 'api', f'repos/structured-world/coordinode/pulls/{args.release_pr}'], text=True))
            source = merged_release_revision(pull)
        subprocess.run(['git', 'merge-base', '--is-ancestor', source, 'origin/main'],
                       cwd=root, check=True)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f'commit={source}\n')
        print(f'Selected source commit: {source}')
        return
    if args.mode == "eligible":
        # This is release-plz's documented release_always=false contract. Check
        # it before expensive preparation or checking an old server-only tag.
        pulls = json.loads(subprocess.check_output(
            ["gh", "api", f"repos/structured-world/coordinode/commits/{source}/pulls"], text=True))
        release = next((p for p in pulls if p["merged_at"] is not None
                        and p["head"]["ref"].startswith("release-plz-")), None)
        publication = source
        if release is not None:
            pages = json.loads(subprocess.check_output([
                'gh', 'api', '--paginate', '--slurp',
                f"repos/structured-world/coordinode/pulls/{release['number']}/commits"], text=True))
            commits = [commit for page in pages for commit in page]
            if not commits:
                raise ValueError('merged release PR has no recorded commits')
            publication = publication_revision(root, source, commits[-1]['sha'])
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"publish={'true' if release is not None else 'false'}\n")
            output.write(f"publication_commit={publication}\nverified_commit={source}\n")
        print("Merged release PR: " + str(release is not None))
        return
    report_path = root / "target/registry-preparation.json"
    report = json.loads(report_path.read_text())
    names = tomllib.loads((root / ".config/registry-packages.toml").read_text())["packages"]
    if set(report["packages"]) != set(names) or report["source_commit"] != source:
        raise ValueError("qualification report does not describe this checkout and inventory")
    if args.mode == "preflight":
        tag = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/v{report['version']}^{{commit}}"],
                             cwd=root, text=True, capture_output=True)
        if tag.returncode == 0 and tag.stdout.strip() != source:
            raise ValueError("version belongs to another source revision; prepare a new release PR")
        check_versions(report, lambda p: request("GET", p), allow_missing=True)
    elif args.mode == "verify":
        check_versions(report, lambda p: request("GET", p), allow_missing=False)
        report["registry_checksums_verified"] = True
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.mode}: verified {len(names)} coordinated packages")


if __name__ == "__main__":
    main()
