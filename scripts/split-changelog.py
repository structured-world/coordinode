#!/usr/bin/env python3
"""Split workspace CHANGELOG.md into per-crate CHANGELOG.md files.

The workspace CHANGELOG.md is maintained by release-plz (via workspace_changelog).
It contains per-crate sections under each version header like:

  ## v0.3.6 - 2026-04-13
  ### coordinode-embed 0.3.6
  #### Added
  - ...
  ### coordinode-query 0.3.6
  ...

This script reads the workspace changelog and writes
crates/<name>/CHANGELOG.md containing only the sections relevant
to that crate, across all versions.

Run after each release (e.g. from the coordinode-release.yml workflow).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
WORKSPACE_CHANGELOG = ROOT / "CHANGELOG.md"
CRATES_DIR = ROOT / "crates"

# A crate heading carries the crate's own version, which is what the per-crate
# file needs as its version header. It is absent in the hand-written
# "## Unreleased" block, so the version is optional here.
CRATE_HEADER_RE = re.compile(
    r'^### (coordinode-[a-z0-9-]+)(?: +(\d+\.\d+\.\d+[^\s]*))?\s*$', re.MULTILINE
)
VERSION_HEADER_RE = re.compile(r'^## (\S+)(?: +- +(\S+))?\s*$', re.MULTILINE)

PREAMBLE = """\
# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

"""


def parse_workspace_changelog(text: str) -> dict[str, list[str]]:
    """
    Returns {crate_name: [entry_str, ...]} where each entry_str is the
    formatted changelog block for one version, newest-first.
    """
    # Locate each top-level version header and the text block that follows it.
    # We use finditer to get positions, then slice the text directly — avoids
    # the re.split() capturing-group indexing pitfall.
    # (release tag, release date, block start, block end)
    version_spans: list[tuple[str, str, int, int]] = []

    matches = list(VERSION_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        tag = m.group(1)            # "v0.3.6" from "## v0.3.6 - 2026-04-13"
        date = m.group(2) or ""
        block_start = m.end() + 1   # character after the newline following the header
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        version_spans.append((tag, date, block_start, block_end))

    crate_sections: dict[str, list[str]] = {}

    for tag, date, block_start, block_end in version_spans:
        block = text[block_start:block_end]

        # Find per-crate sub-sections inside this version block.
        crate_matches = list(CRATE_HEADER_RE.finditer(block))
        if not crate_matches:
            # Every release block names the crates it covers. None means the
            # heading shape changed and this script is silently about to strip
            # the per-crate files down to nothing, which is how they lost their
            # history once already.
            raise SystemExit(
                f"ERROR: release block '{tag}' has no '### <crate>' sections. "
                "The workspace CHANGELOG.md layout changed; update the "
                "heading patterns in this script before regenerating."
            )
        for j, cm in enumerate(crate_matches):
            crate = cm.group(1)
            crate_version = cm.group(2) or tag
            section_start = cm.start()
            section_end = (
                crate_matches[j + 1].start() if j + 1 < len(crate_matches) else len(block)
            )
            section_text = block[section_start:section_end].rstrip()

            # Drop the "### coordinode-<name> <version>" heading and restate the
            # version as this file's own top-level header, since here the crate
            # is the subject rather than one entry among many.
            lines = section_text.split('\n')
            section_body = '\n'.join(lines[1:]).strip().rstrip('-').strip()
            if not section_body:
                continue

            header = f"## {crate_version} - {date}" if date else f"## {crate_version}"
            crate_sections.setdefault(crate, []).append(f"{header}\n\n{section_body}")

    return crate_sections


def write_crate_changelog(crate: str, entries: list[str]) -> None:
    crate_dir = CRATES_DIR / crate
    if not crate_dir.is_dir():
        print(f"  skipping {crate}: directory not found", file=sys.stderr)
        return

    changelog_path = crate_dir / "CHANGELOG.md"
    content = PREAMBLE + "\n\n---\n\n".join(entries) + "\n"
    changelog_path.write_text(content, encoding="utf-8")
    print(f"  wrote {changelog_path.relative_to(ROOT)}")


def main() -> None:
    if not WORKSPACE_CHANGELOG.exists():
        print(f"ERROR: {WORKSPACE_CHANGELOG} not found", file=sys.stderr)
        sys.exit(1)

    text = WORKSPACE_CHANGELOG.read_text(encoding="utf-8")
    crate_sections = parse_workspace_changelog(text)

    if not crate_sections:
        print("No per-crate sections found in workspace CHANGELOG.md")
        sys.exit(0)

    print("Splitting workspace CHANGELOG.md into per-crate files:")
    for crate, entries in sorted(crate_sections.items()):
        write_crate_changelog(crate, entries)

    # Crates with no tracked changes get a minimal pointer to the workspace file.
    all_crates = sorted(
        d.name for d in CRATES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("coordinode-")
    )
    for crate in all_crates:
        if crate not in crate_sections:
            crate_dir = CRATES_DIR / crate
            changelog_path = crate_dir / "CHANGELOG.md"
            changelog_path.write_text(
                PREAMBLE +
                "No changes recorded for this crate in the workspace changelog.\n"
                "See the root [CHANGELOG.md](../../CHANGELOG.md) for full history.\n",
                encoding="utf-8",
            )
            print(f"  wrote {changelog_path.relative_to(ROOT)} (no entries)")


if __name__ == "__main__":
    main()
