"""Calculate library release baselines without confusing server tags with uploads."""
import argparse
import pathlib
import subprocess
import tomllib

import tomlkit

from registry import request


def release_pr_config(document, names, read):
    """Use Git only where no published library API exists to compare against."""
    result = tomlkit.parse(tomlkit.dumps(document))
    defaults = result['workspace']
    packages = result['package']
    configured = {package['name'] for package in packages
                  if package.get('publish', defaults.get('publish', True))}
    if not names or len(set(names)) != len(names) or configured != set(names):
        raise ValueError('publication configuration does not match the package inventory')
    if len({package['name'] for package in packages}) != len(packages):
        raise ValueError('duplicate package configuration in release inventory')
    for package in packages:
        name = package['name']
        if name not in configured:
            continue
        data = read(f'crates/{name}')
        if data is None:
            # This config calculates a PR only. Upload always uses the normal
            # checked-in config; Git-only packages cannot be published.
            package['git_only'] = True
            package['publish'] = False
            package['semver_check'] = False
        elif not isinstance(data, dict) or not isinstance(data.get('crate'), dict) \
                or data['crate'].get('id') != name:
            raise ValueError(f'{name}: unexpected registry response')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.checkout.resolve()
    output = root / 'target/release-pr-config.toml'
    ignored = subprocess.run(['git', 'check-ignore', '--quiet', str(output)], cwd=root)
    if ignored.returncode != 0:
        raise ValueError('release-PR configuration output must be ignored')
    document = tomlkit.parse((root / '.release-plz.toml').read_text())
    names = tomllib.loads((root / '.config/registry-packages.toml').read_text())['packages']
    result = release_pr_config(document, names, lambda path: request('GET', path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tomlkit.dumps(result))
    print(f'Release-PR configuration prepared for {len(names)} libraries')


if __name__ == '__main__':
    main()
