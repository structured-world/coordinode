"""Dependency checks shared by preparation and registry-only verification.

Keep this module standard-library-only: registry verification runs after the
disposable preparation environment has been removed.
"""


def check_native_stack_helpers(packages):
    """Reject the known native stack helpers from the delivered graph."""
    native = sorted({p['name'] for p in packages if p['name'] in {'stacker', 'psm'}})
    if native:
        raise ValueError(f"registry closure contains a native stack helper: {native}")
