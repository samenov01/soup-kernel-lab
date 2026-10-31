"""Reuse the package map transformers already scanned when importing TRL.

The pinned HF packages independently enumerate every installed distribution.
No package is hidden or added: the complete existing mapping is passed through
unchanged during this import, and importlib is restored immediately afterward.
"""


def import_trl_with_existing_package_map():
    import importlib.metadata
    from transformers.utils import import_utils
    mapping = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if not isinstance(mapping, dict):
        import trl
        return trl
    original = importlib.metadata.packages_distributions
    importlib.metadata.packages_distributions = lambda: mapping
    try:
        import trl
        return trl
    finally:
        importlib.metadata.packages_distributions = original
