def test_import_reuses_complete_map_and_restores_importlib():
    import importlib.metadata
    from transformers.utils import import_utils
    from solution.import_metadata import import_trl_with_existing_package_map
    original = importlib.metadata.packages_distributions
    trl = import_trl_with_existing_package_map()
    assert importlib.metadata.packages_distributions is original
    from trl import import_utils as trl_imports
    assert trl_imports.PACKAGE_DISTRIBUTION_MAPPING == import_utils.PACKAGE_DISTRIBUTION_MAPPING
    assert trl.__version__
