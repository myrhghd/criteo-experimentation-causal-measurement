def test_package_import() -> None:
    import criteo_experiment

    assert criteo_experiment.__version__ == "0.1.0"
