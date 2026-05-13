def test_package_import_is_safe() -> None:
    import dockar

    assert dockar.__all__ == ["DocKarConfig", "load_config"]
