from torvex_extract import extract_with_pypdfium2


def test_public_api_import():
    assert callable(extract_with_pypdfium2)