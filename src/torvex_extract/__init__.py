from torvex_extract.pypdfium_extractor import extract_with_pypdfium2, engine


def warm(device: str = "cpu", ocr_backend: str | None = None) -> None:
    engine.warm(device=device, ocr_backend=ocr_backend)


def shutdown() -> None:
    engine.shutdown()


def is_warmed() -> bool:
    return engine.is_warmed()
