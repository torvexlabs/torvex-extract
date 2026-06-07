from torvex_extract.pypdfium_extractor import extract_with_pypdfium2, engine


def warm(device: str = "cpu") -> None:
    engine.warm(device=device)


def shutdown() -> None:
    engine.shutdown()


def is_warmed() -> bool:
    return engine.is_warmed()