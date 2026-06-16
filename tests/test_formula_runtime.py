from __future__ import annotations

import tomllib

import numpy as np
import pytest

from torvex_extract import formula_extractor
from torvex_extract import pypdfium_extractor


def test_formula_extra_installs_unimernet_runtime():
    with open("pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)

    formula_deps = config["project"]["optional-dependencies"]["formula"]

    assert any("unimernet-onnx" in dep for dep in formula_deps)


def test_formula_enabled_raises_before_document_work_when_runtime_missing(monkeypatch):
    class FakeEngine:
        def is_warmed(self):
            return True

    def fail_preflight(*, device=None):
        raise RuntimeError("runtime missing")

    monkeypatch.setattr(pypdfium_extractor, "engine", FakeEngine())
    monkeypatch.setattr(
        pypdfium_extractor,
        "ensure_formula_runtime_available",
        fail_preflight,
    )

    with pytest.raises(RuntimeError, match="runtime missing"):
        pypdfium_extractor.extract_with_pypdfium2(
            "does-not-matter.pdf",
            enable_formula=True,
            formula_device="gpu",
        )


def test_formula_extraction_uses_runtime_batch_api():
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def recognize_batch(self, imgs, *, max_batch_size, sort_by_size):
            self.calls.append(
                {
                    "count": len(imgs),
                    "max_batch_size": max_batch_size,
                    "sort_by_size": sort_by_size,
                }
            )
            return [
                {
                    "latex": "x=1",
                    "tokens": [2],
                    "token_count": 1,
                    "last_token": 2,
                    "eos_reached": True,
                    "truncated": False,
                    "elapsed_ms": 10.0,
                    "ms_per_token": 10.0,
                    "batch_size": len(imgs),
                    "batch_group_index": 0,
                    "active_providers": {"encoder": ["CPUExecutionProvider"]},
                    "io_binding": False,
                }
                for _ in imgs
            ]

    fake_runtime = FakeRuntime()
    extractor = formula_extractor.FormulaMfrExtractor(
        config=formula_extractor.FormulaExtractionConfig(
            max_batch_size=4,
            sort_by_size=True,
        )
    )
    extractor._ocr = fake_runtime

    img_np = np.full((80, 200, 3), 255, dtype=np.uint8)
    img_np[30:40, 50:100] = 0

    artifacts = extractor.extract(
        img_np=img_np,
        formula_bboxes=[
            {
                "formula_id": "f0",
                "type": "display_formula",
                "bbox_px": [40, 20, 120, 50],
                "score": 0.9,
            }
        ],
        page_num=0,
    )

    assert fake_runtime.calls == [
        {
            "count": 1,
            "max_batch_size": 4,
            "sort_by_size": True,
        }
    ]
    assert artifacts[0]["latex"] == "x=1"
    assert artifacts[0]["mfr_batch_size"] == 1
