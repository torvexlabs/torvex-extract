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


def test_formula_extraction_suppresses_display_parent_duplicates():
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def recognize_batch(self, imgs, *, max_batch_size, sort_by_size):
            self.calls.append(len(imgs))
            return [
                {
                    "latex": "x=1",
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
    extractor = formula_extractor.FormulaMfrExtractor()
    extractor._ocr = fake_runtime

    img_np = np.full((120, 220, 3), 255, dtype=np.uint8)
    img_np[20:40, 30:120] = 0
    img_np[60:80, 30:120] = 0

    artifacts = extractor.extract(
        img_np=img_np,
        formula_bboxes=[
            {
                "formula_id": "parent",
                "type": "display_formula",
                "bbox_px": [10, 10, 190, 95],
                "score": 0.9,
            },
            {
                "formula_id": "child_a",
                "type": "display_formula",
                "bbox_px": [25, 15, 130, 45],
                "score": 0.8,
            },
            {
                "formula_id": "child_b",
                "type": "display_formula",
                "bbox_px": [25, 55, 130, 85],
                "score": 0.8,
            },
        ],
        page_num=0,
    )

    assert fake_runtime.calls == [2]
    assert [artifact["formula_id"] for artifact in artifacts] == [
        "child_a",
        "child_b",
    ]


def test_formula_extraction_preserves_strong_display_parent_over_fragments():
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def recognize_batch(self, imgs, *, max_batch_size, sort_by_size):
            self.calls.append(len(imgs))
            return [
                {
                    "latex": "x=1",
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
    extractor = formula_extractor.FormulaMfrExtractor()
    extractor._ocr = fake_runtime

    img_np = np.full((220, 240, 3), 255, dtype=np.uint8)
    img_np[20:40, 30:200] = 0
    img_np[80:100, 30:200] = 0
    img_np[140:160, 30:200] = 0

    artifacts = extractor.extract(
        img_np=img_np,
        formula_bboxes=[
            {
                "formula_id": "parent",
                "type": "display_formula",
                "bbox_px": [10, 10, 220, 180],
                "score": 0.72,
            },
            {
                "formula_id": "child_a",
                "type": "display_formula",
                "bbox_px": [25, 15, 205, 45],
                "score": 0.36,
            },
            {
                "formula_id": "child_b",
                "type": "display_formula",
                "bbox_px": [25, 75, 205, 105],
                "score": 0.40,
            },
            {
                "formula_id": "child_c",
                "type": "display_formula",
                "bbox_px": [25, 135, 205, 165],
                "score": 0.38,
            },
        ],
        page_num=0,
    )

    assert fake_runtime.calls == [4]
    assert [artifact["formula_id"] for artifact in artifacts] == [
        "parent",
        "child_a",
        "child_b",
        "child_c",
    ]
