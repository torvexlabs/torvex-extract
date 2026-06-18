from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "tools" / "omnidocbench_eval" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omnidocbench_markdown import normalized_page_to_markdown


def test_omnidocbench_exporter_skips_unmatched_display_formulas(monkeypatch):
    monkeypatch.delenv("TORVEX_ODB_EXPORT_UNMATCHED_FORMULAS", raising=False)

    markdown = normalized_page_to_markdown(
        {
            "zones": [
                {
                    "type": "text",
                    "bbox_px": [0, 0, 100, 100],
                    "zone_text": "Body text",
                }
            ],
            "formulas": [
                {
                    "type": "display_formula",
                    "status": "accepted",
                    "latex": "x=1",
                    "bbox_px": [200, 200, 300, 240],
                }
            ],
        }
    )

    assert "Body text" in markdown
    assert "$$" not in markdown


def test_omnidocbench_exporter_keeps_matched_display_formulas(monkeypatch):
    monkeypatch.delenv("TORVEX_ODB_EXPORT_UNMATCHED_FORMULAS", raising=False)

    markdown = normalized_page_to_markdown(
        {
            "zones": [
                {
                    "type": "display_formula",
                    "bbox_px": [200, 200, 300, 240],
                }
            ],
            "formulas": [
                {
                    "type": "display_formula",
                    "status": "accepted",
                    "latex": "x=1",
                    "bbox_px": [205, 205, 295, 235],
                }
            ],
        }
    )

    assert "$$\nx=1\n$$" in markdown
