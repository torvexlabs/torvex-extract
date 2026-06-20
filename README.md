# Torvex Extract

Local-first PDF extraction engine for document AI.

Torvex Extract turns PDFs into structured page-level JSON: text, tables, layout zones, spotlight regions, formula boxes, and optional formula LaTeX. It is built for systems that need evidence-grade document processing without sending files to hosted OCR or LLM APIs.

This repository is part of the Torvex Labs document AI stack.

```text
unimernet-onnx
    |
    v
torvex-extract  ->  torvex-bench
    |
    v
ClearVault
```

- [unimernet-onnx](https://github.com/torvexlabs/unimernet-onnx): pure ONNX Runtime UniMERNet formula recognition.
- [torvex-bench](https://github.com/torvexlabs/torvex-bench): benchmark and evaluation harness.
- [ClearVault](https://github.com/sibisrinivasb/clearvault): evidence retrieval and RAG product layer.

## Status

Torvex Extract is in active development. It is usable as an experimental extraction engine, but the public setup is not yet packaged like a polished end-user library.

Important current limits:

- Model artifacts are required and are not committed to this repository.
- The CPU/GPU dependency split is still being cleaned up.
- Formula extraction is optional and depends on `unimernet-onnx`.
- Benchmark claims should be checked through `torvex-bench`, not inferred from this README.

## What It Extracts

Torvex Extract currently targets:

- digital PDF text
- scanned-page OCR text
- page layout zones
- bordered and borderless tables
- table structure
- chart/image/visual "spotlight" regions
- formula bounding boxes
- optional formula LaTeX through UniMERNet ONNX
- timing and quality metadata for diagnostics

The output is JSON, designed to be consumed by downstream retrieval, evaluation, and evidence-citation systems.

## Pipeline

```text
PDF
  |
  v
pypdfium2
  |-- fast page probing
  |-- digital text extraction
  |-- page rendering when visual inference is needed
  |
  v
PP-DocLayoutV3 ONNX
  |-- text/content zones
  |-- table zones
  |-- image/chart/spotlight zones
  |-- formula zones
  |
  v
Table extraction
  |-- pdfplumber for bordered/digital tables
  |-- TATR ONNX for table structure
  |-- OCR-backed table recovery for scanned pages
  |
  v
OCR
  |-- onnxtr_fast_base
  |-- ppocrv6_small
  |
  v
Optional formula recognition
  |-- display formula filtering/splitting
  |-- UniMERNet tiny ONNX runtime
  |
  v
Structured page JSON
```

## Installation

This project uses `uv`.

```bash
git clone https://github.com/torvexlabs/torvex-extract.git
cd torvex-extract
uv sync
```

For development tests:

```bash
uv run pytest
```

Current dependency note: `pyproject.toml` still pins `onnxruntime-gpu[cuda,cudnn]`. A cleaner CPU/GPU packaging split is planned. If you are setting this up on a CPU-only machine, expect dependency/runtime cleanup work.

## Model Artifacts

Large model files are intentionally ignored by git. A working local setup needs model artifacts under `models/` or paths supplied through environment variables.

Expected default layout:

```text
models/
  PP-DocLayoutV3_ir8.onnx
  tatr-v1.1-all.onnx
  ppocrv6-small-onnx/
    det/
      inference.onnx
      inference.yml
    rec/
      inference.onnx
      inference.yml
  unimernet-tiny-onnx/
    artifacts/
      encoder_model.onnx
      decoder_model.onnx
      decoder_with_past_model.onnx
    models/
      unimernet_tiny/
        tokenizer.json
        config.json
```

Useful environment overrides:

| Variable | Purpose |
| --- | --- |
| `DOCLAYOUT_MODEL_PATH` | Override PP-DocLayoutV3 ONNX path |
| `TATR_MODEL_PATH` | Override TATR ONNX path |
| `TORVEX_PPOCRV6_MODEL_DIR` | Override PP-OCRv6 model directory |
| `TORVEX_UNIMERNET_ONNX_MODEL_DIR` | Override UniMERNet ONNX root |
| `TORVEX_UNIMERNET_ONNX_ARTIFACTS_DIR` | Override UniMERNet ONNX artifacts directory |
| `TORVEX_UNIMERNET_TOKENIZER_DIR` | Override UniMERNet tokenizer directory |

## CLI Usage

Run extraction on one PDF:

```bash
uv run torvex-extract path/to/document.pdf --out output.json --pretty
```

Use GPU providers:

```bash
uv run torvex-extract path/to/document.pdf --device gpu --out output.json --pretty
```

Use the PP-OCRv6 small backend for scanned pages:

```bash
uv run torvex-extract path/to/document.pdf --ocr-backend ppocrv6_small --out output.json --pretty
```

Enable optional formula LaTeX extraction:

```bash
uv run torvex-extract path/to/document.pdf --enable-formula --formula-device gpu --out output.json --pretty
```

If `--out` is omitted, the CLI writes `<input>.torvex.json` next to the PDF.

## Python API

```python
from torvex_extract import extract_with_pypdfium2, warm, shutdown

warm(device="cpu", ocr_backend="onnxtr_fast_base")

try:
    pages, errors = extract_with_pypdfium2(
        "path/to/document.pdf",
        enable_formula=False,
    )
finally:
    shutdown()
```

The engine must be warmed before extraction. Model loading is intentionally explicit because ONNX sessions are expensive and should be reused in long-running processes.

## Output Shape

CLI output has this high-level shape:

```json
{
  "pdf": "path/to/document.pdf",
  "device": "cpu",
  "ocr_backend": "onnxtr_fast_base",
  "formula_enabled": false,
  "summary": {
    "pages": 1,
    "errors": 0,
    "text_pages": 1,
    "table_count": 0,
    "spotlight_count": 0,
    "formula_count": 0
  },
  "errors": [],
  "pages": [
    {
      "page_num": 0,
      "needs_ocr": false,
      "final_text": "...",
      "zones": [],
      "tables": [],
      "formula_bboxes": [],
      "formulas": [],
      "metadata": {}
    }
  ]
}
```

Page dictionaries include coordinate metadata such as `bbox_px`, `bbox_pdfium`, and `bbox_plumber` where available. These are used by downstream citation and benchmark tooling.

## Formula Extraction

Formula extraction is optional.

By default, Torvex Extract keeps formula zones as bounding-box artifacts. When `--enable-formula` is used, display formulas are cropped and recognized through UniMERNet ONNX.

Current behavior:

- `display_formula` is enabled by default for formula recognition.
- `inline_formula` and `formula_number` are opt-in through `TORVEX_FORMULA_TYPES`.
- display formula boxes may be split before recognition when a detected box contains stacked equations.
- recognized formulas include LaTeX plus metadata such as token count, EOS status, truncation flag, provider info, and timing.

Important runtime note: install the `unimernet-onnx` runtime carefully. If it pulls a different plain `onnxruntime` package into the same environment, it can break a GPU setup that expects `onnxruntime-gpu`.

## OCR Backends

Torvex Extract currently exposes two OCR backends:

| Backend | Use |
| --- | --- |
| `onnxtr_fast_base` | default OCR backend |
| `ppocrv6_small` | PP-OCRv6 small ONNX backend for OCR comparison and multilingual/scanned-page experiments |

The rest of the pipeline receives the same OCR segment contract from both backends.

## Diagnostics

Useful local scripts:

```bash
uv run python scripts/smoke_engine_contract.py path/to/document.pdf
uv run python scripts/smoke_profile_extract.py path/to/document.pdf
uv run python scripts/overlay_formula_bboxes.py --pdf path/to/document.pdf --smoke-json output.json --page 1 --out overlay.png
```

The smoke scripts are intended for development diagnostics, not as a stable public API.

## Benchmarking

`tools/omnidocbench_eval/` contains helper scripts for OmniDocBench debugging and loss attribution. The broader benchmark orchestration belongs in [torvex-bench](https://github.com/torvexlabs/torvex-bench).

Use `torvex-bench` when you need:

- dataset setup
- reproducible benchmark runs
- comparison against other engines
- score reports
- per-page and per-formula diagnostics

Keep `torvex-extract` focused on the extraction engine itself.

## Development Notes

Run tests:

```bash
uv run pytest
```

Run syntax checks without installing test dependencies:

```bash
python -m compileall -q src tests tools scripts
```

The current tests mostly cover import contracts, formula runtime behavior, exporter policy, and PP-OCRv6 batching logic. Model-backed smoke coverage is still a work in progress.

## Roadmap

- Clean CPU/GPU dependency split.
- Document model artifact download/setup.
- Add minimal model-backed smoke fixture or fixture-free preflight.
- Stabilize CLI output schema.
- Keep extraction engine boundaries separate from benchmark orchestration.
- Improve formula detection/crop/export policy through benchmark evidence.
- Publish reproducible benchmark summaries through `torvex-bench`.

## License

Torvex Extract is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Maintainer

Built by [Torvex Labs](https://github.com/torvexlabs).
