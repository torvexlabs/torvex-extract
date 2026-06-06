from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypdfium2
from PIL import ImageDraw, ImageFont


RENDER_DPI = 200.0


def load_pages(smoke_output_json: Path) -> list[dict]:
    payload = json.loads(smoke_output_json.read_text(encoding="utf-8"))
    return payload["pages"]


def overlay_formula_page(
    pdf_path: Path,
    smoke_output_json: Path,
    page_number_1based: int,
    output_path: Path,
) -> None:
    pages = load_pages(smoke_output_json)

    page_index = page_number_1based - 1
    page_data = pages[page_index]

    formulas = page_data.get("formula_bboxes") or []

    if not formulas:
        print(f"No formula boxes found on page {page_number_1based}")
        return

    pdf = pypdfium2.PdfDocument(str(pdf_path))

    try:
        page = pdf[page_index]
        image = page.render(scale=RENDER_DPI / 72.0).to_pil().convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        for index, formula in enumerate(formulas):
            bbox = formula.get("bbox_px")
            if not bbox or len(bbox) != 4:
                continue

            x0, y0, x1, y1 = [float(v) for v in bbox]

            label = (
                f"{index}: {formula.get('type')} "
                f"{float(formula.get('score', 0.0)):.2f}"
            )

            draw.rectangle(
                [x0, y0, x1, y1],
                outline="red",
                width=4,
            )

            draw.text(
                (x0, max(0, y0 - 22)),
                label,
                fill="red",
                font=font,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        print(f"saved overlay: {output_path}")
        print(f"formula boxes: {len(formulas)}")

    finally:
        pdf.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--smoke-json", required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument(
        "--out",
        default="results/smoke/formula_overlay_page.png",
    )
    args = parser.parse_args()

    overlay_formula_page(
        pdf_path=Path(args.pdf),
        smoke_output_json=Path(args.smoke_json),
        page_number_1based=args.page,
        output_path=Path(args.out),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())