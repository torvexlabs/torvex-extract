"""converter.py — convert raw torvex-extract formula output to the markdown format the OmniDocBench
docker harness expects (the harness's md_tex_filter parses `$$...$$` blocks as display formulas /
equation_isolated). One markdown file per page, formulas in reading order.

Input  : raw predictions JSON  {image_name: [{"content": <latex>, "order": <int>}, ...]}
         (i.e. PP-DocLayout display boxes -> UniMERNet latex, per page, in reading order)
Output : <out_dir>/<image_stem>.md   (one per page)

Usage:
  python converter.py --pred raw_predictions.json --out-dir pred_md [--only-pages pages.json]
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

# --- optional post-process: split a recognized multi-equation block into per-row formulas ---
_SYS_ENV = re.compile(r"\\begin\{(aligned|align|alignat|cases|gather|gathered|split|multline|eqnarray)\}")
_ARR_ENV = re.compile(r"\\begin\{array\}\s*\{([^}]*)\}")
_MATRIX = re.compile(r"\\left\s*[\(\[\|]|\\begin\{(?:p|b|v|V|small)?matrix\}|\\binom")
_ENV = re.compile(r"\\(?:begin|end)\{[^}]*\}")
_ROWSEP = re.compile(r"\\\\|\\newline|\\cr")


def _should_split(latex: str) -> bool:
    if "\\\\" not in latex:        # no row separator -> nothing to split (run-together stays whole)
        return False
    if _MATRIX.search(latex):      # matrix / bracketed -> ONE formula, keep whole
        return False
    if _SYS_ENV.search(latex):
        return True
    m = _ARR_ENV.search(latex)
    if m:
        cols = m.group(1).replace(" ", "").replace("|", "")
        return bool(cols) and ("l" in cols or "r" in cols)   # rl/rcl = aligned system
    return True                    # bare \\ without env -> treat as aligned, split


def _split_rows(latex: str) -> list:
    s = _ARR_ENV.sub("", latex)
    s = _ENV.sub("", s)
    rows = []
    for r in _ROWSEP.split(s):
        r = r.replace("&", " ").strip().strip("\\").strip()
        if r and len(r) > 1:
            rows.append(r)
    return rows


def _as_display(latex: str) -> str:
    c = (latex or "").strip()
    if not c:
        return ""
    # unwrap any pre-existing delimiters, then emit a clean display-formula block
    for lo, hi in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if c.startswith(lo) and c.endswith(hi) and len(c) > len(lo) + len(hi):
            c = c[len(lo):-len(hi)].strip()
            break
    return "$$\n" + c + "\n$$"


# A unified recognizer (e.g. UniRec) emits a MULTI-LINE crop as several delimited
# segments in ONE pred string (\[a\]\n\[b\]\n\[c\]). _as_display would collapse them
# into a single $$ block -> MGAM matches one line, the rest become no-pred. Split them.
_SEG = re.compile(r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$", re.DOTALL)


def _split_segments(content: str) -> list:
    segs = [next(g for g in m.groups() if g is not None).strip() for m in _SEG.finditer(content)]
    segs = [s for s in segs if s]
    return segs if len(segs) >= 2 else []          # only split when genuinely multi-segment


def page_to_markdown(preds: list, split_rows: bool = False, seg_split: bool = False) -> str:
    blocks = []
    for p in sorted(preds, key=lambda x: x.get("order", 0)):
        c = (p.get("content") or "").strip()
        if not c:
            continue
        segs = _split_segments(c) if seg_split else []
        if segs:
            blocks.extend(_as_display(s) for s in segs)             # one $$ per delimited segment
        elif split_rows and _should_split(c):
            blocks.extend(_as_display(r) for r in _split_rows(c))   # one $$ per equation row
        else:
            blocks.append(_as_display(c))
    return "\n\n".join(b for b in blocks if b) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="raw torvex-extract predictions JSON")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only-pages", type=Path, default=None,
                    help="optional JSON list of image names to include (others skipped)")
    ap.add_argument("--seg-split", action="store_true",
                    help="split multi-segment preds (\\[a\\]\\[b\\]) into separate $$ blocks (UniRec granularity fix)")
    args = ap.parse_args()

    data = json.load(open(args.pred, encoding="utf-8"))
    only = set(json.load(open(args.only_pages, encoding="utf-8"))) if args.only_pages else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for image_name, preds in data.items():
        if only is not None and image_name not in only:
            continue
        (args.out_dir / (Path(image_name).stem + ".md")).write_text(
            page_to_markdown(preds, seg_split=args.seg_split), encoding="utf-8")
        n += 1
    print(f"wrote {n} markdown files -> {args.out_dir}")


if __name__ == "__main__":
    main()
