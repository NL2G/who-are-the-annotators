import argparse
import json
import os
from pathlib import Path

import fitz
import logging

# Suppress noisy MuPDF warnings (e.g. "invalid key in dict")
fitz.TOOLS.mupdf_warnings()  # flush any pending warnings
fitz.TOOLS.set_small_glyph_heights(True)  # workaround for some malformed PDFs
logging.getLogger("fitz").setLevel(logging.ERROR)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="data/pdfs", help="Folder with input PDFs")
    parser.add_argument("--out_dir", default="Processed_PDFs/Gold_Processed", help="Where to write .txt files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subfolders for PDFs",
    )
    return parser.parse_args()


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """
    Extract text from a PDF with PyMuPDF.
    Returns dict with 'text', 'pages', and simple per-page stats.
    """
    doc = fitz.open(pdf_path)
    # Drain any warnings produced during open (e.g. "invalid key in dict")
    fitz.TOOLS.mupdf_warnings()

    page_count = doc.page_count

    page_texts = []
    page_stats = []

    for i in range(page_count):
        page = doc.load_page(i)
        # Drain per-page warnings to suppress stderr noise
        fitz.TOOLS.mupdf_warnings()

        # Primary: plain text extraction
        text = page.get_text("text") or ""

        # Fallback 1: block-level extraction
        if not text.strip():
            try:
                blocks = page.get_text("blocks")
                block_texts = [
                    b[4].strip()
                    for b in blocks
                    if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()
                ]
                text = "\n\n".join(block_texts)
            except Exception:
                pass

        # Fallback 2: rawdict word-level extraction for badly structured PDFs
        if not text.strip():
            try:
                words = page.get_text("words")
                text = " ".join(w[4] for w in words if len(w) >= 5 and w[4].strip())
            except Exception:
                pass

        page_texts.append(text)
        page_stats.append(
            {
                "page_index": i,
                "chars": len(text),
                "nonempty": bool(text.strip()),
            }
        )

    doc.close()

    full_text = "\n\n".join(
        [f"--- Page {i+1} ---\n{t}" for i, t in enumerate(page_texts)]
    )

    return {
        "text": full_text,
        "pages": page_count,
        "page_stats": page_stats,
    }



def main():
    args = arg_parser()
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "extraction_manifest.jsonl"

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {pdf_dir}")

    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdf_paths = sorted(pdf_dir.glob(pattern))

    if not pdf_paths:
        print(f"No PDFs found in: {pdf_dir} (recursive={args.recursive})")
        return

    with manifest_path.open("w", encoding="utf-8") as mf:
        for pdf_path in pdf_paths:
            try:
                result = extract_text_from_pdf(pdf_path)

                out_txt = out_dir / f"{pdf_path.stem}.txt"
                out_txt.write_text(result["text"], encoding="utf-8", errors="replace")

                record = {
                    "pdf": str(pdf_path),
                    "out_txt": str(out_txt),
                    "pages": result["pages"],
                    "page_stats": result["page_stats"],
                }
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")

                nonempty_pages = sum(1 for s in result["page_stats"] if s["nonempty"])
                print(f"{pdf_path.name}: {result['pages']} pages, {nonempty_pages} non-empty -> {out_txt.name}")

            except Exception as e:
                err_record = {"pdf": str(pdf_path), "error": repr(e)}
                mf.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                print(f"{pdf_path.name}: {e}")

    print(f"\nDone. Text files in: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()