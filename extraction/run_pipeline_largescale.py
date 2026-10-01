"""
Large-Scale Pipeline Runner.

Processes ALL papers in combined_papers/crawled_and_filtered_papers/{venue}/
across venues: aacl, acl, eacl, emnlp, naacl, tacl.

Two-step workflow (no gold standard needed):

  Step 1 — PDF → Text : convert every PDF to plain text.
  Step 2 — Extract    : run Gemini (or OpenAI batch) to extract annotation metadata.

Modes
─────
  full            Run both steps end-to-end.
  extract_only    Only run extraction (text files must already exist).

Venue filtering
───────────────
  --venues acl emnlp       Only process those two venues.
  (default: all six venues)

Examples
────────
  # Full pipeline, all venues:
  python run_pipeline_largescale.py --mode full --num_workers 4

  # Only ACL + EMNLP with compact prompt:
  python run_pipeline_largescale.py --mode full --venues acl emnlp --compact

  # Extract only (text already converted):
  python run_pipeline_largescale.py --mode extract_only

  # Use OpenAI batch model instead of Gemini:
  python run_pipeline_largescale.py --mode full --model_id gpt-4.1-global-batch --use_blob
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ── project root ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PDFS_ROOT  = SCRIPT_DIR / "Processed_PDFs"
PAPERS_ROOT = SCRIPT_DIR / "combined_papers" / "crawled_and_filtered_papers"

ALL_VENUES = ["aacl", "acl", "eacl", "emnlp", "naacl", "tacl"]

DEFAULT_GEMINI_ID = "gemini-3.1-pro-preview"


# =============================================================================
# HELPERS
# =============================================================================

def banner(msg: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}\n")


def resolve_model_id(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    try:
        from azure_openai_api import AZURE_ENGINE_NAME
        if AZURE_ENGINE_NAME:
            return AZURE_ENGINE_NAME
    except Exception:
        pass
    return DEFAULT_GEMINI_ID


def collect_pdfs(venues: list[str]) -> list[Path]:
    """Gather all PDFs from the selected venue sub-folders."""
    pdfs: list[Path] = []
    for venue in venues:
        venue_dir = PAPERS_ROOT / venue
        if not venue_dir.is_dir():
            print(f"  WARNING: venue directory not found: {venue_dir}")
            continue
        found = sorted(venue_dir.glob("*.pdf"))
        print(f"  {venue:<8s}: {len(found):>4d} PDFs")
        pdfs.extend(found)
    return pdfs


# =============================================================================
# STEP 1 — PDF → Text
# =============================================================================

def step_pdf_to_text(
    pdfs: list[Path],
    txt_dir: Path,
) -> None:
    """Convert every PDF to a .txt file in txt_dir."""
    from pdf_to_text import extract_text_from_pdf

    banner("Step 1 — PDF → Text")
    txt_dir.mkdir(parents=True, exist_ok=True)

    already = {p.stem for p in txt_dir.glob("*.txt")}
    to_process = [p for p in pdfs if p.stem not in already]

    if not to_process:
        print(f"All {len(pdfs)} PDFs already converted → {txt_dir}")
        return

    print(f"Converting {len(to_process)}/{len(pdfs)} PDFs …")
    manifest = txt_dir / "extraction_manifest.jsonl"
    success = 0
    with manifest.open("a", encoding="utf-8") as mf:
        for i, pdf_path in enumerate(to_process, 1):
            try:
                result = extract_text_from_pdf(pdf_path)
                out_txt = txt_dir / f"{pdf_path.stem}.txt"
                out_txt.write_text(
                    result["text"], encoding="utf-8", errors="replace"
                )
                mf.write(
                    json.dumps(
                        {
                            "pdf": str(pdf_path),
                            "out_txt": str(out_txt),
                            "pages": result["pages"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                nonempty = sum(1 for s in result["page_stats"] if s["nonempty"])
                success += 1
                if i % 50 == 0 or i == len(to_process):
                    print(f"  [{i}/{len(to_process)}] {pdf_path.name}: "
                          f"{result['pages']} pp, {nonempty} non-empty")
            except Exception as exc:
                print(f"  [{i}/{len(to_process)}] {pdf_path.name}: ERROR – {exc}")

    print(f"Done: {success} converted, {len(to_process) - success} failed.")
    print(f"Text files in: {txt_dir}")


# =============================================================================
# STEP 2 — Extraction (Gemini or OpenAI batch)
# =============================================================================

def _step_extraction_gemini(
    model_id: str,
    txt_dir: Path,
    extractions_dir: Path,
    *,
    compact: bool = False,
    thinking_level: str | None = None,
    num_workers: int = 1,
) -> None:
    """Run Gemini extraction on all text files — no gold-standard filter."""
    import llm_extraction_gemini as ext_mod

    fake_args = argparse.Namespace(
        work_dir=str(SCRIPT_DIR),
        model_id=model_id,
        temperature=1.0,
        top_p=0.95,
        max_tokens=8000,
        compact_prompt=compact,
        no_chunking=False,
        chunk_tokens=8000,
        input_dir=str(txt_dir),
        output_dir=str(extractions_dir),
        single_file=None,
        skip_existing=True,
        thinking_level=thinking_level,
        num_workers=num_workers,
        gold_standard_csv=None,          # ← no filtering
    )

    original_parser = ext_mod.arg_parser
    ext_mod.arg_parser = lambda: fake_args
    try:
        ext_mod.main()
    finally:
        ext_mod.arg_parser = original_parser


def _step_extraction_openai(
    model_id: str,
    txt_dir: Path,
    extractions_dir: Path,
    *,
    compact: bool = False,
    submit_only: bool = False,
    resume_batch_id: str | None = None,
    use_blob: bool = False,
) -> None:
    """Run / submit OpenAI batch extraction — no gold-standard filter."""
    import llm_extraction_batch as ext_mod

    fake_args = argparse.Namespace(
        work_dir=str(SCRIPT_DIR),
        model_id=model_id,
        temperature=0.2,
        top_p=0.1,
        max_tokens=10000,
        compact_prompt=compact,
        input_dir=str(txt_dir),
        output_dir=str(extractions_dir),
        single_file=None,
        skip_existing=True,
        use_blob=use_blob,
        batch_size=0,
        sync=False,
        submit_only=submit_only and (resume_batch_id is None),
        resume=resume_batch_id,
        gold_standard_csv=None,          # ← no filtering
    )

    original_parser = ext_mod.arg_parser
    ext_mod.arg_parser = lambda: fake_args
    try:
        ext_mod.main()
    finally:
        ext_mod.arg_parser = original_parser


def step_extraction(
    model_id: str,
    txt_dir: Path,
    extractions_dir: Path,
    *,
    compact: bool = False,
    thinking_level: str | None = None,
    num_workers: int = 1,
    submit_only: bool = False,
    resume_batch_id: str | None = None,
    use_blob: bool = False,
) -> None:
    """Route to Gemini or OpenAI extraction based on model_id prefix."""
    banner(f"Step 2 — Extraction ({model_id})")
    extractions_dir.mkdir(parents=True, exist_ok=True)

    if model_id.startswith("gemini"):
        _step_extraction_gemini(
            model_id, txt_dir, extractions_dir,
            compact=compact,
            thinking_level=thinking_level,
            num_workers=num_workers,
        )
    else:
        _step_extraction_openai(
            model_id, txt_dir, extractions_dir,
            compact=compact,
            submit_only=submit_only,
            resume_batch_id=resume_batch_id,
            use_blob=use_blob,
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Large-Scale Pipeline: process ALL papers in combined_papers/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["full", "extract_only"],
        default="full",
        help="full = PDF→text + extraction (default); extract_only = skip conversion",
    )
    p.add_argument(
        "--venues",
        nargs="+",
        default=ALL_VENUES,
        choices=ALL_VENUES,
        help=f"Venues to process (default: all — {', '.join(ALL_VENUES)}). "
             "Ignored when --papers_dir is set.",
    )
    p.add_argument(
        "--papers_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Path to a flat directory of PDFs (e.g. from sample_acl_anthology.py). "
             "When set, --venues is ignored and PDFs are read from this directory "
             "recursively.",
    )
    p.add_argument(
        "--model_id",
        default=None,
        help="Model ID: gemini-* uses Gemini, anything else uses OpenAI/Azure batch "
             f"(default: env var or {DEFAULT_GEMINI_ID})",
    )
    p.add_argument("--compact", action="store_true", help="Use compact extraction prompt")
    p.add_argument(
        "--thinking_level",
        default=None,
        choices=["low", "medium", "high", "minimal"],
        help="Gemini thinking level (Gemini 3 models only)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Parallel worker threads for Gemini extraction (default: 1)",
    )
    p.add_argument("--use_blob", action="store_true", help="Azure Blob Storage for batch I/O")
    p.add_argument("--skip_pdf", action="store_true", help="Skip PDF→text even in full mode")
    p.add_argument(
        "--submit_only",
        action="store_true",
        help="Submit OpenAI batch and exit without waiting",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="BATCH_ID",
        help="Resume a previously submitted OpenAI batch job",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    model_id = resolve_model_id(args.model_id)
    venues = args.venues

    tag = "LargeScale"
    txt_dir         = PDFS_ROOT / f"{tag}_Processed_{model_id}"
    extractions_dir = PDFS_ROOT / f"{tag}_Extractions_{model_id}"

    print("=" * 60)
    print("  Large-Scale Pipeline")
    print("=" * 60)

    # Collect PDFs — either from a custom flat dir or from venue structure
    if args.papers_dir:
        papers_dir = Path(args.papers_dir)
        if not papers_dir.is_dir():
            print(f"ERROR: --papers_dir does not exist: {papers_dir}", file=sys.stderr)
            sys.exit(1)
        pdfs = sorted(papers_dir.rglob("*.pdf"))
        tag = f"RandomSample_{papers_dir.name}"
        txt_dir         = PDFS_ROOT / f"{tag}_Processed_{model_id}"
        extractions_dir = PDFS_ROOT / f"{tag}_Extractions_{model_id}"
        print(f"  Papers dir      : {papers_dir}")
        print(f"  PDFs found      : {len(pdfs)}")
    else:
        print(f"  Venues          : {', '.join(venues)}")
        print(f"  Papers root     : {PAPERS_ROOT}")
    print(f"  Text dir        : {txt_dir}")
    print(f"  Extractions dir : {extractions_dir}")
    print(f"  Model           : {model_id}")
    print(f"  Mode            : {args.mode}")
    print(f"  Workers         : {args.num_workers}")
    print()

    # Collect PDFs — already done above for --papers_dir; else scan venues
    if not args.papers_dir:
        print("Scanning venue directories …")
        pdfs = collect_pdfs(venues)
        print(f"Total: {len(pdfs)} PDFs\n")

    if not pdfs:
        print("ERROR: No PDFs found. Check combined_papers/crawled_and_filtered_papers/",
              file=sys.stderr)
        sys.exit(1)

    # Step 1 — PDF → Text
    if args.mode == "full" and not args.skip_pdf:
        step_pdf_to_text(pdfs, txt_dir)

    # Step 2 — Extraction
    step_extraction(
        model_id,
        txt_dir,
        extractions_dir,
        compact=args.compact,
        thinking_level=args.thinking_level,
        num_workers=args.num_workers,
        submit_only=args.submit_only,
        resume_batch_id=args.resume,
        use_blob=args.use_blob,
    )

    banner("Large-Scale Pipeline Complete")
    print(f"  Extractions → {extractions_dir}")
    print(f"  Total papers scanned: {len(pdfs)}")


if __name__ == "__main__":
    main()
