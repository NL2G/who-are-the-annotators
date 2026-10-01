"""
Gold-Standard Pipeline Runner.

Three-step workflow based on the new gold standard CSV (Paper Link format):

  Step 1 — Download  : fetch PDFs from the Paper Link column of the gold CSV.
  Step 2 — Extract   : run Gemini to extract annotation metadata from each PDF.
  Step 3 — Agree     : use Azure OpenAI GPT-as-Judge to compare every LLM-extracted
                        field against the gold standard and get a correct/wrong verdict
                        for each column, then aggregate to an overall agreement rate.

Modes
─────
  full            Run all three steps end-to-end.
  download_only   Only download PDFs.
  extract_only    Only run Gemini extraction (PDFs/text must already exist).
  agree_only      Only run GPT-judge agreement (extractions must already exist).

Examples
────────
  # Full pipeline (download → extract → agree):
  python run_pipeline_gold.py --mode full

  # Skip download if PDFs are already on disk:
  python run_pipeline_gold.py --mode full --skip_download

  # Only the extraction step (re-use already-downloaded PDFs):
  python run_pipeline_gold.py --mode extract_only

  # Only the agreement step (re-use already-generated extractions):
  python run_pipeline_gold.py --mode agree_only

  # Use a specific Gemini model and judge model:
  python run_pipeline_gold.py --mode full \\
      --gemini_model_id gemini-3.1-pro-preview \\
      --judge_model_id gpt-4.1 \\
      --num_workers 4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ── project root ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent        # …/Who_Are_Annotators
PDFS_ROOT  = SCRIPT_DIR / "Processed_PDFs"

DEFAULT_GOLD_CSV   = SCRIPT_DIR / "data" / "gold_standard.csv"
DEFAULT_PDF_DIR    = SCRIPT_DIR / "data" / "pdfs"
DEFAULT_GEMINI_ID  = "gemini-3.1-pro-preview"


# =============================================================================
# HELPERS
# =============================================================================

def banner(msg: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}\n")


def resolve_judge_model(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    try:
        from azure_openai_api import AZURE_ENGINE_NAME
        if AZURE_ENGINE_NAME:
            return AZURE_ENGINE_NAME
    except Exception:
        pass
    return "gpt-4.1"


# =============================================================================
# STEP 1 — Download PDFs
# =============================================================================

def step_download(
    gold_csv: Path,
    pdf_dir: Path,
    delay: float = 1.0,
) -> None:
    """Download all PDFs referenced in the gold CSV."""
    from download_papers import download_papers_from_csv

    banner("Step 1 — Downloading PDFs")
    download_papers_from_csv(
        csv_path=gold_csv,
        out_dir=pdf_dir,
        delay=delay,
    )


# =============================================================================
# STEP 2a — PDF → Text
# =============================================================================

def step_pdf_to_text(
    pdf_dir: Path,
    txt_dir: Path,
) -> None:
    """Convert every PDF in pdf_dir to a .txt file in txt_dir."""
    from pdf_to_text import extract_text_from_pdf

    banner("Step 2a — PDF → Text")
    txt_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    already = {p.stem for p in txt_dir.glob("*.txt")}
    to_process = [p for p in pdfs if p.stem not in already]
    if not to_process:
        print(f"All {len(pdfs)} PDFs already converted → {txt_dir}")
        return

    print(f"Converting {len(to_process)}/{len(pdfs)} PDFs …")
    manifest = txt_dir / "extraction_manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as mf:
        for pdf_path in to_process:
            try:
                result = extract_text_from_pdf(pdf_path)
                out_txt = txt_dir / f"{pdf_path.stem}.txt"
                out_txt.write_text(result["text"], encoding="utf-8", errors="replace")
                mf.write(
                    json.dumps(
                        {"pdf": str(pdf_path), "out_txt": str(out_txt), "pages": result["pages"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                nonempty = sum(1 for s in result["page_stats"] if s["nonempty"])
                print(f"  {pdf_path.name}: {result['pages']} pp, {nonempty} non-empty")
            except Exception as exc:
                print(f"  {pdf_path.name}: ERROR – {exc}")
    print(f"Text files in: {txt_dir}")


# =============================================================================
# STEP 2b — Gemini Extraction
# =============================================================================

def step_extraction(
    gemini_model_id: str,
    gold_csv: Path,
    txt_dir: Path,
    extractions_dir: Path,
    *,
    compact: bool = False,
    thinking_level: str | None = None,
    num_workers: int = 1,
) -> None:
    """Run Gemini extraction on all text files in txt_dir."""
    import llm_extraction_gemini as ext_mod

    banner(f"Step 2b — Gemini Extraction ({gemini_model_id})")
    extractions_dir.mkdir(parents=True, exist_ok=True)

    # Pass the new gold CSV directly; load_gold_standard_ids now handles URL-based CSVs.
    # This ensures only papers from the gold CSV are processed, in case txt_dir ever
    # contains extra files.
    fake_args = argparse.Namespace(
        work_dir=str(SCRIPT_DIR),
        model_id=gemini_model_id,
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
        gold_standard_csv=str(gold_csv),
    )

    original_parser = ext_mod.arg_parser
    ext_mod.arg_parser = lambda: fake_args
    try:
        ext_mod.main()
    finally:
        ext_mod.arg_parser = original_parser


# =============================================================================
# STEP 3 — GPT-Judge Agreement
# =============================================================================

def step_agreement(
    gold_csv: Path,
    extractions_dir: Path,
    output_dir: Path,
    gemini_model_id: str,
    judge_model_id: str,
    *,
    temperature: float = 0.0,
    use_blob: bool = False,
    submit_only: bool = False,
    resume: Optional[str] = None,
) -> None:
    """Run GPT-as-Judge agreement between Gemini extractions and the gold CSV."""
    import llm_human_agreement_gpt_judge as judge_mod

    banner("Step 3 — GPT-Judge Agreement")
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_args = argparse.Namespace(
        work_dir=str(SCRIPT_DIR),
        gold_csv=str(gold_csv),
        gemini_model_id=gemini_model_id,
        llm_extractions_dir=str(extractions_dir),
        judge_model_id=judge_model_id,
        output_dir=str(output_dir),
        temperature=temperature,
        use_blob=use_blob,
        submit_only=submit_only,
        resume=resume,
    )

    original_parser = judge_mod.arg_parser
    judge_mod.arg_parser = lambda: fake_args
    try:
        judge_mod.main()
    finally:
        judge_mod.arg_parser = original_parser


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gold-Standard Pipeline Runner (download → extract → GPT-judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["full", "download_only", "extract_only", "agree_only"],
        default="full",
        help=(
            "full          = run all three steps end-to-end (default)\n"
            "download_only = only download PDFs from the gold CSV\n"
            "extract_only  = only run Gemini extraction\n"
            "agree_only    = only run GPT-judge agreement"
        ),
    )
    p.add_argument(
        "--gold_csv",
        default=str(DEFAULT_GOLD_CSV),
        help=(
            "Path to the gold standard CSV with a 'Paper Link' column "
            f"(default: {DEFAULT_GOLD_CSV.name})"
        ),
    )
    p.add_argument(
        "--pdf_dir",
        default=str(DEFAULT_PDF_DIR),
        help=f"Directory for downloaded PDFs (default: {DEFAULT_PDF_DIR})",
    )
    p.add_argument(
        "--gemini_model_id",
        default=DEFAULT_GEMINI_ID,
        help=f"Gemini model used for extraction (default: {DEFAULT_GEMINI_ID})",
    )
    p.add_argument(
        "--judge_model_id",
        default=None,
        help="Azure OpenAI model used for field-level judging "
             "(default: AZURE_ENGINE_NAME env var or gpt-4.1)",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip the PDF download step (PDFs already present in pdf_dir)",
    )
    p.add_argument(
        "--skip_pdf_to_text",
        action="store_true",
        help="Skip the PDF-to-text conversion step",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Use compact extraction prompt for Gemini",
    )
    p.add_argument(
        "--thinking_level",
        default=None,
        choices=["low", "medium", "high", "minimal"],
        help="Gemini thinking level for extraction (Gemini 3 models only)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel threads for Gemini extraction (default: 1)",
    )
    p.add_argument(
        "--download_delay",
        type=float,
        default=1.0,
        help="Seconds to wait between PDF downloads (default: 1.0)",
    )
    p.add_argument(
        "--judge_temperature",
        type=float,
        default=0.0,
        help="Temperature for GPT judge calls (default: 0.0)",
    )
    p.add_argument(
        "--use_blob",
        action="store_true",
        default=False,
        help="Use Azure Blob Storage for batch I/O (recommended for large jobs)",
    )
    p.add_argument(
        "--submit_only",
        action="store_true",
        help="Submit the GPT-judge batch job and exit without waiting",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="BATCH_ID",
        help="Resume a previously submitted GPT-judge batch job",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory for GPT-judge agreement results",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    gold_csv       = Path(args.gold_csv)
    pdf_dir        = Path(args.pdf_dir)
    gemini_id      = args.gemini_model_id
    judge_id       = resolve_judge_model(args.judge_model_id)

    txt_dir        = PDFS_ROOT / f"Gold_Processed_{gemini_id}"
    extractions_dir = PDFS_ROOT / f"Gold_Extractions_{gemini_id}"
    agreement_dir  = (
        Path(args.output_dir)
        if args.output_dir
        else PDFS_ROOT / f"Gold_Agreement_gpt_judge_{gemini_id}"
    )

    print("=" * 60)
    print("  Gold-Standard Pipeline")
    print("=" * 60)
    print(f"  Gold CSV        : {gold_csv}")
    print(f"  PDF dir         : {pdf_dir}")
    print(f"  Text dir        : {txt_dir}")
    print(f"  Extractions dir : {extractions_dir}")
    print(f"  Agreement dir   : {agreement_dir}")
    print(f"  Gemini model    : {gemini_id}")
    print(f"  Judge model     : {judge_id}")
    print(f"  Mode            : {args.mode}")
    print()

    if not gold_csv.exists():
        print(f"ERROR: Gold CSV not found: {gold_csv}", file=sys.stderr)
        sys.exit(1)

    run_download    = args.mode in ("full", "download_only") and not args.skip_download
    run_pdf_to_text = args.mode in ("full", "extract_only") and not args.skip_pdf_to_text
    run_extract     = args.mode in ("full", "extract_only")
    run_agree       = args.mode in ("full", "agree_only")

    if run_download:
        step_download(gold_csv, pdf_dir, delay=args.download_delay)

    if run_pdf_to_text:
        step_pdf_to_text(pdf_dir, txt_dir)

    if run_extract:
        step_extraction(
            gemini_id,
            gold_csv,
            txt_dir,
            extractions_dir,
            compact=args.compact,
            thinking_level=args.thinking_level,
            num_workers=args.num_workers,
        )

    if run_agree:
        step_agreement(
            gold_csv=gold_csv,
            extractions_dir=extractions_dir,
            output_dir=agreement_dir,
            gemini_model_id=gemini_id,
            judge_model_id=judge_id,
            temperature=args.judge_temperature,
            use_blob=args.use_blob,
            submit_only=args.submit_only,
            resume=args.resume,
        )

    banner("Pipeline Complete")


if __name__ == "__main__":
    main()
