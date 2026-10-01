"""
LLM Extraction Script - Using Google Gemini 3.
Extracts human annotation information from papers.
Drop-in replacement for llm_extraction_v2.py, using Gemini instead of OpenAI.
"""

import argparse
import concurrent.futures
import json
import re
import threading
from pathlib import Path
from typing import Dict, Optional, Set

from tqdm import tqdm

from gemini_api import GeminiApi
from prompts_v2 import (
    build_extraction_prompt,
    build_extraction_prompt_compact,
    build_json_fix_prompt,
    extract_annotation_chunks,
    extract_annotation_chunks_embedding,
    get_extraction_json_schema,
    get_system_prompt,
    truncate_paper,
)
from validation import validate_full_extraction


# Regex to extract the canonical paper base ID from a filename stem or CSV value.
# Handles 3- and 4-segment IDs (e.g. 2021.eacl.14.WiCTSVAn_Evalua).
_PAPER_ID_RE = re.compile(
    r"^(\d{4}\.[A-Za-z0-9-]+\.[A-Za-z0-9]+(?:\.[A-Za-z0-9_]+)*)"
)

# Explicit aliases: filename-derived numeric IDs -> gold-standard title-based IDs.
# Add an entry here whenever a paper in Combined/ uses a numeric 3rd segment
# (e.g. "3") while the gold-standard CSV uses the title form (e.g. "TikZill").
_PAPER_ID_ALIASES: Dict[str, str] = {
    "2026.iclr-main.3": "2026.iclr-main.TikZill",
}


def _base_id(raw: str) -> str:
    """Return canonical paper base ID, stripping annotator suffixes."""
    m = _PAPER_ID_RE.match(raw.strip())
    return m.group(1) if m else raw.strip()


def load_gold_standard_ids(csv_path: Path) -> Set[str]:
    """Return the set of canonical paper base IDs present in the gold standard CSV.

    Handles two CSV formats:
    1. Old format: PaperID column (section labels in row 0, real column names in row 1).
    2. New gold format: Paper Link column (URLs like https://aclanthology.org/2023.acl-long.711.pdf).
       The paper stem is extracted from the URL (e.g. '2023.acl-long.711').
    """
    _URL_ACL_RE = re.compile(
        r"aclanthology\.org/([A-Za-z0-9._-]+?)(?:\.pdf)?/?$"
    )
    _URL_ARXIV_RE = re.compile(
        r"arxiv\.org/(?:pdf|abs)/([0-9]+\.[0-9]+(?:v\d+)?)"
    )

    def _stem_from_url(url: str) -> str:
        url = url.strip().rstrip("/")
        m = _URL_ACL_RE.search(url)
        if m:
            return m.group(1)
        m = _URL_ARXIV_RE.search(url)
        if m:
            return m.group(1)
        last = url.split("/")[-1]
        return last[:-4] if last.endswith(".pdf") else last

    with csv_path.open(encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline()
        second_line = fh.readline()

    # Detect format: new gold CSV has "Paper Link" in row 0's first cell
    first_cell_row0 = first_line.split(",")[0].strip().strip('"').lower()
    is_url_format = "paper link" in first_cell_row0 or first_cell_row0.startswith("http")

    second_first_cell = second_line.split(",")[0].strip().lower()
    skip_rows_old = [0] if second_first_cell.startswith("paperid") else []

    ids: Set[str] = set()
    with csv_path.open(encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    if is_url_format:
        # New format: header in row 0, URLs in col 0 of data rows
        for line in lines[1:]:  # skip header row
            line = line.strip()
            if not line:
                continue
            first_cell = line.split(",")[0].strip().strip('"')
            if not first_cell.startswith("http"):
                continue
            stem = _stem_from_url(first_cell)
            bid = _base_id(stem)
            if bid:
                ids.add(bid)
    else:
        # Old format: paper IDs in col 0 (optionally with section header in row 0)
        header_line = 1 if skip_rows_old else 0
        for line in lines[header_line + 1:]:  # skip header itself
            line = line.strip()
            if not line:
                continue
            first_cell = line.split(",")[0].strip().strip('"')
            if not first_cell or not first_cell[0].isdigit():
                continue
            # Strip any ":task" suffix before normalising
            first_cell = first_cell.split(":")[0].strip()
            bid = _base_id(first_cell)
            if bid:
                ids.add(bid)
    return ids


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract human annotation data from papers using Gemini API"
    )
    parser.add_argument("--work_dir", required=True, help="Workspace directory")
    parser.add_argument(
        "--model_id",
        default="gemini-3.1-pro-preview",
        help="Model ID (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature (Gemini 3 default is 1.0; avoid lower values)")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--thinking_level", default=None,
                        choices=["low", "medium", "high", "minimal"],
                        help="Gemini 3 thinking level (default: model decides)")
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--compact_prompt", action="store_true")
    parser.add_argument(
        "--no_chunking",
        action="store_true",
        help="Disable annotation-section chunking and send the full paper "
             "(up to 20 000 tokens). Chunking is ON by default.",
    )
    parser.add_argument(
        "--chunk_tokens",
        type=int,
        default=8000,
        help="Token budget for annotation-relevant chunks (default: 8000). "
             "Ignored when --no_chunking is set.",
    )
    parser.add_argument("--input_dir", help="Override input directory")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--single_file", help="Process only this single file")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel worker threads (default: 1)")
    parser.add_argument(
        "--gold_standard_csv",
        default="data/gold_standard.csv",
        help="Path to gold standard CSV. When set (or auto-detected), only papers "
             "in that CSV will be processed.",
    )
    return parser.parse_args()


def fix_json_with_llm(api: GeminiApi, raw_json: str, max_retries: int = 3) -> str:
    if not raw_json:
        return ""
    current = raw_json
    for attempt in range(1, max_retries + 1):
        try:
            json.loads(current)
            return current
        except json.JSONDecodeError as e:
            if attempt == max_retries:
                print(f"JSON unfixable after {max_retries} attempts")
                return ""
            print(f"JSON fix attempt {attempt}/{max_retries}: {e}")
            prompt = build_json_fix_prompt(current, str(e))
            current = api.request(prompt)
    return ""


def extract_from_paper(
    api: GeminiApi,
    paper_content: str,
    filename: str,
    compact: bool = False,
) -> Optional[dict]:
    if compact:
        prompt = build_extraction_prompt_compact(paper_content, filename)
        schema = None  # compact mode doesn't use structured output
    else:
        prompt = build_extraction_prompt(paper_content, filename)
        schema = get_extraction_json_schema()

    system_prompt = get_system_prompt()

    try:
        response = api.request(
            prompt,
            json_mode=True,
            system_prompt=system_prompt,
            response_json_schema=schema,
        )
    except Exception as e:
        print(f"API request failed: {e}")
        return None

    try:
        extraction = json.loads(response)
    except json.JSONDecodeError:
        print("Initial JSON parse failed, attempting fix...")
        fixed = fix_json_with_llm(api, response)
        if not fixed:
            return None
        try:
            extraction = json.loads(fixed)
        except json.JSONDecodeError:
            print("JSON fix failed")
            return None

    validated = validate_full_extraction(extraction)
    return validated


def main():
    args = arg_parser()
    work_dir = Path(args.work_dir)

    name = "Combined"
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = work_dir / "Processed_PDFs" / f"{name}_Processed"

    # Fallback: check Combined_Processed if default dir is empty
    if not args.input_dir and not list(input_dir.glob("*.txt")):
        alt = work_dir / "Processed_PDFs" / "Combined_Processed"
        if alt.exists() and list(alt.glob("*.txt")):
            input_dir = alt

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            work_dir
            / "Processed_PDFs"
            / f"{name}_Extractions_{args.model_id}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing {args.model_id} API...")
    api = GeminiApi(
        model_id=args.model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )

    if args.single_file:
        files = [input_dir / args.single_file]
        if not files[0].exists():
            raise FileNotFoundError(f"File not found: {files[0]}")
    else:
        files = sorted(input_dir.glob("*.txt"))

    if not files:
        print(f"No .txt files found in {input_dir}")
        return

    # --- Gold standard filter ---
    # Auto-detect the CSV if not overridden via --single_file (single_file skips filter)
    if not args.single_file:
        gold_csv_path: Optional[Path] = None
        if args.gold_standard_csv:
            gold_csv_path = Path(args.gold_standard_csv)
        else:
            candidate = work_dir / "data" / "gold_standard.csv"
            if candidate.exists():
                gold_csv_path = candidate
        if gold_csv_path is not None:
            gold_ids = load_gold_standard_ids(gold_csv_path)
            # Expand gold_ids with aliased numeric IDs so that files like
            # 2026.iclr-main.3-christan-steffen.txt are kept when the gold
            # standard CSV uses the title form (2026.iclr-main.TikZill).
            for alias_from, alias_to in _PAPER_ID_ALIASES.items():
                if alias_to in gold_ids:
                    gold_ids.add(alias_from)
            before = len(files)
            files = [
                f for f in files
                if _base_id(f.stem) in gold_ids
            ]
            print(
                f"Gold standard filter: {len(files)}/{before} files kept "
                f"({before - len(files)} skipped, {len(gold_ids)} unique IDs in CSV)"
            )
        else:
            print("No gold standard CSV found; processing all files.")
    # ----------------------------

    print(f"Found {len(files)} files to process")

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    stats_lock = threading.Lock()

    def process_file(file_path: Path) -> tuple[str, str]:
        filename = file_path.stem
        output_path = output_dir / f"{filename}_annotations.json"

        if args.skip_existing and output_path.exists():
            with stats_lock:
                stats["skipped"] += 1
            return "skipped", filename

        paper_content = file_path.read_text(encoding="utf-8", errors="replace")
        if args.no_chunking:
            paper_content = truncate_paper(paper_content, max_tokens=30000)
        else:
            original_chars = len(paper_content)
            if args.chunk_tokens > 4000:
                paper_content = extract_annotation_chunks_embedding(
                    paper_content, max_tokens=args.chunk_tokens
                )
            else:
                paper_content = extract_annotation_chunks(
                    paper_content, max_tokens=args.chunk_tokens
                )
            reduction = 100 * (1 - len(paper_content) / max(original_chars, 1))
            print(
                f"[chunk] {filename}: {len(paper_content)//4} tokens "
                f"(reduced {reduction:.0f}% vs full paper)"
            )

        extraction = extract_from_paper(
            api, paper_content, filename, compact=args.compact_prompt
        )

        if extraction is None:
            with stats_lock:
                stats["failed"] += 1
            return "failed", filename

        output_path.write_text(
            json.dumps(extraction, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with stats_lock:
            stats["processed"] += 1
        return "processed", filename

    num_workers = args.num_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_file, fp): fp for fp in files}
        with tqdm(total=len(files), desc="Extracting", unit="file") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                status, fname = future.result()
                pbar.set_postfix(status=status, file=fname[:40])
                pbar.update(1)

    print(f"\nDone: {stats['processed']} processed, {stats['skipped']} skipped, {stats['failed']} failed")


if __name__ == "__main__":
    main()
