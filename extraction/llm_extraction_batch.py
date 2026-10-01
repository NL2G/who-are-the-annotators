# /// script
# dependencies = [
#     "tqdm",
#     "openai>=1.0.0",
#     "python-dotenv",
#     "azure-storage-blob",
#     "azure-identity",
# ]
# ///

"""
LLM Extraction Script (Batch) - Azure OpenAI Batch API.

Instead of processing papers one-by-one synchronously, this script:
1. Reads all paper text files
2. Builds one JSONL with all extraction prompts
3. Submits the JSONL as an Azure Batch job (via Azure Blob Storage)
4. Polls until completion
5. Downloads results, validates, and saves per-paper JSON files

Usage:
    python llm_extraction_batch.py \\
        --work_dir /path/to/workspace \\
        --model_id gpt-4.1 \\
        --use_blob
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from azure_openai_api import AzureOpenAIApi
from prompts_v2 import (
    build_extraction_prompt,
    build_extraction_prompt_compact,
    build_json_fix_prompt,
    get_system_prompt,
    truncate_paper,
)
from validation import validate_full_extraction


# =============================================================================
# CLI
# =============================================================================


def arg_parser():
    p = argparse.ArgumentParser(
        description="Batch extraction of annotation data via Azure OpenAI Batch API"
    )
    p.add_argument("--work_dir", required=True, help="Workspace root directory")
    p.add_argument(
        "--model_id",
        default=None,
        help="Azure deployment / model name (default: AZURE_ENGINE_NAME env var)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature"
    )
    p.add_argument(
        "--top_p", type=float, default=0.1, help="Nucleus sampling parameter"
    )
    p.add_argument(
        "--max_tokens", type=int, default=10000, help="Max response tokens"
    )
    p.add_argument(
        "--compact_prompt",
        action="store_true",
        help="Use compact prompt for shorter context windows",
    )
    p.add_argument("--input_dir", help="Override input directory")
    p.add_argument("--output_dir", help="Override output directory")
    p.add_argument(
        "--single_file", help="Process only this filename (no path)"
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip papers that already have an output JSON",
    )
    p.add_argument(
        "--use_blob",
        action="store_true",
        default=False,
        help="Use Azure Blob Storage for batch I/O (recommended for large jobs)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Split into sub-batches of this size (0 = single batch)",
    )
    p.add_argument(
        "--sync",
        action="store_true",
        help="Fall back to synchronous per-paper calls instead of batch API",
    )
    p.add_argument(
        "--submit_only",
        action="store_true",
        help="Submit the batch job and exit without waiting for results. "
             "Prints the batch_id so you can resume later with --resume.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="BATCH_ID",
        help="Resume a previously submitted batch job. "
             "Pass the batch_id printed by --submit_only.",
    )
    return p.parse_args()


# =============================================================================
# BATCH REQUEST BUILDER
# =============================================================================


def build_extraction_request(
    api: AzureOpenAIApi,
    paper_content: str,
    filename: str,
    compact: bool = False,
) -> Dict:
    """Build a single batch-request dict for one paper."""
    paper_content = truncate_paper(paper_content, max_tokens=20000)

    if compact:
        prompt = build_extraction_prompt_compact(paper_content, filename)
    else:
        prompt = build_extraction_prompt(paper_content, filename)

    system_prompt = get_system_prompt()

    return AzureOpenAIApi.build_batch_line(
        custom_id=f"extract-{filename}",
        user_prompt=prompt,
        model=api.model_id,
        temperature=api.temperature,
        top_p=api.top_p,
        max_tokens=api.max_tokens,
        json_mode=True,
        system_prompt=system_prompt,
    )


# =============================================================================
# SYNCHRONOUS FALLBACK (same as llm_extraction_v2.py)
# =============================================================================


def fix_json_with_llm(api: AzureOpenAIApi, raw_json: str, max_retries: int = 3) -> str:
    """Attempt to fix malformed JSON using the LLM."""
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


def extract_sync(
    api: AzureOpenAIApi,
    paper_content: str,
    filename: str,
    compact: bool = False,
) -> Optional[dict]:
    """Synchronous per-paper extraction (same logic as llm_extraction_v2)."""
    paper_content = truncate_paper(paper_content, max_tokens=20000)
    prompt = (
        build_extraction_prompt_compact(paper_content, filename)
        if compact
        else build_extraction_prompt(paper_content, filename)
    )
    system_prompt = get_system_prompt()

    try:
        response = api.request(prompt, json_mode=True, system_prompt=system_prompt)
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

    return validate_full_extraction(extraction)


# =============================================================================
# BATCH RESULT PROCESSING
# =============================================================================


def process_batch_results(
    id_to_result: Dict[str, dict],
    filename_map: Dict[str, Path],
    output_dir: Path,
    api: Optional[AzureOpenAIApi] = None,
) -> Dict[str, str]:
    """
    Process batch results: parse JSON, validate, save per-paper outputs.

    Args:
        id_to_result: {custom_id: batch_result_dict}
        filename_map: {custom_id: original_txt_path}
        output_dir: Where to save per-paper JSON
        api: Optional API for JSON-fix retries

    Returns:
        Dict with stats {processed, failed}
    """
    stats = {"processed": 0, "failed": 0}

    for custom_id, txt_path in tqdm(
        filename_map.items(), desc="Processing results"
    ):
        filename = txt_path.stem
        output_path = output_dir / f"{filename}_annotations.json"

        result = id_to_result.get(custom_id)
        raw_content = AzureOpenAIApi.parse_batch_response(result)

        if raw_content is None:
            print(f"FAILED (no content): {filename}")
            stats["failed"] += 1
            continue

        # Parse JSON
        try:
            extraction = json.loads(raw_content)
        except json.JSONDecodeError:
            # Try LLM fix if API is available
            if api:
                print(f"JSON parse failed for {filename}, attempting fix...")
                fixed = fix_json_with_llm(api, raw_content, max_retries=2)
                if fixed:
                    try:
                        extraction = json.loads(fixed)
                    except json.JSONDecodeError:
                        print(f"FAILED (unfixable JSON): {filename}")
                        stats["failed"] += 1
                        continue
                else:
                    print(f"FAILED (unfixable JSON): {filename}")
                    stats["failed"] += 1
                    continue
            else:
                print(f"FAILED (invalid JSON, no API for fix): {filename}")
                stats["failed"] += 1
                continue

        # Validate
        validated = validate_full_extraction(extraction)

        # Save
        output_path.write_text(
            json.dumps(validated, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        n_exp = len(validated.get("experiments", []))
        skip = " (SKIPPED)" if validated.get("skip", False) else ""
        print(f"OK: {filename} - {n_exp} experiments{skip}")
        stats["processed"] += 1

    return stats


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = arg_parser()
    work_dir = Path(args.work_dir)
    name = "Combined"

    # Resolve model name from env if not provided via CLI
    from azure_openai_api import AZURE_ENGINE_NAME
    model_id = args.model_id or AZURE_ENGINE_NAME or "gpt-4.1"

    # Directories
    input_dir = (
        Path(args.input_dir)
        if args.input_dir
        else work_dir
        / "Who_Are_Annotators"
        / "Processed_PDFs"
        / f"{name}_Processed"
    )
    # Fallback: check Combined_Processed if default dir is empty
    if not args.input_dir and not list(input_dir.glob("*.txt")):
        alt = work_dir / "Who_Are_Annotators" / "Processed_PDFs" / "Combined_Processed"
        if alt.exists() and list(alt.glob("*.txt")):
            input_dir = alt
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else work_dir
        / "Who_Are_Annotators"
        / "Processed_PDFs"
        / f"{name}_Extractions_{model_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # API
    print(f"Initializing Azure OpenAI ({model_id})...")
    api = AzureOpenAIApi(
        model_id=model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Files
    if args.single_file:
        files = [input_dir / args.single_file]
        if not files[0].exists():
            raise FileNotFoundError(f"File not found: {files[0]}")
    else:
        files = sorted(input_dir.glob("*.txt"))

    if not files:
        print(f"No .txt files found in {input_dir}")
        return

    # Filter already-processed
    to_process: List[Path] = []
    for fp in files:
        out = output_dir / f"{fp.stem}_annotations.json"
        if args.skip_existing and out.exists():
            continue
        to_process.append(fp)

    print(
        f"Found {len(files)} files, {len(to_process)} to process "
        f"({len(files) - len(to_process)} skipped)"
    )

    if not to_process and not args.resume:
        print("Nothing to do.")
        return

    # ── resume mode ───────────────────────────────────────────────────────
    if args.resume:
        batch_id = args.resume
        print(f"\n=== Resuming batch {batch_id} ===")

        # Rebuild filename_map from input_dir (all .txt files)
        filename_map = {}
        for txt in sorted(input_dir.glob("*.txt")):
            cid = f"extract-{txt.stem}"
            filename_map[cid] = txt

        results_path = output_dir / "extraction_results.jsonl"
        id_to_result = api.resume_batch(
            batch_id, results_path, use_blob=args.use_blob
        )

        print(f"\n=== Processing {len(id_to_result)} results ===")
        stats = process_batch_results(id_to_result, filename_map, output_dir, api=api)

        print("\n=== Extraction Complete ===")
        print(f"Processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Output: {output_dir}")
        return

    # ── synchronous fallback ──────────────────────────────────────────────
    if args.sync:
        print("\n=== Running in synchronous mode ===")
        stats = {"processed": 0, "skipped": 0, "failed": 0}
        for fp in tqdm(to_process, desc="Extracting"):
            filename = fp.stem
            paper = fp.read_text(encoding="utf-8", errors="replace")
            extraction = extract_sync(api, paper, filename, compact=args.compact_prompt)
            if extraction is None:
                stats["failed"] += 1
                continue
            out = output_dir / f"{filename}_annotations.json"
            out.write_text(
                json.dumps(extraction, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            stats["processed"] += 1
        print(f"\nDone: {stats}")
        return

    # ── batch mode ────────────────────────────────────────────────────────
    print("\n=== Building batch requests ===")

    all_requests = []
    filename_map: Dict[str, Path] = {}  # custom_id -> original Path

    for fp in tqdm(to_process, desc="Building prompts"):
        paper = fp.read_text(encoding="utf-8", errors="replace")
        req = build_extraction_request(
            api, paper, fp.stem, compact=args.compact_prompt
        )
        all_requests.append(req)
        filename_map[req["custom_id"]] = fp

    print(f"Total batch requests: {len(all_requests)}")

    # Optional sub-batching
    batches: List[List[Dict]] = []
    if args.batch_size and args.batch_size > 0:
        for i in range(0, len(all_requests), args.batch_size):
            batches.append(all_requests[i : i + args.batch_size])
        print(f"Split into {len(batches)} sub-batches of up to {args.batch_size}")
    else:
        batches = [all_requests]

    # Execute batches
    all_results: Dict[str, dict] = {}

    if args.submit_only:
        # Submit and exit — save manifest for each sub-batch
        for batch_idx, batch in enumerate(batches):
            prefix = f"extraction_batch{batch_idx}"
            print(f"\n=== Submitting batch {batch_idx + 1}/{len(batches)} ({len(batch)} requests) ===")
            api.submit_only(
                batch,
                output_dir=output_dir,
                prefix=prefix,
                use_blob=args.use_blob,
                extra_manifest={"filename_map": {r['custom_id']: str(filename_map[r['custom_id']]) for r in batch}},
            )
        print("\nAll batches submitted. Run again with --resume <batch_id> to collect results.")
        return

    for batch_idx, batch in enumerate(batches):
        prefix = f"extraction_batch{batch_idx}"
        print(f"\n=== Submitting batch {batch_idx + 1}/{len(batches)} ({len(batch)} requests) ===")
        results = api.run_batch(
            batch,
            output_dir=output_dir,
            prefix=prefix,
            use_blob=args.use_blob,
        )
        all_results.update(results)

    # Process results
    print(f"\n=== Processing {len(all_results)} results ===")
    stats = process_batch_results(all_results, filename_map, output_dir, api=api)

    print("\n=== Extraction Complete ===")
    print(f"Processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
