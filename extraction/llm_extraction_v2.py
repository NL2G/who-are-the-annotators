"""
LLM Extraction Script (v2) - Using OpenAI API.
Extracts human annotation information from papers using GPT-4.1 or GPT-5-mini.
Includes validation and post-processing.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from openai_api import OpenAIApi
from prompts_v2 import (
    build_extraction_prompt,
    build_extraction_prompt_compact,
    build_json_fix_prompt,
    get_system_prompt,
    truncate_paper,
)
from validation import validate_full_extraction


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract human annotation data from papers using OpenAI API"
    )
    parser.add_argument(
        "--work_dir", 
        required=True, 
        help="Workspace directory"
    )
    parser.add_argument(
        "--model_id", 
        default="gpt-4.1",
        help="Model ID (default: gpt-4.1, alternatives: gpt-5-mini, gpt-4-turbo)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.2, 
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.1, 
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8000,
        help="Maximum tokens in response"
    )
    parser.add_argument(
        "--compact_prompt",
        action="store_true",
        help="Use compact prompt for token-limited contexts"
    )
    parser.add_argument(
        "--input_dir",
        help="Override input directory (default: {work_dir}/Who_Are_Annotators/Processed_PDFs/Combined_Processed)"
    )
    parser.add_argument(
        "--output_dir",
        help="Override output directory"
    )
    parser.add_argument(
        "--single_file",
        help="Process only this single file (filename without path)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files that already have output"
    )
    return parser.parse_args()


def fix_json_with_llm(api: OpenAIApi, raw_json: str, max_retries: int = 3) -> str:
    """
    Attempt to fix malformed JSON using the LLM.
    
    Args:
        api: OpenAI API client
        raw_json: Broken JSON string
        max_retries: Maximum fix attempts
        
    Returns:
        Fixed JSON string, or empty string if unfixable
    """
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
    api: OpenAIApi,
    paper_content: str,
    filename: str,
    compact: bool = False,
) -> Optional[dict]:
    """
    Extract human annotation information from a paper.
    
    Args:
        api: OpenAI API client
        paper_content: Paper text content
        filename: Paper filename/ID
        compact: Use compact prompt
        
    Returns:
        Validated extraction dictionary, or None on failure
    """
    # Build prompt
    if compact:
        prompt = build_extraction_prompt_compact(paper_content, filename)
    else:
        prompt = build_extraction_prompt(paper_content, filename)
    
    # Get system prompt
    system_prompt = get_system_prompt()
    
    # Make API request with JSON mode
    try:
        response = api.request(
            prompt, 
            json_mode=True,
            system_prompt=system_prompt
        )
    except Exception as e:
        print(f"API request failed: {e}")
        return None
    
    # Parse JSON
    try:
        extraction = json.loads(response)
    except json.JSONDecodeError:
        # Try to fix JSON
        print(f"Initial JSON parse failed, attempting fix...")
        fixed = fix_json_with_llm(api, response)
        if not fixed:
            return None
        try:
            extraction = json.loads(fixed)
        except json.JSONDecodeError:
            print("JSON fix failed")
            return None
    
    # Validate and normalize
    validated = validate_full_extraction(extraction)
    
    return validated


def main():
    args = arg_parser()
    work_dir = Path(args.work_dir)
    
    # Set up directories
    name = "Combined"
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = work_dir / "Who_Are_Annotators" / "Processed_PDFs" / f"{name}_Processed"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = work_dir / "Who_Are_Annotators" / "Processed_PDFs" / f"{name}_Extractions_{args.model_id}"
    
    # Also check data/pdfs_processed if Combined_Processed is empty
    if not list(input_dir.glob("*.txt")) and not args.input_dir:
        alt = work_dir / "Who_Are_Annotators" / "Processed_PDFs" / "Combined_Processed"
        if alt.exists() and list(alt.glob("*.txt")):
            input_dir = alt
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize API
    print(f"Initializing {args.model_id} API...")
    api = OpenAIApi(
        model_id=args.model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    # Get files to process
    if args.single_file:
        files = [input_dir / args.single_file]
        if not files[0].exists():
            raise FileNotFoundError(f"File not found: {files[0]}")
    else:
        files = sorted(input_dir.glob("*.txt"))
    
    if not files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    
    for file_path in files:
        filename = file_path.stem
        output_path = output_dir / f"{filename}_annotations.json"
        
        # Skip if exists
        if args.skip_existing and output_path.exists():
            print(f"Skipping (exists): {filename}")
            stats["skipped"] += 1
            continue
        
        print(f"\nProcessing: {filename}")
        
        # Read paper content
        paper_content = file_path.read_text(encoding="utf-8", errors="replace")
        
        # Truncate if too long
        paper_content = truncate_paper(paper_content, max_tokens=20000)
        
        # Extract
        extraction = extract_from_paper(
            api, 
            paper_content, 
            filename,
            compact=args.compact_prompt,
        )
        
        if extraction is None:
            print(f"FAILED: {filename}")
            stats["failed"] += 1
            continue
        
        # Save
        output_path.write_text(
            json.dumps(extraction, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        n_experiments = len(extraction.get("experiments", []))
        skip_status = " (SKIPPED)" if extraction.get("skip", False) else ""
        print(f"SUCCESS: {filename} - {n_experiments} experiments{skip_status}")
        stats["processed"] += 1
    
    # Summary
    print(f"\n=== Extraction Complete ===")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
