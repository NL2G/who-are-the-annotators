"""
LLM-Human Agreement Pipeline — GPT-as-Judge (Gold Standard CSV).

Instead of deterministic field comparison, this module sends both the
human gold-standard annotations and the LLM extractions to Azure OpenAI
GPT and asks it to:
  1. Match each LLM experiment to the closest human experiment (by subtask).
  2. Judge each field in the matched pair as "correct" or "wrong".
  3. Return a structured JSON result.

Agreement is then aggregated across all papers.

Usage
-----
    python llm_human_agreement_gpt_judge.py \\
        --work_dir . \\
        --gold_csv data/gold_standard.csv \\
        --gemini_model_id gemini-3.1-pro-preview

    # Override the Azure model used for judging:
    python llm_human_agreement_gpt_judge.py \\
        --work_dir . \\
        --gold_csv data/gold_standard.csv \\
        --judge_model_id gpt-4.1
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from azure_openai_api import AzureOpenAIApi, AZURE_ENGINE_NAME
from download_papers import url_to_stem

# =============================================================================
# CONSTANTS
# =============================================================================

# Fields present in the gold CSV (in evaluation order).
# disagreement_resolution_adjudication is NOT in the gold CSV — skip it.
EVAL_FIELDS = [
    "nlp_task_type",
    "guidelines_released",
    "iaa_metric_name",
    "iaa_value",
    "annotators_per_item",
    "total_annotators",
    "total_annotated_items",
    "items_per_annotator",
    "recruitment",
    "crowd_qualifications_screening",
    "annotator_training",
    "language_proficiency",
    "expertise_level",
    "reported_compensation",
    "payment_rate",
    "age_reported",
    "gender_reported",
    "nation_origin_reported",
    "nation_residence_reported",
    "education_reported",
    "political_orientation_reported",
    "quality_control_filtering",
]

# Gold CSV column → internal field name
GOLD_COL_TO_FIELD: Dict[str, str] = {
    "Task type":                          "nlp_task_type",
    "Subtask name":                       "subtask_name",
    "Annotation guidelines released":     "guidelines_released",
    "IAA metric name":                    "iaa_metric_name",
    "IAA value":                          "iaa_value",
    "Number of annotators per item":      "annotators_per_item",
    "Total number of annotators":         "total_annotators",
    "Total number of annotated items":    "total_annotated_items",
    "Number of items per annotator":      "items_per_annotator",
    "Recruitment":                        "recruitment",
    "Crowd qualifications & screening":   "crowd_qualifications_screening",
    "Annotator training":                 "annotator_training",
    "Annotation task language proficiency": "language_proficiency",
    "Level of expertise":                 "expertise_level",
    "Reported compensation":              "reported_compensation",
    "Payment rate":                       "payment_rate",
    "Age":                                "age_reported",
    "Gender":                             "gender_reported",
    "Nation of origin":                   "nation_origin_reported",
    "Nation of residence":                "nation_residence_reported",
    "Education":                          "education_reported",
    "Political orientation":              "political_orientation_reported",
    "Quality control & filtering":        "quality_control_filtering",
}

# =============================================================================
# GOLD CSV LOADING
# =============================================================================

def _detect_link_col(df: pd.DataFrame) -> str:
    """Return the name of the Paper Link column."""
    for col in df.columns:
        if col.strip().lower() in ("paper link", "paperlink", "paper_link"):
            return col
    for col in df.columns:
        if "paper" in col.lower() and "link" in col.lower():
            return col
    raise ValueError(
        f"No 'Paper Link' column found. Available: {list(df.columns)}"
    )


def load_gold_csv(csv_path: Path) -> pd.DataFrame:
    """Load the new gold standard CSV into a DataFrame."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="warn")
    df = df.dropna(how="all")
    return df


def _row_to_exp(row: pd.Series, col_map: Dict[str, str]) -> Dict[str, Any]:
    """Convert one gold CSV row to an experiment dict."""
    exp: Dict[str, Any] = {}
    for csv_col, field in col_map.items():
        val = row.get(csv_col, None)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            exp[field] = str(val).strip()
        else:
            exp[field] = "Not reported"
    return exp


def get_human_exps_for_url(
    df: pd.DataFrame,
    url: str,
    link_col: str,
    col_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Return all human experiment dicts for a paper URL."""
    mask = df[link_col].astype(str).str.strip() == url.strip()
    rows = df[mask]
    exps = []
    for _, row in rows.iterrows():
        exp = _row_to_exp(row, col_map)
        exps.append(exp)
    return exps


# =============================================================================
# LLM EXTRACTION LOADING
# =============================================================================

def load_llm_extraction(url: str, extractions_dir: Path) -> Optional[dict]:
    """Load the Gemini extraction JSON for a paper URL."""
    stem = url_to_stem(url)
    # Try exact stem, then with common suffixes
    candidates = [
        extractions_dir / f"{stem}_annotations.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                print(f"  Failed to load {path.name}: {e}")
                return None
    # Fuzzy: look for any file whose stem starts with our stem
    matches = list(extractions_dir.glob(f"{stem}*_annotations.json"))
    if matches:
        try:
            return json.loads(matches[0].read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            print(f"  Failed to load {matches[0].name}: {e}")
    return None


def get_llm_experiments(extraction: dict) -> List[Dict[str, Any]]:
    """Extract the list of experiments from a Gemini extraction result."""
    exps = extraction.get("experiments") or extraction.get("annotations") or []
    if isinstance(exps, list):
        return exps
    return []


# =============================================================================
# GPT JUDGE PROMPT
# =============================================================================

_SYSTEM_PROMPT = """\
You are a precise evaluator of automated annotation-metadata extraction quality.

You will receive:
  - HUMAN GOLD STANDARD ANNOTATIONS for a paper (the ground truth, list of experiments).
  - LLM EXTRACTED ANNOTATIONS for the same paper (what you are evaluating).

Your task has two parts:
  1. MATCH  – For each LLM experiment, find the best matching human experiment based
              on subtask_name and nlp_task_type similarity. Matching is 1-to-1.
              If an LLM experiment has no reasonable match, skip it.
  2. JUDGE  – For each matched pair, evaluate every field as "correct" or "wrong".

Field evaluation guidelines:
  - nlp_task_type      : "correct" if semantically equivalent (ignore extra description text
                         in parentheses such as "(Binary, multi-class…)").
  - subtask_name       : "correct" if both describe the same annotation task (minor wording OK).
  - guidelines_released: exact Yes/No match required.
  - iaa_metric_name    : "correct" if the set of metric names substantially overlaps.
                         Accept Unicode equivalents (κ = kappa, α = alpha).
                         Order does not matter.
  - iaa_value          : "correct" if the numeric values substantially overlap
                         (pipe-separated human values vs. list LLM values both accepted).
                         Allow rounding differences up to 0.01. "Not reported" = "Not reported".
  - Workload fields    : "correct" if numerically equivalent.
                         "15k" = "15000" = "15K"; "2,864" = "2864"; ranges like "2-3" must match.
                         "Not reported" must equal "Not reported".
  - recruitment        : "correct" if canonical category matches
                         (e.g. "Crowdsourcing" covers all crowdsourcing platform names).
  - crowd_qualifications_screening, annotator_training, quality_control_filtering:
                         exactly Yes / No / NA / Not reported.
  - language_proficiency, expertise_level: "correct" if semantically equivalent.
  - reported_compensation: "correct" if Paid / Free / Not reported matches.
  - payment_rate       : "correct" if Specific numeric rate / General mention / NA matches.
  - Demographic fields (age_reported, gender_reported, nation_origin_reported,
    nation_residence_reported, education_reported, political_orientation_reported):
                         Yes / No match required (treat "Not reported" as "No").

Return ONLY valid JSON — no markdown fences, no explanatory text outside the JSON.
"""

_FIELDS_LIST = ", ".join(f'"{f}"' for f in EVAL_FIELDS)


def build_judge_prompt(
    paper_url: str,
    human_exps: List[Dict[str, Any]],
    llm_exps: List[Dict[str, Any]],
) -> str:
    """Build the user-turn prompt for the GPT judge."""
    human_json = json.dumps(human_exps, indent=2, ensure_ascii=False)
    llm_json = json.dumps(llm_exps, indent=2, ensure_ascii=False)

    fields_schema = "\n".join(
        f'        "{f}": "correct" or "wrong"' for f in EVAL_FIELDS
    )

    return f"""\
PAPER URL: {paper_url}

## HUMAN GOLD STANDARD ANNOTATIONS ({len(human_exps)} experiment(s), indexed 0–{len(human_exps) - 1})
{human_json}

## LLM EXTRACTED ANNOTATIONS ({len(llm_exps)} experiment(s), indexed 0–{len(llm_exps) - 1})
{llm_json}

## REQUIRED OUTPUT FORMAT
Return a JSON object exactly as shown below.
For every matched pair include all {len(EVAL_FIELDS)} fields listed.
{{
  "matches": [
    {{
      "llm_exp_idx": <integer>,
      "human_exp_idx": <integer>,
      "fields": {{
{fields_schema}
      }}
    }}
  ]
}}
"""


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_judge_response(raw: str) -> dict:
    """Parse the GPT judge JSON response, with fallback strip of markdown fences."""
    text = raw.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [warn] Failed to parse judge response: {e}")
        return {"matches": []}


def validate_judge_result(
    result: dict,
    n_llm: int,
    n_human: int,
) -> List[dict]:
    """Validate and filter the matches list from a judge response."""
    matches = result.get("matches", [])
    valid = []
    for m in matches:
        llm_idx = m.get("llm_exp_idx")
        human_idx = m.get("human_exp_idx")
        fields = m.get("fields", {})
        if not isinstance(llm_idx, int) or not isinstance(human_idx, int):
            continue
        if llm_idx < 0 or llm_idx >= n_llm:
            continue
        if human_idx < 0 or human_idx >= n_human:
            continue
        if not isinstance(fields, dict):
            continue
        # Normalise field verdicts to lowercase "correct"/"wrong"
        normalised = {}
        for f in EVAL_FIELDS:
            raw_v = str(fields.get(f, "wrong")).strip().lower()
            normalised[f] = "correct" if raw_v == "correct" else "wrong"
        valid.append({
            "llm_exp_idx": llm_idx,
            "human_exp_idx": human_idx,
            "fields": normalised,
        })
    return valid


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_results(paper_results: List[dict]) -> Dict[str, Any]:
    """Aggregate field-level correctness across all paper results."""
    field_matches: Dict[str, int] = defaultdict(int)
    field_totals: Dict[str, int] = defaultdict(int)
    total_correct = 0
    total_all = 0

    for res in paper_results:
        for match in res.get("matches", []):
            for field, verdict in match.get("fields", {}).items():
                if field not in EVAL_FIELDS:
                    continue
                field_totals[field] += 1
                if verdict == "correct":
                    field_matches[field] += 1
                    total_correct += 1
                total_all += 1

    field_agreement: Dict[str, dict] = {}
    for field in EVAL_FIELDS:
        t = field_totals[field]
        m = field_matches[field]
        field_agreement[field] = {
            "matches": m,
            "total": t,
            "rate": round(m / t, 4) if t > 0 else 0.0,
        }

    return {
        "total_pairs": sum(len(r.get("matches", [])) for r in paper_results),
        "overall_matches": total_correct,
        "overall_total": total_all,
        "overall_rate": round(total_correct / total_all, 4) if total_all > 0 else 0.0,
        "field_agreement": field_agreement,
    }


def print_summary(stats: Dict[str, Any]) -> None:
    """Print a human-readable agreement summary."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  AGREEMENT SUMMARY (GPT-as-Judge vs Gold Standard)")
    print(sep)
    print(
        f"  Overall: {stats['overall_matches']}/{stats['overall_total']} "
        f"= {stats['overall_rate']:.1%}"
    )
    print(f"  Matched pairs: {stats['total_pairs']}")
    print()
    print(f"  {'Field':<40}  {'Rate':>7}  {'Match/Total':>12}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*12}")
    for field, fa in stats["field_agreement"].items():
        bar = f"{fa['matches']}/{fa['total']}"
        print(f"  {field:<40}  {fa['rate']:>6.1%}  {bar:>12}")
    print(sep)


# =============================================================================
# MAIN
# =============================================================================

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPT-as-Judge agreement between Gemini extractions and gold standard CSV "
                    "(uses Azure OpenAI Batch API)"
    )
    p.add_argument("--work_dir", required=True, help="Workspace root (Who_Are_Annotators folder)")
    p.add_argument(
        "--gold_csv",
        default=None,
        help="Path to the new gold standard CSV with Paper Link column "
             "(default: data/gold_standard.csv)",
    )
    p.add_argument(
        "--gemini_model_id",
        default="gemini-3.1-pro-preview",
        help="Gemini model ID used during extraction (determines extraction directory name)",
    )
    p.add_argument(
        "--llm_extractions_dir",
        default=None,
        help="Override path to LLM extractions directory",
    )
    p.add_argument(
        "--judge_model_id",
        default=None,
        help="Azure OpenAI model used for field-level judging "
             "(default: AZURE_ENGINE_NAME env var or gpt-4.1)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory for per-paper JSON results and aggregate stats",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for GPT judge calls (default: 0.0 for determinism)",
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
        help="Submit the batch job and exit without waiting. "
             "Use --resume <BATCH_ID> to collect results later.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="BATCH_ID",
        help="Resume a previously submitted batch job by its batch_id.",
    )
    return p.parse_args()


# =============================================================================
# RESULT PROCESSING HELPER
# =============================================================================

def _process_batch_results(
    id_to_result: Dict[str, Any],
    task_meta: List[Dict[str, Any]],
    output_dir: Path,
    gemini_model_id: str,
    judge_model_id: str,
) -> List[dict]:
    """Parse raw batch results, validate, save per-paper JSONs, return all_paper_results."""
    all_paper_results: List[dict] = []
    stats = {"processed": 0, "skipped_no_result": 0, "failed_parse": 0}

    for meta in tqdm(task_meta, desc="Processing judge results"):
        url = meta["url"]
        stem = meta["stem"]
        human_exps = meta["human_exps"]
        llm_exps = meta["llm_exps"]
        cid = meta["custom_id"]

        result = id_to_result.get(cid)
        if result is None:
            print(f"  [warn] No batch result for {cid}")
            stats["skipped_no_result"] += 1
            continue

        raw_response = AzureOpenAIApi.parse_batch_response(result)
        if not raw_response:
            print(f"  [warn] Empty/error response for {cid} "
                  f"(status={result.get('response', {}).get('status_code')})")
            stats["failed_parse"] += 1
            continue

        parsed = parse_judge_response(raw_response)
        valid_matches = validate_judge_result(parsed, len(llm_exps), len(human_exps))

        paper_result = {
            "url": url,
            "paper_stem": stem,
            "n_human_exps": len(human_exps),
            "n_llm_exps": len(llm_exps),
            "matches": valid_matches,
            "llm_model": gemini_model_id,
            "judge_model": judge_model_id,
        }

        out_path = output_dir / f"{stem}_judge_result.json"
        out_path.write_text(
            json.dumps(paper_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        all_paper_results.append(paper_result)
        stats["processed"] += 1
        print(
            f"  {stem}: {len(valid_matches)} matched pairs "
            f"(human={len(human_exps)}, llm={len(llm_exps)})"
        )

    print(
        f"\nBatch processing: {stats['processed']} processed, "
        f"{stats['skipped_no_result']} missing results, "
        f"{stats['failed_parse']} parse failures"
    )
    return all_paper_results


def _aggregate_and_save(
    all_paper_results: List[dict],
    output_dir: Path,
    gemini_model_id: str,
    judge_model_id: str,
) -> None:
    """Aggregate results and write summary files."""
    if not all_paper_results:
        print("No results to aggregate.")
        return

    agg = aggregate_results(all_paper_results)
    agg["paper_count"] = len(all_paper_results)
    agg["gemini_model"] = gemini_model_id
    agg["judge_model"] = judge_model_id

    agg_path = output_dir / "agreement_stats.json"
    agg_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAggregate stats saved to: {agg_path}")

    all_results_path = output_dir / "all_paper_results.json"
    all_results_path.write_text(
        json.dumps(all_paper_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_summary(agg)


def main() -> None:
    args = arg_parser()
    work_dir = Path(args.work_dir)

    # ── Resolve paths ─────────────────────────────────────────────────────
    gold_csv_path = (
        Path(args.gold_csv)
        if args.gold_csv
        else work_dir / "data" / "gold_standard.csv"
    )

    extractions_dir = (
        Path(args.llm_extractions_dir)
        if args.llm_extractions_dir
        else work_dir / "Processed_PDFs" / f"Combined_Extractions_{args.gemini_model_id}"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else work_dir / "Processed_PDFs" / f"Combined_Agreement_gpt_judge_{args.gemini_model_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_model_id = args.judge_model_id or AZURE_ENGINE_NAME or "gpt-4.1"

    api = AzureOpenAIApi(
        model_id=judge_model_id,
        temperature=args.temperature,
        max_tokens=4096,
    )

    # ── RESUME MODE ───────────────────────────────────────────────────────
    if args.resume:
        batch_id = args.resume
        print(f"\n=== Resuming batch {batch_id} ===")

        results_path = output_dir / "judge_results.jsonl"
        id_to_result = api.resume_batch(
            batch_id, results_path, use_blob=args.use_blob
        )

        meta_path = output_dir / "judge_task_meta.json"
        if not meta_path.exists():
            print(f"ERROR: task metadata not found at {meta_path}. "
                  "Cannot process results without it.")
            return

        task_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(task_meta)} task metadata entries, "
              f"{len(id_to_result)} batch results")

        all_paper_results = _process_batch_results(
            id_to_result, task_meta, output_dir, args.gemini_model_id, judge_model_id
        )
        _aggregate_and_save(
            all_paper_results, output_dir, args.gemini_model_id, judge_model_id
        )
        return

    # ── NORMAL FLOW ───────────────────────────────────────────────────────
    if not gold_csv_path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {gold_csv_path}")

    print(f"Gold CSV          : {gold_csv_path}")
    print(f"Extractions dir   : {extractions_dir}")
    print(f"Output dir        : {output_dir}")
    print(f"Judge model       : {judge_model_id}")
    print(f"Gemini model      : {args.gemini_model_id}")

    # ── Load gold CSV ─────────────────────────────────────────────────────
    print("\nLoading gold CSV …")
    gold_df = load_gold_csv(gold_csv_path)
    link_col = _detect_link_col(gold_df)
    print(f"  {len(gold_df)} rows, {len(gold_df[link_col].unique())} unique papers")

    # ── Build column mapping ──────────────────────────────────────────────
    actual_cols = {c.strip(): c for c in gold_df.columns}
    col_map: Dict[str, str] = {}
    for canonical_col, field in GOLD_COL_TO_FIELD.items():
        if canonical_col in actual_cols:
            col_map[actual_cols[canonical_col]] = field
        else:
            lc = canonical_col.lower()
            for c in actual_cols:
                if c.lower() == lc:
                    col_map[actual_cols[c]] = field
                    break

    # ── Phase 1: Build batch requests ────────────────────────────────────
    print("\n=== Phase 1: Building batch requests ===")
    unique_urls = list(
        dict.fromkeys(
            str(u).strip()
            for u in gold_df[link_col].dropna().tolist()
            if str(u).strip().startswith("http")
        )
    )

    batch_requests: List[Dict[str, Any]] = []
    task_meta: List[Dict[str, Any]] = []

    for url in tqdm(unique_urls, desc="Preparing requests"):
        human_exps = get_human_exps_for_url(gold_df, url, link_col, col_map)
        extraction = load_llm_extraction(url, extractions_dir)

        if extraction is None:
            print(f"  [skip] No extraction found for {url}")
            continue

        llm_exps = get_llm_experiments(extraction)
        if not llm_exps:
            print(f"  [skip] Empty extraction for {url}")
            continue

        stem = url_to_stem(url)
        cid = f"judge-{stem}"

        prompt = build_judge_prompt(url, human_exps, llm_exps)
        batch_requests.append(
            AzureOpenAIApi.build_batch_line(
                custom_id=cid,
                user_prompt=prompt,
                model=judge_model_id,
                temperature=args.temperature,
                max_tokens=4096,
                json_mode=True,
                system_prompt=_SYSTEM_PROMPT,
            )
        )
        task_meta.append({
            "custom_id": cid,
            "url": url,
            "stem": stem,
            "human_exps": human_exps,
            "llm_exps": llm_exps,
        })

    print(f"Prepared {len(batch_requests)} batch requests "
          f"({len(unique_urls) - len(batch_requests)} skipped)")

    if not batch_requests:
        print("No requests to submit.")
        return

    # Save task metadata so resume mode can use it
    meta_path = output_dir / "judge_task_meta.json"
    meta_path.write_text(
        json.dumps(task_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Task metadata saved -> {meta_path}")

    # ── Phase 2: Submit batch ─────────────────────────────────────────────
    print("\n=== Phase 2: Submitting batch ===")

    if args.submit_only:
        batch_id = api.submit_only(
            batch_requests,
            output_dir,
            prefix="judge",
            use_blob=args.use_blob,
            extra_manifest={
                "gemini_model_id": args.gemini_model_id,
                "judge_model_id": judge_model_id,
            },
        )
        print(f"\nBatch submitted: {batch_id}")
        print(f"Resume with:  python llm_human_agreement_gpt_judge.py "
              f"--work_dir {args.work_dir} "
              f"--gemini_model_id {args.gemini_model_id} "
              f"--resume {batch_id}")
        return

    # Full run: submit + poll + download + process
    results_path = output_dir / "judge_results.jsonl"
    id_to_result = api.run_batch(
        batch_requests,
        output_dir,
        prefix="judge",
        use_blob=args.use_blob,
    )

    # ── Phase 3: Process results ──────────────────────────────────────────
    print("\n=== Phase 3: Processing results ===")
    all_paper_results = _process_batch_results(
        id_to_result, task_meta, output_dir, args.gemini_model_id, judge_model_id
    )
    _aggregate_and_save(
        all_paper_results, output_dir, args.gemini_model_id, judge_model_id
    )


if __name__ == "__main__":
    main()
