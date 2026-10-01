"""
LLM vs Individual Human Annotators — Inter-Annotator Agreement.

The paper compares LLM-vs-gold (alpha=0.606) to human-vs-human,
pre-adjudication (alpha=0.585) and calls the LLM "human-comparable". A
reviewer pointed out these two alphas have different reference points.
human_gold_iaa.py addressed this by comparing individual humans to the
SAME adjudicated gold used for the LLM — but the gold standard is itself
built from these annotators' own judgments (via discussion), so human-vs-
gold agreement is somewhat structurally inflated relative to a fully
independent rater.

This script sidesteps the gold standard entirely: it compares the LLM's
raw extraction directly against each individual human annotator's raw
(pre-adjudication) extraction, on the same papers, using the same
independent-rater structure as the original human-vs-human alpha (0.585).
The LLM is treated as "just another annotator" — neither side had any
part in producing the other's answers or the gold standard.

Usage:
    uv run human_llm_iaa.py
    uv run human_llm_iaa.py --extractions-dir Processed_PDFs/Combined_Extractions_gpt-5-batch
    uv run human_llm_iaa.py --output llm_vs_human_results.json
"""
# /// script
# dependencies = ["pandas"]
# ///

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from human_llm_agreement import (
    FIELDS,
    _GOLD_COL_MAP,
    _LIST_FIELDS,
    _norm,
    _split_list_value,
    _match_experiments,
    _detect_link_col,
    load_llm_extraction,
    _krippendorff_alpha_nominal,
)

GOLD_CSV_DEFAULT = "data/gold_standard.csv"
DOUBLE_CSV_DEFAULT = "data/double_annotated.csv"
EXTRACTIONS_DIR_DEFAULT = "Processed_PDFs/Combined_Extractions_gpt-4.1-global-batch"


def _build_col_map(headers: List[str]) -> Dict[str, str]:
    header_lc = {h.strip().lower(): h.strip() for h in headers}
    col_map: Dict[str, str] = {}
    for canonical, field in _GOLD_COL_MAP.items():
        if canonical in header_lc.values():
            col_map[canonical] = field
        elif canonical.lower() in header_lc:
            col_map[header_lc[canonical.lower()]] = field
    return col_map


def load_paperid_to_url(gold_csv_path: Path) -> Dict[str, str]:
    """Map PaperID -> Paper Link URL, using the adjudicated gold CSV (has both columns)."""
    text = gold_csv_path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    headers = list(reader.fieldnames or [])
    link_col = _detect_link_col(headers)

    mapping: Dict[str, str] = {}
    for row in reader:
        pid = (row.get("PaperID") or "").strip()
        url = (row.get(link_col) or "").strip()
        if pid and url:
            mapping[pid] = url
    return mapping


def load_double_csv_by_annotator(path: Path) -> Dict[str, Dict[str, List[dict]]]:
    """Load the raw double-annotated CSV -> {annotator: {paper_id: [row_dict, ...]}}."""
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    headers = list(reader.fieldnames or [])
    col_map = _build_col_map(headers)

    by_annotator: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in reader:
        annotator = (row.get("annotator") or row.get("Annotator") or "").strip().lower().capitalize()
        paper_id = (row.get("PaperID") or row.get("PaperID: task") or "").strip()
        if not annotator or not paper_id:
            continue
        rec: dict = {}
        for csv_col, field in col_map.items():
            rec[field] = (row.get(csv_col) or "").strip()
        by_annotator[annotator][paper_id].append(rec)

    return {a: dict(papers) for a, papers in by_annotator.items()}


def compute_llm_vs_humans(
    human_by_annotator: Dict[str, Dict[str, List[dict]]],
    paperid_to_url: Dict[str, str],
    extractions_dir: Path,
) -> dict:
    llm_cache: Dict[str, Optional[List[dict]]] = {}

    per_annotator_pairs: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    pooled_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_matched: Dict[str, int] = defaultdict(int)
    match_count, total_count = 0, 0
    missing_papers = set()
    no_url_papers = set()

    for annotator, papers in human_by_annotator.items():
        for paper_id, human_rows in papers.items():
            url = paperid_to_url.get(paper_id)
            if not url:
                no_url_papers.add(paper_id)
                continue

            if paper_id not in llm_cache:
                llm_cache[paper_id] = load_llm_extraction(url, extractions_dir)
            llm_recs = llm_cache[paper_id]

            if not llm_recs:
                missing_papers.add(paper_id)
                continue

            pairs = _match_experiments(human_rows, llm_recs)
            n_matched[annotator] += len(pairs)

            for h, l in pairs:
                for field in FIELDS:
                    vh, vl = h.get(field, ""), l.get(field, "")
                    if field in _LIST_FIELDS:
                        norms_h = sorted(set(_norm(field, v) for v in _split_list_value(vh))) or ["not reported"]
                        norms_l = sorted(set(_norm(field, v) for v in _split_list_value(vl))) or ["not reported"]
                        pair = ("|".join(norms_h), "|".join(norms_l))
                    else:
                        pair = (_norm(field, vh), _norm(field, vl))

                    per_annotator_pairs[annotator][field].append(pair)
                    pooled_pairs[field].append(pair)
                    total_count += 1
                    match_count += int(pair[0] == pair[1])

    per_annotator_alpha = {}
    for annotator, field_pairs in per_annotator_pairs.items():
        field_alphas = {f: _krippendorff_alpha_nominal(field_pairs.get(f, [])) for f in FIELDS}
        valid = [a for a in field_alphas.values() if a is not None]
        macro = round(sum(valid) / len(valid), 4) if valid else None
        per_annotator_alpha[annotator] = {
            "n_matched_subtasks": n_matched[annotator],
            "macro_alpha": macro,
        }

    pooled_field_alpha = {f: _krippendorff_alpha_nominal(pooled_pairs.get(f, [])) for f in FIELDS}
    valid_pooled = [a for a in pooled_field_alpha.values() if a is not None]
    pooled_macro = round(sum(valid_pooled) / len(valid_pooled), 4) if valid_pooled else None

    return {
        "per_annotator": per_annotator_alpha,
        "pooled_macro_alpha": pooled_macro,
        "pooled_n_matched_subtasks": sum(n_matched.values()),
        "pooled_agreement_pct": round(match_count / total_count * 100, 1) if total_count else None,
        "missing_llm_extraction_papers": sorted(missing_papers),
        "papers_without_url": sorted(no_url_papers),
    }


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="LLM-vs-individual-human IAA: an apples-to-apples alternative to LLM-vs-gold"
    )
    parser.add_argument("--double-csv", default=DOUBLE_CSV_DEFAULT,
                        help="Path to the raw (pre-adjudication) double-annotated CSV")
    parser.add_argument("--gold-csv", default=GOLD_CSV_DEFAULT,
                        help="Path to the adjudicated gold CSV (used only for its PaperID -> Paper Link mapping)")
    parser.add_argument("--extractions-dir", default=EXTRACTIONS_DIR_DEFAULT,
                        help="Directory of *_annotations.json LLM extraction files")
    parser.add_argument("--min-n", type=int, default=10,
                        help="Minimum matched subtasks for an annotator to count toward the mean-of-annotators summary")
    parser.add_argument("--output", help="Save JSON report to this path")
    args = parser.parse_args()

    double_path = Path(args.double_csv)
    gold_path = Path(args.gold_csv)
    extractions_dir = Path(args.extractions_dir)
    for p in (double_path, gold_path):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
    if not extractions_dir.exists():
        raise FileNotFoundError(f"Extractions dir not found: {extractions_dir}")

    print(f"Loading {double_path.name} ...")
    human_by_annotator = load_double_csv_by_annotator(double_path)
    n_rows = sum(len(rows) for papers in human_by_annotator.values() for rows in papers.values())
    print(f"  {n_rows} human annotation rows across {len(human_by_annotator)} annotators loaded")

    print(f"Loading {gold_path.name} for PaperID -> URL mapping ...")
    paperid_to_url = load_paperid_to_url(gold_path)
    print(f"  {len(paperid_to_url)} paper URLs resolved")

    results = compute_llm_vs_humans(human_by_annotator, paperid_to_url, extractions_dir)

    W = 78
    print(f"\n{'=' * W}")
    print("  LLM vs INDIVIDUAL HUMAN ANNOTATORS")
    print("  (apples-to-apples: same independent-rater structure as human-vs-human)")
    print(f"{'=' * W}")

    if results["papers_without_url"]:
        print(f"\n  [warn] {len(results['papers_without_url'])} paper(s) had no URL in the gold CSV, skipped.")
    if results["missing_llm_extraction_papers"]:
        print(f"  [warn] {len(results['missing_llm_extraction_papers'])} paper(s) had no LLM extraction found, skipped.")

    pooled = results["pooled_macro_alpha"]
    pooled_str = f"{pooled:.4f}" if pooled is not None else "N/A"
    print(f"\n  Pooled, all annotators vs. LLM ({results['pooled_n_matched_subtasks']} matched subtasks):")
    print(f"    Krippendorff's alpha (macro-avg over fields): {pooled_str}")
    print(f"    Raw agreement:                                {results['pooled_agreement_pct']}%")

    rows_for_summary = [
        (ann, d["n_matched_subtasks"], d["macro_alpha"])
        for ann, d in results["per_annotator"].items()
    ]
    well_powered = [a for _, n, a in rows_for_summary if n >= args.min_n and a is not None]
    if well_powered:
        mean_a = sum(well_powered) / len(well_powered)
        print(f"\n  Mean of per-annotator macro-alpha "
              f"(annotators with >= {args.min_n} matched subtasks, n={len(well_powered)}): {mean_a:.4f}")

    print("\n  Reference points already reported / computed:")
    print("    LLM vs. adjudicated gold:            alpha = 0.606")
    print("    human vs. human (pre-adjudication):  alpha = 0.585")

    print(f"\n{'-' * W}")
    print(f"  {'ANNOTATOR':<15s} {'N matched':>10s}  {'Macro alpha vs LLM':>22s}")
    print(f"{'-' * W}")
    for ann, n, a in sorted(rows_for_summary, key=lambda x: (x[2] is None, -(x[2] or 0))):
        a_str = f"{a:.4f}" if a is not None else "N/A (too few matched subtasks)"
        print(f"  {ann:<15s} {n:>10d}  {a_str:>22s}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()
