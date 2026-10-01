"""
Human vs Adjudicated-Gold Inter-Annotator Agreement.

The paper currently compares:
  - LLM vs. adjudicated gold           alpha = 0.606
  - human vs. human, pre-adjudication  alpha = 0.585
and calls the LLM "human-comparable". A reviewer pointed out these two
alphas have different reference points (adjudicated consensus vs. raw
independent annotators), so the comparison isn't apples-to-apples.

This script computes the missing, fairer reference point: each individual
human annotator's Krippendorff's alpha against the SAME adjudicated gold
standard used to score the LLM, plus a pooled figure (all annotators'
rows vs. gold treated as one dataset) that is directly comparable to the
LLM's 0.606.

Usage:
    uv run human_gold_iaa.py
    uv run human_gold_iaa.py --output human_vs_gold_results.json
"""
# /// script
# dependencies = []
# ///

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from human_human_iaa import (
    FIELDS,
    _match_subtasks,
    _norm,
    _split_list_value,
    _LIST_FIELDS,
    _krippendorff_alpha_nominal,
    load_csv,
)

GOLD_ANNOTATOR = "Gold"  # load_csv() strips + capitalizes the raw "gold" value to this


def compute_alpha_vs_gold(rows: List[dict], gold_name: str = GOLD_ANNOTATOR) -> dict:
    """Per-annotator and pooled Krippendorff's alpha against the adjudicated gold rows."""
    paper_rows: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        paper_rows[r["_paper_id"]][r["_annotator"]].append(r)

    per_annotator_pairs: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    pooled_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_matched_subtasks: Dict[str, int] = defaultdict(int)
    match_count, total_count = 0, 0

    for paper_id, ann_dict in sorted(paper_rows.items()):
        if gold_name not in ann_dict:
            continue
        gold_rows = ann_dict[gold_name]

        for annotator, rows_a in ann_dict.items():
            if annotator == gold_name:
                continue
            matched = _match_subtasks(rows_a, gold_rows)
            n_matched_subtasks[annotator] += len(matched)

            for ra, rg in matched:
                for field in FIELDS:
                    va, vg = ra.get(field, ""), rg.get(field, "")
                    if field in _LIST_FIELDS:
                        norms_a = sorted(set(_norm(field, v) for v in _split_list_value(va))) or ["not reported"]
                        norms_g = sorted(set(_norm(field, v) for v in _split_list_value(vg))) or ["not reported"]
                        pair = ("|".join(norms_a), "|".join(norms_g))
                    else:
                        pair = (_norm(field, va), _norm(field, vg))

                    per_annotator_pairs[annotator][field].append(pair)
                    pooled_pairs[field].append(pair)
                    total_count += 1
                    match_count += int(pair[0] == pair[1])

    per_annotator_alpha = {}
    for annotator, field_pairs in per_annotator_pairs.items():
        field_alphas = {
            field: _krippendorff_alpha_nominal(field_pairs.get(field, []))
            for field in FIELDS
        }
        valid = [a for a in field_alphas.values() if a is not None]
        macro = round(sum(valid) / len(valid), 4) if valid else None
        per_annotator_alpha[annotator] = {
            "n_matched_subtasks": n_matched_subtasks[annotator],
            "per_field_alpha": field_alphas,
            "macro_alpha": macro,
        }

    pooled_field_alpha = {
        field: _krippendorff_alpha_nominal(pooled_pairs.get(field, []))
        for field in FIELDS
    }
    valid_pooled = [a for a in pooled_field_alpha.values() if a is not None]
    pooled_macro_alpha = round(sum(valid_pooled) / len(valid_pooled), 4) if valid_pooled else None

    return {
        "per_annotator": per_annotator_alpha,
        "pooled_per_field_alpha": pooled_field_alpha,
        "pooled_macro_alpha": pooled_macro_alpha,
        "pooled_n_matched_subtasks": sum(n_matched_subtasks.values()),
        "pooled_agreement_pct": round(match_count / total_count * 100, 1) if total_count else None,
    }


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Human-vs-adjudicated-gold IAA: a fairer reference point for the LLM-vs-gold alpha"
    )
    parser.add_argument(
        "--double-csv",
        default=r"data/double_annotated.csv",
        help="Path to the raw (pre-adjudication) double-annotated CSV",
    )
    parser.add_argument(
        "--gold-csv",
        default=r"data/gold_standard.csv",
        help="Path to the adjudicated gold-standard CSV",
    )
    parser.add_argument(
        "--min-n", type=int, default=5,
        help="Minimum matched subtasks for an annotator to count toward the mean-of-annotators summary",
    )
    parser.add_argument("--output", help="Save JSON report to this path")
    args = parser.parse_args()

    double_path = Path(args.double_csv)
    gold_path = Path(args.gold_csv)
    for p in (double_path, gold_path):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    print(f"Loading {double_path.name} ...")
    human_rows = load_csv(double_path)
    print(f"  {len(human_rows)} human annotation rows loaded")

    print(f"Loading {gold_path.name} ...")
    gold_rows = load_csv(gold_path)
    for r in gold_rows:
        r["_annotator"] = GOLD_ANNOTATOR
    print(f"  {len(gold_rows)} gold (adjudicated) rows loaded")

    results = compute_alpha_vs_gold(human_rows + gold_rows)

    W = 78
    print(f"\n{'=' * W}")
    print("  HUMAN vs ADJUDICATED-GOLD AGREEMENT")
    print("  (fairer reference point for the LLM-vs-gold alpha)")
    print(f"{'=' * W}")

    pooled = results["pooled_macro_alpha"]
    pooled_str = f"{pooled:.4f}" if pooled is not None else "N/A"
    print(f"\n  Pooled, all annotators vs. gold ({results['pooled_n_matched_subtasks']} matched subtasks):")
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

    print("\n  Reference points already reported in the paper:")
    print("    LLM vs. adjudicated gold:            alpha = 0.606")
    print("    human vs. human (pre-adjudication):  alpha = 0.585")

    print(f"\n{'-' * W}")
    print(f"  {'ANNOTATOR':<15s} {'N matched':>10s}  {'Macro alpha vs gold':>22s}")
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
