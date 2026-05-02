"""
15 Jul 2025

USAGE:
python3 template.py

"""

import argparse
import json
import os
import time
from pathlib import Path

from hum_hum_iaa import make_dirs, my_compute_iaa, _match_subtasks, _pretty, generate_visualizations
from hum_hum_iaa import fix_crowd_screening
from configs import FIELD_GROUPS
import pandas as pd


def format_args(args):
    parts = []
    for k, v in vars(args).items():
        if v is None or v is False:
            continue
        if v is True:
            parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', help="", default='data/original/Gold_Extractions_gemini-3.1-pro-preview/')
    parser.add_argument('--link', default='data/original/gold_standard.csv')
    parser.add_argument('--gold', default='data/gold.tsv')
    parser.add_argument('--res', default='data/')
    parser.add_argument("--no_viz", action="store_true", help="Skip generating visualizations")
    parser.add_argument("--viz_dir", default="res/pics/iaa")
    parser.add_argument('--temp', default='temp/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res, args.temp], args=args)

    csv = pd.read_csv(args.link)
    print(csv.shape)

    gold = pd.read_csv(args.gold, sep="\t")
    print(gold.shape)

    def clean_id(s):
        return (
            s.str.lower().str.strip()
        )

    gold["PaperID"] = clean_id(gold["PaperID"])
    csv["PaperID"] = clean_id(csv["PaperID"])

    gold = gold.merge(
        csv[["PaperID", "Paper Link"]],
        on="PaperID",
        how="left",
        indicator=True
    )
    gold = gold.drop_duplicates()

    assert not gold["Paper Link"].isna().any(), "llm_id contains NaN values"

    #  ACL Anthology (with or without .pdf, with or without trailing slash)
    acl = gold["Paper Link"].str.extract(r"aclanthology\.org/(?:.*?/)?([^/]+)\.pdf")[0]
    arxiv = gold["Paper Link"].str.extract(r"arxiv\.org/pdf/([^/]+)")[0]

    gold["llm_id"] = acl.fillna(arxiv).fillna(gold["Paper Link"])
    assert not gold["llm_id"].isna().any(), "llm_id contains NaN values"

    gold.insert(0, "annotator", "gold")
    print(gold.columns.tolist())

    fh = [f for f in os.listdir(args.indir) if f.endswith(".json")]
    print(f"\nFound {len(fh)} files in {args.indir}")
    llm_docs = []
    for f in fh:
        df = pd.read_json(args.indir + f)
        exp_df = pd.json_normalize(df["experiments"])
        df = df.drop(columns=["experiments"]).join(exp_df)
        llm_docs.append(df)

    llm_recs = pd.concat(llm_docs)

    llm_recs["paper_id"] = llm_recs["paper_id"].astype("string").str.strip().str.lower()
    gold["llm_id"] = gold["llm_id"].astype("string").str.strip().str.lower()

    for id in gold["llm_id"].unique():
        if id not in llm_recs["paper_id"].values:
            # if str(id) not in gold["llm_id"].values:
            print(f"Missing {id} in LLM records")
    llm_recs.insert(0, "annotator", "gemini")

    rename_map = {
        "paper_id": "llm_id",
        "papers_topic": "paper_topic",
        "human_annotation_type": "task_type",
        "iaa_metric_name": "iaa_metric",
        "annotators_per_item": "annotators_per_item",
        "total_annotated_items": "total_items",
        "human_evaluated_sample": "Human-evaluated sample",
        "crowd_qualifications_screening": "crowd_screening",
        "annotator_training": "training",
        "age_reported": "age",
        "gender_reported": "gender",
        "reported_compensation": "compensation",
        "nation_origin_reported": "nation_origin",
        "nation_residence_reported": "nation_residence",
        "education_reported": "education",
        "political_orientation_reported": "political_orientation",
        "quality_control_filtering": "post_filtering",
        "disagreement_resolution_adjudication": "adjudication",
        "subtask_name": "Subtask name",
    }
    llm_recs = llm_recs.rename(columns=rename_map)
    llm_recs = llm_recs.loc[:, ~llm_recs.columns.str.startswith("numerical_evidence")]
    for lst in ["paper_topic", "intended_use", 'iaa_metric', 'iaa_value']:
        llm_recs[lst] = llm_recs[lst].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    common_cols = gold.columns.intersection(llm_recs.columns)

    only_gold = gold.columns.difference(llm_recs.columns)

    only_llm_recs = llm_recs.columns.difference(gold.columns)

    print("\ncommon:", list(common_cols))

    print("\nonly gold:", list(only_gold))

    print("\nonly llm_recs:", list(only_llm_recs))

    print(len(common_cols))

    all_cols = list(dict.fromkeys(gold.columns.tolist() + llm_recs.columns.tolist()))

    humllm = pd.concat(
        [gold.reindex(columns=all_cols), llm_recs.reindex(columns=all_cols)],
        ignore_index=True
    )

    humllm = humllm.drop(columns=["_merge"], errors="ignore")
    humllm = humllm.rename(columns={'PaperID': 'gold_PaperID', 'llm_id': 'PaperID', 'Subtask name': "subtask_name"})

    def safe_lower(col):
        if col.dtype == "object":
            try:
                return col.str.lower()
            except:
                return col
        return col


    humllm = humllm.apply(safe_lower)

    humllm.to_csv(args.res + "hum_llm_iaa.tsv", sep="\t", index=False)

    # === INTERDEPENDENT FIELDS CLEANUP ===
    humllm = fix_crowd_screening(humllm)

    mask = (humllm["total_annotators"].astype(str) == "1") | (humllm["annotators_per_item"].astype(str) == "1")
    humllm.loc[mask, "adjudication"] = "none"

    # mask = humllm["recruitment"].eq("not reported")
    # humllm.loc[mask, "crowd_screening"] = "no"

    mask = humllm["expertise_level"].isin(["high", "medium"])
    humllm.loc[mask, "education"] = "yes"

    results = my_compute_iaa(humllm, orderly=False, explore=False, matcher=_match_subtasks)

    # ── Console report ───────────────────────────────────────────────
    W = 85
    print(f"\n{'=' * W}")
    print(f"  HUMAN–LLM INTER-ANNOTATOR AGREEMENT REPORT")
    print(f"  ({results['total_subtask_pairs']} subtask pairs evaluated)")
    print(f"{'=' * W}")

    # Overall
    alpha = results.get('overall_alpha')
    alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
    print(f"\n  Overall agreement: {results['overall_agreement_pct']}%    Krippendorff's α (macro-avg): {alpha_str}\n")
    kappa = results.get('overall_kappa')
    kappa_str = f"{kappa:.4f}" if kappa is not None else "N/A"
    print(f"Cohens's k (macro-avg): {kappa_str}\n")

    # Per-field
    print(f"{'─' * W}")
    _kripp_hdr = "Kripp " + chr(945)
    _cohen_hdr = "Cohen's " + chr(945)
    print(f"  {'FIELD':<40s} {'Agree%':>7s}  {_kripp_hdr:>8s} {_cohen_hdr:>8s} {'Match':>5s}/{'Total':>5s}")
    print(f"{'─' * W}")
    for grp, fields_in_grp in FIELD_GROUPS.items():
        if grp == "Year_and_Venue":
            continue
        print(f"  [{grp}]")
        for field in fields_in_grp:
            pct = results["per_field_agreement_pct"].get(field)
            fa = results.get("per_field_alpha", {}).get(field)
            fk = results.get("per_field_kappa", {}).get(field)
            m = results["per_field_matches"].get(field, 0)
            t = results["per_field_totals"].get(field, 0)
            pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
            a_str = f"{fa:.3f}" if fa is not None else "  N/A"
            k_str = f"{fk:.3f}" if fk is not None else "  N/A"
            indicator = "✓" if pct and pct >= 80 else ("~" if pct and pct >= 60 else "✗")
            print(f"    {indicator} {_pretty(field):<38s} {pct_str:>7s}  {a_str:>8s} {k_str:>8s} {m:>5d} / {t:>5d}")
        print()

    # Category summary
    print(f"{'─' * W}")
    print("  CATEGORY SUMMARY")
    print(f"{'─' * W}")
    cat_alphas = results.get("category_alpha", {})
    for cat, pct in results["category_agreement_pct"].items():
        bar_len = int((pct or 0) / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        ca = cat_alphas.get(cat)
        ca_str = f"α={ca:.3f}" if ca is not None else "α=N/A"
        print(f"    {cat:<20s} {bar}  {pct_str}  {ca_str}")

    # Per-annotator-pair
    print(f"\n{'─' * W}")
    print("  PER-ANNOTATOR-PAIR AGREEMENT")
    print(f"{'─' * W}")
    for pair, pct in sorted(results["pair_agreement_pct"].items(),
                            key=lambda x: x[1] or 0, reverse=True):
        bar_len = int((pct or 0) / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        print(f"    {pair:<30s} {bar}  {pct_str}")

    # Per-paper
    print(f"\n{'─' * W}")
    print("  PER-PAPER AGREEMENT")
    print(f"{'─' * W}")
    for paper, pct in sorted(results["paper_agreement_pct"].items(),
                             key=lambda x: x[1] or 0, reverse=True):
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        print(f"    {paper:<50s} {pct_str}")

    # Not-reported analysis
    nr_both = results.get("not_reported_both", {})
    if nr_both:
        print(f"\n{'─' * W}")
        print("  'NOT REPORTED' ANALYSIS  (both annotators → not reported)")
        print(f"{'─' * W}")
        for field in sorted(nr_both, key=nr_both.get, reverse=True):
            total = results["per_field_totals"].get(field, 1)
            frac = nr_both[field] / total * 100 if total else 0
            print(f"    {_pretty(field):<40s} {nr_both[field]:>3d} / {total:>3d}  ({frac:.0f}%)")

    # Top disagreement patterns
    disag = results.get("disagreement_patterns", {})
    fields_with_disag = sorted(disag, key=lambda f: sum(p["count"] for p in disag[f]), reverse=True)
    if fields_with_disag:
        print(f"\n{'─' * W}")
        print("  TOP DISAGREEMENT PATTERNS (field → value_A ↔ value_B)")
        print(f"{'─' * W}")
        for field in fields_with_disag[:10]:
            print(f"    {_pretty(field)}:")
            for p in disag[field][:3]:
                print(f"      {p['values'][0]} ↔ {p['values'][1]}  (×{p['count']})")

    print(f"\n{'=' * W}\n")

    # ── Save JSON ────────────────────────────────────────────────────
    if args.res:
        out = Path("res") / "hum_llm_iaa_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Strip raw values from pair_details to keep JSON manageable
        clean_details = []
        for d in results["pair_details"]:
            clean = {k: v for k, v in d.items() if not k.startswith("_")}
            clean_details.append(clean)
        save_data = {k: v for k, v in results.items() if k != "pair_details"}
        save_data["pair_details"] = clean_details
        out.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON report saved to {out}")

    # ── Visualizations ───────────────────────────────────────────────
    if not args.no_viz:
        print("\nGenerating visualizations ...")
        viz_dir = Path(args.viz_dir)
        generate_visualizations(results, 'llm', viz_dir)

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
