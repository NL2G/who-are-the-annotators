"""
27 April 2026
adaptation of Gagan's script

Human–Human Inter-Annotator Agreement (IAA) — Extended Analytics & Visualizations.

Computes per-field, per-annotator-pair, per-paper, and per-category agreement
between human annotators in the double-annotated gold standard CSV.
Produces a comprehensive console report plus publication-quality visualizations.

Usage:
    uv run hum_hum_iaa.py                                        # defaults

"""
# /// script
# dependencies = [
#   "requests",
#   "beautifulsoup4",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.26",
# ]
# ///

import csv
import io
import json
import math

import re

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd

import argparse
import os
import sys
from datetime import datetime
from configs import _HEADER_MAP, FIELDS, FIELD_LABELS, TAX, FIELD_GROUPS, PREFIXES
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


def fix_crowd_screening(df):
    df = df.copy()
    cs = df["crowd_screening"]
    rec = df["recruitment"].str.lower()

    mask1 = (rec == "crowdsourcing") & (cs == "na")
    mask2 = (rec != "crowdsourcing") & (cs == "no")

    df.loc[mask1, "crowd_screening"] = "no"
    df.loc[mask2, "crowd_screening"] = "na"

    return df


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


def make_dirs(logs=None, make_them=None, args=None):
    for i in make_them:
        if i:
            os.makedirs(i, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logs}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    if args:
        print(f"Run settings: python3 {sys.argv[0]} {format_args(args)}")
    else:
        print(f"Run settings: python3 {sys.argv[0]}")


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def input_sanity_check_column(df, col, top_k=10):
    vals = df[col].dropna().astype(str)

    counter = Counter(vals)
    total = len(vals)

    print(f"Column: {col}")
    print(f"Total non-null: {total}")
    print("-" * 40)

    for k, v in counter.most_common(top_k):
        print(f"{k:25s} {v:5d}  ({v/total:.1%})")

    print("-" * 40)
    print(f"Unique values: {len(counter)}")


def column_distributions(pairs):
    col_a = Counter(a for a, _ in pairs)
    col_b = Counter(b for _, b in pairs)
    return col_a, col_b


def format_counter(counter, top_k=None):
    total = sum(counter.values())
    items = counter.most_common(top_k)
    return ", ".join(
        f"{k}:{v} ({v/total:.1%})" for k, v in items
    )


# ── Normalization helpers ────────────────────────────────────────────

_YES_NO_FIELDS = TAX['binary']


def _norm(field: str, raw: str) -> str:
    """Lightweight normalization for comparison."""
    s = raw.strip().lower()
    if not s or s in ("", "-"):
        s = "not reported"

    # Yes/No fields
    if field in _YES_NO_FIELDS:
        if s in ("yes", "true", "1", 'Yes'):
            return "yes"
        if s in ("na", "n/a", "not applicable", 'No'):
            return "no"
        return "no"

    # NLP task type — strip parenthetical
    if field == "task_type":
        for k, v in PREFIXES.items():
            if s.startswith(k):
                return v
        return s

    # IAA metric — coarse buckets
    if field == "iaa_metric":
        if "fleiss" in s:
            return "fleiss"
        if "cohen" in s:
            return "cohen"
        if "krippendorff" in s:
            return "krippendorff"
        if s in ("f1", "f1-score", "f1 score", "f1 agreement"):
            return "f1"
        if "pearson" in s:
            return "pearson"
        if "spearman" in s:
            return "spearman"
        if "kendall" in s:
            return "kendall"
        if any(k in s for k in ("total agreement", "percent agreement",
                                 "percentage agreement")):
            return "total agreement"
        if "majority" in s:
            return "majority"
        if s in ("not reported", "nr", "none", "n/a"):
            return "not reported"
        return s

    # IAA value — strip %, normalise blanks
    if field == "iaa_value":
        if s in ("not reported", "nr", "na", "n/a", "none"):
            return "not reported"
        return s.rstrip("%").strip()

    # Recruitment
    if field == "recruitment":
        if any(k in s for k in ("mturk", "mechanical turk", "amazon")):
            return "amazon turks"
        if "prolific" in s or "profilic" in s:
            return "prolific"
        if "crowdsourc" in s:
            return "crowdsourcing"
        if "author" in s:
            return "authors"
        if "other" in s:
            return "other"
        if s in ("not reported", "nr", "unknown"):
            return "not reported"
        return s

    # Expertise
    if field == "expertise_level":
        if any(k in s for k in ("high", "expert")):
            return "high"
        if any(k in s for k in ("medium", "student")):
            return "medium"
        # if any(k in s for k in ("low", "crowd")):
        #     return "low"
        if s in ("na", "n/a"):
            return "na"
        if s in ("not reported", "nr", "unknown", ""):
            return "not reported"
        return s

    # Language proficiency
    if field == "language_proficiency":
        if "non" in s and "native" in s:
            return "non-native"
        if "native" in s:
            return "native"
        if s in ("mixed", "both"):
            return "mixed"
        return "not reported"

    # Reported compensation
    if field == "compensation":
        if s == "paid":
            return "paid"
        if s in ("free", "voluntary"):
            return "free"
        return "not reported"

    # Payment rate
    if field == "payment_rate":
        if "specific" in s:
            return "specific numeric rate"
        if "general" in s:
            return "general mention"
        return "na"

    # Workload fields — extract first integer for comparison
    if field in ("annotators_per_item", "total_annotators",
                 "total_items", "items_per_annotator"):
        if s in ("not reported", "nr", "unknown", "na", "n/a"):
            return "not reported"
        # Expand K suffix
        m = re.search(r"(\d+(?:\.\d+)?)\s*[kK]", s)
        if m:
            try:
                return str(int(float(m.group(1)) * 1000))
            except ValueError:
                pass
        nums = re.findall(r"\d+", s.replace(",", ""))
        if nums:
            return nums[0].lstrip("0") or "0"
        return s

    return s


# ── Matching subtasks across annotators ──────────────────────────────

def _subtask_key(subtask_name: str) -> str:
    """Coarse key for fuzzy subtask matching."""
    s = subtask_name.strip().lower()
    # Remove common filler words
    s = re.sub(r"\b(the|a|an|of|for|in|and|or)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _match_subtasks(
    rows_a: List[dict], rows_b: List[dict]
) -> List[Tuple[dict, dict]]:
    """Match subtask rows between two annotators for the same paper.

    Strategy:
    1. Exact subtask_name match (case-insensitive)
    2. Fuzzy key match
    3. Remaining unmatched rows paired by order
    """
    pairs: List[Tuple[dict, dict]] = []
    used_b = set()

    # Build fuzzy key lookup for B
    key_to_b: Dict[str, List[int]] = defaultdict(list)
    for i, rb in enumerate(rows_b):
        key_to_b[_subtask_key(rb.get("subtask_name", ""))].append(i)

    for ra in rows_a:
        name_a = ra.get("subtask_name", "").strip().lower()
        key_a = _subtask_key(ra.get("subtask_name", ""))
        matched = False

        # Exact match
        for i, rb in enumerate(rows_b):
            if i in used_b:
                continue
            if rb.get("subtask_name", "").strip().lower() == name_a and name_a:
                pairs.append((ra, rb))
                used_b.add(i)
                matched = True
                break

        if matched:
            continue

        # Fuzzy key match
        for i in key_to_b.get(key_a, []):
            if i not in used_b and key_a:
                pairs.append((ra, rows_b[i]))
                used_b.add(i)
                matched = True
                break

    # Order-based fallback for remaining
    remaining_a = [ra for ra in rows_a if not any(ra is p[0] for p in pairs)]
    remaining_b = [rows_b[i] for i in range(len(rows_b)) if i not in used_b]
    for ra, rb in zip(remaining_a, remaining_b):
        pairs.append((ra, rb))

    return pairs


def match_subtasks(df) -> List[Tuple[Dict, Dict]]:
    pairs = []

    # group by paper_id
    for paper_id, group in df.groupby("paper_id"):
        # split by annotator
        annotators = list(group["annotator"].unique())
        if len(annotators) != 2:
            exit('I dont see exactly two annotators. Something is wrong!')  

        a_name, b_name = annotators

        rows_a = group[group["annotator"] == a_name].to_dict("records")
        rows_b = group[group["annotator"] == b_name].to_dict("records")

        if len(rows_a) != len(rows_b):
            raise ValueError(f"Mismatch in rows for {paper_id}")

        # strict positional matching
        pairs.extend(zip(rows_a, rows_b))

    return pairs


# ── Multi-value field comparison ─────────────────────────────────────
_PIPED_FIELDS = ["iaa_metric"]
_STRING_DIGITS_FIELDS = TAX['string_digits']
MULTILABEL_FIELDS = TAX['multilabel']  # {"paper_topic", 'intended_use', "iaa_metric"}


def _split_list_value(raw: str) -> List[str]:
    """Split pipe/comma-separated IAA values."""
    # print(raw)
    # exit()
    parts = re.split(r"\s*\|\s*", raw)
    if len(parts) == 1:
        parts = re.split(r"\s*,\s*(?![^(]*\))", raw)  # skip commas inside parens

    return [p.strip() for p in parts if p.strip()]


_DIGIT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def match_digits(val: str, max: int = 2) -> List[float]:
    """
    Extract up to `max` numeric values from a string.
    Returns floats in order of appearance.
    """
    if val is None:
        return []

    val = str(val).strip().lower()

    # common missing cases
    if val in {"", "na", "n/a", "not reported", "none", "null"}:
        return []

    matches = _DIGIT_RE.findall(val)

    if not matches:
        return []

    # convert safely
    out = []
    for m in matches[:max]:
        try:
            out.append(float(m))
        except ValueError:
            continue

    return out


def match_multi_selection(val: str) -> List[str]:
    """
    Parse comma-separated (or mixed separator) multi-label fields
    and normalize them for comparison.
    """
    if val is None:
        return []

    val = str(val).strip().lower()

    if val in {"", "na", "n/a", "not reported", "none", "null"}:
        return []

    # split on comma, semicolon, or pipe (robust to formats)
    parts = re.split(r"[,\|;]", val)

    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        # normalize internal whitespace
        p = re.sub(r"\s+", " ", p)

        cleaned.append(p)

    return cleaned


def _field_agrees_extended(field: str, val_a: str, val_b: str) -> bool:
    """Return True when two annotator values agree for a field."""
    # -------------------------
    # numeric extraction fields
    # -------------------------
    if field in _STRING_DIGITS_FIELDS:
        a_nums = match_digits(val_a, max=2)
        b_nums = match_digits(val_b, max=2)

        # both empty / non-parsable → only agree if both fail
        if not a_nums and not b_nums:
            return True

        return a_nums == b_nums

    # -------------------------
    # comma-separated multi-choice fields
    # -------------------------
    elif field in MULTILABEL_FIELDS:
        a_set = set(match_multi_selection(val_a))
        b_set = set(match_multi_selection(val_b))

        # treat empty as no signal → only agree if both empty
        if not a_set and not b_set:
            return True

        return len(a_set.intersection(b_set)) > 0

    # -------------------------
    # default comparison
    # -------------------------
    a = _norm(field, val_a)
    b = _norm(field, val_b)

    # unify missing values
    missing = {"", "na", "n/a", "not reported", "none"}

    if a in missing and b in missing:
        return True

    return a == b


# ── Krippendorff's Alpha ─────────────────────────────────────────────

def _krippendorff_alpha_nominal(
    pairs: List[Tuple[str, str]],
) -> Optional[float]:
    """Compute Krippendorff's Alpha for nominal data with 2 coders per field."""
    n = len(pairs)
    if n < 2:
        return None

    # Category frequencies across all values (both coders)
    freq: Counter = Counter()
    for a, b in pairs:
        freq[a] += 1
        freq[b] += 1

    N = 2 * n  # total number of assigned values

    # Observed disagreement: fraction of within-unit pairs that disagree
    disagree = sum(1 for a, b in pairs if a != b)
    D_o = disagree / n

    # Expected disagreement from marginal frequencies
    sum_nc_nc1 = sum(nc * (nc - 1) for nc in freq.values())
    D_e = 1.0 - sum_nc_nc1 / (N * (N - 1))

    if D_e == 0:
        return 1.0 if D_o == 0 else 0.0

    return round(1.0 - D_o / D_e, 4)


def _cohen_kappa_nominal(
    pairs: List[Tuple[str, str]],
) -> Optional[float]:
    """Compute Cohen's kappa for nominal data with 2 coders per unit."""
    n = len(pairs)
    if n < 2:
        return None

    # observed agreement
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n

    # marginal frequencies
    freq_a = Counter()
    freq_b = Counter()

    for a, b in pairs:
        freq_a[a] += 1
        freq_b[b] += 1

    # expected agreement
    p_e = 0.0
    labels = set(freq_a) | set(freq_b)
    for l in labels:
        p_e += (freq_a[l] / n) * (freq_b[l] / n)

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    return 1.0 - (1.0 - p_o) / (1.0 - p_e)


# ── Core computation ─────────────────────────────────────────────────

def my_compute_iaa(df, orderly: bool = True, explore: bool = False, matcher: Callable = None):
    rows = df.to_dict("records")

    # group by paper → annotator
    paper_rows = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        r["_row_id"] = i
        paper_rows[r["PaperID"]][r["annotator"]].append(r)

    field_match = defaultdict(int)
    field_total = defaultdict(int)
    field_value_pairs = defaultdict(list)
    pair_details = []
    total_pairs = 0

    # changed_order_ids = []
    for paper_id, ann_dict in sorted(paper_rows.items()):
        annotators = sorted(ann_dict.keys())
        if len(annotators) < 2:
            continue

        for ann_a, ann_b in combinations(annotators, 2):
            rows_a = ann_dict[ann_a]
            rows_b = ann_dict[ann_b]

            # --- pairing strategy ---
            if orderly:
                rows_a = sorted(rows_a, key=lambda x: x["_row_id"])
                rows_b = sorted(rows_b, key=lambda x: x["_row_id"])

                if len(rows_a) != len(rows_b):
                    raise ValueError(f"Mismatch in {paper_id}: {ann_a} vs {ann_b}")
                matched = []
                for ra, rb in zip(rows_a, rows_b):  # or your custom logic
                    matched.append((ra, rb))
            else:
                if matcher is None:
                    raise ValueError("Non-orderly mode requires a matcher function")
                matched = matcher(rows_a, rows_b)

            for ra, rb in matched:
                total_pairs += 1
                per_field = {}
                raw_a = {}
                raw_b = {}
                for field in FIELDS:
                    va = ra.get(field, "")
                    vb = rb.get(field, "")
                    agrees = _field_agrees_extended(field, va, vb)
                    field_match[field] += int(agrees)
                    field_total[field] += 1
                    per_field[field] = agrees

                    raw_a[field] = va
                    raw_b[field] = vb

                    # Collect normalized value pairs for Krippendorff's Alpha
                    if field in _PIPED_FIELDS:
                        try:
                            norms_a = sorted(set(
                                _norm(field, v) for v in _split_list_value(va)
                            )) or ["not reported"]
                        except TypeError:
                            print(raw_a)
                        norms_b = sorted(set(
                            _norm(field, v) for v in _split_list_value(vb)
                        )) or ["not reported"]
                        field_value_pairs[field].append(
                            ("|".join(norms_a), "|".join(norms_b))
                        )
                    elif field in _STRING_DIGITS_FIELDS:
                        # serialization (consistent + stable)
                        a_nums = match_digits(va, max=2)
                        b_nums = match_digits(vb, max=2)
                        norms_a = [f"{v:g}" for v in a_nums] if a_nums else ["not reported"]
                        norms_b = [f"{v:g}" for v in b_nums] if b_nums else ["not reported"]

                        field_value_pairs[field].append(
                            ("|".join(norms_a), "|".join(norms_b))
                        )

                    elif field in MULTILABEL_FIELDS:
                        norms_a = sorted(v for v in match_multi_selection(va))
                        norms_b = sorted(v for v in match_multi_selection(vb))

                        try:
                            field_value_pairs[field].append(
                                ("|".join(norms_a), "|".join(norms_b))
                            )
                        except Exception as e:
                            print(field, va, vb, norms_a, norms_b)
                            exit()

                    else:
                        field_value_pairs[field].append(
                            (_norm(field, va), _norm(field, vb))
                        )

                pair_details.append({
                    "paper_id": paper_id,
                    "annotator_a": ann_a,
                    "annotator_b": ann_b,
                    "subtask_a": ra.get("subtask_name", ""),
                    "subtask_b": rb.get("subtask_name", ""),
                    "field_agreement": per_field,
                    "_val_a": raw_a,
                    "_val_b": raw_b,
                })

    # Aggregate
    per_field_pct = {}
    for field in FIELDS:
        total = field_total.get(field, 0)
        if total > 0:
            per_field_pct[field] = round(field_match[field] / total * 100, 1)
        else:
            per_field_pct[field] = None

    overall_matches = sum(field_match.values())
    overall_total = sum(field_total.values())
    overall_pct = round(overall_matches / overall_total * 100, 1) if overall_total else None
    if explore:
        # pairs sanity check
        for field in FIELDS:
            col_a, col_b = column_distributions(field_value_pairs.get(field, []))
            print(f"\n *** Field: {field} ({FIELD_LABELS.get(field, field)}) *** \n")
            print("A:", format_counter(col_a))
            print("B:", format_counter(col_b))
            input('Press Enter to continue...')

    # ── Krippendorff's Alpha and Cohen's Kappa per field ───────────────────────────────

    per_field_alpha: Dict[str, Optional[float]] = {}
    for field in FIELDS:
        per_field_alpha[field] = _krippendorff_alpha_nominal(
            field_value_pairs.get(field, [])
        )

    per_field_kappa: Dict[str, Optional[float]] = {}
    for field in FIELDS:
        per_field_kappa[field] = _cohen_kappa_nominal(
            field_value_pairs.get(field, [])
        )

    # Macro-average (mean of per-field vals)
    valid_alphas = [a for a in per_field_alpha.values() if a is not None]
    overall_alpha = round(sum(valid_alphas) / len(valid_alphas), 4) if valid_alphas else None

    valid_kappa = [a for a in per_field_kappa.values() if a is not None]
    overall_kappa = round(sum(valid_kappa) / len(valid_kappa), 4) if valid_kappa else None

    # ── Per-annotator-pair aggregate ─────────────────────────────────
    pair_agg: Dict[str, Dict] = defaultdict(lambda: {"match": 0, "total": 0})
    for d in pair_details:
        key = f"{d['annotator_a']} × {d['annotator_b']}"
        for field, agreed in d["field_agreement"].items():
            pair_agg[key]["match"] += int(agreed)
            pair_agg[key]["total"] += 1
    pair_agreement = {
        k: round(v["match"] / v["total"] * 100, 1) if v["total"] else None
        for k, v in pair_agg.items()
    }

    # ── Per-paper aggregate ──────────────────────────────────────────
    paper_agg: Dict[str, Dict] = defaultdict(lambda: {"match": 0, "total": 0})
    for d in pair_details:
        pid = d["paper_id"]
        for field, agreed in d["field_agreement"].items():
            paper_agg[pid]["match"] += int(agreed)
            paper_agg[pid]["total"] += 1
    paper_agreement = {
        k: round(v["match"] / v["total"] * 100, 1) if v["total"] else None
        for k, v in paper_agg.items()
    }

    # ── Per-annotator-pair per-field (for heatmap) ───────────────────
    pair_field: Dict[str, Dict[str, Dict]] = defaultdict(
        lambda: defaultdict(lambda: {"match": 0, "total": 0})
    )
    for d in pair_details:
        key = f"{d['annotator_a']} × {d['annotator_b']}"
        for field, agreed in d["field_agreement"].items():
            pair_field[key][field]["match"] += int(agreed)
            pair_field[key][field]["total"] += 1

    # ── Per-category aggregate ───────────────────────────────────────
    category_agreement: Dict[str, Optional[float]] = {}
    category_alpha: Dict[str, Optional[float]] = {}
    for cat, fields_in_cat in FIELD_GROUPS.items():
        if cat == "Year_and_Venue":
            continue
        m = sum(field_match.get(f, 0) for f in fields_in_cat)
        t = sum(field_total.get(f, 0) for f in fields_in_cat)
        category_agreement[cat] = round(m / t * 100, 1) if t else None
        cat_as = [per_field_alpha[f] for f in fields_in_cat
                  if per_field_alpha.get(f) is not None]
        category_alpha[cat] = round(
            sum(cat_as) / len(cat_as), 4
        ) if cat_as else None

    # ── Confusion / disagreement analysis ────────────────────────────
    disagreements: Dict[str, List[dict]] = defaultdict(list)
    not_reported_both: Dict[str, int] = defaultdict(int)
    not_reported_one: Dict[str, int] = defaultdict(int)
    for d in pair_details:
        for field in FIELDS:
            va_raw = d.get("_val_a", {}).get(field, "")
            vb_raw = d.get("_val_b", {}).get(field, "")
            # try:
            na = _norm(field, va_raw)
            # except:
            #     print(field, va_raw)
            #     print(d)
            #     exit()
            nb = _norm(field, vb_raw)
            if na == "not reported" and nb == "not reported":
                not_reported_both[field] += 1
            elif na == "not reported" or nb == "not reported":
                not_reported_one[field] += 1
            if not d["field_agreement"].get(field, True):
                disagreements[field].append({
                    "paper": d["paper_id"],
                    "a": f"{d['annotator_a']}: {va_raw}",
                    "b": f"{d['annotator_b']}: {vb_raw}",
                    "norm_a": na,
                    "norm_b": nb,
                })

    # ── Top disagreement patterns per field ──────────────────────────
    disagreement_patterns: Dict[str, List[dict]] = {}
    for field, disag_list in disagreements.items():
        pattern_counts: Counter = Counter()
        for d in disag_list:
            pair_key = tuple(sorted([d["norm_a"], d["norm_b"]]))
            pattern_counts[pair_key] += 1
        disagreement_patterns[field] = [
            {"values": list(k), "count": c}
            for k, c in pattern_counts.most_common(5)
        ]

    return {
        "total_subtask_pairs": total_pairs,
        "overall_agreement_pct": overall_pct,
        "overall_alpha": overall_alpha,
        "overall_kappa": overall_kappa,
        "per_field_agreement_pct": per_field_pct,
        "per_field_alpha": per_field_alpha,
        "per_field_kappa": per_field_kappa,
        "per_field_matches": dict(field_match),
        "per_field_totals": dict(field_total),
        "pair_agreement_pct": pair_agreement,
        "paper_agreement_pct": paper_agreement,
        "category_agreement_pct": category_agreement,
        "category_alpha": category_alpha,
        "pair_field_data": {
            pk: {fk: round(fv["match"] / fv["total"] * 100, 1) if fv["total"] else None
                 for fk, fv in fdict.items()}
            for pk, fdict in pair_field.items()
        },
        "not_reported_both": dict(not_reported_both),
        "not_reported_one": dict(not_reported_one),
        "disagreement_patterns": {
            k: v for k, v in disagreement_patterns.items()
        },
        "pair_details": pair_details,
    }


# ── Visualization ────────────────────────────────────────────────────

def _pretty(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").title())


def _rate_color(pct: float) -> str:
    if pct >= 90:
        return "#2ecc71"
    if pct >= 70:
        return "#f39c12"
    return "#e74c3c"


def _save(fig, name: str, out_dir: Path):
    fig.savefig(out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  saved {name}.png")


def generate_visualizations(results: dict, who: str, out_dir: Path):
    """Produce 3 insight-rich human–human IAA visualisations."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    PAL = sns.color_palette("colorblind")
    out_dir.mkdir(parents=True, exist_ok=True)

    pf_pct = results["per_field_agreement_pct"]
    nr_both = results.get("not_reported_both", {})
    nr_one = results.get("not_reported_one", {})
    total_pairs = results["total_subtask_pairs"]

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 1 — "The Annotation Difficulty Landscape"
    #
    # A single horizontal chart per field showing three stacked components:
    #   ■ Genuine agreement (both annotators agree on a real value)
    #   ■ Vacuous agreement (both say "not reported" — inflates IAA)
    #   ■ Disagreement
    # This is the key insight: some fields look high-agreement only because
    # both annotators couldn't find the info. Reveals true extraction difficulty.
    # Or the info was truly not reported (you don't think of reporting Political orientation in most NLP tasks)
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 1 — Annotation difficulty landscape ...")

    # Build ordered fields by category
    ordered_fields = []
    group_boundaries = []  # (start_idx, group_name)
    for grp, flds in FIELD_GROUPS.items():
        if grp == "Year_and_Venue":
            continue
        group_boundaries.append((len(ordered_fields), grp))
        ordered_fields.extend(flds)

    n = len(ordered_fields)
    genuine_agree = []
    vacuous_agree = []
    disagree_vals = []

    for f in ordered_fields:
        t = results["per_field_totals"].get(f, total_pairs)
        nr_b = nr_both.get(f, 0)
        nr_o = nr_one.get(f, 0)
        m = results["per_field_matches"].get(f, 0)
        # Genuine agreement = matches minus the "both not reported" cases
        gen = max(m - nr_b, 0)
        vac = nr_b
        dis = t - m
        genuine_agree.append(gen / t * 100 if t else 0)
        vacuous_agree.append(vac / t * 100 if t else 0)
        disagree_vals.append(dis / t * 100 if t else 0)

    genuine_agree = np.array(genuine_agree)
    vacuous_agree = np.array(vacuous_agree)
    disagree_vals = np.array(disagree_vals)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    y = np.arange(n)
    bh = 0.7

    C_GEN = "#27ae60"   # real agreement — green
    C_VAC = "#95a5a6"   # vacuous — grey
    C_DIS = "#e74c3c"   # disagreement — red

    ax.barh(y, genuine_agree, height=bh, color=C_GEN, label="Genuine agreement")
    ax.barh(y, vacuous_agree, height=bh, left=genuine_agree, color=C_VAC,
            label="Both → 'not reported'  (vacuous)")
    ax.barh(y, disagree_vals, height=bh,
            left=genuine_agree + vacuous_agree, color=C_DIS, label="Disagreement")

    # Annotate headline % on the right
    for i, f in enumerate(ordered_fields):
        total_agree = genuine_agree[i] + vacuous_agree[i]
        ax.text(101, i, f"{total_agree:.0f}%", va="center", fontsize=8,
                color="black", fontweight="bold")

    # Category dividers & labels
    for idx, (start, grp) in enumerate(group_boundaries):
        if start > 0:
            ax.axhline(y=start - 0.5, color="#bdc3c7", linestyle="-", linewidth=0.8)
        # Find end of this group
        end = group_boundaries[idx + 1][0] if idx + 1 < len(group_boundaries) else n
        mid = (start + end - 1) / 2
        ax.text(-2, mid, grp, va="center", ha="right", fontsize=8,
                fontstyle="italic", color="#7f8c8d", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([_pretty(f) for f in ordered_fields], fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Percentage of subtask pairs", fontsize=10)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    ax.set_title(
        "Where Do Annotators Actually Agree?\n"
        "Separating genuine agreement from vacuous 'both couldn't find it' agreement",
        fontweight="bold", fontsize=12, pad=12,
    )
    fig.tight_layout()
    _save(fig, f"{who}_fig1_difficulty_landscape", out_dir)

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 2 — "Annotator × Field Heatmap with Marginals"
    #
    # The full pair×field heatmap, but with marginal summaries on right
    # (per-pair overall) and bottom (per-field overall) so you can
    # instantly see WHO struggles and WHAT is hard in one glance.
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 2 — Annotator × Field heatmap with marginals ...")

    pair_field_data = results.get("pair_field_data", {})
    pair_pct = results.get("pair_agreement_pct", {})

    # Sort pairs by their overall agreement (worst at bottom → stands out)
    pairs = sorted(pair_field_data.keys(),
                   key=lambda p: pair_pct.get(p, 0) or 0, reverse=True)
    fields_hm = ordered_fields  # already in category order

    matrix = np.array([
        [pair_field_data[p].get(f, 0) or 0 for f in fields_hm]
        for p in pairs
    ])

    fig = plt.figure(figsize=(16, max(5, len(pairs) * 0.65 + 3)))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 2], height_ratios=[len(pairs), 2],
                           hspace=0.04, wspace=0.04)

    # Main heatmap
    ax_main = fig.add_subplot(gs[0, 0])
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=0, vmax=100, linewidths=0.6, linecolor="white",
                xticklabels=[_pretty(f) for f in fields_hm],
                yticklabels=pairs, ax=ax_main, cbar=False,
                annot_kws={"fontsize": 7.5})
    ax_main.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha="left", fontsize=8)
    plt.setp(ax_main.get_yticklabels(), fontsize=9)

    # Add category dividers on x-axis
    for idx, (start, grp) in enumerate(group_boundaries):
        if start > 0:
            ax_main.axvline(x=start, color="black", linewidth=1.5, alpha=0.3)

    # Right marginal: per-pair overall %
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
    pair_vals = [pair_pct.get(p, 0) or 0 for p in pairs]
    pair_colors = [_rate_color(v) for v in pair_vals]
    bars = ax_right.barh(range(len(pairs)), pair_vals, color=pair_colors,
                         edgecolor="white", height=0.8)
    for bar, v in zip(bars, pair_vals):
        ax_right.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                      f"{v:.0f}%", va="center", fontsize=8)
    ax_right.set_xlim(0, 110)
    ax_right.set_xlabel("Pair %", fontsize=8)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.invert_yaxis()

    # Bottom marginal: per-field overall %
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
    field_vals = [pf_pct.get(f, 0) or 0 for f in fields_hm]
    field_colors = [_rate_color(v) for v in field_vals]
    ax_bottom.bar(range(len(fields_hm)), field_vals, color=field_colors,
                  edgecolor="white", width=0.8)
    for i, v in enumerate(field_vals):
        ax_bottom.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=7, rotation=90)
    ax_bottom.set_ylim(0, 115)
    ax_bottom.set_ylabel("Field %", fontsize=8)
    ax_bottom.tick_params(axis="x", bottom=False, labelbottom=False)
    # Category dividers
    for idx, (start, grp) in enumerate(group_boundaries):
        if start > 0:
            ax_bottom.axvline(x=start - 0.5, color="black", linewidth=1.5, alpha=0.3)

    # Hide unused corner
    ax_corner = fig.add_subplot(gs[1, 1])
    ax_corner.axis("off")
    overall = results["overall_agreement_pct"] or 0
    ax_corner.text(0.5, 0.5, f"Overall\n{overall:.1f}%",
                   ha="center", va="center", fontsize=14, fontweight="bold",
                   transform=ax_corner.transAxes,
                   bbox=dict(boxstyle="round,pad=0.4",
                             facecolor=_rate_color(overall), alpha=0.3))

    fig.suptitle(
        "Who Agrees on What?  —  Annotator Pair × Extraction Field\n"
        "Marginals show per-pair (right) and per-field (bottom) totals",
        fontweight="bold", fontsize=13, y=1.02,
    )
    _save(fig, f"{who}_fig2_pair_field_heatmap", out_dir)

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 3 — "Root Causes of Disagreement"
    #
    # Two-panel figure:
    #   Left:  For the hardest fields, shows *what kind* of disagreement
    #          (value-vs-value, value-vs-not-reported, na-vs-no, etc.)
    #   Right: Category radar overlaid with a "not-reported" radar
    #          to show which categories are hard vs. just unreportable
    # ═══════════════════════════════════════════════════════════════════
    print("Figure 3 — Root causes of disagreement ...")

    disag_patterns = results.get("disagreement_patterns", {})

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.3)

    # ── Left panel: disagreement breakdown for hardest fields ────────
    ax_left = fig.add_subplot(gs[0, 0])

    # Pick the 8 fields with most disagreement
    field_disag_count = {
        f: results["per_field_totals"].get(f, 0) - results["per_field_matches"].get(f, 0)
        for f in FIELDS
    }
    hard_fields = sorted(field_disag_count, key=field_disag_count.get, reverse=True)[:8]

    # For each hard field, categorize disagreements
    y_positions = []
    y_labels = []
    bar_data = []   # list of (val_vs_val, val_vs_nr, na_ambiguity)

    C_VV = "#e74c3c"   # genuine value conflict
    C_VNR = "#3498db"  # one found it, other didn't
    C_NA = "#f39c12"   # na vs no/yes ambiguity

    for f in hard_fields:
        patterns = disag_patterns.get(f, [])
        val_vs_val = 0
        val_vs_nr = 0
        na_ambig = 0
        for p in patterns:
            a, b = p["values"]
            c = p["count"]
            if "not reported" in (a, b):
                val_vs_nr += c
            elif "na" in (a, b):
                na_ambig += c
            else:
                val_vs_val += c
        bar_data.append((val_vs_val, val_vs_nr, na_ambig))
        y_labels.append(_pretty(f))

    y_pos = np.arange(len(hard_fields))
    vv = [d[0] for d in bar_data]
    vnr = [d[1] for d in bar_data]
    na_a = [d[2] for d in bar_data]

    bh = 0.6
    ax_left.barh(y_pos, vv, height=bh, color=C_VV, label="Value ↔ Value  (genuine conflict)")
    ax_left.barh(y_pos, vnr, height=bh, left=vv, color=C_VNR,
                 label="Value ↔ 'Not reported'  (extraction miss)")
    ax_left.barh(y_pos, na_a, height=bh,
                 left=[a + b for a, b in zip(vv, vnr)], color=C_NA,
                 label="'N/A' ambiguity  (na vs yes/no)")

    # Total disagreement count on right
    for i in range(len(hard_fields)):
        total = vv[i] + vnr[i] + na_a[i]
        ax_left.text(total + 0.3, i, f"{total}", va="center", fontsize=9, fontweight="bold")

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(y_labels, fontsize=10)
    ax_left.set_xlabel("Number of disagreeing subtask pairs", fontsize=10)
    ax_left.invert_yaxis()
    ax_left.legend(fontsize=8.5, loc="lower right", frameon=True)
    ax_left.set_title("Why Annotators Disagree\n(8 hardest fields)", fontweight="bold", fontsize=11)

    # ── Right panel: category radar with not-reported overlay ────────
    ax_right = fig.add_subplot(gs[0, 1], polar=True)

    cat_pct = results.get("category_agreement_pct", {})
    cats = list([k for k in FIELD_GROUPS.keys() if k != "Year_and_Venue"])
    N = len(cats)
    angles = [i / N * 2 * math.pi for i in range(N)]
    angles_closed = angles + [angles[0]]

    # Category agreement
    cat_vals = [cat_pct.get(c, 0) or 0 for c in cats]
    cat_closed = cat_vals + [cat_vals[0]]

    # Category "not-reported" rate (how much of agreement is vacuous)
    cat_nr_rate = []
    for cat, flds in FIELD_GROUPS.items():
        if cat == "Year_and_Venue":
            continue
        nr_b_sum = sum(nr_both.get(f, 0) for f in flds)
        total_sum = sum(results["per_field_totals"].get(f, 0) for f in flds)
        cat_nr_rate.append(nr_b_sum / total_sum * 100 if total_sum else 0)
    nr_closed = cat_nr_rate + [cat_nr_rate[0]]

    # "Genuine" agreement (subtract not-reported from total agreement)
    cat_genuine = []
    for cat, flds in FIELD_GROUPS.items():
        if cat == "Year_and_Venue":
            continue
        m_sum = sum(results["per_field_matches"].get(f, 0) for f in flds)
        nr_b_sum = sum(nr_both.get(f, 0) for f in flds)
        t_sum = sum(results["per_field_totals"].get(f, 0) for f in flds)
        genuine = (m_sum - nr_b_sum) / t_sum * 100 if t_sum else 0
        cat_genuine.append(max(genuine, 0))
    gen_closed = cat_genuine + [cat_genuine[0]]

    ax_right.fill(angles_closed, cat_closed, alpha=0.15, color=PAL[0])
    ax_right.plot(angles_closed, cat_closed, "o-", color=PAL[0],
                  linewidth=2, label="Total agreement")

    ax_right.fill(angles_closed, gen_closed, alpha=0.15, color="#27ae60")
    ax_right.plot(angles_closed, gen_closed, "s--", color="#27ae60",
                  linewidth=2, label="Genuine agreement")

    ax_right.fill(angles_closed, nr_closed, alpha=0.2, color="#95a5a6")
    ax_right.plot(angles_closed, nr_closed, "^:", color="#95a5a6",
                  linewidth=2, label="'Not reported' rate")

    ax_right.set_xticks(angles)
    ax_right.set_xticklabels(cats, fontsize=9)
    ax_right.set_ylim(0, 100)
    ax_right.set_yticks([20, 40, 60, 80, 100])
    ax_right.set_yticklabels(["20%", "40%", "60%", "80%", "100%"],
                              fontsize=7, color="grey")

    # Annotate values
    for angle, tv, gv in zip(angles, cat_vals, cat_genuine):
        ax_right.text(angle, tv + 5, f"{tv:.0f}%", ha="center", fontsize=8,
                      fontweight="bold", color=PAL[0])
        ax_right.text(angle, gv - 7, f"{gv:.0f}%", ha="center", fontsize=8,
                      color="#27ae60")

    ax_right.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.35, 1.12),
                    frameon=True)
    ax_right.set_title("Total vs. Genuine Agreement\nby Category",
                       fontweight="bold", fontsize=11, pad=20)

    fig.suptitle(
        "Root Causes of Disagreement — What Makes Annotation Hard?",
        fontweight="bold", fontsize=13, y=1.02,
    )
    fig.subplots_adjust(wspace=0.3)
    _save(fig, f"{who}_fig3_disagreement_causes", out_dir)

    print(f"\nAll 3 figures saved to {out_dir}/")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 output on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Human–Human IAA from double-annotated CSV")
    parser.add_argument("--tab", default=r"data/double_annotated.tsv")
    parser.add_argument("--res", help="Save JSON report to this path", default="res/")
    parser.add_argument("--explore", action="store_true", help="run exploratory data analysis")
    parser.add_argument("--no_viz", action="store_true", help="Skip generating visualizations")
    parser.add_argument("--viz_dir", default="res/pics/iaa")
    parser.add_argument('--logs', default='logs/')
    args = parser.parse_args()

    make_dirs(logs=args.logs, make_them=[args.res, args.viz_dir], args=args)

    csv_path = Path(args.tab)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        sep="\t",
        engine="python",
        on_bad_lines="warn",
        dtype=str,
        keep_default_na=False
    )
    if df.isna().any().any():
        print('Warning: NaN values detected in the CSV after loading. Check the data for missing values.')
    else:
        print('\n ***** No NaN values detected in the CSV after loading. *****\n')

    df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

    df = df.rename(columns=_HEADER_MAP)

    if args.explore:
        print(df.head())
        print(df.columns)

        DROP_DOWNS = list(set(FIELDS) - {"items_per_annotator", "total_items", "total_annotators",
                                         "annotators_per_item", "iaa_value"})
        for col in DROP_DOWNS:
            input_sanity_check_column(df, col)

    # === INTERDEPENDENT FIELDS CLEANUP ===
    df = fix_crowd_screening(df)

    mask = (df["total_annotators"].astype(str) == "1") | (df["annotators_per_item"].astype(str) == "1")
    df.loc[mask, "adjudication"] = "none"

    # mask = df["recruitment"].eq("not reported")
    # df.loc[mask, "crowd_screening"] = "no"

    mask = df["expertise_level"].isin(["high", "medium"])
    df.loc[mask, "education"] = "yes"

    results = my_compute_iaa(df, orderly=False, explore=False, matcher=_match_subtasks)

    # ── Console report ───────────────────────────────────────────────
    W = 85
    print(f"\n{'=' * W}")
    print(f"  HUMAN–HUMAN INTER-ANNOTATOR AGREEMENT REPORT")
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
        out = Path(args.res) / "hum_hum_iaa_report.json"
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
        generate_visualizations(results, "hh", viz_dir)


if __name__ == "__main__":
    main()
