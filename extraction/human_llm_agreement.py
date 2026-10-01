"""
Human–LLM Inter-Annotator Agreement (Deterministic, no judge LLM).

Compares LLM extraction JSON files against the gold standard CSV using
the same field normalization and subtask-matching logic as human_human_iaa.py.
Reports per-field, per-category, per-paper agreement rates.

Usage:
    uv run human_llm_agreement.py
    uv run human_llm_agreement.py --gold_csv "data/gold_standard.csv"
    uv run human_llm_agreement.py --extractions_dir Processed_PDFs/Combined_Extractions_gpt-5-batch
    uv run human_llm_agreement.py --output results.json --no-viz
"""
# /// script
# dependencies = [
#   "pandas",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.26",
# ]
# ///

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from download_papers import url_to_stem

# ── Fields to evaluate ───────────────────────────────────────────────
FIELDS = [
    "papers_topic",
    "human_annotation_type",
    "intended_use",
    "guidelines_released",
    "iaa_metric_name",
    "iaa_value",
    "annotators_per_item",
    "total_annotators",
    "total_annotated_items",
    "items_per_annotator",
    "human_evaluated_sample",
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
    "disagreement_resolution_adjudication",
]

FIELD_GROUPS = {
    "Task & Data": [
        "papers_topic", "human_annotation_type", "intended_use",
        "guidelines_released",
        "iaa_metric_name", "iaa_value",
    ],
    "Annotator Count": [
        "annotators_per_item", "total_annotators",
        "total_annotated_items", "items_per_annotator",
        "human_evaluated_sample",
    ],
    "Process & Quality": [
        "recruitment", "crowd_qualifications_screening",
        "annotator_training", "quality_control_filtering",
        "disagreement_resolution_adjudication",
    ],
    "Demographics": [
        "language_proficiency", "expertise_level",
        "age_reported", "gender_reported",
        "nation_origin_reported", "nation_residence_reported",
        "education_reported", "political_orientation_reported",
    ],
    "Compensation": [
        "reported_compensation", "payment_rate",
    ],
}

FIELD_LABELS = {
    "papers_topic":                "Paper's Topic",
    "human_annotation_type":       "Human Annotation Type",
    "intended_use":                "Intended Use",
    "guidelines_released":         "Guidelines Released",
    "iaa_metric_name":             "IAA Metric Name",
    "iaa_value":                   "IAA Value",
    "annotators_per_item":         "Annotators / Item",
    "total_annotators":            "Total Annotators",
    "total_annotated_items":       "Total Items",
    "items_per_annotator":         "Items / Annotator",
    "human_evaluated_sample":      "Human-Evaluated Sample",
    "recruitment":                 "Recruitment",
    "crowd_qualifications_screening": "Crowd Screening",
    "annotator_training":          "Training",
    "language_proficiency":        "Language Proficiency",
    "expertise_level":             "Expertise Level",
    "reported_compensation":       "Reported Compensation",
    "payment_rate":                "Payment Rate",
    "age_reported":                "Age Reported",
    "gender_reported":             "Gender Reported",
    "nation_origin_reported":      "Nation Origin",
    "nation_residence_reported":   "Nation Residence",
    "education_reported":          "Education",
    "political_orientation_reported": "Political Orient.",
    "quality_control_filtering":   "Quality Filtering",
    "disagreement_resolution_adjudication": "Adjudication",
}

# Gold CSV column → field key
_GOLD_COL_MAP: Dict[str, str] = {
    "Paper's topic":                        "papers_topic",
    "Task type":                            "human_annotation_type",
    "Annotation task type":                 "human_annotation_type",
    "Subtask name":                         "subtask_name",
    "Intended use":                         "intended_use",
    "Annotation guidelines released":       "guidelines_released",
    "IAA metric name":                      "iaa_metric_name",
    "IAA value":                            "iaa_value",
    "Number of annotators per item":        "annotators_per_item",
    "Total number of annotators":           "total_annotators",
    "Total number of annotated items":      "total_annotated_items",
    "Number of items per annotator":        "items_per_annotator",
    "Human-evaluated sample":               "human_evaluated_sample",
    "Recruitment":                          "recruitment",
    "Crowd qualifications & screening":     "crowd_qualifications_screening",
    "Annotator training":                   "annotator_training",
    "Annotation task language proficiency": "language_proficiency",
    "Level of expertise":                   "expertise_level",
    "Reported compensation":                "reported_compensation",
    "Payment rate":                         "payment_rate",
    "Age":                                  "age_reported",
    "Gender":                               "gender_reported",
    "Nation of origin":                     "nation_origin_reported",
    "Nation of residence":                  "nation_residence_reported",
    "Education":                            "education_reported",
    "Political orientation":                "political_orientation_reported",
    "Quality control & filtering":          "quality_control_filtering",
    "Adjudication":                         "disagreement_resolution_adjudication",
}

# ── Normalization ────────────────────────────────────────────────────

_YES_NO_FIELDS = {
    "guidelines_released",
    # crowd_qualifications_screening is NOT a pure yes/no field in prompts_v2:
    # it allows "Yes" | "Not reported" | "NA"; human annotators often write
    # "no" to mean "not applicable" (non-crowdsourcing study), so it is handled
    # separately in _norm below.
    "annotator_training",
    "age_reported",
    "gender_reported",
    "nation_origin_reported",
    "nation_residence_reported",
    "education_reported",
    "political_orientation_reported",
    "quality_control_filtering",
}


def _norm(field: str, raw: str) -> str:
    """Normalize a field value to a canonical form for comparison."""
    s = str(raw).strip().lower()
    if not s or s == "-":
        s = "not reported"

    if field in _YES_NO_FIELDS:
        if s in ("yes", "true", "1"):
            return "yes"
        if s in ("na", "n/a", "not applicable"):
            return "na"
        return "no"

    if field == "human_annotation_type":
        for prefix in (
            "closed-label classification",
            "ordinal / scalar rating",
            "span-level or structural annotation",
            "comparative / preference judgments",
            "open-ended generation or transformation",
        ):
            if s.startswith(prefix) or prefix.split()[0] in s:
                return prefix
        return s

    if field == "iaa_metric_name":
        if "fleiss" in s:
            return "fleiss"
        if "cohen" in s:
            return "cohen"
        if "krippendorff" in s:
            return "krippendorff"
        if s in ("f1", "f1-score", "f1 score", "f1 agreement") or "f1" in s:
            return "f1"
        if "pearson" in s:
            return "pearson"
        if "spearman" in s:
            return "spearman"
        if "kendall" in s:
            return "kendall"
        if any(k in s for k in ("total agreement", "percent agreement", "percentage agreement")):
            return "total agreement"
        if "majority" in s:
            return "majority"
        if s in ("not reported", "nr", "none", "n/a"):
            return "not reported"
        return s

    if field == "iaa_value":
        if s in ("not reported", "nr", "na", "n/a", "none"):
            return "not reported"
        return s.rstrip("%").strip()

    if field == "crowd_qualifications_screening":
        if s in ("yes",):
            return "yes"
        # "No", "NA", "Not reported" all mean no screening was applied
        return "no"

    if field == "recruitment":
        # IMPORTANT: "not via crowdsourcing" (old-taxonomy "Other") MUST be checked
        # BEFORE the generic "crowdsourc" substring check, otherwise it maps to "crowdsourcing".
        if "not via crowdsourc" in s or ("crowdsourc" in s and "not" in s.split("crowdsourc")[0]):
            return "not reported"
        # All crowdsourcing platforms → canonical "crowdsourcing"
        if any(k in s for k in ("mturk", "mechanical turk", "amazon turk",
                                 "prolific", "toloka", "crowdflower",
                                 "figure eight", "appen", "profilic")):
            return "crowdsourcing"
        if "crowdsourc" in s or "crowd platform" in s or "crowd-based" in s:
            return "crowdsourcing"
        if "author" in s or "researcher" in s:
            return "authors"
        if "mixed" in s:
            return "mixed"
        if "other" in s:
            return "not reported"
        if any(k in s for k in ("volunteer", "student", "university", "hired",
                                 "recruited", "employ", "freelanc", "intern",
                                 "colleague", "linguist", "annotator")):
            return "not reported"
        if s in ("not reported", "nr", "unknown", ""):
            return "not reported"
        return s

    if field == "expertise_level":
        if "mixed" in s:
            return "mixed"
        if any(k in s for k in ("high", "expert")):
            return "high"
        if any(k in s for k in ("medium", "student")):
            return "medium"
        # "General language/knowledge task" — task requires no specialisation
        if any(k in s for k in ("general", "crowd", "non-expert", "nonexpert",
                                 "layperson", "lay person", "low", "naïve",
                                 "naive", "non expert", "novice", "untrained")):
            return "general language/knowledge task"
        if s in ("na", "n/a", "not applicable"):
            return "not reported"
        if s in ("not reported", "nr", "unknown", ""):
            return "not reported"
        return s

    if field == "language_proficiency":
        if "non" in s and "native" in s:
            return "non-native"
        if "native" in s:
            return "native"
        if s in ("mixed", "both"):
            return "mixed"
        if s == "medium":
            return "medium"
        return "not reported"

    if field == "reported_compensation":
        if s in ("paid", "yes", "compensated") or re.search(r"\d", s):
            return "paid"
        if s in ("free", "voluntary", "unpaid", "volunteer", "no"):
            return "free"
        return "not reported"

    if field == "payment_rate":
        if "specific" in s or re.search(r"\d", s):  # specific amount or number
            return "specific numeric rate"
        if "general" in s:
            return "general mention"
        # Unify all "no payment rate" variants into one canonical value
        return "no"

    if field in ("annotators_per_item", "total_annotators",
                 "total_annotated_items", "items_per_annotator"):
        if s in ("not reported", "nr", "unknown", "na", "n/a"):
            return "not reported"
        m = re.search(r"(\d+(?:\.\d+)?)\s*[kK]", s)
        if m:
            try:
                return str(int(float(m.group(1)) * 1000))
            except ValueError:
                pass
        nums = re.findall(r"\d+", s.replace(",", ".").replace(".", ""))
        if nums:
            return nums[0].lstrip("0") or "0"
        return s

    if field == "papers_topic":
        # Map to canonical categories from prompts_v2
        if any(k in s for k in ("linguist", "morpho", "syntax", "phonol", "lexic")):
            return "linguistics"
        if any(k in s for k in ("discourse", "pragmat", "dialog", "rhetor", "coherence")):
            return "discourse & pragmatics"
        if any(k in s for k in ("information extraction", "relation extraction",
                                 "event extraction", "knowledge graph", "ie ")):
            return "information extraction"
        if any(k in s for k in ("question answer", "qa ", "q&a", "reading comprehension",
                                 "information access", "information retrieval")):
            return "qa & information access"
        if any(k in s for k in ("semantic", "inference", "similarity", "entailment",
                                 "nli", "natural language inference", "sts",
                                 "paraphrase detection", "textual similarity")):
            return "semantics, inference & similarity"
        if any(k in s for k in ("subjective", "sentiment", "emotion", "hate",
                                 "offensive", "toxic", "bias", "stance", "opinion",
                                 "social meaning", "social bias", "fairness",
                                 "abusive", "irony", "sarcasm")):
            return "subjective language & social meaning"
        if any(k in s for k in ("generation", "open-ended", "open ended", "creative",
                                 "story", "data-to-text", "text generation")):
            return "open-ended generation"
        if any(k in s for k in ("compression", "reformulat", "summar", "simplif",
                                 "paraphras")):
            return "compression & reformulation tasks"
        if any(k in s for k in ("multilingual", "translat", "cross-lingual",
                                 "crosslingual", "machine translation", "mt ")):
            return "multilingual & translation"
        if any(k in s for k in ("multimodal", "vision", "image", "video", "audio",
                                 "speech", "visual")):
            return "multimodal tasks"
        return s

    if field == "intended_use":
        if any(k in s for k in ("resource", "dataset", "corpus", "benchmark")):
            return "resource creation"
        if "human performance" in s or "human evaluation" in s:
            return "human performance"
        if any(k in s for k in ("model output", "model evaluation", "system evaluation",
                                 "automatic evaluation", "output evaluation")):
            return "model output evaluation"
        if s in ("not reported", "nr", "na", "n/a", "none", ""):
            return "not reported"
        if "none of the above" in s:
            return "none of the above"
        return s

    if field == "human_evaluated_sample":
        if s in ("not reported", "nr", "na", "n/a", "none", ""):
            return "not reported"
        # Normalize "all" variants
        if s in ("all", "all items", "entire", "full", "complete", "entire dataset",
                 "all data", "full dataset", "whole"):
            return "all"
        # Try to extract the number from any string containing digits
        nums = re.findall(r"\d+", s.replace(",", ""))
        if nums:
            return nums[0].lstrip("0") or "0"
        return s

    if field == "disagreement_resolution_adjudication":
        if "majority" in s:
            return "majority vote"
        if "expert" in s and "adjud" in s:
            return "expert adjudication"
        if "third" in s and "annot" in s:
            return "third annotator"
        if "consensus" in s:
            return "consensus discussion"
        if "soft" in s or "distributional" in s:
            return "soft labels / distributional labels"
        if "weight" in s:
            return "weighted voting"
        if "other" in s or "mixed" in s:
            return "other / mixed"
        if s in ("none", "no", "na", "n/a", "not reported", "nr", ""):
            return "none"
        return s

    return s


# ── Multi-value field handling ───────────────────────────────────────

_LIST_FIELDS = {"iaa_metric_name", "iaa_value", "papers_topic", "intended_use"}


def _split_list_value(raw: str) -> List[str]:
    """Split pipe/comma-separated IAA values into a list."""
    parts = re.split(r"\s*\|\s*", raw)
    if len(parts) == 1:
        parts = re.split(r"\s*,\s*(?![^(]*\))", raw)
    return [p.strip() for p in parts if p.strip()]


_NUMERIC_FIELDS = {
    "annotators_per_item", "total_annotators",
    "total_annotated_items", "items_per_annotator",
    "human_evaluated_sample", "iaa_value",
}


def _numeric_token_overlap(a: str, b: str) -> bool:
    """Return True if the two strings share at least one integer token."""
    na = _norm("_dummy", a) if False else a  # use raw values
    nb = _norm("_dummy", b) if False else b
    toks_a = set(re.findall(r"\d+", str(na).replace(",", "")))
    toks_b = set(re.findall(r"\d+", str(nb).replace(",", "")))
    if not toks_a or not toks_b:
        return False
    return bool(toks_a & toks_b)


def _field_agrees(field: str, val_a: str, val_b: str) -> bool:
    """Return True if the two values agree for the given field."""
    if field in _LIST_FIELDS:
        norms_a = [_norm(field, v) for v in _split_list_value(val_a)] or ["not reported"]
        norms_b = [_norm(field, v) for v in _split_list_value(val_b)] or ["not reported"]
        return bool(set(norms_a) & set(norms_b))

    na = _norm(field, val_a)
    nb = _norm(field, val_b)
    if na == nb:
        return True

    # Numeric fallback: if both contain numbers, check for shared integer tokens
    if field in _NUMERIC_FIELDS:
        return _numeric_token_overlap(val_a, val_b)

    return False


# ── LLM experiment → canonical field dict ────────────────────────────

def _llm_exp_to_rec(exp: dict) -> dict:
    """Convert an LLM extraction experiment dict to the canonical field dict."""
    rec: dict = {}

    # List fields → join to pipe-separated string for _field_agrees
    for field in ("iaa_metric_name", "iaa_value", "papers_topic", "intended_use"):
        val = exp.get(field, "Not reported")
        if isinstance(val, list):
            val = " | ".join(str(v) for v in val if v)
        rec[field] = val or "Not reported"

    # Direct-mapped string fields
    direct = [
        "human_annotation_type",
        "guidelines_released",
        "annotators_per_item",
        "total_annotators",
        "total_annotated_items",
        "items_per_annotator",
        "human_evaluated_sample",
        "recruitment",
        "crowd_qualifications_screening",
        "annotator_training",
        "language_proficiency",
        "expertise_level",
        "age_reported",
        "gender_reported",
        "nation_origin_reported",
        "nation_residence_reported",
        "education_reported",
        "political_orientation_reported",
        "quality_control_filtering",
        "disagreement_resolution_adjudication",
    ]
    for field in direct:
        rec[field] = str(exp.get(field, "Not reported") or "Not reported").strip()

    # prompts_v2 uses dedicated "reported_compensation" and "payment_rate" fields.
    # Fall back to legacy "compensation" key for older extraction formats.
    rec["reported_compensation"] = str(
        exp.get("reported_compensation") or exp.get("compensation") or "Not reported"
    ).strip()
    rec["payment_rate"] = str(exp.get("payment_rate") or "Not reported").strip()

    # subtask_name from paper_experiment_id as fallback
    rec["subtask_name"] = str(exp.get("subtask_name") or exp.get("paper_experiment_id") or "")

    return rec


# ── Subtask matching ─────────────────────────────────────────────────

def _subtask_key(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\b(the|a|an|of|for|in|and|or)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _match_experiments(
    gold_rows: List[dict], llm_recs: List[dict]
) -> List[Tuple[dict, dict]]:
    """Match gold rows to LLM experiment records by subtask name, falling back to order."""
    pairs: List[Tuple[dict, dict]] = []
    used_llm = set()

    key_to_llm: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(llm_recs):
        key_to_llm[_subtask_key(r.get("subtask_name", ""))].append(i)

    for gold in gold_rows:
        name_g = gold.get("subtask_name", "").strip().lower()
        key_g = _subtask_key(gold.get("subtask_name", ""))
        matched = False

        # Exact match on subtask_name
        for i, r in enumerate(llm_recs):
            if i in used_llm:
                continue
            if r.get("subtask_name", "").strip().lower() == name_g and name_g:
                pairs.append((gold, r))
                used_llm.add(i)
                matched = True
                break

        if not matched:
            # Fuzzy key match
            for i in key_to_llm.get(key_g, []):
                if i not in used_llm and key_g:
                    pairs.append((gold, llm_recs[i]))
                    used_llm.add(i)
                    matched = True
                    break

    # Positional fallback for remaining un-matched items
    remaining_gold = [g for g in gold_rows if not any(g is p[0] for p in pairs)]
    remaining_llm = [llm_recs[i] for i in range(len(llm_recs)) if i not in used_llm]
    for g, l in zip(remaining_gold, remaining_llm):
        pairs.append((g, l))

    return pairs


# ── Gold CSV loading ─────────────────────────────────────────────────

def _detect_link_col(headers: List[str]) -> str:
    for h in headers:
        if h.strip().lower() in ("paper link", "paperlink", "paper_link"):
            return h
    for h in headers:
        if "paper" in h.lower() and "link" in h.lower():
            return h
    raise ValueError(f"No 'Paper Link' column found. Available: {headers}")


def load_gold_csv(path: Path) -> Dict[str, List[dict]]:
    """Load gold CSV → {url: [row_dict, ...]} using canonical field keys."""
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    headers = list(reader.fieldnames or [])

    link_col = _detect_link_col(headers)

    # Build case-insensitive column map
    col_map: Dict[str, str] = {}
    header_lc = {h.strip().lower(): h.strip() for h in headers}
    for canonical, field in _GOLD_COL_MAP.items():
        if canonical in header_lc.values():
            col_map[canonical] = field
        elif canonical.lower() in header_lc:
            col_map[header_lc[canonical.lower()]] = field

    by_url: Dict[str, List[dict]] = defaultdict(list)
    for row in reader:
        url = (row.get(link_col) or "").strip()
        if not url or not url.startswith("http"):
            continue
        rec: dict = {"_url": url}
        for csv_col, field in col_map.items():
            rec[field] = (row.get(csv_col) or "").strip()
        by_url[url].append(rec)

    return dict(by_url)


# ── LLM extraction loading ───────────────────────────────────────────

def load_llm_extraction(url: str, extractions_dir: Path) -> Optional[List[dict]]:
    """Load and return the list of experiment records for a paper URL."""
    stem = url_to_stem(url)
    # Exact match
    exact = extractions_dir / f"{stem}_annotations.json"
    if exact.exists():
        candidates = [exact]
    else:
        candidates = sorted(extractions_dir.glob(f"{stem}*_annotations.json"))

    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if data.get("skip"):
                return []
            exps = data.get("experiments") or data.get("annotations") or []
            return [_llm_exp_to_rec(e) for e in exps if isinstance(e, dict)]
        except Exception as e:
            print(f"  [warn] Failed to load {path.name}: {e}")

    return None  # not found


# ── Krippendorff's Alpha ─────────────────────────────────────────────

def _krippendorff_alpha_nominal(
    pairs: List[Tuple[str, str]],
) -> Optional[float]:
    """Compute Krippendorff's Alpha for nominal data with 2 coders per unit."""
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


# ── Core computation ─────────────────────────────────────────────────

def compute_agreement(
    gold_by_url: Dict[str, List[dict]],
    extractions_dir: Path,
    exclude_fields: frozenset = frozenset(),
) -> dict:
    field_match: Dict[str, int] = defaultdict(int)
    field_total: Dict[str, int] = defaultdict(int)
    field_value_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    pair_details: List[dict] = []
    paper_stats: Dict[str, dict] = {}
    missing_papers: List[str] = []
    total_pairs = 0

    for url, gold_rows in sorted(gold_by_url.items()):
        stem = url_to_stem(url)
        llm_recs = load_llm_extraction(url, extractions_dir)

        if llm_recs is None:
            missing_papers.append(stem)
            continue
        if not llm_recs:
            continue

        pairs = _match_experiments(gold_rows, llm_recs)
        paper_field_match: Dict[str, int] = defaultdict(int)
        paper_field_total: Dict[str, int] = defaultdict(int)

        for gold, llm in pairs:
            total_pairs += 1
            per_field: Dict[str, bool] = {}
            for field in FIELDS:
                if field in exclude_fields:
                    continue
                v_gold = gold.get(field, "")
                v_llm = llm.get(field, "")
                agrees = _field_agrees(field, v_gold, v_llm)
                field_match[field] += int(agrees)
                field_total[field] += 1
                paper_field_match[field] += int(agrees)
                paper_field_total[field] += 1
                per_field[field] = agrees

                # Collect normalized value pairs for Krippendorff's Alpha
                if field in _LIST_FIELDS:
                    norms_g = sorted(set(
                        _norm(field, v) for v in _split_list_value(v_gold)
                    )) or ["not reported"]
                    norms_l = sorted(set(
                        _norm(field, v) for v in _split_list_value(v_llm)
                    )) or ["not reported"]
                    field_value_pairs[field].append(
                        ("|".join(norms_g), "|".join(norms_l))
                    )
                else:
                    field_value_pairs[field].append(
                        (_norm(field, v_gold), _norm(field, v_llm))
                    )

            pair_details.append({
                "url": url,
                "paper_stem": stem,
                "subtask_gold": gold.get("subtask_name", ""),
                "subtask_llm": llm.get("subtask_name", ""),
                "field_agreement": per_field,
                "_val_gold": {f: gold.get(f, "") for f in FIELDS if f not in exclude_fields},
                "_val_llm": {f: llm.get(f, "") for f in FIELDS if f not in exclude_fields},
            })

        # Per-paper aggregate
        pm = sum(paper_field_match.values())
        pt = sum(paper_field_total.values())
        paper_stats[stem] = {
            "pairs": len(pairs),
            "gold_count": len(gold_rows),
            "llm_count": len(llm_recs),
            "overall_pct": round(pm / pt * 100, 1) if pt else None,
        }

    # Overall aggregates
    per_field_pct: Dict[str, Optional[float]] = {}
    for field in FIELDS:
        if field in exclude_fields:
            continue
        t = field_total.get(field, 0)
        per_field_pct[field] = round(field_match[field] / t * 100, 1) if t > 0 else None

    overall_matches = sum(field_match.values())
    overall_total = sum(field_total.values())
    overall_pct = round(overall_matches / overall_total * 100, 1) if overall_total else None

    # Krippendorff's Alpha per field
    per_field_alpha: Dict[str, Optional[float]] = {}
    for field in FIELDS:
        if field in exclude_fields:
            continue
        per_field_alpha[field] = _krippendorff_alpha_nominal(
            field_value_pairs.get(field, [])
        )

    # Macro-average alpha (mean of per-field alphas)
    valid_alphas = [a for a in per_field_alpha.values() if a is not None]
    overall_alpha = round(sum(valid_alphas) / len(valid_alphas), 4) if valid_alphas else None

    # Category agreement
    category_pct: Dict[str, Optional[float]] = {}
    category_alpha: Dict[str, Optional[float]] = {}
    for cat, cat_fields in FIELD_GROUPS.items():
        cm = sum(field_match.get(f, 0) for f in cat_fields)
        ct = sum(field_total.get(f, 0) for f in cat_fields)
        category_pct[cat] = round(cm / ct * 100, 1) if ct else None
        cat_as = [per_field_alpha[f] for f in cat_fields
                  if per_field_alpha.get(f) is not None]
        category_alpha[cat] = round(
            sum(cat_as) / len(cat_as), 4
        ) if cat_as else None

    return {
        "total_pairs": total_pairs,
        "overall_pct": overall_pct,
        "overall_alpha": overall_alpha,
        "per_field_pct": per_field_pct,
        "per_field_alpha": per_field_alpha,
        "field_match": dict(field_match),
        "field_total": dict(field_total),
        "category_agreement_pct": category_pct,
        "category_alpha": category_alpha,
        "paper_agreement_pct": {s: d["overall_pct"] for s, d in paper_stats.items()},
        "paper_experiment_counts": {
            s: {"gold": d["gold_count"], "llm": d["llm_count"]}
            for s, d in paper_stats.items()
        },
        "pair_details": pair_details,
        "missing_papers": missing_papers,
    }


# ── Console report ───────────────────────────────────────────────────

def print_report(results: dict, extractions_dir: Path) -> None:
    W = 85
    total = results["total_pairs"]
    overall = results["overall_pct"]
    overall_str = f"{overall:.1f}%" if overall is not None else "N/A"

    print()
    print("=" * W)
    print("  HUMAN–LLM AGREEMENT REPORT")
    print(f"  Extractions : {extractions_dir}")
    print(f"  ({total} matched experiment pairs evaluated)")
    print("=" * W)
    print()
    alpha = results.get("overall_alpha")
    alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
    print(f"  Overall agreement: {overall_str}    Krippendorff's α (macro-avg): {alpha_str}")
    print()

    sep = "─" * W
    print(sep)
    _kripp_hdr = "Kripp " + chr(945)
    print(f"  {'FIELD':<42} {'Agree%':>7}  {_kripp_hdr:>8}  {'Match / Total':>14}")
    print(sep)

    for group, group_fields in FIELD_GROUPS.items():
        visible = [f for f in group_fields if results["field_total"].get(f, 0) > 0]
        if not visible:
            continue
        print(f"\n  [{group}]")
        for field in visible:
            label = FIELD_LABELS.get(field, field)
            pct = results["per_field_pct"].get(field)
            fa = results.get("per_field_alpha", {}).get(field)
            m = results["field_match"].get(field, 0)
            t = results["field_total"].get(field, 0)
            if pct is not None:
                tick = "✓" if pct >= 70 else "✗"
                pct_str = f"{pct:.1f}%"
            else:
                tick = "✗"
                pct_str = "N/A"
            a_str = f"{fa:.3f}" if fa is not None else "  N/A"
            print(f"    {tick} {label:<40} {pct_str:>7}  {a_str:>8}   {m:>4} / {t:>4}")

    print()
    print(sep)
    print("  CATEGORY SUMMARY")
    print(sep)
    cat_alphas = results.get("category_alpha", {})
    for cat, pct in results["category_agreement_pct"].items():
        bar_len = int((pct or 0) / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        ca = cat_alphas.get(cat)
        ca_str = f"α={ca:.3f}" if ca is not None else "α=N/A"
        print(f"    {cat:<20s} {bar}  {pct_str}  {ca_str}")

    print()
    print(sep)
    print("  PER-PAPER AGREEMENT")
    print(sep)
    for stem, pct in sorted(
        results["paper_agreement_pct"].items(),
        key=lambda x: x[1] or 0, reverse=True,
    ):
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        print(f"    {stem:<55s} {pct_str}")

    if results["missing_papers"]:
        print()
        print(sep)
        print("  PAPERS WITH NO EXTRACTION FILE")
        print(sep)
        for stem in results["missing_papers"]:
            print(f"    {stem}")

    # ── Experiment count section ────────────────────────────────────
    exp_counts = results.get("paper_experiment_counts", {})
    if exp_counts:
        print()
        print(sep)
        print("  EXPERIMENTS PER PAPER  (Gold vs LLM)")
        print(sep)
        print(f"    {'Paper':<45}  {'Gold':>4}  {'LLM':>4}  {'Match':>5}")
        print(f"    {'─' * 45}  {'────':>4}  {'────':>4}  {'─────':>5}")
        exact_matches = 0
        llm_over = 0
        llm_under = 0
        for stem in sorted(exp_counts):
            g = exp_counts[stem]["gold"]
            l = exp_counts[stem]["llm"]
            match_sym = "✓" if g == l else ("↑" if l > g else "↓")
            print(f"    {stem:<45}  {g:>4}  {l:>4}  {match_sym:>5}")
            if g == l:
                exact_matches += 1
            elif l > g:
                llm_over += 1
            else:
                llm_under += 1
        n = len(exp_counts)
        avg_gold = sum(v["gold"] for v in exp_counts.values()) / n if n else 0
        avg_llm  = sum(v["llm"]  for v in exp_counts.values()) / n if n else 0
        print()
        print(f"    Summary ({n} papers):  "
              f"avg gold={avg_gold:.2f}  avg LLM={avg_llm:.2f}  "
              f"exact match={exact_matches}/{n} ({exact_matches/n*100:.0f}%)  "
              f"LLM over-extracts={llm_over}  under-extracts={llm_under}")
        print(f"    Legend: ✓ count matches  ↑ LLM extracted more  ↓ LLM extracted fewer")

    print()
    print("=" * W)


def print_disagreements(results: dict, top_n: int = 20) -> None:
    """Print examples where human and LLM disagree."""
    W = 75
    print()
    print("=" * W)
    print(f"  DISAGREEMENT EXAMPLES (up to {top_n} per field with <70% agreement)")
    print("=" * W)

    low_fields = [
        f for f in FIELDS
        if (results["per_field_pct"].get(f) or 0) < 70
        and results["field_total"].get(f, 0) > 0
    ]

    for field in low_fields:
        label = FIELD_LABELS.get(field, field)
        examples = [
            d for d in results["pair_details"]
            if not d["field_agreement"].get(field, True)
        ][:top_n]
        if not examples:
            continue
        print(f"\n  {label}  ({results['per_field_pct'].get(field):.1f}%)")
        print(f"  {'─' * 60}")
        for ex in examples[:5]:
            g = _norm(field, ex["_val_gold"].get(field, ""))
            l = _norm(field, ex["_val_llm"].get(field, ""))
            st = ex.get("subtask_gold") or ex.get("paper_stem", "")
            print(f"    {ex['paper_stem'][:30]:<30}  gold={g!r}  llm={l!r}")
            if st:
                print(f"      subtask: {st[:60]}")


# ── Main ─────────────────────────────────────────────────────────────
# ── Main ─────────────────────────────────────────────────────────────


def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic human–LLM agreement (no judge LLM required)"
    )
    p.add_argument(
        "--gold_csv",
        default="data/gold_standard.csv",
        help="Path to gold standard CSV with 'Paper Link' column",
    )
    p.add_argument(
        "--extractions_dir",
        default="Processed_PDFs/Combined_Extractions_gpt-4.1-global-batch",
        help="Directory containing *_annotations.json extraction files",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to save results JSON",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization (not yet implemented, accepted for compatibility)",
    )
    p.add_argument(
        "--show-disagreements",
        action="store_true",
        help="Print per-field disagreement examples",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        metavar="FIELD",
        default=[],
        help="Field names to exclude from agreement computation and report "
             "(e.g. --exclude papers_topic payment_rate)",
    )
    args = p.parse_args()
    return args


def main() -> None:
    args = arg_parser()
    gold_path = Path(args.gold_csv)
    if not gold_path.exists():
        print(f"ERROR: Gold CSV not found: {gold_path}", file=sys.stderr)
        sys.exit(1)

    extractions_dir = Path(args.extractions_dir)
    if not extractions_dir.exists():
        print(f"ERROR: Extractions directory not found: {extractions_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading gold CSV: {gold_path.name} …")
    gold_by_url = load_gold_csv(gold_path)
    print(f"  {len(gold_by_url)} unique paper URLs, "
          f"{sum(len(v) for v in gold_by_url.values())} experiment rows")

    print(f"Extractions dir : {extractions_dir}")

    exclude_fields = frozenset(args.exclude)
    if exclude_fields:
        print(f"Excluding fields: {', '.join(sorted(exclude_fields))}")

    results = compute_agreement(gold_by_url, extractions_dir, exclude_fields=exclude_fields)

    print_report(results, extractions_dir)

    if args.show_disagreements:
        print_disagreements(results)

    if args.output:
        out_path = Path(args.output)
        # Remove large pair_details from saved JSON to keep it compact
        save = {k: v for k, v in results.items() if k != "pair_details"}
        out_path.write_text(json.dumps(save, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
