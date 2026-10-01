"""
Standardize Human CSV Annotations.

Normalizes ALL values in the human CSV to the exact canonical taxonomy values
so that agreement scoring with LLM outputs produces fair comparisons.

Run:
    python standardize_human_csv.py                       # dry-run (prints changes)
    python standardize_human_csv.py --apply               # overwrites original CSV
    python standardize_human_csv.py --output cleaned.csv  # writes to separate file

Changes are printed to stdout so the user can audit every single edit.
"""

import argparse
import csv
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

# ======================================================================
# Canonical taxonomy values
# ======================================================================

CANONICAL_NLP_TASK = [
    "Closed-label classification",
    "Ordinal / scalar rating",
    "Span-level or structural annotation",
    "Comparative / preference judgments",
    "Open-ended generation or transformation",
    "Other",
]

CANONICAL_IAA_METRIC = [
    "Fleiss' κ",
    "Cohen's κ",
    "Krippendorff's α",
    "F1 agreement",
    "Pearson",
    "Spearman",
    "Majority agreement",
    "Kendall's tau",
    "Total agreement",
    "Other",
    "Not reported",
]

CANONICAL_RECRUITMENT = [
    "Amazon Turks",
    "Profilic",
    "Toloka",
    "Yahoo! Crowdsourcing",
    "Other crowdsourcing platform",
    "Recruiting not via crowdsourcing platform",
    "Any authors as annotators",
    "Not reported",
]

CANONICAL_EXPERTISE = [
    "High",
    "Medium",
    "Mixed",
    "General Language/Knowledge task",
    "Not reported",
]

CANONICAL_LANG_PROF = ["Native", "Non-Native", "Mixed", "Medium", "Not reported"]

CANONICAL_REPORTED_COMPENSATION = ["Paid", "Free", "NA", "Not reported"]

CANONICAL_PAYMENT_RATE = ["Specific numeric rate", "General mention", "No", "NA"]

CANONICAL_ADJUDICATION = [
    "Majority vote",
    "Expert adjudication",
    "Third annotator",
    "Consensus discussion",
    "Soft labels / Distributional labels",
    "Weighted voting",
    "Other / mixed",
    "None",
]

CANONICAL_PAPERS_TOPIC = [
    "Linguistics",
    "Discourse & Pragmatics",
    "Information Extraction",
    "QA & information access",
    "Semantics, inference & similarity",
    "Subjective language & social meaning",
    "Open-ended generation",
    "Compression & reformulation tasks",
    "Multilingual & translation",
    "Multimodal tasks",
]

CANONICAL_INTENDED_USE = [
    "Resource creation",
    "Human performance",
    "Model output evaluation",
    "None of the above",
]

# ======================================================================
# Column-name detection helpers
# ======================================================================

COL_ALIASES = {
    "nlp_task":        ["NLP task", "NLP Task", "NLP task type"],
    "subtask_name":    ["Subtask name", "Subtask", "Task name"],
    "guidelines":      ["Annotation Guideline released", "Guidelines released", "Guidelines"],
    "iaa_metric":      ["IAA Metric Name", "IAA Metric", "Metric"],
    "iaa_value":       ["IAA Value", "IAA"],
    "ann_per_item":    ["Number of Annotators per Item", "Annotators per item", "Annotators/item",
                        "Number of Annotators per item"],
    "total_ann":       ["Total number of annotators", "Total annotators", "# Annotators"],
    "total_items":     ["Total number of annotated items", "Total annotated items", "# Items"],
    "items_per_ann":   ["Number of items per annotator", "Items per annotator"],
    "recruitment":     ["Recruitment", "Recruiting"],
    "crowd_qual":      ["Crowd Qualifications & Screening", "Crowd Qualifications", "Qualifications"],
    "training":        ["Annotator training", "Annotator Training", "Training"],
    "lang_prof":       ["Annotation task language proficiency", "Annotation task language proficency",
                        "Language proficiency", "Language"],
    "expertise":       ["Level of expertise", "Expertise level", "Expertise"],
    "reported_comp":   ["Reported compensation", "Reported Compensation", "Payment Currency", "Currency"],
    "payment_rate":    ["Payment rate", "Payment Rate", "Compensation", "Payment"],
    "age":             ["Age", "Age reported"],
    "gender":          ["Gender", "Gender reported"],
    "nation_origin":   ["Nation of origin", "Nationality"],
    "nation_res":      ["Nation of residence", "Residence"],
    "education":       ["Education", "Education reported"],
    "political":       ["Political orientation", "Political"],
    "qc":              ["Quality Control & Filtering", "Quality Control", "QC"],
    "disagree":        ["Disagreement Resolution / Adjudication", "Disagreement Resolution",
                        "Adjudication"],
    "papers_topic":    ["Paper's topic", "Papers topic", "Topic"],
    "intended_use":    ["Intended use", "Intended Use"],
    "human_eval":      ["Human-evaluated sample", "Human evaluated sample"],
}


def _find_col(headers: List[str], key: str) -> Optional[str]:
    """Return the actual header that matches one of the aliases for *key*."""
    aliases = COL_ALIASES.get(key, [])
    lower_map = {h.lower().strip(): h for h in headers}
    for alias in aliases:
        if alias in headers:
            return alias
        if alias.lower().strip() in lower_map:
            return lower_map[alias.lower().strip()]
    return None


# ======================================================================
# Normalization functions (field-level)
# ======================================================================

def _strip(v: str) -> str:
    """Strip whitespace + remove BOM / zero-width chars."""
    return v.strip().strip("\ufeff").strip()


def norm_nlp_task(v: str) -> str:
    """Strip parenthetical descriptions from NLP task type."""
    v = _strip(v)
    if not v:
        return "Other"
    # Match the canonical prefix before any parenthetical
    for canon in CANONICAL_NLP_TASK:
        if v.lower().startswith(canon.lower()):
            return canon
    # Fuzzy fallbacks
    vl = v.lower()
    if any(k in vl for k in ("classif", "label", "tagging", "categor", "binary", "multi-class")):
        return "Closed-label classification"
    if any(k in vl for k in ("ordinal", "scalar", "rating", "likert", "scale")):
        return "Ordinal / scalar rating"
    if any(k in vl for k in ("span", "ner", "structur", "sequence label", "coreference",
                              "dependency", "constituency")):
        return "Span-level or structural annotation"
    if any(k in vl for k in ("compar", "preference", "ranking", "pairwise", "best")):
        return "Comparative / preference judgments"
    if any(k in vl for k in ("generat", "transform", "open-ended", "open ended",
                              "summar", "paraphras", "translat", "simplif")):
        return "Open-ended generation or transformation"
    if vl == "other":
        return "Other"
    return v  # leave as-is if truly unknown


def _fix_greek(v: str) -> str:
    """Fix mojibake Greek letters from UTF-8 read as latin-1 / cp1252."""
    v = v.replace("Îº", "κ").replace("α", "α")
    v = v.replace("Îš", "κ").replace("Î'", "α")
    v = re.sub(r"(?i)\bkappa\b", "κ", v)
    v = re.sub(r"(?i)\balpha\b", "α", v)
    v = v.replace("\u2019", "'").replace("\u2018", "'")
    v = v.replace("\u201c", '"').replace("\u201d", '"')
    return v


def _norm_single_iaa_metric(v: str) -> str:
    """Normalize one IAA metric name to canonical value."""
    v = _fix_greek(_strip(v))
    if not v:
        return "Not reported"
    for canon in CANONICAL_IAA_METRIC:
        if v.lower() == canon.lower():
            return canon
    vl = v.lower()
    if vl in ("none", "na", "n/a", "not reported", "nr", "-", ""):
        return "Not reported"
    if "fleiss" in vl:
        return "Fleiss' κ"
    if "cohen" in vl:
        return "Cohen's κ"
    if "krippendorff" in vl:
        return "Krippendorff's α"
    if vl in ("f1", "f1-score", "f1 score", "f1 agreement"):
        return "F1 agreement"
    if "pearson" in vl:
        return "Pearson"
    if "spearman" in vl:
        return "Spearman"
    if "kendall" in vl:
        return "Kendall's tau"
    if "majority" in vl:
        return "Majority agreement"
    if any(k in vl for k in ("percent agreement", "percentage agreement", "total agreement",
                              "exact agreement", "observed agreement", "raw agreement")):
        return "Total agreement"
    if vl in ("icc", "scott", "gwet", "bleu"):
        return "Other"
    return "Other"


def norm_iaa_metric(v: str) -> str:
    """Normalize IAA metric name (handles pipe/comma separated lists)."""
    v = _fix_greek(_strip(v))
    if not v:
        return "Not reported"
    parts = re.split(r"\s*[|]\s*", v)
    if len(parts) == 1:
        parts = re.split(r"\s*,\s*", v)
    normed = [_norm_single_iaa_metric(p) for p in parts if _strip(p)]
    if not normed:
        return "Not reported"
    return " | ".join(normed)


def norm_iaa_value(v: str) -> str:
    """Normalize IAA value (handles pipe/comma separated lists)."""
    v = _strip(v)
    if not v or v.lower() in ("none", "na", "n/a", "not reported", "nr", "-"):
        return "Not reported"
    parts = re.split(r"\s*[|]\s*", v)
    if len(parts) == 1:
        parts = re.split(r"\s*,\s*", v)
    normed = []
    for p in parts:
        p = _strip(p)
        if not p:
            continue
        if p.lower() in ("na", "n/a"):
            normed.append("NA")
        elif p.lower() in ("not reported", "nr", "none", "-"):
            normed.append("Not reported")
        else:
            normed.append(p)
    if not normed:
        return "Not reported"
    return " | ".join(normed)


def norm_recruitment(v: str) -> str:
    """Normalize recruitment to canonical values."""
    v = _strip(v)
    if not v or v.lower() in ("not reported", "nr", "unknown", "-", ""):
        return "Not reported"
    vl = v.lower()
    if "," in v:
        parts = [_strip(p) for p in v.split(",")]
        normed_parts = [norm_recruitment(p) for p in parts if _strip(p)]
        real_parts = [p for p in normed_parts if p != "Not reported"]
        if real_parts:
            return real_parts[0]
        return "Not reported"
    if any(k in vl for k in ("mturk", "mechanical turk", "amazon turk", "amazon mechanical")):
        return "Amazon Turks"
    if vl == "amt":
        return "Amazon Turks"
    if "prolific" in vl or "profilic" in vl:
        return "Profilic"
    if "toloka" in vl:
        return "Toloka"
    if "yahoo" in vl:
        return "Yahoo! Crowdsourcing"
    if "author" in vl:
        return "Any authors as annotators"
    if "local expert" in vl:
        return "Recruiting not via crowdsourcing platform"
    if any(k in vl for k in ("not via", "in-house", "internal", "lab ", "university",
                              "volunteer", "hired", "recruit")):
        return "Recruiting not via crowdsourcing platform"
    if "crowdsourc" in vl:
        return "Other crowdsourcing platform"
    return v


def norm_yes_no(v: str) -> str:
    """Normalize to Yes/No (demographics, guidelines_released)."""
    v = _strip(v)
    vl = v.lower()
    if vl in ("yes", "true", "1"):
        return "Yes"
    return "No"


def norm_guidelines(v: str) -> str:
    """Normalize guidelines_released → Yes or No."""
    return norm_yes_no(v)


def norm_yes_no_na(v: str) -> str:
    """Normalize to Yes/No/NA (annotator_training, quality_control)."""
    v = _strip(v)
    vl = v.lower()
    if vl in ("yes", "true", "1"):
        return "Yes"
    if vl in ("no", "false", "0"):
        return "No"
    return "NA"


def norm_quality_control(v: str) -> str:
    """quality_control_filtering → Yes, No, or NA."""
    return norm_yes_no_na(v)


def norm_crowd_qual(v: str, recruitment_val: str) -> str:
    """crowd_qualifications_screening → Yes, No, or NA."""
    v = _strip(v)
    vl = v.lower()
    rec_lower = recruitment_val.lower() if recruitment_val else ""
    is_crowd = any(k in rec_lower for k in ("crowdsourcing", "amazon turks", "profilic",
                                             "toloka", "yahoo", "other crowdsourcing"))
    if not is_crowd:
        return "NA"
    if vl in ("yes", "true", "1"):
        return "Yes"
    if vl in ("no", "false", "0"):
        return "No"
    return "No"


def norm_training(v: str) -> str:
    """annotator_training → Yes, No, or NA."""
    return norm_yes_no_na(v)


def norm_lang_prof(v: str) -> str:
    """language_proficiency → Native, Non-Native, Mixed, Medium, Not reported."""
    v = _strip(v)
    vl = v.lower()
    if vl in ("not reported", "nr", "unknown", "-", "", "na", "n/a"):
        return "Not reported"
    if "non" in vl and "native" in vl:
        return "Non-Native"
    if vl == "medium":
        return "Medium"
    if "native" in vl:
        return "Native"
    if vl in ("mixed", "both"):
        return "Mixed"
    if vl == "yes":
        return "Not reported"
    if vl == "no":
        return "Not reported"
    return "Not reported"


def norm_disagreement(v: str) -> str:
    """disagreement_resolution_adjudication → canonical adjudication method or None."""
    v = _strip(v)
    if not v or v.lower() in ("not reported", "nr", "na", "n/a", "no", "-", "", "none"):
        return "None"
    vl = v.lower()
    for canon in CANONICAL_ADJUDICATION:
        if vl == canon.lower():
            return canon
    if "majority" in vl:
        return "Majority vote"
    if "expert" in vl:
        return "Expert adjudication"
    if "third" in vl:
        return "Third annotator"
    if "consensus" in vl or "discussion" in vl:
        return "Consensus discussion"
    if "soft" in vl or "distribut" in vl:
        return "Soft labels / Distributional labels"
    if "weight" in vl:
        return "Weighted voting"
    if vl in ("yes", "true", "1"):
        return "Other / mixed"
    return "Other / mixed"


def norm_expertise(v: str) -> str:
    """expertise_level → canonical value."""
    v = _strip(v)
    if not v or v.lower() in ("not reported", "nr", "unknown", "-", ""):
        return "Not reported"
    vl = v.lower()
    for canon in CANONICAL_EXPERTISE:
        if vl == canon.lower():
            return canon
    if any(k in vl for k in ("expert", "professional", "high", "researcher", "physician",
                              "linguist", "domain")):
        return "High"
    if any(k in vl for k in ("student", "medium", "graduate", "research assistant")):
        return "Medium"
    if "mixed" in vl:
        return "Mixed"
    if any(k in vl for k in ("crowd", "lay", "non-expert", "low", "mturk",
                              "general language", "general knowledge")):
        return "General Language/Knowledge task"
    if vl in ("na", "n/a"):
        return "General Language/Knowledge task"
    if "," in v:
        parts = [_strip(p) for p in v.split(",")]
        normed_parts = [norm_expertise(p) for p in parts]
        return normed_parts[0] if normed_parts else "Not reported"
    return "Not reported"


def norm_reported_compensation(v: str) -> str:
    """reported_compensation → Paid, Free, NA, or Not reported."""
    v = _strip(v)
    vl = v.lower()
    if vl in ("paid",):
        return "Paid"
    if vl in ("free", "voluntary", "volunteer"):
        return "Free"
    if vl in ("na", "n/a", "not applicable"):
        return "NA"
    return "Not reported"


def norm_payment_rate(v: str) -> str:
    """payment_rate → Specific numeric rate, General mention, No, or AA."""
    v = _strip(v)
    vl = v.lower()
    if vl in ("specific numeric rate", "specific", "numeric"):
        return "Specific numeric rate"
    if vl in ("general mention", "general"):
        return "General mention"
    if vl in ("no", "none", "false"):
        return "No"
    return "NA"


def norm_workload(v: str) -> str:
    """Workload fields → keep as-is if valid, else 'Not reported'."""
    v = _strip(v)
    if not v or v.lower() in ("not reported", "nr", "unknown", "-", "", "na", "n/a",
                               "not applicable"):
        return "Not reported"
    return v


def norm_papers_topic(v: str) -> str:
    """Normalize a single papers_topic value to canonical form."""
    v = _strip(v).strip('"')
    if not v:
        return v
    vl = v.lower()
    for canon in CANONICAL_PAPERS_TOPIC:
        if vl == canon.lower():
            return canon
    return v


def norm_intended_use(v: str) -> str:
    """Normalize a single intended_use value to canonical form."""
    v = _strip(v)
    if not v:
        return v
    vl = v.lower()
    for canon in CANONICAL_INTENDED_USE:
        if vl == canon.lower():
            return canon
    return v


def norm_multi_value_field(v: str, normalizer) -> str:
    """Normalize a comma-separated multi-value field, applying normalizer to each part."""
    import csv as _csv
    import io as _io
    v = _strip(v)
    if not v:
        return v
    parts = next(_csv.reader(_io.StringIO(v)))
    normed = [normalizer(_strip(p)) for p in parts if _strip(p)]
    return ", ".join(normed) if normed else v


# ======================================================================
# Main standardization logic
# ======================================================================

def standardize_row(row: dict, headers: List[str]) -> dict:
    """Standardize one CSV row in-place. Returns the modified row."""
    out = dict(row)

    # --- NLP task type ---
    col = _find_col(headers, "nlp_task")
    if col and col in out:
        out[col] = norm_nlp_task(out[col])

    # --- Guidelines released ---
    col = _find_col(headers, "guidelines")
    if col and col in out:
        out[col] = norm_guidelines(out[col])

    # --- IAA Metric Name ---
    col_metric = _find_col(headers, "iaa_metric")
    if col_metric and col_metric in out:
        out[col_metric] = norm_iaa_metric(out[col_metric])

    # --- IAA Value ---
    col_val = _find_col(headers, "iaa_value")
    if col_val and col_val in out:
        out[col_val] = norm_iaa_value(out[col_val])

    # --- Workload fields ---
    for key in ("ann_per_item", "total_ann", "total_items", "items_per_ann"):
        col = _find_col(headers, key)
        if col and col in out:
            out[col] = norm_workload(out[col])

    # --- Recruitment (normalize BEFORE crowd_qual so we can pass it) ---
    col_rec = _find_col(headers, "recruitment")
    if col_rec and col_rec in out:
        out[col_rec] = norm_recruitment(out[col_rec])

    # --- Crowd Qualifications ---
    col_cq = _find_col(headers, "crowd_qual")
    rec_val = out.get(col_rec, "") if col_rec else ""
    if col_cq and col_cq in out:
        out[col_cq] = norm_crowd_qual(out[col_cq], rec_val)

    # --- Annotator training ---
    col = _find_col(headers, "training")
    if col and col in out:
        out[col] = norm_training(out[col])

    # --- Language proficiency ---
    col = _find_col(headers, "lang_prof")
    if col and col in out:
        out[col] = norm_lang_prof(out[col])

    # --- Expertise ---
    col = _find_col(headers, "expertise")
    if col and col in out:
        out[col] = norm_expertise(out[col])

    # --- Reported compensation ---
    col = _find_col(headers, "reported_comp")
    if col and col in out:
        out[col] = norm_reported_compensation(out[col])

    # --- Payment rate ---
    col = _find_col(headers, "payment_rate")
    if col and col in out:
        out[col] = norm_payment_rate(out[col])

    # --- Demographics (all Yes/No only) ---
    for key in ("age", "gender", "nation_origin", "nation_res", "education", "political"):
        col = _find_col(headers, key)
        if col and col in out:
            out[col] = norm_yes_no(out[col])

    # --- Quality Control ---
    col = _find_col(headers, "qc")
    if col and col in out:
        out[col] = norm_quality_control(out[col])

    # --- Disagreement Resolution ---
    col = _find_col(headers, "disagree")
    if col and col in out:
        out[col] = norm_disagreement(out[col])

    # --- Paper's topic (multi-value) ---
    col = _find_col(headers, "papers_topic")
    if col and col in out:
        out[col] = norm_multi_value_field(out[col], norm_papers_topic)

    # --- Intended use (multi-value) ---
    col = _find_col(headers, "intended_use")
    if col and col in out:
        out[col] = norm_multi_value_field(out[col], norm_intended_use)

    return out


def diff_rows(original: dict, standardized: dict, row_idx: int) -> List[str]:
    """Return human-readable list of changes for one row."""
    changes = []
    for key in original:
        old = original[key]
        new = standardized.get(key, old)
        if old != new:
            changes.append(f"  Row {row_idx:>3} | {key:45s} | {old!r:60s} → {new!r}")
    return changes


def run(csv_path: Path, output_path: Optional[Path] = None, apply: bool = False):
    """Run standardization on a CSV file."""
    text = csv_path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    headers = reader.fieldnames or []
    rows = list(reader)

    print(f"Read {len(rows)} rows, {len(headers)} columns from {csv_path.name}")
    print(f"Columns: {headers}\n")

    all_changes = []
    standardized_rows = []

    for i, row in enumerate(rows, start=2):  # row 1 = header
        original = dict(row)
        normed = standardize_row(row, headers)
        standardized_rows.append(normed)
        changes = diff_rows(original, normed, i)
        all_changes.extend(changes)

    if all_changes:
        print(f"=== {len(all_changes)} field changes ===\n")
        for c in all_changes:
            print(c)
    else:
        print("No changes needed — CSV is already standardized.")

    dest = output_path or (csv_path if apply else None)
    if dest:
        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(standardized_rows)
        print(f"\nWrote standardized CSV to {dest}")
    elif not apply:
        print(f"\nDry-run complete. Use --apply to overwrite or --output <file> to save.")


def main():
    parser = argparse.ArgumentParser(description="Standardize human CSV annotations to canonical taxonomy values")
    parser.add_argument(
        "--csv",
        default="Processed_PDFs/Combined_CSV_Annotations/annotations_combined_total.csv",
        help="Path to the human CSV file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write standardized CSV to this path (leave empty for dry-run or --apply)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Overwrite the original CSV in-place",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    run(csv_path, output_path, args.apply)


if __name__ == "__main__":
    main()
