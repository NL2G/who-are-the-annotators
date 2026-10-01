"""
Large-Scale Extraction Analysis.

Reads all *_annotations.json files produced by the large-scale extraction
pipeline and generates descriptive statistics + visualizations.

Usage
─────
  python analyze_largescale.py --extractions_dir Processed_PDFs/LargeScale_Extractions_gemini-3.1-pro-preview
  python analyze_largescale.py   # auto-detect most recent LargeScale_Extractions_*

Output
──────
  • Console report with per-field value distributions
  • CSV summary table  (analysis_output/field_distributions.csv)
  • Visualisations     (analysis_output/*.png)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

# ── project root ──────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
PDFS_ROOT     = SCRIPT_DIR / "Processed_PDFs"
MANIFEST_DEFAULT = SCRIPT_DIR / "combined_papers" / "random_acl_sample" / "manifest.csv"


def load_manifest(manifest_path: Optional[Path]) -> dict[str, dict]:
    """Load manifest.csv and return a lookup dict.

    Indexed by BOTH ``full_id`` (e.g. ``2025.naacl-long.425``) AND by the
    bare PDF filename (e.g. ``425.IHEval....pdf``) so that extraction JSON
    paper_ids derived from PDF filenames also match.

    Returns an empty dict if no manifest path is given or the file does not exist.
    """
    import csv
    if manifest_path is None:
        manifest_path = MANIFEST_DEFAULT
    if not manifest_path.is_file():
        return {}
    index: dict[str, dict] = {}
    with manifest_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fid = row.get("full_id", "").strip()
            if fid:
                index[fid] = row
            # Also index by bare PDF filename so filenames-as-paper_ids match
            pdf_fn = row.get("pdf_filename", "")
            if pdf_fn:
                pdf_name = Path(pdf_fn.replace("\\", "/")).name  # e.g. "425.Title.pdf"
                if pdf_name and pdf_name not in index:
                    index[pdf_name] = row
    print(f"  Loaded manifest: {len(index):,} entries from {manifest_path}")
    return index

# ── field taxonomy (mirrors human_llm_agreement.py) ──────────────────────────
FIELD_GROUPS = {
    "Task & Data": [
        "human_annotation_type",
        "guidelines_released",
        "iaa_metric_name",
        "iaa_value",
    ],
    "Annotator Count": [
        "annotators_per_item",
        "total_annotators",
        "total_annotated_items",
        "items_per_annotator",
    ],
    "Process & Quality": [
        "recruitment",
        "crowd_qualifications_screening",
        "annotator_training",
        "quality_control_filtering",
        "disagreement_resolution_adjudication",
    ],
    "Demographics": [
        "language_proficiency",
        "expertise_level",
        "age_reported",
        "gender_reported",
        "nation_origin_reported",
        "nation_residence_reported",
        "education_reported",
        "political_orientation_reported",
    ],
    "Compensation": [
        "compensation",
        "payment_currency",
    ],
}

_ALL_FIELDS: list[str] = [f for fs in FIELD_GROUPS.values() for f in fs]

_NUMERIC_FIELDS = {
    "annotators_per_item", "total_annotators",
    "total_annotated_items", "items_per_annotator",
}

_YES_NO_FIELDS = {
    "guidelines_released", "annotator_training",
    "quality_control_filtering",
    "age_reported", "gender_reported",
    "nation_origin_reported", "nation_residence_reported",
    "education_reported", "political_orientation_reported",
}

_LIST_FIELDS = {"iaa_metric_name", "iaa_value"}

FIELD_LABELS = {
    "human_annotation_type": "Human Annotation Task Type",
    "guidelines_released": "Guidelines Released",
    "iaa_metric_name": "IAA Metric Name",
    "iaa_value": "IAA Value",
    "annotators_per_item": "Annotators / Item",
    "total_annotators": "Total Annotators",
    "total_annotated_items": "Total Items",
    "items_per_annotator": "Items / Annotator",
    "recruitment": "Recruitment",
    "crowd_qualifications_screening": "Crowd Screening",
    "annotator_training": "Training",
    "language_proficiency": "Language Proficiency",
    "expertise_level": "Expertise Level",
    "compensation": "Compensation",
    "payment_currency": "Payment Currency",
    "age_reported": "Age Reported",
    "gender_reported": "Gender Reported",
    "nation_origin_reported": "Nation Origin Reported",
    "nation_residence_reported": "Nation Residence Reported",
    "education_reported": "Education Reported",
    "political_orientation_reported": "Political Orient. Reported",
    "quality_control_filtering": "Quality Filtering",
    "disagreement_resolution_adjudication": "Adjudication",
}

# ── venue extraction from paper_id ────────────────────────────────────────────
_VENUE_RE = re.compile(
    r"^\d{4}\."
    r"(aacl|acl|eacl|emnlp|naacl|tacl|"
    r"findings|eval4nlp|nips|iclr|iccv|"
    r"[a-z0-9]+)"
    r"[-.]",
    re.IGNORECASE,
)


def _extract_venue(paper_id: str) -> str:
    # Path format: "emnlp/2024/2024emnlp-main/1224.Title.pdf"
    if "/" in paper_id or "\\" in paper_id:
        parts = Path(paper_id.replace("\\", "/")).parts
        return parts[0].lower() if parts else "unknown"
    # ACL anthology format: "2024.emnlp-main.1224"
    m = _VENUE_RE.match(paper_id)
    return m.group(1).lower() if m else "unknown"


def _extract_year(paper_id: str) -> str:
    # Path format: "emnlp/2024/2024emnlp-main/1224.Title.pdf"
    if "/" in paper_id or "\\" in paper_id:
        parts = Path(paper_id.replace("\\", "/")).parts
        if len(parts) > 1 and re.match(r"^\d{4}$", parts[1]):
            y = int(parts[1])
            if 2000 <= y <= 2035:
                return parts[1]
        return "unknown"
    # ACL anthology format: "2024.emnlp-main.1224"
    m = re.match(r"^(\d{4})\.", paper_id)
    if m:
        y = int(m.group(1))
        if 2000 <= y <= 2035:
            return m.group(1)
    return "unknown"


def _parse_pdf_filename(pdf_filename: str) -> tuple[str, str, str]:
    """Parse manifest pdf_filename into (venue, year, pdf_name).

    Example: ``naacl\\2025\\2025naacl-long\\425.Title.pdf``
    → venue="naacl", year="2025", pdf_name="425.Title.pdf"
    """
    parts = Path(pdf_filename.replace("\\", "/")).parts
    venue = parts[0] if len(parts) > 0 else ""
    year  = parts[1] if len(parts) > 1 else ""
    name  = parts[-1]
    return venue, year, name


# =============================================================================
# LOADING
# =============================================================================

def load_extractions(
    extractions_dir: Path,
    manifest: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Load all *_annotations.json files. Returns list of experiment dicts
    augmented with paper_id, venue, year.

    If *manifest* is provided, venue and year are parsed from the manifest's
    pdf_filename path (e.g. ``naacl/2025/2025naacl-long/425.Title.pdf``),
    falling back to regex extraction from paper_id.
    """
    experiments: list[dict[str, Any]] = []
    json_files = sorted(extractions_dir.glob("*_annotations.json"))
    skipped = 0
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"  WARN: could not parse {jf.name}: {exc}")
            continue

        paper_id = data.get("paper_id", jf.stem.replace("_annotations", ""))
        if data.get("skip", False):
            skipped += 1
            continue

        # Prefer manifest-derived venue/year; fall back to regex
        m = (manifest or {}).get(paper_id, {})
        pdf_filename = m.get("pdf_filename", "")
        if pdf_filename:
            m_venue, m_year, _ = _parse_pdf_filename(pdf_filename)
            # Also accept explicit manifest columns as fallback
            venue = m_venue or m.get("venue", "") or _extract_venue(paper_id)
            year  = m_year  or m.get("year",  "") or _extract_year(paper_id)
        elif m:
            # Manifest matched but has no pdf_filename — use direct columns
            venue = m.get("venue", "") or _extract_venue(paper_id)
            year  = m.get("year",  "") or _extract_year(paper_id)
        else:
            venue = _extract_venue(paper_id)
            year  = _extract_year(paper_id)

        for exp in data.get("experiments", []):
            exp["_paper_id"] = paper_id
            exp["_venue"] = venue
            exp["_year"] = year
            experiments.append(exp)

    print(f"Loaded {len(experiments)} experiments from {len(json_files)} files "
          f"({skipped} skipped papers)")
    return experiments


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def _norm_val(val: Any) -> str:
    """Normalize a single field value for counting."""
    if val is None:
        return "Not reported"
    s = str(val).strip()
    if not s or s.lower() in ("na", "n/a", "not reported", "none", "null"):
        return "Not reported"
    return s


def compute_field_distributions(experiments: list[dict]) -> dict[str, Counter]:
    """Per-field value counters."""
    dist: dict[str, Counter] = {f: Counter() for f in _ALL_FIELDS}
    for exp in experiments:
        for field in _ALL_FIELDS:
            raw = exp.get(field)
            if field in _LIST_FIELDS:
                vals = raw if isinstance(raw, list) else [raw]
                for v in vals:
                    dist[field][_norm_val(v)] += 1
            else:
                dist[field][_norm_val(raw)] += 1
    return dist


def compute_reporting_rates(experiments: list[dict]) -> dict[str, float]:
    """Fraction of experiments where each field is actually reported (not NA)."""
    total = len(experiments)
    if total == 0:
        return {}
    rates: dict[str, float] = {}
    for field in _ALL_FIELDS:
        reported = 0
        for exp in experiments:
            raw = exp.get(field)
            if field in _LIST_FIELDS:
                vals = raw if isinstance(raw, list) else [raw]
                if any(_norm_val(v) != "Not reported" for v in vals):
                    reported += 1
            else:
                if _norm_val(raw) != "Not reported":
                    reported += 1
        rates[field] = round(reported / total * 100, 1)
    return rates


def compute_venue_stats(experiments: list[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for exp in experiments:
        c[exp["_venue"]] += 1
    return dict(c.most_common())


def compute_year_stats(experiments: list[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for exp in experiments:
        c[exp["_year"]] += 1
    return dict(sorted(c.items()))


def compute_numeric_stats(experiments: list[dict]) -> dict[str, dict]:
    """For numeric fields, compute min/max/median/mean over parseable values."""
    import statistics

    out: dict[str, dict] = {}
    for field in _NUMERIC_FIELDS:
        vals: list[float] = []
        for exp in experiments:
            raw = _norm_val(exp.get(field))
            try:
                vals.append(float(raw))
            except (ValueError, TypeError):
                pass
        if vals:
            out[field] = {
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean": round(statistics.mean(vals), 1),
                "median": round(statistics.median(vals), 1),
            }
        else:
            out[field] = {"count": 0}
    return out


def compute_demographics_reporting(experiments: list[dict]) -> dict[str, dict]:
    """Per-venue breakdown of how often demographic fields are reported (Yes)."""
    demo_fields = [f for f in FIELD_GROUPS["Demographics"] if f in _YES_NO_FIELDS]
    venue_demo: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    venue_total: dict[str, int] = Counter()

    for exp in experiments:
        venue = exp["_venue"]
        venue_total[venue] += 1
        for df in demo_fields:
            v = _norm_val(exp.get(df)).lower()
            if v == "yes":
                venue_demo[venue][df] += 1

    result: dict[str, dict] = {}
    for venue in sorted(venue_total):
        total = venue_total[venue]
        result[venue] = {
            df: round(venue_demo[venue].get(df, 0) / total * 100, 1)
            for df in demo_fields
        }
        result[venue]["_total"] = total
    return result


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_report(
    experiments: list[dict],
    distributions: dict[str, Counter],
    reporting_rates: dict[str, float],
    venue_stats: dict[str, int],
    year_stats: dict[str, int],
    numeric_stats: dict[str, dict],
    demographics: dict[str, dict],
) -> None:
    W = 90
    print(f"\n{'=' * W}")
    print("  LARGE-SCALE EXTRACTION ANALYSIS")
    print(f"  {len(experiments)} annotation experiments across "
          f"{len(set(e['_paper_id'] for e in experiments))} papers")
    print(f"{'=' * W}")

    # ── Venue breakdown ───────────────────────────────────────────────────
    print(f"\n  {'VENUE BREAKDOWN':─<{W - 2}}")
    for venue, count in venue_stats.items():
        bar_len = int(count / max(venue_stats.values()) * 40)
        bar = "█" * bar_len
        print(f"    {venue:<12s} {bar}  {count}")

    # ── Year breakdown ────────────────────────────────────────────────────
    print(f"\n  {'YEAR BREAKDOWN':─<{W - 2}}")
    for year, count in year_stats.items():
        bar_len = int(count / max(year_stats.values()) * 40)
        bar = "█" * bar_len
        print(f"    {year}  {bar}  {count}")

    # ── Reporting rates ───────────────────────────────────────────────────
    print(f"\n  {'REPORTING RATES (% of experiments with a value)':─<{W - 2}}")
    print(f"  {'Field':<40s}  {'Reported%':>10s}")
    print(f"  {'─' * 52}")
    for group, fields in FIELD_GROUPS.items():
        print(f"  [{group}]")
        for field in fields:
            rate = reporting_rates.get(field, 0)
            indicator = "✓" if rate >= 70 else ("~" if rate >= 40 else "✗")
            label = FIELD_LABELS.get(field, field)
            print(f"    {indicator} {label:<38s}  {rate:>9.1f}%")
        print()

    # ── Numeric field stats ───────────────────────────────────────────────
    print(f"  {'NUMERIC FIELDS':─<{W - 2}}")
    print(f"  {'Field':<30s}  {'N':>5s}  {'Min':>8s}  {'Max':>8s}  "
          f"{'Mean':>8s}  {'Median':>8s}")
    print(f"  {'─' * 72}")
    for field, stats in numeric_stats.items():
        label = FIELD_LABELS.get(field, field)
        if stats["count"] == 0:
            print(f"  {label:<30s}  {'0':>5s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'—':>8s}")
        else:
            print(f"  {label:<30s}  {stats['count']:>5d}  {stats['min']:>8.0f}  "
                  f"{stats['max']:>8.0f}  {stats['mean']:>8.1f}  {stats['median']:>8.1f}")

    # ── Top values per field ──────────────────────────────────────────────
    print(f"\n  {'TOP VALUES PER FIELD (up to 10)':─<{W - 2}}")
    for group, fields in FIELD_GROUPS.items():
        print(f"\n  [{group}]")
        for field in fields:
            label = FIELD_LABELS.get(field, field)
            print(f"    {label}:")
            top = distributions[field].most_common(10)
            total_vals = sum(distributions[field].values())
            for val, cnt in top:
                pct = cnt / total_vals * 100 if total_vals else 0
                print(f"      {val:<45s}  {cnt:>5d}  ({pct:>5.1f}%)")
            if len(distributions[field]) > 10:
                print(f"      … and {len(distributions[field]) - 10} more unique values")

    # ── Demographics by venue ─────────────────────────────────────────────
    print(f"\n  {'DEMOGRAPHICS REPORTING BY VENUE (% Yes)':─<{W - 2}}")
    demo_fields = [f for f in FIELD_GROUPS["Demographics"] if f in _YES_NO_FIELDS]
    header_labels = [FIELD_LABELS.get(f, f)[:10] for f in demo_fields]
    print(f"  {'Venue':<12s}  {'N':>4s}  " + "  ".join(f"{h:>10s}" for h in header_labels))
    print(f"  {'─' * (20 + 12 * len(demo_fields))}")
    for venue, data in demographics.items():
        n = data["_total"]
        vals = "  ".join(f"{data.get(f, 0):>9.1f}%" for f in demo_fields)
        print(f"  {venue:<12s}  {n:>4d}  {vals}")

    print(f"\n{'=' * W}\n")


# =============================================================================
# VISUALISATIONS
# =============================================================================

def generate_visualizations(
    experiments: list[dict],
    distributions: dict[str, Counter],
    reporting_rates: dict[str, float],
    venue_stats: dict[str, int],
    year_stats: dict[str, int],
    numeric_stats: dict[str, dict],
    demographics: dict[str, dict],
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed — skipping visualizations")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.max_open_warning": 0})

    # ── 1. Reporting rates bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    fields_sorted = sorted(reporting_rates, key=reporting_rates.get, reverse=True)
    labels = [FIELD_LABELS.get(f, f) for f in fields_sorted]
    values = [reporting_rates[f] for f in fields_sorted]
    colors = ["#2ecc71" if v >= 70 else "#f39c12" if v >= 40 else "#e74c3c"
              for v in values]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("% Experiments Reporting This Field")
    ax.set_title("Field Reporting Rates Across All Papers")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "reporting_rates.png", dpi=150)
    plt.close(fig)
    print(f"  Saved reporting_rates.png")

    # ── 2. Venue distribution pie chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    v_labels = list(venue_stats.keys())
    v_sizes = list(venue_stats.values())
    ax.pie(v_sizes, labels=[f"{l} ({s})" for l, s in zip(v_labels, v_sizes)],
           autopct="%1.1f%%", startangle=140)
    ax.set_title("Experiments by Venue")
    fig.tight_layout()
    fig.savefig(out_dir / "venue_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved venue_distribution.png")

    # ── 3. Year trend ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    years = list(year_stats.keys())
    counts = list(year_stats.values())
    ax.bar(years, counts, color="#3498db")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Experiments")
    ax.set_title("Experiments by Year")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "year_trend.png", dpi=150)
    plt.close(fig)
    print(f"  Saved year_trend.png")

    # ── 4. Recruitment methods ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    recruit = distributions.get("recruitment", Counter())
    if recruit:
        top_recruit = recruit.most_common(8)
        r_labels = [v for v, _ in top_recruit]
        r_counts = [c for _, c in top_recruit]
        ax.barh(r_labels[::-1], r_counts[::-1], color="#9b59b6")
        ax.set_xlabel("Count")
        ax.set_title("Top Recruitment Methods")
        fig.tight_layout()
        fig.savefig(out_dir / "recruitment_methods.png", dpi=150)
        plt.close(fig)
        print(f"  Saved recruitment_methods.png")
    else:
        plt.close(fig)

    # ── 5. Expertise level distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    expertise = distributions.get("expertise_level", Counter())
    if expertise:
        top_exp = expertise.most_common(8)
        e_labels = [v for v, _ in top_exp]
        e_counts = [c for _, c in top_exp]
        ax.bar(e_labels, e_counts, color="#e67e22")
        ax.set_ylabel("Count")
        ax.set_title("Expertise Level Distribution")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "expertise_distribution.png", dpi=150)
        plt.close(fig)
        print(f"  Saved expertise_distribution.png")
    else:
        plt.close(fig)

    # ── 6. Demographics heatmap by venue ──────────────────────────────────
    if demographics:
        demo_fields = [f for f in FIELD_GROUPS["Demographics"] if f in _YES_NO_FIELDS]
        venues_list = sorted(demographics.keys())
        matrix = []
        for venue in venues_list:
            row = [demographics[venue].get(f, 0) for f in demo_fields]
            matrix.append(row)

        fig, ax = plt.subplots(figsize=(12, max(4, len(venues_list) * 0.8)))
        import numpy as np
        data = np.array(matrix)
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(demo_fields)))
        ax.set_xticklabels([FIELD_LABELS.get(f, f)[:12] for f in demo_fields],
                           rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(venues_list)))
        ax.set_yticklabels(venues_list)
        for i in range(len(venues_list)):
            for j in range(len(demo_fields)):
                ax.text(j, i, f"{data[i, j]:.0f}%", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_title("Demographics Reporting by Venue (% Yes)")
        fig.colorbar(im, ax=ax, label="% Reporting Yes", shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / "demographics_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  Saved demographics_heatmap.png")

    # ── 7. Human annotation task type distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    task_types = distributions.get("human_annotation_type", Counter())
    if task_types:
        top_tasks = task_types.most_common(10)
        t_labels = [v for v, _ in top_tasks]
        t_counts = [c for _, c in top_tasks]
        ax.barh(t_labels[::-1], t_counts[::-1], color="#1abc9c")
        ax.set_xlabel("Count")
        ax.set_title("Top Human Annotation Task Types")
        fig.tight_layout()
        fig.savefig(out_dir / "task_types.png", dpi=150)
        plt.close(fig)
        print(f"  Saved task_types.png")
    else:
        plt.close(fig)

    # ── 8. Guidelines + Training + QC stacked bar by year ─────────────────
    process_fields = ["guidelines_released", "annotator_training",
                      "quality_control_filtering"]
    year_process: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    year_totals: dict[str, int] = Counter()
    for exp in experiments:
        y = exp["_year"]
        year_totals[y] += 1
        for pf in process_fields:
            if _norm_val(exp.get(pf)).lower() == "yes":
                year_process[y][pf] += 1

    if year_totals:
        years_sorted = sorted(year_totals.keys())
        fig, ax = plt.subplots(figsize=(12, 5))
        import numpy as np
        x = np.arange(len(years_sorted))
        width = 0.25
        colors_p = ["#2ecc71", "#3498db", "#e74c3c"]
        for i, pf in enumerate(process_fields):
            rates = [year_process[y].get(pf, 0) / year_totals[y] * 100
                     if year_totals[y] else 0 for y in years_sorted]
            ax.bar(x + i * width, rates, width,
                   label=FIELD_LABELS.get(pf, pf), color=colors_p[i])
        ax.set_xlabel("Year")
        ax.set_ylabel("% Yes")
        ax.set_title("Process & Quality Practices Over Time")
        ax.set_xticks(x + width)
        ax.set_xticklabels(years_sorted)
        ax.legend()
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(out_dir / "process_by_year.png", dpi=150)
        plt.close(fig)
        print(f"  Saved process_by_year.png")


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_csv(
    distributions: dict[str, Counter],
    reporting_rates: dict[str, float],
    out_dir: Path,
) -> None:
    """Write per-field value distributions to CSV."""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "field_distributions.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "field", "field_label", "reporting_rate_pct",
                     "value", "count"])
        for group, fields in FIELD_GROUPS.items():
            for field in fields:
                label = FIELD_LABELS.get(field, field)
                rate = reporting_rates.get(field, 0)
                for val, cnt in distributions[field].most_common():
                    w.writerow([group, field, label, rate, val, cnt])
    print(f"  Saved {path}")

    # Also export a one-row-per-experiment flat CSV
    path2 = out_dir / "all_experiments.csv"
    print(f"  (all_experiments.csv will be created on next run with data)")


def export_experiments_csv(
    experiments: list[dict],
    out_dir: Path,
    manifest: dict[str, dict] | None = None,
) -> None:
    """Write every experiment as one row to a flat CSV.

    If *manifest* is supplied (keyed by full_id / paper_id):
    - _venue and _year are taken from the manifest (parsed from pdf_filename
      path components), falling back to values from the extraction JSON.
    - Extra columns are added: web_url, pdf_filename, pdf_name, downloaded.
      pdf_name is just the bare filename (no directory prefix).
    """
    import csv
    if manifest is None:
        manifest = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "all_experiments.csv"

    meta_cols = ["_paper_id", "_venue", "_year", "paper_experiment_id"]
    manifest_cols = ["web_url", "pdf_filename", "pdf_name", "downloaded"] if manifest else []
    cols = meta_cols + _ALL_FIELDS + manifest_cols

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for exp in experiments:
            paper_id = exp.get("_paper_id", "")
            m = manifest.get(paper_id, {})

            # Derive venue/year/pdf_name from manifest pdf_filename path
            pdf_filename = m.get("pdf_filename", "")
            if pdf_filename:
                m_venue, m_year, pdf_name = _parse_pdf_filename(pdf_filename)
            else:
                m_venue = m.get("venue", "")
                m_year  = m.get("year", "")
                pdf_name = ""

            # Build row — override _venue/_year from manifest when available
            row = []
            for c in meta_cols:
                if c == "_venue":
                    val = m_venue or exp.get("_venue", "")
                elif c == "_year":
                    val = m_year or exp.get("_year", "")
                else:
                    val = exp.get(c, "")
                row.append(val)
            for c in _ALL_FIELDS:
                val = exp.get(c, "")
                if isinstance(val, list):
                    val = "; ".join(str(v) for v in val)
                row.append(val)
            if manifest_cols:
                row.append(m.get("web_url", ""))
                row.append(pdf_filename)
                row.append(pdf_name)
                row.append(m.get("downloaded", ""))
            w.writerow(row)
    print(f"  Saved {path}  ({len(experiments)} rows)")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze large-scale extraction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--extractions_dir",
        default=None,
        help="Path to the extractions directory (default: auto-detect latest LargeScale_*)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for CSV + visualizations (default: analysis_output/)",
    )
    p.add_argument(
        "--no_viz",
        action="store_true",
        help="Skip visualization generation",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help=(
            "Path to manifest.csv produced by sample_acl_anthology.py. "
            f"Columns web_url, pdf_filename, downloaded are merged into "
            "all_experiments.csv on full_id. "
            f"Defaults to {MANIFEST_DEFAULT} if that file exists."
        ),
    )
    return p.parse_args()


def _auto_detect_extractions_dir() -> Optional[Path]:
    """Find the most recent LargeScale_Extractions_* directory."""
    candidates = sorted(PDFS_ROOT.glob("LargeScale_Extractions_*"))
    if candidates:
        return candidates[-1]
    # Fall back: any Combined_Extractions_*
    candidates = sorted(PDFS_ROOT.glob("Combined_Extractions_*"))
    if candidates:
        return candidates[-1]
    return None


def main() -> None:
    args = parse_args()

    # Resolve extractions dir
    if args.extractions_dir:
        extractions_dir = Path(args.extractions_dir)
    else:
        extractions_dir = _auto_detect_extractions_dir()
        if extractions_dir is None:
            print("ERROR: No extractions directory found. "
                  "Run the large-scale pipeline first or pass --extractions_dir.",
                  file=sys.stderr)
            sys.exit(1)
    print(f"Extractions dir: {extractions_dir}")

    if not extractions_dir.is_dir():
        print(f"ERROR: Not a directory: {extractions_dir}", file=sys.stderr)
        sys.exit(1)

    # Load manifest first so venue/year can be resolved during extraction load
    manifest_path = Path(args.manifest) if args.manifest else None
    manifest = load_manifest(manifest_path)

    # Load
    experiments = load_extractions(extractions_dir, manifest=manifest)
    if not experiments:
        print("ERROR: No experiments loaded.", file=sys.stderr)
        sys.exit(1)

    # Compute
    distributions = compute_field_distributions(experiments)
    reporting_rates = compute_reporting_rates(experiments)
    venue_stats = compute_venue_stats(experiments)
    year_stats = compute_year_stats(experiments)
    numeric_stats = compute_numeric_stats(experiments)
    demographics = compute_demographics_reporting(experiments)

    # Report
    print_report(
        experiments, distributions, reporting_rates,
        venue_stats, year_stats, numeric_stats, demographics,
    )

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "analysis_output"
    print(f"Output dir: {out_dir}\n")

    export_csv(distributions, reporting_rates, out_dir)
    export_experiments_csv(experiments, out_dir, manifest=manifest)

    if not args.no_viz:
        print("\nGenerating visualizations …")
        generate_visualizations(
            experiments, distributions, reporting_rates,
            venue_stats, year_stats, numeric_stats, demographics,
            out_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
