# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "acl-anthology>=1.1.0",
# ]
# ///
"""
Random Sample from ACL Anthology (2016-2026).

Uses the official `acl-anthology` Python package to enumerate all papers,
randomly samples N of them from the 2016-2026 window (stratified by year
for balanced coverage), downloads the PDFs from aclanthology.org, and
writes a manifest CSV.

Requires `git` to be installed -- the library clones the ACL Anthology repo
to a local cache the first time it runs (~600 MB, subsequent runs just pull).

Downloaded PDFs mirror the layout of the crawled-and-filtered papers:
    combined_papers/random_acl_sample/<venue>/<year>/<collection>/paper.pdf

e.g.  combined_papers/random_acl_sample/acl/2022/2022acl-long/4.Paper Title.pdf

Usage
-----
  uv run sample_acl_anthology.py                        # defaults (n=3000, seed=42)
  uv run sample_acl_anthology.py --n 3000 --seed 42
  uv run sample_acl_anthology.py --no_stratify          # pure random (not year-stratified)
  uv run sample_acl_anthology.py --year_min 2018 --year_max 2024
  uv run sample_acl_anthology.py --manifest_only        # skip download, write manifest only
  uv run sample_acl_anthology.py --workers 8            # parallel download workers

Next step after running this script
------------------------------------
  python run_pipeline_largescale.py --mode full \
         --papers_dir combined_papers/random_acl_sample \
         --model_id gemini-3.1-pro-preview --num_workers 4
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple

# -- paths --------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "combined_papers" / "random_acl_sample"

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; academic-research-bot/1.0)"}

# -- venue allowlist ----------------------------------------------------------
# Only main *ACL conferences and journals; excludes workshops, tutorials, demos
# (which make up the majority of anthology entries).
MAIN_VENUES: frozenset = frozenset({
    "acl",       # Annual Meeting of the ACL
    "emnlp",     # Empirical Methods in NLP
    "naacl",     # NAACL-HLT
    "eacl",      # European Chapter of the ACL
    "aacl",      # AACL-IJCNLP
    "coling",    # International Conference on Computational Linguistics
    "ijcnlp",    # International Joint Conference on NLP
    "tacl",      # Transactions of the ACL (journal)
    "cl",        # Computational Linguistics (journal)
    "findings",  # Findings of ACL/EMNLP/NAACL (associated findings papers)
})


# =============================================================================
# Paper record (lightweight; avoids keeping Anthology objects in memory)
# =============================================================================

class PaperRecord(NamedTuple):
    full_id:  str
    year:     int
    venue:    str          # primary venue id (e.g. "acl", "emnlp")
    title:    str
    web_url:  str          # https://aclanthology.org/2022.acl-long.220
    abstract: str


# =============================================================================
# Load papers via acl-anthology package
# =============================================================================

def load_papers(year_min: int, year_max: int) -> List[PaperRecord]:
    """Enumerate all non-deleted, non-frontmatter papers from the ACL Anthology
    that fall within [year_min, year_max] and have an aclanthology.org URL.

    On first run, Anthology.from_repo() clones the ACL Anthology data repo
    (~600 MB).  Subsequent runs do a fast `git pull`.
    """
    from acl_anthology import Anthology  # type: ignore[import]

    print("Initialising ACL Anthology (may clone/pull repo on first run) ...")
    anthology = Anthology.from_repo()

    print("Loading all metadata ...")
    anthology.load_all()

    papers: List[PaperRecord] = []
    total_seen = 0

    for collection in anthology.collections.values():
        for volume in collection.volumes():
            for paper in volume.papers():
                total_seen += 1
                if paper.is_frontmatter or paper.is_deleted:
                    continue

                # year is stored as a string on the volume
                try:
                    year = int(paper.year)
                except (TypeError, ValueError):
                    continue

                if not (year_min <= year <= year_max):
                    continue

                url: str = paper.web_url or ""
                if "aclanthology.org" not in url:
                    continue

                # primary venue: first element of venue_ids list, fall back to
                # parsing the collection ID (e.g. "2022.acl" -> "acl")
                venue_ids: List[str] = list(paper.venue_ids) if paper.venue_ids else []
                if venue_ids:
                    venue = venue_ids[0].lower()
                else:
                    cid = paper.collection_id or ""
                    m = re.search(r"\.(\w+)$", cid)
                    venue = m.group(1).lower() if m else cid.lower()

                # skip workshops, tutorials, demos, SRW, etc.
                if venue not in MAIN_VENUES:
                    continue

                title    = str(paper.title) if paper.title else ""
                abstract = str(paper.abstract) if paper.abstract else ""

                papers.append(
                    PaperRecord(
                        full_id=paper.full_id,
                        year=year,
                        venue=venue,
                        title=title,
                        web_url=url,
                        abstract=abstract,
                    )
                )

    print(f"  Scanned {total_seen:,} papers total.")
    print(f"  {len(papers):,} papers match the {year_min}-{year_max} window "
          f"and are from main *ACL venues.")
    return papers


# =============================================================================
# Sampling
# =============================================================================

def stratified_sample(
    papers: List[PaperRecord],
    n: int,
    seed: int,
) -> List[PaperRecord]:
    """Sample *n* papers stratified by year (proportional allocation)."""
    rng = random.Random(seed)

    by_year: Dict[int, List[PaperRecord]] = defaultdict(list)
    for p in papers:
        by_year[p.year].append(p)

    years = sorted(by_year)
    total_available = sum(len(v) for v in by_year.values())

    if n >= total_available:
        print(f"  WARNING: requested {n} but only {total_available} available; using all.")
        result = list(papers)
        rng.shuffle(result)
        return result

    # Proportional allocation with largest-remainder rounding
    allocations: Dict[int, int] = {}
    remainders: List[tuple] = []
    allocated = 0
    for yr in years:
        exact = n * len(by_year[yr]) / total_available
        floored = int(exact)
        allocations[yr] = floored
        allocated += floored
        remainders.append((exact - floored, yr))

    remainders.sort(reverse=True)
    for _, yr in remainders[: n - allocated]:
        allocations[yr] += 1

    sampled: List[PaperRecord] = []
    for yr in years:
        k = min(allocations[yr], len(by_year[yr]))
        sampled.extend(rng.sample(by_year[yr], k))

    rng.shuffle(sampled)
    return sampled


def pure_random_sample(
    papers: List[PaperRecord],
    n: int,
    seed: int,
) -> List[PaperRecord]:
    rng = random.Random(seed)
    if n >= len(papers):
        return list(papers)
    return rng.sample(papers, n)


# =============================================================================
# Path / filename helpers  (mirrors crawled-and-filtered-papers layout)
# =============================================================================

def _collection_folder(paper: PaperRecord) -> str:
    """Derive the collection/volume folder name from full_id.

    Examples
    --------
    '2022.acl-long.4'   -> '2022acl-long'
    '2021.emnlp-main.3' -> '2021emnlp-main'
    'P18-1001'          -> 'P18-1001'  (legacy IDs, no dots → use as-is)
    """
    parts = paper.full_id.split(".")
    if len(parts) >= 3:
        # "<year>.<venue-track>.<num>"  →  "<year><venue-track>"
        return parts[0] + parts[1]
    if len(parts) == 2:
        return parts[0] + parts[1]
    return re.sub(r"[^\w\-]", "_", paper.full_id)


def _paper_dir(paper: PaperRecord, base_dir: Path) -> Path:
    """Return <base_dir>/<venue>/<year>/<collection_folder>/"""
    return base_dir / paper.venue / str(paper.year) / _collection_folder(paper)


def _safe_filename(paper: PaperRecord) -> str:
    """Return '<paper_num>.<safe_title>.pdf', matching crawled paper naming.

    e.g. full_id='2022.acl-long.4', title='Toward Annotator Group Bias...'
         → '4.Toward Annotator Group Bias....pdf'
    """
    parts = paper.full_id.split(".")
    # paper number is the last segment of the full_id
    paper_num = parts[-1] if len(parts) >= 2 else paper.full_id
    # strip characters illegal on Windows / Unix filesystems
    safe_title = re.sub(r'[\\/:*?"<>|]', "", paper.title).strip()
    safe_title = safe_title[:200]   # cap to avoid path-length issues
    return f"{paper_num}.{safe_title}.pdf"


def _pdf_url(paper: PaperRecord) -> str:
    """Convert the landing-page URL to a direct PDF link."""
    url = paper.web_url.rstrip("/")
    return url if url.endswith(".pdf") else url + ".pdf"


# =============================================================================
# Download
# =============================================================================

def _download_one(
    paper: PaperRecord,
    base_dir: Path,
    delay: float,
    timeout: int = 60,
) -> bool:
    out_dir  = _paper_dir(paper, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _safe_filename(paper)

    if out_path.exists() and out_path.stat().st_size > 4096:
        return True

    pdf_url = _pdf_url(paper)
    try:
        req = urllib.request.Request(pdf_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        out_path.write_bytes(data)
        time.sleep(delay)
        return True
    except (urllib.error.URLError, OSError):
        return False


def download_all(
    sampled: List[PaperRecord],
    base_dir: Path,
    delay: float,
    workers: int,
) -> Dict[str, bool]:
    """Download all sampled PDFs.  Returns {full_id: success}."""
    results: Dict[str, bool] = {}
    total = len(sampled)

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _dl(p: PaperRecord):
            return p.full_id, _download_one(p, base_dir, delay)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_dl, p): p for p in sampled}
            done = 0
            for fut in as_completed(futures):
                fid, ok = fut.result()
                results[fid] = ok
                done += 1
                if done % 100 == 0 or done == total:
                    success = sum(results.values())
                    print(f"  [{done}/{total}] {success} ok, {done - success} failed")
    else:
        for i, paper in enumerate(sampled, 1):
            results[paper.full_id] = _download_one(paper, base_dir, delay)
            if i % 100 == 0 or i == total:
                success = sum(results.values())
                print(f"  [{i}/{total}] {success} ok, {i - success} failed")

    return results


# =============================================================================
# Manifest CSV
# =============================================================================

def write_manifest(
    sampled: List[PaperRecord],
    download_results: Dict[str, bool],
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "full_id", "year", "venue", "title",
                "web_url", "pdf_filename", "downloaded", "abstract",
            ],
        )
        writer.writeheader()
        for p in sampled:
            # relative path matches the nested directory layout
            rel_path = str(
                Path(p.venue) / str(p.year) / _collection_folder(p) / _safe_filename(p)
            )
            writer.writerow(
                {
                    "full_id":      p.full_id,
                    "year":         p.year,
                    "venue":        p.venue,
                    "title":        p.title,
                    "web_url":      p.web_url,
                    "pdf_filename": rel_path,
                    "downloaded":   download_results.get(p.full_id, False),
                    "abstract":     p.abstract,
                }
            )
    print(f"Manifest written -> {manifest_path}")


# =============================================================================
# CLI
# =============================================================================

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Randomly sample N papers from the ACL Anthology (2016-2026) "
            "using the official acl-anthology Python package, then download their PDFs."
        )
    )
    p.add_argument(
        "--n", type=int, default=3000,
        help="Number of papers to sample (default: 3000).",
    )
    p.add_argument(
        "--year_min", type=int, default=2016,
        help="Earliest publication year to include (default: 2016).",
    )
    p.add_argument(
        "--year_max", type=int, default=2026,
        help="Latest publication year to include (default: 2026).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--no_stratify", action="store_true",
        help="Use pure random sampling instead of year-stratified sampling.",
    )
    p.add_argument(
        "--out_dir", type=Path, default=None,
        help="Root output directory for PDFs and manifest "
             f"(default: {DEFAULT_OUT}). "
             "PDFs land at <out_dir>/<venue>/<year>/<collection>/paper.pdf",
    )
    p.add_argument(
        "--manifest_only", action="store_true",
        help="Write the manifest CSV without downloading any PDFs.",
    )
    p.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds to wait between downloads per worker (default: 0.3). "
             "Be polite to aclanthology.org.",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download workers (default: 4). Keep <= 8.",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = arg_parser()

    out_dir       = args.out_dir or DEFAULT_OUT
    manifest_path = out_dir / "manifest.csv"

    print("=" * 60)
    print("  ACL Anthology Random Sampler")
    print(f"  n={args.n}  years={args.year_min}-{args.year_max}"
          f"  seed={args.seed}  stratified={not args.no_stratify}")
    print("=" * 60)

    # 1. Load from ACL Anthology package
    all_papers = load_papers(args.year_min, args.year_max)

    # Show year breakdown
    by_year: Dict[int, int] = defaultdict(int)
    for p in all_papers:
        by_year[p.year] += 1
    print("\n  Year breakdown:")
    for yr in sorted(by_year):
        print(f"    {yr}: {by_year[yr]:,}")

    # 2. Sample
    if args.no_stratify:
        sampled = pure_random_sample(all_papers, args.n, args.seed)
        print(f"\nPure random sample: {len(sampled):,} papers (seed={args.seed})")
    else:
        sampled = stratified_sample(all_papers, args.n, args.seed)
        print(f"\nStratified sample: {len(sampled):,} papers (seed={args.seed})")

    venue_counts: Dict[str, int] = defaultdict(int)
    year_counts:  Dict[int, int] = defaultdict(int)
    for p in sampled:
        venue_counts[p.venue] += 1
        year_counts[p.year]   += 1
    print("  Venue spread:", dict(sorted(venue_counts.items())))
    print("  Year spread: ", dict(sorted(year_counts.items())))

    # 3. Download (or skip)
    if args.manifest_only:
        print("\n--manifest_only: skipping download.")
        download_results: Dict[str, bool] = {p.full_id: False for p in sampled}
    else:
        print(f"\nDownloading {len(sampled):,} PDFs -> {out_dir}")
        print(f"  Layout: <venue>/<year>/<collection>/paper.pdf")
        print(f"  workers={args.workers}  delay={args.delay}s per worker")
        download_results = download_all(
            sampled, out_dir, delay=args.delay, workers=args.workers
        )
        success = sum(download_results.values())
        failed  = len(download_results) - success
        print(f"\nDownload complete: {success} succeeded, {failed} failed.")

    # 4. Manifest
    write_manifest(sampled, download_results, manifest_path)

    # 5. Next step hint
    print("\nNext step -- run the large-scale pipeline on the sample:")
    print(f'  python run_pipeline_largescale.py --mode full \\')
    print(f'         --papers_dir "{out_dir}" \\')
    print(f'         --model_id gemini-3.1-pro-preview --num_workers 4')


if __name__ == "__main__":
    main()
