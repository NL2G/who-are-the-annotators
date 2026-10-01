"""
Download PDFs from paper links listed in the gold standard CSV.

Usage:
    python download_papers.py \
        --csv "data/gold_standard.csv" \
        --out_dir data/pdfs
"""

import argparse
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


# =============================================================================
# URL HELPERS
# =============================================================================

def url_to_pdf_url(url: str) -> str:
    """Return a direct PDF URL, converting ACL anthology pages and arXiv abs links."""
    url = url.strip()
    # ACL Anthology: ensure it ends with .pdf
    if "aclanthology.org/" in url:
        url = url.rstrip("/")
        if not url.endswith(".pdf"):
            url = url + ".pdf"
        return url
    # arXiv abstract page → PDF
    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/")
    return url


def url_to_stem(url: str) -> str:
    """Derive a clean filename stem (no extension) from a paper URL.

    Examples
    --------
    'https://aclanthology.org/2023.acl-long.711.pdf'  → '2023.acl-long.711'
    'https://aclanthology.org/2021.acl-long.540/'     → '2021.acl-long.540'
    'https://arxiv.org/pdf/2603.03072'                 → '2603.03072'
    'https://arxiv.org/abs/2503.11509'                 → '2503.11509'
    """
    url = url.strip().rstrip("/")
    # ACL Anthology
    m = re.search(r"aclanthology\.org/([A-Za-z0-9._-]+?)(?:\.pdf)?/?$", url)
    if m:
        return m.group(1)
    # arXiv (both /pdf/ and /abs/)
    m = re.search(r"arxiv\.org/(?:pdf|abs)/([0-9]+\.[0-9]+(?:v\d+)?)", url)
    if m:
        return m.group(1)
    # Generic fallback: strip .pdf and use last path component
    last = url.split("/")[-1]
    return last[:-4] if last.endswith(".pdf") else (last or "paper")


# =============================================================================
# DOWNLOAD
# =============================================================================

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; academic-research-bot/1.0)"}


def download_pdf(url: str, out_path: Path, delay: float = 1.0) -> bool:
    """Download the PDF at *url* to *out_path*.  Returns True on success."""
    pdf_url = url_to_pdf_url(url)

    if out_path.exists() and out_path.stat().st_size > 4096:
        print(f"  [skip] already exists: {out_path.name}")
        return True

    try:
        req = urllib.request.Request(pdf_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        if len(data) < 1024:
            print(f"  [warn] suspiciously small response ({len(data)} B) for {pdf_url}")
            return False

        out_path.write_bytes(data)
        print(f"  [ok]   {out_path.name}  ({len(data) // 1024} KB)")
        time.sleep(delay)
        return True

    except urllib.error.HTTPError as exc:
        print(f"  [err]  HTTP {exc.code} – {pdf_url}")
        return False
    except Exception as exc:
        print(f"  [err]  {exc} – {pdf_url}")
        return False


# =============================================================================
# CSV LOADING
# =============================================================================

def load_paper_links(csv_path: Path) -> list:
    """Return an ordered, deduplicated list of paper-link URLs from the CSV.

    Accepts the new gold-standard format (Paper Link as first column) as well
    as any CSV that has a column whose name contains "paper" and "link".
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="warn")

    # Find the Paper Link column (case-insensitive)
    link_col: str | None = None
    for col in df.columns:
        if col.strip().lower() in ("paper link", "paperlink", "paper_link"):
            link_col = col
            break
    if link_col is None:
        for col in df.columns:
            if "paper" in col.lower() and "link" in col.lower():
                link_col = col
                break
    if link_col is None:
        raise ValueError(
            f"No 'Paper Link' column found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    urls = (
        df[link_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    seen: set = set()
    unique: list = []
    for u in urls:
        if u and u.startswith("http") and u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


# =============================================================================
# MAIN DOWNLOAD FUNCTION
# =============================================================================

def download_papers_from_csv(
    csv_path: Path,
    out_dir: Path,
    delay: float = 1.0,
) -> dict:
    """Download all unique papers referenced in *csv_path*.

    Returns
    -------
    dict
        Mapping ``{url: local_pdf_path}`` for every successfully downloaded paper.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = load_paper_links(csv_path)
    print(f"Found {len(urls)} unique paper links in {csv_path.name}")

    url_to_path: dict = {}
    for i, url in enumerate(urls, 1):
        stem = url_to_stem(url)
        out_path = out_dir / f"{stem}.pdf"
        print(f"[{i}/{len(urls)}] {url}")
        if download_pdf(url, out_path, delay=delay):
            url_to_path[url] = out_path

    failed = len(urls) - len(url_to_path)
    print(
        f"\nResult: {len(url_to_path)}/{len(urls)} downloaded"
        + (f", {failed} failed" if failed else "")
        + f" → {out_dir}"
    )
    return url_to_path


# =============================================================================
# CLI
# =============================================================================

def arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download PDFs from the gold standard CSV Paper Link column"
    )
    p.add_argument(
        "--csv",
        required=True,
        help="Path to the gold standard CSV (must have a 'Paper Link' column)",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory where downloaded PDFs will be saved",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between downloads (default: 1.0)",
    )
    return p.parse_args()


def main() -> None:
    args = arg_parser()
    download_papers_from_csv(
        csv_path=Path(args.csv),
        out_dir=Path(args.out_dir),
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
