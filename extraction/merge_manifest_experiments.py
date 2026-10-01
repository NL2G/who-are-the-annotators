"""
Merge manifest.csv and all_experiments.csv on paper_id.

Paper ID is derived from manifest's pdf_filename by taking the text after
the last '/' and stripping the '.pdf' extension. This matches the PaperID
column in all_experiments.csv.

For findings papers, the venue is extracted from the pdf_filename path
(e.g. findings/2023/2023findings-emnlp/... -> venue = emnlp).
"""

import argparse
import os

import pandas as pd

DEFAULT_DIR = "Processed_PDFs/RandomSample_Analysis_Output"

ALLOWED_VENUES = {"coling", "emnlp", "cl", "acl", "aacl", "eacl", "ijcnlp", "tacl", "naacl"}
YEAR_MIN = 2018
YEAR_MAX = 2025


def extract_venue(row: pd.Series) -> str:
    """
    If venue is 'findings', extract the real venue from the pdf_filename path.
    E.g. findings/2023/2023findings-emnlp/... -> 'emnlp'
    Otherwise return the venue as-is.
    """
    if row["venue"] == "findings":
        parts = row["pdf_filename"].split("/")
        # parts[2] looks like '2023findings-emnlp'
        if len(parts) > 2 and "findings-" in parts[2]:
            return parts[2].split("findings-")[-1]
    return row["venue"]


def extract_paper_id(pdf_filename: str) -> str:
    """
    Derive the paper_id from pdf_filename:
      - take the text after the last '/'
      - strip the '.pdf' extension
    """
    return pdf_filename.split("/")[-1].removesuffix(".pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=os.path.join(DEFAULT_DIR, "manifest.csv"))
    parser.add_argument("--experiments", default=os.path.join(DEFAULT_DIR, "all_experiments.csv"))
    parser.add_argument("--output", default=os.path.join(DEFAULT_DIR, "merged_output.csv"))
    args = parser.parse_args()
    MANIFEST_PATH, ALL_EXPERIMENTS_PATH, OUTPUT_PATH = args.manifest, args.experiments, args.output

    # --- Load manifest ---
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str)

    # Fix venue for findings papers
    manifest["venue"] = manifest.apply(extract_venue, axis=1)

    # Derive paper_id merge key
    manifest["paper_id"] = manifest["pdf_filename"].apply(extract_paper_id)

    print(f"Manifest rows: {len(manifest)}")
    print(f"Unique manifest paper_ids: {manifest['paper_id'].nunique()}")

    # --- Load all_experiments ---
    experiments = pd.read_csv(ALL_EXPERIMENTS_PATH, dtype=str, keep_default_na=False)
    print(f"\nAll-experiments rows: {len(experiments)}")
    print(f"Unique PaperIDs in all_experiments: {experiments['PaperID'].nunique()}")

    # --- Merge ---
    # Drop Venue and Year from experiments — use manifest's versions instead
    experiments.drop(columns=["Venue", "Year"], inplace=True)

    merged = experiments.merge(
        manifest,
        left_on="PaperID",
        right_on="paper_id",
        how="left",
    )

    matched = merged["paper_id"].notna().sum()
    print(f"\nMerged rows: {len(merged)}")
    print(f"Rows with a manifest match: {matched}")
    print(f"Rows without a manifest match: {len(merged) - matched}")

    # Drop the redundant paper_id helper column
    merged.drop(columns=["paper_id"], inplace=True)

    # --- Filter by venue and year ---
    before = len(merged)
    merged = merged[merged["venue"].isin(ALLOWED_VENUES)]
    merged = merged[merged["year"].notna()]
    merged = merged[merged["year"].astype(int).between(YEAR_MIN, YEAR_MAX)]
    print(f"\nAfter venue/year filter: {len(merged)} rows (removed {before - len(merged)})")
    print(f"Venues kept: {sorted(merged['venue'].unique())}")
    print(f"Years kept:  {sorted(merged['year'].unique())}")

    # Drop unwanted columns before saving
    merged.drop(columns=["title", "pdf_filename", "downloaded", "abstract"], inplace=True, errors="ignore")

    # --- Save ---
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved merged file to:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
