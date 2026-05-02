"""
01 May 2026
get a table with hits per each value of each column, to check the data and get a sense of the distribution of values.
This is also useful for sanity checking the data, as it can reveal any unexpected or inconsistent values that may need to be cleaned or corrected before analysis.

USAGE:
python3 cats_counts.py

"""

import argparse
import time
from collections import Counter
from pathlib import Path

from configs import FIELDS, TAX
from configs import make_dirs
import pandas as pd
from reportage_score_tempo import compute_fair_score, compute_blunt_score

from scipy.stats import chi2_contingency


def cross_sample_value_distributions(df_dict, cols, multilabel, strings, top_k=None):
    df = pd.concat(
        [get_cross_sample_distribution(df_dict, c, multilabel, strings, top_k) for c in cols],
        ignore_index=True
    )

    return df.sort_values(
        ["category", "Δ% (filt-rand)"],
        key=lambda x: x.abs() if x.name == "Δ% (filt-rand)" else x,
        ascending=[True, False]
    )


def get_cross_sample_distribution(df_dict, col, multilabel, strings, top_k=None):
    multilabel = multilabel or []
    strings = strings or []

    df_filt = df_dict["filter"]
    df_rand = df_dict["random"]

    print("FILTER types:")
    print(df_filt[col].apply(type).value_counts())

    print("\nRANDOM types:")
    print(df_rand[col].apply(type).value_counts())

    # ---------- helper to get counts ----------
    def normalize_multilabel(series):
        return (
            series.dropna()
            .apply(lambda x: ";".join(x) if isinstance(x, list) else str(x))
            .str.split(";")
            .explode()
            .str.strip()
        )

    def extract_counts(df):
        if col in multilabel:
            vals = normalize_multilabel(df[col])
            vals = vals[vals != ""]
            return Counter(vals)

        elif col in strings:
            vals = df[col].astype("string")
            reported = (
                vals.notna()
                & (vals.str.strip() != "")
                & (vals.str.lower() != "not reported")
            )
            mapped = reported.map({True: "reported", False: "not_reported"})
            return Counter(mapped)

        else:
            vals = df[col].dropna().astype(str)
            return Counter(vals)

    # ---------- counts ----------
    counts_f = extract_counts(df_filt)
    counts_r = extract_counts(df_rand)

    all_values = set(counts_f.keys()) | set(counts_r.keys())

    n_f = sum(counts_f.values())
    n_r = sum(counts_r.values())

    rows = []
    min_n = 30

    for val in all_values:
        c_f = counts_f.get(val, 0)
        c_r = counts_r.get(val, 0)

        p_f = (c_f / n_f * 100) if n_f else 0
        p_r = (c_r / n_r * 100) if n_r else 0

        # signed deltas
        delta_count = c_f - c_r
        delta_pct = p_f - p_r

        # chi2 on proportions
        sig = "NA"
        if c_f >= min_n and c_r >= min_n:
            table = [[c_f, n_f - c_f],
                     [c_r, n_r - c_r]]
            try:
                _, pval, _, _ = chi2_contingency(table)
                sig = "*" if pval < 0.05 else ""
            except:
                sig = "NA"

        rows.append({
            "category": col,
            "value": val,
            "Δ# (filt-rand)": int(delta_count),
            "Δ% (filt-rand)": round(delta_pct, 1),
            "p<0.05": sig
        })

    out = pd.DataFrame(rows)

    if top_k:
        out = out.sort_values("Δ# (filt-rand)", ascending=False).head(top_k)

    out = out.sort_values("Δ% (filt-rand)", key=lambda x: x.abs(), ascending=False)

    return out


def input_sanity_check_column(df, col, top_k=100):
    vals = df[col].dropna().astype(str)

    counter = Counter(vals)
    total = len(vals)

    print(f"Column: {col}")
    print(f"Total non-null: {total}")
    print("-" * 40)

    for k, v in counter.most_common(top_k):
        print(f"{k:25s} {v:5d}  ({v / total:.1%})")

    print("-" * 40)
    print(f"Unique values: {len(counter)}\n")


def load_df(path, sanity):
    df = pd.read_csv(Path(path), sep="\t", engine="python", on_bad_lines="warn", dtype=str, keep_default_na=False)
    if df.isna().any().any():
        exit('Exiting: NaN values detected in the CSV after loading. Check the data for missing values.')

    print(f"\tUnique papers: {df['PaperID'].nunique()}, total rows: {len(df)}")

    if sanity:
        for col in FIELDS:
            input_sanity_check_column(df, col)

    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year_num", "year"])
    df["year_num"] = df["year_num"].astype(int)

    df = df.astype("string")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help="", default='data/filter.tsv')
    parser.add_argument('--gold', help="", default='data/gold.tsv')
    parser.add_argument('--random', help="", default='data/random.tsv')
    parser.add_argument("--sanity", action="store_true", help="run exploratory data analysis")
    parser.add_argument('--res', default='res/cross_sample/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res], args=args)

    dfs = {
        "filter": load_df(args.filter, args.sanity),
        "gold": load_df(args.gold, args.sanity),
        "random": load_df(args.random, args.sanity),
    }

    freq_df = cross_sample_value_distributions(dfs, FIELDS, TAX['multilabel'],
                                               TAX['string_digits'],
                                               top_k=50)
    print(freq_df.head())
    print(freq_df.shape)
    freq_df.to_csv(f"{args.res}cross_sample_value_distributions.tsv",
                   sep="\t", index=False)

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
