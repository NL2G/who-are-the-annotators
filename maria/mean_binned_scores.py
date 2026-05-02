"""
01 May 2026
get a table with hits per each value of each column, to check the data and get a sense of the distribution of values.
This is also useful for sanity checking the data, as it can reveal any unexpected or inconsistent values that may need to be cleaned or corrected before analysis.

USAGE:
python3 mean_binned_scores.py

"""

import argparse
import time
from collections import Counter
from pathlib import Path

from configs import FIELDS, TAX
from configs import make_dirs
import pandas as pd
from reportage_score_tempo import compute_fair_score, compute_blunt_score


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

    return df


def binned_category_frequencies(df, cat_cols, year_col="year"):
    df = df.copy()

    # define time bins
    bins = {
        "2018-2021": (2018, 2021),
        "2022-2025": (2022, 2025)
    }

    out = []

    for cat in cat_cols:
        tmp = {}

        # total
        tmp["category"] = cat
        tmp["total"] = df[cat].sum()

        # per period
        for label, (start, end) in bins.items():
            mask = df[year_col].between(start, end)
            tmp[label] = df.loc[mask, cat].sum()

        out.append(tmp)

    return pd.DataFrame(out)

def bin_summary(df, score_cols, year_col=None):
    df = df.copy()

    df["period"] = pd.cut(
        df[year_col],
        bins=[2017, 2021, 2025],
        labels=["2018-2021", "2022-2025"]
    )

    out = []

    for col in score_cols:
        tmp = (
            df.groupby("period")
              .agg(
                  tasks=(col, "size"),          # total rows
                  papers=("PaperID", "nunique"),# unique papers
                  mean=(col, "mean")
              )
              .reset_index()
              .rename(columns={
                  "tasks": f"tasks_{col}",
                  "papers": f"papers_{col}",
                  "mean": f"mean_{col}"
              })
        )
        out.append(tmp)

    # merge all on period
    result = out[0]
    for t in out[1:]:
        result = result.merge(t, on="period")

    # keep one tasks/papers column (they are identical across scores)
    task_cols = [c for c in result.columns if c.startswith("tasks_")]
    paper_cols = [c for c in result.columns if c.startswith("papers_")]

    result["tasks"] = result[task_cols[0]]
    result["papers"] = result[paper_cols[0]]

    result = result.drop(columns=task_cols + paper_cols)

    result["tasks/papers"] = (result["tasks"] / result["papers"]).round(2)

    return result


RUN = 'random'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help="", default='data/filter.tsv')
    parser.add_argument('--gold', help="", default='data/gold.tsv')
    parser.add_argument('--random', help="", default='data/random.tsv')
    parser.add_argument("--sanity", action="store_true", help="run exploratory data analysis")
    parser.add_argument('--res', default='res/check_impact/')
    parser.add_argument('--pics', default='res/pics/check_impact/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res, args.pics], args=args)

    dfs = {
        "filter": load_df(args.filter, args.sanity),
        "gold": load_df(args.gold, args.sanity),
        "random": load_df(args.random, args.sanity),
    }
    results = {}

    for name, df in dfs.items():
        if name != "filter":
            continue

        df = df.copy()

        for score_type in ["fair_score", "blunt_score"]:

            if score_type == "fair_score":
                df[["fair_reported", "expected", "fair_score"]] = df.apply(
                    compute_fair_score, axis=1
                )
            else:
                df[["blunt_reported", "reportable", "blunt_score"]] = df.apply(
                    compute_blunt_score, axis=1
                )
        data = df.copy()
        data["year_num"] = pd.to_numeric(data["year"], errors="coerce")
        data = data.dropna(subset=["year_num", "year"])
        data["year_num"] = data["year_num"].astype(int)

        summary = bin_summary(data, ["fair_score", "blunt_score"], year_col="year_num")
        print(summary)

        results[name] = data

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
