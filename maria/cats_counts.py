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


def get_all_binned_value_distributions(df, cols, multilabel, strings, year_col=None, top_k=None):
    return pd.concat(
        [get_binned_value_distribution(df, c, multilabel, strings, year_col, top_k) for c in cols],
        ignore_index=True
    )


def get_binned_value_distribution(df, col, multilabel, strings, year_col=None, top_k=None):
    multilabel = multilabel or []
    strings = strings or []

    df = df.copy()
    df[col] = df[col].astype("string")

    # define periods
    df["period"] = pd.cut(
        df[year_col],
        bins=[2017, 2021, 2025],
        labels=["2018-2021", "2022-2025"]
    )

    def get_counts(series):
        return Counter(series)

    # --- MULTILABEL ---
    if col in multilabel:
        df_exp = df[[col, "period"]].dropna().copy()
        df_exp[col] = df_exp[col].astype(str).str.split(";")
        df_exp = df_exp.explode(col)
        df_exp[col] = df_exp[col].str.strip()
        df_exp = df_exp[df_exp[col] != ""]

        counts = (
            df_exp.groupby(["period", col], observed=False)
            .size()
            .unstack(fill_value=0)
        )

    elif col in strings:
        vals = df[col].astype("string")

        reported = (
                vals.notna()
                & (vals.str.strip() != "")
                & (vals.str.lower() != "not reported")
        )

        df_tmp = df.copy()
        df_tmp["value"] = reported.map({True: "reported", False: "not_reported"})

        counts = (
            df_tmp.groupby(["period", "value"], observed=False)  # includes all category combinations
            .size()
            .unstack(fill_value=0)
        )

    else:
        df_tmp = df[[col, "period"]].dropna().copy()
        df_tmp[col] = df_tmp[col].astype(str)

        counts = (
            df_tmp.groupby(["period", col], observed=False)
            .size()
            .unstack(fill_value=0)
        )

    pivot = counts.T
    pivot = pivot.reindex(columns=["2018-2021", "2022-2025"], fill_value=0)

    n1 = pivot["2018-2021"].sum()
    n2 = pivot["2022-2025"].sum()

    rows = []
    min_n = 30

    for val, row in pivot.iterrows():
        c1 = row["2018-2021"]
        c2 = row["2022-2025"]

        p1 = (c1 / n1 * 100) if n1 else 0
        p2 = (c2 / n2 * 100) if n2 else 0

        sig = "NA"
        if c1 >= min_n and c2 >= min_n:
            table = [[c1, n1 - c1],
                     [c2, n2 - c2]]
            try:
                _, pval, _, _ = chi2_contingency(table)
                sig = "*" if pval < 0.05 else ""
            except:
                sig = "NA"

        rows.append({
            "column": col,
            "value": val,
            "2018-2021 #": int(c1),
            "2022-2025 #": int(c2),
            "2018-2021 %": round(p1, 1),
            "2022-2025 %": round(p2, 1),
            "p<0.05": sig
        })

    out = pd.DataFrame(rows)

    if top_k:
        out = out.sort_values("2018-2021 #", ascending=False).head(top_k)

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


def get_all_distributions(df, cols, multilabel, strings, top_k=None):
    return pd.concat(
        [get_value_distribution(df, c, multilabel, strings, top_k) for c in cols],
        ignore_index=True
    )


def get_value_distribution(df, col, multilabel, strings, top_k=None):
    multilabel = multilabel or []
    strings = strings or []

    # --- MULTILABEL CASE ---
    if col in multilabel:
        vals = (
            df[col]
            .dropna()
            .astype(str)
            .str.split(";")
            .explode()
            .str.strip()
        )
        vals = vals[vals != ""]  # drop empty tokens

        total = len(vals)
        counter = Counter(vals)

        items = counter.most_common(top_k)
        out = pd.DataFrame(items, columns=["value", "count"])
        out["%"] = (out["count"] / total).round(3) * 100
        out["column"] = col
        out = out.rename(columns={"count": "#"})
        out = out[["column", "value", "#", "%"]]

        return out

    # --- STRING (BINARIZED) CASE ---
    elif col in strings:
        vals = df[col].astype(str)

        reported = vals.notna() & (vals.str.strip() != "") & (vals.str.lower() != "not reported")
        not_reported = ~reported

        counts = {
            "reported": reported.sum(),
            "not_reported": not_reported.sum()
        }

        total = sum(counts.values())

        out = pd.DataFrame(
            counts.items(), columns=["value", "count"]
        )
        out["%"] = (out["count"] / total).round(3) * 100
        out["column"] = col
        out = out.rename(columns={"count": "#"})
        out = out[["column", "value", "#", "%"]]

        return out

    # --- DEFAULT CASE ---
    else:
        vals = df[col].dropna().astype(str)
        total = len(vals)

        counter = Counter(vals)
        items = counter.most_common(top_k)

        out = pd.DataFrame(items, columns=["value", "count"])
        out["%"] = (out["count"] / total).round(3) * 100
        out["column"] = col
        out = out.rename(columns={"count": "#"})
        out = out[["column", "value", "#", "%"]]

        return out


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help="", default='data/filter.tsv')
    parser.add_argument('--gold', help="", default='data/gold.tsv')
    parser.add_argument('--random', help="", default='data/random.tsv')
    parser.add_argument("--sanity", action="store_true", help="run exploratory data analysis")
    parser.add_argument('--res', default='res/check_impact/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res], args=args)

    dfs = {
        "filter": load_df(args.filter, args.sanity),
        "gold": load_df(args.gold, args.sanity),
        "random": load_df(args.random, args.sanity),
    }

    for name, df in dfs.items():
        if name != "filter":
            continue

        dist_tab = get_all_distributions(df, FIELDS, TAX['multilabel'], TAX['string_digits'], top_k=20)
        print(dist_tab.head())
        dist_tab.to_csv(f"{args.res}value_distributions_{name}.tsv", sep="\t", index=False)

        data = df.copy()

        data["year_num"] = pd.to_numeric(data["year"], errors="coerce")
        data = data.dropna(subset=["year_num", "year"])
        data["year_num"] = data["year_num"].astype(int)

        freq_df = get_all_binned_value_distributions(data, FIELDS, TAX['multilabel'],
                                                     TAX['string_digits'],
                                                     year_col="year_num",
                                                     top_k=50)
        print(freq_df.head())
        print(freq_df.shape)
        freq_df.to_csv(f"{args.res}binned_value_distributions_{name}.tsv",
                       sep="\t", index=False)

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
