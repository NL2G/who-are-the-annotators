"""
01 May 2026
get a table with hits per each value of each column, to check the data and get a sense of the distribution of values.
This is also useful for sanity checking the data, as it can reveal any unexpected or inconsistent values that may need to be cleaned or corrected before analysis.

USAGE:
python3 cross_sample_bars.py

"""

import argparse
import time
from collections import Counter
from pathlib import Path

import seaborn as sns

from configs import FIELDS, TAX
from configs import make_dirs
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
import scienceplots

plt.style.use(["science", "no-latex"])

sns.set_theme(context="talk", style="white")

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})


def cross_sample_value_distributions(df_dict, cols, multilabel, strings, top_k=None):
    df = pd.concat(
        [get_cross_sample_distribution(df_dict, c, multilabel, strings, top_k) for c in cols],
        ignore_index=True
    )

    col = df["Δ% (main - random)"]

    print("min:", col.min())
    print("max:", col.max())
    print("mean (abs):", col.abs().mean())
    #
    # exit()

    return df.sort_values(
        ["category", "Δ% (main - random)"],
        key=lambda x: x.abs() if x.name == "Δ% (main - random)" else x,
        ascending=[True, False]
    )


def get_cross_sample_distribution(df_dict, col, multilabel, strings, top_k=None):
    multilabel = multilabel or []
    strings = strings or []

    df_filt = df_dict["filter"]
    df_rand = df_dict["random"]

    # print("FILTER types:")
    # print(df_filt[col].apply(type).value_counts())
    #
    # print("\nRANDOM types:")
    # print(df_rand[col].apply(type).value_counts())

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
            "Δ# (main - random)": int(delta_count),
            "Δ% (main - random)": round(delta_pct, 1),
            "p<0.05": sig
        })

    out = pd.DataFrame(rows)

    if top_k:
        out = out.sort_values("Δ# (main - random)", ascending=False).head(top_k)

    out = out.sort_values("Δ% (main - random)", key=lambda x: x.abs(), ascending=False)

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


def plot_significant_deltas(
    df,
    binary_lst=None,
    positive_values=("yes", "reported", "present", "true", "1"),
    top_k_per_category=None,
    exclude_categories=("payment_rate",),
    min_abs_pct=None,
    save_as=None
):
    # plot
    figsize = (16, 20)
    d = df.copy()
    binary_lst = set(binary_lst or [])

    # normalize value strings for matching
    d["value_norm"] = d["value"].str.lower().str.strip()

    if exclude_categories:
        d = d[~d["category"].isin(exclude_categories)]

    # keep only significant
    d = d[d["p<0.05"] == "*"]

    # optional filter by effect size
    if min_abs_pct is not None:
        d = d[d["Δ% (main - random)"].abs() >= min_abs_pct]

    # --- handle binary categories: keep ONLY positive values ---
    if binary_lst:
        binary_mask = d["category"].isin(binary_lst)

        d_bin = d[binary_mask & d["value_norm"].isin(positive_values)]
        d_nonbin = d[~binary_mask]

        d = pd.concat([d_nonbin, d_bin], ignore_index=True)

    # --- top-k per category (after filtering) ---
    if top_k_per_category is not None:
        d = (
            d.assign(abs_pct=d["Δ% (main - random)"].abs())
             .sort_values(["category", "abs_pct"], ascending=[True, False])
             .groupby("category")
             .head(top_k_per_category)
        )

    # sort globally
    d = d.sort_values("Δ% (main - random)")

    # labels
    d["label"] = d["category"] + " | " + d["value"]
    plt.figure(figsize=figsize)
    bars = plt.barh(
        d["label"],
        d["Δ% (main - random)"]
    )

    plt.axvline(0, linewidth=1)

    # faint horizontal dashed grid
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    # larger fonts
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate at bar beginnings
    for i, v in enumerate(d["Δ% (main - random)"]):

        if v >= 0:
            # positive bars -> label left of bar start
            x = -0.3
            ha = "right"
        else:
            # negative bars -> label right of bar start
            x = 0.3
            ha = "left"

        plt.text(
            x,
            i,
            f"{v:.1f}",
            va="center",
            ha=ha,
            fontsize=16
        )
    ax = plt.gca()
    ax.set_xlabel(
        r'$\Delta\% \; (\mathrm{Annotated}_{LLM} - \mathrm{random})$',
        fontsize=24
    )
    # plt.xlabel(r'$\Delta\% \; (\mathrm{Annotated}_{extract} - \mathrm{random})$', fontsize=18)
    plt.ylabel("")
    plt.title("")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
    plt.show()

    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help="", default='data/filter.tsv')
    parser.add_argument('--gold', help="", default='data/gold.tsv')
    parser.add_argument('--random', help="", default='data/random.tsv')
    parser.add_argument("--sanity", action="store_true", help="run exploratory data analysis")
    parser.add_argument('--res', default='res/cross_sample/')
    parser.add_argument('--pics', default='res/pics/cross_sample/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res, args.pics], args=args)

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
    top_k = 2
    save_pic = f"{args.pics}main_rand_divergence_pct.png"
    plot_significant_deltas(freq_df, binary_lst=TAX["binary"]+TAX['string_digits'],
                            top_k_per_category=None, save_as=save_pic)

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
