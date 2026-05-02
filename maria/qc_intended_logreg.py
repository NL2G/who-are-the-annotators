"""
29 April 2026
RQ4: Does the intended use of human judgment affect reportage quality?

USAGE:
python3 qc_intended_logreg.py

"""

import argparse
import matplotlib.ticker as mticker
import time
from collections import Counter
from pathlib import Path

import numpy as np
import seaborn as sns

from configs import FIELDS
from configs import SAMPLES_PALETTE
from configs import make_dirs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf


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


def lighten_color(color, amount=0.5):
    """
    Mix color with white.
    amount=0 → original color
    amount=1 → white
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color

    c = mcolors.to_rgb(c)
    white = (1, 1, 1)

    return tuple((1 - amount) * x + amount * y for x, y in zip(c, white))


def load_df(path, sanity):

    df = pd.read_csv(Path(path), sep="\t", engine="python", on_bad_lines="warn", dtype=str, keep_default_na=False)
    if df.isna().any().any():
        exit('Exiting: NaN values detected in the CSV after loading. Check the data for missing values.')

    print(f"\tUnique papers: {df['PaperID'].nunique()}, total rows: {len(df)}")

    if sanity:
        for col in FIELDS:
            input_sanity_check_column(df, col)

    return df


def plot_uses_odds(summary_df, color=None, dvar=None, save_as=None):
    fig, ax = plt.subplots(figsize=(5, 4))

    summary_df = summary_df.sort_values("OR")

    ax.errorbar(
        summary_df.index,
        summary_df["OR"],
        yerr=[
            summary_df["OR"] - summary_df["lower"],
            summary_df["upper"] - summary_df["OR"]
        ],
        fmt="o",
        capsize=5,
        color=color
    )

    ax.set_yscale("log")

    # reference line at OR = 1
    ax.axhline(1, linestyle="--", color="gray")  # OR = 1 → no effect
    # ax.text(
    #     0.01, 1,
    #     "no effect",
    #     transform=ax.get_yaxis_transform(),
    #     va="bottom",
    #     ha="left",
    #     fontsize=9,
    #     color="gray"
    # )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # how many times the odds of QC change relative to baseline (OR = 1), y=10 tenfold increase
    if "recruit" in dvar:
        ax.set_ylabel("", labelpad=0)
    else:
        ax.set_ylabel("log(odds ratio)", labelpad=-1)
    ax.set_title("")  # Predictors of QC usage
    for i, val in enumerate(summary_df["OR"]):
        ax.text(i, val, f"  OR={val:.1f}", ha="left", va="bottom", fontsize=11)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
    plt.show()


def plot_tempo_qc_odds(df, model, baseline=None, color=None, save_as=None):

    years = np.sort(df["year_num"].dropna().unique())

    if baseline == "zero":
        base = pd.DataFrame({
            "year_num": years,
            "is_resource": 0,
            "is_model": 0,
            "is_human": 0
        })
    else:  # mean-based baseline
        base = pd.DataFrame({
            "year_num": years,
            "is_resource": df["is_resource"].mean(),
            "is_model": df["is_model"].mean(),
            "is_human": df["is_human"].mean()
        })

    base["p_qc"] = model.predict(base)
    pred = model.get_prediction(base)
    summary = pred.summary_frame(alpha=0.05)
    base["p_qc"] = summary["predicted"]
    base["lower"] = summary["ci_lower"]
    base["upper"] = summary["ci_upper"]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(base["year_num"], base["p_qc"], color=color, linewidth=2)
    ax.fill_between(
        base["year_num"],
        base["lower"],
        base["upper"],
        alpha=0.2
    )

    ax.set_xlabel("")
    ax.set_ylabel("predicted probability of QC")
    ax.set_title("")  # controlled temporal effect of year on QC probability, holding task distribution fixed at mean values
    ax.set_ylim(0, 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help="", default='data/filter.tsv')
    parser.add_argument('--gold', help="", default='data/gold.tsv')
    parser.add_argument('--random', help="", default='data/random.tsv')
    parser.add_argument("--sanity", action="store_true", help="run exploratory data analysis")
    parser.add_argument('--res', default='res/logreg_odds/')
    parser.add_argument('--pics', default='res/pics/logreg_odds/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res, args.pics], args=args)

    dfs = {
        "filter": load_df(args.filter, args.sanity),
        "gold": load_df(args.gold, args.sanity),
        "random": load_df(args.random, args.sanity),
    }

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })

    for name, df in dfs.items():
        if name != "filter":
            continue
        for dvar in ["recruit", "recruit-comp", "qc"]:
            if dvar == "recruit-comp":
                df["has_qc"] = (
                    (df["recruitment"].str.strip() != "not reported") |
                    (df["compensation"].str.strip() != "not reported")
                )
            elif dvar == "recruit":
                df["has_qc"] = (df["recruitment"].str.strip() != "not reported")
            else:  # "qc"
                df["has_qc"] = (
                        (df["post_filtering"] == "yes") |
                        (df["adjudication"] != "none") |
                        (df["training"] == "yes") |
                        (df["iaa_metric"] != "not reported"))

            df["has_qc"] = df["has_qc"].astype(int)

            df["is_resource"] = df["intended_use"].str.contains("resource creation", case=False, na=False).astype(int)
            df["is_model"] = df["intended_use"].str.contains("model output evaluation", case=False, na=False).astype(int)
            df["is_human"] = df["intended_use"].str.contains("human performance", case=False, na=False).astype(int)
            df["year_num"] = pd.to_numeric(df["year"], errors="coerce")

            model = smf.logit(
                "has_qc ~ is_resource + is_model + is_human + year_num",
                data=df
            ).fit()

            print(model.summary())

            odds_ratios = np.exp(model.params)
            print("\nOdds Ratios:")
            print(odds_ratios)

            params = model.params
            conf = model.conf_int()

            summary_df = pd.DataFrame({
                "OR": np.exp(params),
                "lower": np.exp(conf[0]),
                "upper": np.exp(conf[1])
            }).drop("Intercept")

            summary_df = summary_df.rename(index={
                "is_resource": "resource",
                "is_model": "output eval.",
                "is_human": "human level",
                "year_num": "year"
            })

            summary_df.to_csv(f"{args.res}{name}uses~{dvar}_odds.tsv", sep='\t')

            save_odds = f"{args.pics}{name}_uses~{dvar}_odds+years.png"
            plot_uses_odds(summary_df, color=SAMPLES_PALETTE[name], dvar=dvar, save_as=save_odds)

            # save_tempo_odds = f"{args.res}{name}_qc_over_time.png"
            # plot_tempo_qc_odds(df, model, baseline='mean', color=SAMPLES_PALETTE[name], save_as=save_tempo_odds)

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
