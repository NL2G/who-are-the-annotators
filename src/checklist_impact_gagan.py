"""
29 April 2026
association between characteristics
USAGE:
python3 checklist_impact.py

"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "scienceplots>=1.0",
#     "seaborn>=0.12",
#     "statsmodels>=0.14"
# ]
# ///

import argparse
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
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
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


def prepare_its(df, year_col="year", score_col="score"):
    data = df.copy()

    data["year_num"] = pd.to_numeric(data[year_col], errors="coerce")
    data = data.dropna(subset=["year_num", score_col])
    data["year_num"] = data["year_num"].astype(int)

    # intervention: ACL checklist (2022)
    data["post_2022"] = (data["year_num"] >= 2022).astype(int)

    # time centered at intervention (optional but cleaner)
    data["time"] = data["year_num"] - 2022

    return data


def fit_its(data, score_col="score"):
    model = smf.ols(
        f"{score_col} ~ time + post_2022 + time:post_2022",
        data=data
    ).fit()

    print(model.summary())
    return model


def aggregate_yearly(data, score_col="score"):
    yearly = (
        data.groupby("year_num")[score_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("year_num")
    )
    yearly["sem"] = yearly["std"] / (yearly["count"] ** 0.5)
    return yearly


# interrupted time series
def plot_its(yearly, model, color="#1f77b4", save_as=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    # --- observed data ---
    ax.errorbar(
        yearly["year_num"],
        yearly["mean"],
        yerr=yearly["sem"],
        fmt="o",
        capsize=3,
        color=color,
        markersize=7,
        linewidth=1.5,
    )

    # --- prediction grid ---
    years = np.arange(yearly["year_num"].min(), yearly["year_num"].max() + 1)

    pred_df = pd.DataFrame({
        "year_num": years,
        "time": years - 2022,
        "post_2022": (years >= 2022).astype(int)
    })

    pred = model.get_prediction(pred_df).summary_frame()

    # --- fitted line + CI ---
    ax.plot(years, pred["mean"], color=color, linewidth=2.5)
    ax.fill_between(
        years,
        pred["mean_ci_lower"],
        pred["mean_ci_upper"],
        color=color,
        alpha=0.15
    )

    # --- intervention line ---
    ax.axvline(2022, linestyle="--", color="gray", linewidth=1.2)
    ax.text(2022.1, 0.97, "ACL checklist", transform=ax.get_xaxis_transform(),
            va="top", fontsize=16, color="black")

    # what would have happened without the checklist
    pred_df_cf = pred_df.copy()
    pred_df_cf["post_2022"] = 0  # no intervention

    cf = model.get_prediction(pred_df_cf).summary_frame()

    ax.plot(years, cf["mean"], linestyle="--", color=color, alpha=0.55,
            linewidth=1.8)

    # --- manual legend ---
    obs_handle = Line2D([0], [0], color=color, marker="o", markersize=7,
                        linewidth=1.5, label="Yearly average")
    fit_handle = Line2D([0], [0], color=color, linewidth=2.5,
                        label="Trend estimated by the model")
    cf_handle  = Line2D([0], [0], color=color, linewidth=1.8,
                        linestyle="--", alpha=0.55, label="Projected trend without the checklist")
    ci_handle  = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.15, ec="none",
                                label="95% confidence band")

    ax.legend(handles=[obs_handle, fit_handle, cf_handle, ci_handle],
              frameon=False, loc="upper left")

    ax.set_xlabel("")
    ax.set_ylabel("Average reportage score")
    ax.set_title("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
        plt.savefig(save_as.replace(".png", ".pdf"), bbox_inches="tight")

    plt.close()


def plot_venues(df_sub, cur_score, save_as=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.lineplot(
        data=df_sub,
        x="year_num",
        y="fair_score",
        hue="venue",
        estimator="mean",
        errorbar=None,
        linewidth=3,
    )

    ax.legend(frameon=False)
    ax.set_xlabel("")
    ax.set_ylabel("Reportage score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, linestyle="--", alpha=0.3)

    plt.title("")
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
    plt.close()


def its_by_venue(df, cur_score, top_venues, save_as=None):
    df = df.copy()
    df["post_2022"] = (df["year_num"] >= 2022).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10.colors  # stable palette

    venue_color = {v: colors[i % len(colors)] for i, v in enumerate(top_venues)}

    years_global = np.arange(df["year_num"].min(), df["year_num"].max() + 1)

    for v in top_venues:
        sub = df[df["venue"] == v].copy()
        if sub.empty:
            continue

        model = smf.ols(
            f"{cur_score} ~ year_num + post_2022 + year_num:post_2022",
            data=sub
        ).fit()

        # --- observed ---
        yearly = (
            sub.groupby("year_num")[cur_score]
            .agg(["mean", "sem"])
            .reset_index()
        )

        ax.errorbar(
            yearly["year_num"],
            yearly["mean"],
            yerr=yearly["sem"],
            fmt="o",
            color=venue_color[v],
            capsize=3,
            markersize=7,
            linewidth=1.5,
        )

        # --- prediction grid ---
        pred_df = pd.DataFrame({
            "year_num": years_global,
            "post_2022": (years_global >= 2022).astype(int)
        })

        pred = model.get_prediction(pred_df).summary_frame()

        # fitted (coloured solid) + CI band
        ax.plot(
            years_global,
            pred["mean"],
            color=venue_color[v],
            linewidth=2.5
        )
        ax.fill_between(
            years_global,
            pred["mean_ci_lower"],
            pred["mean_ci_upper"],
            color=venue_color[v],
            alpha=0.12
        )

        # counterfactual (dashed)
        pred_cf = pred_df.copy()
        pred_cf["post_2022"] = 0

        cf = model.get_prediction(pred_cf).summary_frame()

        ax.plot(
            years_global,
            cf["mean"],
            linestyle="--",
            color=venue_color[v],
            alpha=0.55,
            linewidth=1.8
        )

    # intervention marker
    ax.axvline(2022, linestyle="--", color="gray", linewidth=1.2)
    ax.text(2022.1, 0.97, "ACL checklist\nintroduced", transform=ax.get_xaxis_transform(),
            va="top", fontsize=13, color="gray")
    # ---------------------------
    # MANUAL LEGEND
    # ---------------------------
    venue_handles = [
        Line2D([0], [0], color=venue_color[v], lw=2.5, marker="o",
               markersize=7, label=v.upper())
        for v in top_venues
    ]

    fit_handle = Line2D([0], [0], color="darkgray", lw=2.5, label="Trend estimated by the model")
    cf_handle = Line2D(
        [0], [0],
        linestyle="--",
        color="darkgray",
        alpha=0.55,
        label="Projected trend without the checklist"
    )

    ax.legend(
        handles=venue_handles + [fit_handle, cf_handle],
        loc="upper left",
        frameon=False
    )

    ax.set_xlabel("")
    ax.set_ylabel("Average reportage score\n(0 = nothing reported, 1 = fully reported)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")

    plt.close()


RUN = "filter"  # "filter", "gold", "random"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help="", default='data/filter.tsv')
    parser.add_argument('--gold', help="", default='data/gold.tsv')
    parser.add_argument('--random', help="", default=None)
    parser.add_argument("--sanity", action="store_true", help="run exploratory data analysis")
    parser.add_argument('--res', default=f'res/check_impact/{RUN}/')
    parser.add_argument('--pics', default=f'res/pics/check_impact/{RUN}/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res, args.pics], args=args)

    dfs = {
        "filter": load_df(args.filter, args.sanity),
        "gold": load_df(args.gold, args.sanity),
        # "random": load_df(args.random, args.sanity),
    }

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16
    })
    for cur_score in ["fair_score", "blunt_score"]:
        for name, df in dfs.items():
            if name != RUN:
                continue

            if cur_score == "fair_score":
                df[["fair_reported", "expected", "fair_score"]] = df.apply(compute_fair_score, axis=1)
            else:
                df[["blunt_reported", "reportable", "blunt_score"]] = df.apply(compute_blunt_score, axis=1)

            data = prepare_its(df, score_col=cur_score)
            model = fit_its(data, score_col=cur_score)
            yearly = aggregate_yearly(data, score_col=cur_score)
            plot_its(yearly, model, color=SAMPLES_PALETTE[name], save_as=f"{args.pics}its_{name}_{cur_score}.png")

            # decomposition by venue
            top_venues = ["acl", "emnlp", "naacl"]
            df_sub = data[data["venue"].isin(top_venues)].copy()
            # print(df_sub.head())
            # exit()

            # plot_venues(df_sub, cur_score, save_as=f"{args.pics}venues_{name}_{cur_score}.png")

            # ols_venues(df_sub, cur_score, top_venues, save_as=f"{args.pics}ols_venues_{name}_{cur_score}.png")

            its_by_venue(df_sub, cur_score, top_venues, save_as=f"{args.pics}its_by_venue_{name}_{cur_score}.png")

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
