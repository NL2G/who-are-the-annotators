"""
29 April 2026

USAGE:
python3 reportage_score_tempo.py --bars

"""

import argparse
import os
import re
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
#
import scienceplots

plt.style.use(["science", "no-latex"])

sns.set_theme(context="talk", style="white")

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})


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


def split_multi(val):
    if pd.isna(val) or str(val).strip() == "":
        return ["missing"]
    return [v.strip() for v in re.split(r"[;|,]", str(val)) if v.strip()]


FAIR_DICT = {
    'universal': ["guidelines_released", "training", "education", "post_filtering",  # yes/no
                  "total_annotators", "total_items",  # string_digits or "not reported"
                  "recruitment", "language_proficiency", "expertise_level", "compensation"],
    # not reported or categorical
    'opinionated': ['age', 'gender', 'nation_residence', 'nation_origin', 'political_orientation'],  # yes/no
    'conditional': ['iaa_metric', 'iaa_value', "annotators_per_item", "items_per_annotator",  # not reported
                    "crowd_screening", "adjudication"]  # no or none
}


# rules:
# if "total_annotators" == 1 or "annotators_per_item" == 1, then skip "Agreement Level" and Adjudication
# if "iaa_metric" == "not_reported", then skip "iaa_value"
# if 'iaa_metric' != 'other' and 'iaa_value' == 'na', then 0, else 1  # unclear reportage

# excessive info, not to be expected along with other reporting
# if "total_annotators" == "annotators_per_item" and != "not reported", then skip "annotators_per_item"  # duplicate info
# if "annotators_per_item" == "total_annotators" and "total_items" != "not reported", skip "items_per_annotator"
# if "recruitment" != "crowdsourcing", skip "crowd_screening"
# if "task_type"!= "open_ended", skip "adjudication"

# payment_rate is a hierarchical extension of compensation: not its own dimension, exclude entirely


def compute_blunt_score(row):
    def is_reported(v):
        return pd.notna(v) and str(v).strip().lower() not in {"not reported", "no", "na", "none"}

    reported_count = 0
    _total = 0

    reportables = set(FAIR_DICT["universal"] + FAIR_DICT["conditional"] + FAIR_DICT["opinionated"])  # - {'political_orientation', 'nation_origin'}  # exclude payment_rate entirely

    # --- always include universal ---
    for col in reportables:
        _total += 1
        reported_count += int(is_reported(row.get(col, np.nan)))

    score = reported_count / _total if _total > 0 else np.nan

    return pd.Series({
        "blunt_reported": reported_count,
        "reportable": _total,
        "blunt_score": score
    })


# fair_total range is 10 to 21 depending on conditions, reported_count is 0 to fair_total, score is 0 to 1
def compute_fair_score(row):
    def is_reported(v):
        return pd.notna(v) and str(v).strip().lower() not in {"not reported", "no", "na", "none"}

    reported_count = 0
    fair_total = 0

    # --- always include universal ---
    for col in FAIR_DICT["universal"]:
        fair_total += 1
        reported_count += int(is_reported(row.get(col, np.nan)))

    # --- conditional rules ---
    skip = set()

    # rule: annotator structure
    if row.get("total_annotators") == "1" or row.get("annotators_per_item") == "1":
        skip.update(["iaa_metric", "iaa_value", "adjudication"])

    # rule: iaa dependency
    if str(row.get("iaa_metric", "")).lower() == "not reported":
        skip.add("iaa_value")

    # rule: unclear iaa value reportage
    if (str(row.get("iaa_metric", "")).lower() != "other" and
            str(row.get("iaa_value", "")).lower() == "na"):
        reported_count += 0

    # rule: redundancy
    if (row.get("total_annotators") == row.get("annotators_per_item") and
            row.get("total_annotators") != "not reported"):
        skip.add("annotators_per_item")

    if (row.get("total_annotators") == row.get("annotators_per_item") and
            row.get("total_items") != "not reported"):
        skip.add("items_per_annotator")

    if str(row.get("recruitment", "")).lower() != "crowdsourcing":
        skip.add("crowd_screening")

    if str(row.get("task_type", "")).lower() != "open-ended generation":
        skip.add("adjudication")

    # --- conditional fields ---
    for col in FAIR_DICT["conditional"]:
        if col in skip:
            continue
        fair_total += 1
        reported_count += int(is_reported(row.get(col, np.nan)))

    # --- opinionated only if condition satisfied ---
    include_opinionated = (
            str(row.get("paper_topic", "")).lower()
            == "subjective language & social meaning"
    )

    if include_opinionated:
        for col in FAIR_DICT["opinionated"]:
            fair_total += 1
            reported_count += int(is_reported(row.get(col, np.nan)))

    score = reported_count / fair_total if fair_total > 0 else np.nan

    return pd.Series({
        "fair_reported": reported_count,
        "expected": fair_total,
        "fair_score": score
    })


def normalize_year(y):
    if not isinstance(y, str):
        return None
    y = y.strip()
    if y.startswith("<"):
        return int(y[1:]) - 1  # e.g. "<2018" -> 2017
    if y.endswith("+"):
        return int(y[:-1])  # e.g. "2025+" -> 2025
    if y.isdigit():
        return int(y)
    return None


def make_dataset_year_table(
        df,
        dataset_col="dataset",
        year_col="year",
        score_col="score",
        datasets=None
):
    data = df.copy()

    # --- strict year parsing ---
    data["year_num"] = pd.to_numeric(data[year_col], errors="coerce")
    data = data.dropna(subset=["year_num", score_col, dataset_col])
    data["year_num"] = data["year_num"].astype(int)

    # optionally filter datasets
    if datasets is not None:
        data = data[data[dataset_col].isin(datasets)]

    # --- aggregate mean score ---
    agg = (
        data.groupby([dataset_col, "year_num"])[score_col]
        .mean()
        .reset_index()
    )

    # --- pivot ---
    table = agg.pivot(
        index=dataset_col,
        columns="year_num",
        values=score_col
    ).sort_index(axis=1)

    order = ["gold", "filter", "random"]
    table = table.reindex(order).reset_index()

    return table


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


# def plot_score_trend_with_counts(df, year_col="year", score_col="score",
#                                  main_color=None,
#                                  save_as=None):
#     data = df.copy()
#
#     # --- strict year parsing ---
#     data["year_num"] = pd.to_numeric(data[year_col], errors="coerce")
#     data = data.dropna(subset=["year_num", score_col])
#     data["year_num"] = data["year_num"].astype(int)
#
#     # --- aggregate ---
#     yearly = (
#         data.groupby("year_num")[score_col]
#         .agg(["mean", "std", "count"])
#         .reset_index()
#         .sort_values("year_num")
#     )
#     yearly["sem"] = yearly["std"] / (yearly["count"] ** 0.5)
#
#     # alternative simpler
#     yearly_mean = (
#         data.groupby("year_num", as_index=False)[score_col]
#         .mean()
#         .sort_values("year_num")
#     )
#
#     # --- plot ---
#     fig, ax1 = plt.subplots(figsize=(9, 5))
#
#     sns.lineplot(data=yearly_mean, x="year_num", y=score_col,
#                  color=main_color,
#                  marker="o", ax=ax1)
#     # error bars = SEM Standard Error of the Mean
#     # how uncertain your estimate of the mean is per year, not the variability of individual scores (which is SD)
#     ax1.errorbar(
#         yearly["year_num"],
#         yearly["mean"],
#         yerr=yearly["sem"],
#         fmt="none",
#         ecolor=main_color,
#         capsize=3
#     )
#
#     n_rows = len(data)
#
#     # textbox
#     # ax1.text(
#     #     0.02, 0.95,
#     #     f"Main 'filter' dataset\nTotal annotated tasks: {n_rows}",
#     #     transform=ax1.transAxes,
#     #     fontsize=10,
#     #     verticalalignment="top",
#     #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none")
#     # )
#
#     # # smoothed trend
#     # sns.regplot(
#     #     data=yearly,
#     #     x="year_num",
#     #     y="mean",
#     #     lowess=True,
#     #     scatter=True,
#     #     ax=ax1,
#     #     line_kws={"linewidth": 2}
#     # )
#
#     # # add SD error bars
#     # ax1.errorbar(
#     #     yearly["year_num"],
#     #     yearly["mean"],
#     #     yerr=yearly["std"],
#     #     fmt="none",   # no extra markers
#     #     capsize=3
#     # )
#
#     ax1.set_xlabel("")
#     ax1.set_ylabel("Average score")
#
#     # secondary axis for counts
#     bar_color = lighten_color(main_color, amount=0.2)
#     ax2 = ax1.twinx()
#     ax2.bar(
#         yearly["year_num"],
#         yearly["count"],
#         color=bar_color,
#         alpha=0.3
#     )
#     ax2.set_ylabel("# Annotated tasks")
#
#     # clean spines
#     ax1.spines["top"].set_visible(False)
#     ax2.spines["top"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax2.spines["left"].set_visible(False)
#
#     plt.title("Reportage score over time with SEM intervals and paper counts")
#     plt.tight_layout()
#     if save_as:
#         plt.savefig(save_as)
#     plt.show()


def compare_trends_bars(df, year_col="year", score_col="score",
                        main_color="#1f77b4", focus_color=None, labels=None,
                        my_split_by='paper_topic', bars=True,
                        cat="subjective language & social meaning",
                        save_as=None):
    data = df.copy()

    data["is_focused"] = data[my_split_by].str.contains(cat, case=False, na=False)

    data["year_num"] = pd.to_numeric(data[year_col], errors="coerce")
    data = data.dropna(subset=["year_num", score_col])
    data["year_num"] = data["year_num"].astype(int)

    # --- aggregate for lines ---
    def agg(sub):
        return (
            sub.groupby("year_num")[score_col]
            .mean()
            .reset_index()
        )

    full = agg(data)

    subj = agg(data[data["is_focused"]])
    if "Other" not in labels[1]:
        data["double_focused"] = data[my_split_by].str.contains("resource creation", case=False, na=False)
        nonsubj = agg(data[data["double_focused"]])
    else:
        nonsubj = agg(data[~data["is_focused"]])

    # --- aggregate for bars ---
    counts = (
        data.groupby(["year_num", "is_focused"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    counts.columns = ["other", "focus"]

    colors = {
        "all": "#4D4D4D",   # dark gray
        "focus": focus_color,  # "#d62728",
        "other": main_color
    }

    subj_yearly = (
        data[data["is_focused"]]
        .groupby("year_num")[score_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("year_num")
    )

    subj_yearly["sem"] = subj_yearly["std"] / (subj_yearly["count"] ** 0.5)

    # sns.set_context("paper")
    print(plt.rcParams["font.size"])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    if bars:
        ax2 = ax1.twinx()

        # -------------------------------
        # background stacked bars
        # -------------------------------
        ax2.bar(
            counts.index,
            counts["focus"],
            color=colors["focus"],
            alpha=0.35,
            width=0.8,
            edgecolor="white",
            label="_nolegend_",
            zorder=0,
        )

        ax2.bar(
            counts.index,
            counts["other"],
            bottom=counts["focus"],
            color=colors["other"],
            alpha=0.35,
            edgecolor="white",
            width=0.8,
            label="_nolegend_",
            zorder=0,
        )

        ax2.set_ylabel("# annotation tasks")

        years = range(data["year_num"].min(), data["year_num"].max() + 1)
        ax2.set_xticks(years)

    # -------------------------------
    # overall trend
    # -------------------------------
    sns.lineplot(
        data=full,
        x="year_num",
        y=score_col,
        ax=ax1,
        label="All annotation tasks",
        color=colors["all"],
        linestyle="--",
        linewidth=3,
        zorder=3,
    )

    # -------------------------------
    # focused category
    # -------------------------------
    sns.lineplot(
        data=subj_yearly,
        x="year_num",
        y="mean",
        ax=ax1,
        label=labels[0],
        color=colors["focus"],
        linewidth=3,
        zorder=4,
    )

    ax1.errorbar(
        subj_yearly["year_num"],
        subj_yearly["mean"],
        yerr=subj_yearly["sem"],
        fmt="o",
        markersize=4,
        ecolor=colors["focus"],
        color=colors["focus"],
        elinewidth=1.5,
        capsize=2,
        zorder=5,
    )

    # -------------------------------
    # other categories
    # -------------------------------
    sns.lineplot(
        data=nonsubj,
        x="year_num",
        y=score_col,
        ax=ax1,
        label=labels[1],
        color=colors["other"],
        linewidth=3,
        zorder=3,
    )

    # -------------------------------
    # cosmetics
    # -------------------------------
    ax1.set_xlabel("")
    ax1.set_ylabel("Average reportage score")
    ax1.set_title("")

    ax1.legend(frameon=False, loc="upper left")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if bars:
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
    plt.rcParams["xtick.top"] = False
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.tick_params(axis='x', pad=8)
    # ax2.tick_params(axis='both', which='both', length=0)
    ax1.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )
    # ax2.grid(False)

    plt.tight_layout()

    if save_as:
        if save_as.endswith(".pdf"):
            plt.savefig(save_as, bbox_inches="tight")
        else:
            plt.savefig(save_as, dpi=300, bbox_inches="tight")

    plt.show()


# def compare_trends(df, year_col="year", score_col="score",
#                    main_color=None,
#                    save_as=None):
#
#     data = df.copy()
#     data["is_focused"] = data["paper_topic"].str.contains(
#         "subjective language & social meaning",
#         case=False,
#         na=False
#     )
#
#     group_col = "is_focused"
#
#     data["year_num"] = pd.to_numeric(data[year_col], errors="coerce")
#     data = data.dropna(subset=["year_num", score_col, group_col])
#     data["year_num"] = data["year_num"].astype(int)
#
#     def agg(sub):
#         return (
#             sub.groupby("year_num")[score_col]
#             .mean()
#             .reset_index()
#         )
#
#     full = agg(data)
#     subj = agg(data[data[group_col]])
#     nonsubj = agg(data[~data[group_col]])
#
#     colors = {
#         "all": "#4c4c4c",  # neutral dark gray,
#         "focus": "#d62728",  # red (Matplotlib default red)
#         "other": main_color
#     }
#
#     plt.figure(figsize=(9, 5))
#
#     sns.lineplot(data=full, x="year_num", y=score_col, label="All", color=colors["all"],
#                  linestyle="--", linewidth=2)
#     sns.lineplot(data=subj, x="year_num", y=score_col, label="Sociolinguistic papers", color=colors["focus"],
#                  linewidth=2)
#     sns.lineplot(data=nonsubj, x="year_num", y=score_col, label="Other papers",
#                  color=colors["other"], linewidth=2)
#     ax = plt.gca()
#     ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
#     plt.xlabel("year")
#     plt.ylabel("Average score")
#     plt.title("")
#     plt.legend(frameon=False)
#     plt.tight_layout()
#
#     if save_as:
#         plt.savefig(save_as)
#     plt.show()


def load_df(path, sanity):

    df = pd.read_csv(Path(path), sep="\t", engine="python", on_bad_lines="warn", dtype=str, keep_default_na=False)
    if df.isna().any().any():
        exit('Exiting: NaN values detected in the CSV after loading. Check the data for missing values.')

    print(f"\tUnique papers: {df['PaperID'].nunique()}, total rows: {len(df)}")

    if sanity:
        for col in FIELDS:
            input_sanity_check_column(df, col)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help="", default='data/filter.tsv')
    parser.add_argument('--gold', help="", default='data/gold.tsv')
    parser.add_argument('--random', help="", default='data/random.tsv')
    parser.add_argument("--sanity", action="store_true", help="run exploratory data analysis")
    parser.add_argument("--bars", action="store_true", help="Plot bars in the background.")
    parser.add_argument('--res', default='res/score_tempo/')
    parser.add_argument('--pics', default='res/pics/score_tempo/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res, args.pics], args=args)

    dfs = {
        "gold": load_df(args.gold, args.sanity),
        "filter": load_df(args.filter, args.sanity),
        "random": load_df(args.random, args.sanity),
    }

    for cur_score in ["fair_score", "blunt_score"]:
        print(f'\n{cur_score.upper()}')

        for name, df in dfs.items():
            if args.bars:
                out_pic = str(Path(args.pics) / f"uses_{name}_{cur_score}+bars.pdf")
            else:
                out_pic = str(Path(args.pics) / f"uses_{name}_{cur_score}.pdf")
            if name != 'filter':
                continue
            print(f"\n\t{name}\n")
            if cur_score == "fair_score":
                df[["fair_reported", "expected", "fair_score"]] = df.apply(compute_fair_score, axis=1)
            else:
                df[["blunt_reported", "reportable", "blunt_score"]] = df.apply(compute_blunt_score, axis=1)

            # print(df.head())

            # save_pic = f"{args.pics}{name}_blunt_score_trend_tasks.png"
            # plot_score_trend_with_counts(gold, year_col="year",
            #                              score_col="blunt_score",
            #                              main_color=SAMPLES_PALETTE[dataset],
            #                              save_as=save_pic)

            # compare_trends(df, year_col="year", score_col="blunt_score",
            #                main_color=SAMPLES_PALETTE[dataset],
            #                save_as=f"{args.pics}{name}_ss_vs_full_blunt_score.png")

            # Sociopapers: blue  -> "#1f77b4" green -> #2ca02c  (focus: socio)
            compare_trends_bars(df, year_col="year", score_col=cur_score,
                                main_color="#0072B2",  # SAMPLES_PALETTE[name],
                                focus_color="#009E73",  # green
                                bars=args.bars,
                                my_split_by='paper_topic', labels=["Socio-linguistic topics", "Other topics"],
                                cat="subjective language & social meaning",
                                save_as=out_pic)
            # Intended use re-coloring:
            # orange -> "#ff7f0e" + red -> "#d62728"  inten
            # purple -> "#9467bd" + olive  -> "#bcbd22"
            # "#0072B2",  # blue "#D55E00",  # orange

            compare_trends_bars(df, year_col="year", score_col=cur_score,
                                main_color="#0072B2",  # SAMPLES_PALETTE[name],
                                focus_color="#D55E00",
                                bars=args.bars,
                                my_split_by='intended_use', labels=["Model output evaluation", "Resource creation"],
                                # my_split_by='intended_use', labels=["model output evaluation", "other purposes"],
                                cat="model output evaluation",
                                save_as=out_pic)
            dfs[name] = df

        mega_df = pd.concat([
            dfs["filter"].assign(dataset="filter"),
            dfs["gold"].assign(dataset="gold"),
            dfs["random"].assign(dataset="random")
        ], ignore_index=True)

        assert "dataset" in mega_df.columns

        save_scores = f"{args.res}mean_{cur_score}_by_year.tsv"
        if os.path.exists(save_scores):
            print("Scores already saved. To recompute, delete the existing file.")
        else:
            table = make_dataset_year_table(
                mega_df,
                dataset_col="dataset",
                year_col="year",
                score_col=cur_score,
                datasets=["gold", "filter", "random"]
            )

            print(table)
            table.to_csv(save_scores, index=False, sep='\t')

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
