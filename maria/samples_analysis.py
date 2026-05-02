"""
28 April 2026

USAGE:
python3 samples_analysis.py

"""

import argparse
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import seaborn as sns
from configs import _HEADER_MAP, FIELDS, FIELD_LABELS, TAX, FIELD_GROUPS, PREFIXES, ALL_DROP_DOWNS
from configs import SAMPLES_PALETTE
import pandas as pd
import matplotlib.pyplot as plt


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


def format_args(args):
    parts = []
    for k, v in vars(args).items():
        if v is None or v is False:
            continue
        if v is True:
            parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


def make_dirs(logs=None, make_them=None, args=None):
    for i in make_them:
        if i:
            os.makedirs(i, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logs}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    if args:
        print(f"Run settings: python3 {sys.argv[0]} {format_args(args)}")
    else:
        print(f"Run settings: python3 {sys.argv[0]}")


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def plot_field_distributions(field_value_pairs, fields, field_labels=None):
    field_labels = field_labels or {}

    n = len(fields)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    axes = axes.flatten()

    for i, field in enumerate(fields):
        pairs = field_value_pairs.get(field, [])

        col_a = Counter(a for a, _ in pairs)
        col_b = Counter(b for _, b in pairs)

        labels = sorted(set(col_a) | set(col_b))

        a_vals = [col_a.get(l, 0) for l in labels]
        b_vals = [col_b.get(l, 0) for l in labels]

        x = range(len(labels))
        ax = axes[i]

        ax.bar([j - 0.2 for j in x], a_vals, width=0.4, label="A")
        ax.bar([j + 0.2 for j in x], b_vals, width=0.4, label="B")

        ax.set_title(field_labels.get(field, field))
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def load_df(path):
    print(path)
    my_df = pd.read_csv(
        Path(path),
        sep="\t",
        engine="python",
        on_bad_lines="warn",
        dtype=str,
        keep_default_na=False
    )
    my_df = my_df.rename(columns=_HEADER_MAP)
    MORE_FIXES = {
        "Year": "year",
        "Venue": "venue",
        "Human Annotation task type": "task_type",
        "Quality control & filtering": "post_filtering"}
    my_df = my_df.rename(columns=MORE_FIXES)
    print(my_df.columns.tolist())

    if my_df.isna().any().any():
        exit('Exiting: NaN values detected in the CSV after loading. Check the data for missing values.')

    my_df = my_df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)
    my_df["task_type"] = my_df["task_type"].apply(normalize_task_type)
    my_df["expertise_level"] = my_df["expertise_level"].replace(
        {"general language/knowledge task": "general task"})
    my_df["adjudication"] = my_df["adjudication"].replace(
        {"soft labels / distributional labels": "soft labels", "other / mixed": "other/mixed"})
    
    return my_df


def compare_columns(dfs: dict):
    cols = {name: set(df.columns) for name, df in dfs.items()}

    all_cols = set.union(*cols.values())

    print("\n=== Column comparison ===\n")

    for name, c in cols.items():
        missing = all_cols - c
        extra = c - (set.intersection(*cols.values()))

        print(f"[{name}]")
        print(f"  n_cols: {len(c)}")

        if missing:
            print(f"  missing: {sorted(missing)}")
        if extra:
            print(f"  extra: {sorted(extra)}")
        print()

    common = set.intersection(*cols.values())
    print(f"Common columns: {len(common)}")


def split_multi(val):
    if pd.isna(val) or str(val).strip() == "":
        return ["missing"]
    return [v.strip() for v in re.split(r"[;\|,]", str(val)) if v.strip()]


def binarize_string_fields(my_df, string_fields: list):
    # print(my_df["iaa_value"].unique())

    # separate BINARY treatment of string_digits fields if needed, e.g., distribution of iaa_value or annotators_per_item
    for field in string_fields:
        s = my_df[field].astype(str).str.strip().str.lower()

        if field == "iaa_value":
            mask_no = s.isin({"not reported", "na"})
        else:
            mask_no = s.eq("not reported")

        my_df[field] = mask_no.map({True: "not reported", False: "reported"})

        my_df = my_df.rename(columns={"iaa_value": "iaa_value_binarized"})

    return my_df


def build_long_df(my_dfs: dict,
        field_groups,
        multi_label_fields
):
    frames = []

    for name, df in my_dfs.items():
        tmp = df.copy()
        tmp["sample"] = name
        frames.append(tmp)

    df_all = pd.concat(frames, ignore_index=True)
    
    df_all = binarize_string_fields(df_all, TAX['string_digits'])

    # --- long format
    long = df_all.melt(
        id_vars=["sample"],
        var_name="field",
        value_name="value"
    )

    field_groups["Workload (binarized)"] = field_groups.pop("Workload")
    for k, fields in field_groups.items():
        field_groups[k] = [
            "iaa_value_binarized" if f == "iaa_value" else f
            for f in fields
        ]

    # --- map groups
    field_to_group = {
        f: g for g, fields in field_groups.items() for f in fields
    }

    long["group"] = long["field"].map(field_to_group)
    long = long.dropna(subset=["group"])

    # --- total number of items per (field, sample) BEFORE exploding
    base = (
        long.groupby(["group", "field", "sample"])
        .size()
        .reset_index(name="n_items")
    )

    # --- split multi-label vs single
    multi_mask = long["field"].isin(multi_label_fields)

    multi = long[multi_mask].copy()
    single = long[~multi_mask].copy()

    # --- explode multi-label
    # What fraction of items contain label X?
    multi["value"] = multi["value"].apply(split_multi)
    multi = multi.explode("value")

    # --- normalize single
    single["value"] = single["value"].fillna("missing")

    long = pd.concat([single, multi], ignore_index=True)

    # --- count label occurrences
    counts = (
        long.groupby(["group", "field", "sample", "value"])
        .size()
        .reset_index(name="count")
    )

    # --- merge with base item counts
    df = counts.merge(base, on=["group", "field", "sample"])

    # --- Option B normalization
    df["pct"] = df["count"] / df["n_items"]

    return df


def plot_group_facets(long_df, group_name, save_to=None):
    save_as = f"{save_to}{group_name.replace(' ', '_').lower()}.png" if save_to else None
    data = long_df[long_df["group"] == group_name]

    if data.empty:
        print(f"No data to plot for group '{group_name}'. Skipping.")
        return

    g = sns.FacetGrid(
        data,
        col="field",
        col_wrap=1,  # vertical stack
        sharex=True,
        sharey=False,
        height=3,
        aspect=2.2
    )

    def plot_with_order(data, **kwargs):
        ax = plt.gca()
        order = (
            data.groupby("value")["pct"]
            .mean()
            .sort_values(ascending=True)
            .index
            .tolist()
        )

        sns.barplot(
            data=data,
            y="value",
            x="pct",
            hue="sample",
            palette=SAMPLES_PALETTE,
            hue_order=["gold", "filter", "random"],
            order=order,
            errorbar=None,
            ax=ax
        )
        # flip so highest is visually on top
        # ax.invert_yaxis()
        top_value = (
            data.groupby("value")["pct"]
            .mean()
            .idxmax()
        )
        # find y position of that category (after inversion!)
        yticks = [t.get_text() for t in ax.get_yticklabels()]
        y_index = yticks.index(top_value)
        # annotate bars belonging to that category
        for container in ax.containers:  # one per hue (gold/filter/random)
            for bar in container:
                y = bar.get_y() + bar.get_height() / 2

                # match by y-position (robust)
                if abs(y - y_index) < 0.5:
                    width = bar.get_width()

                    ax.text(
                        width + 0.01,
                        y,
                        f"{width:.2f}",
                        va="center",
                        ha="left",
                        fontsize=8
                    )

        ax.set_xlim(0, 1)
        ax.set_xlabel("percentage")
        ax.set_ylabel("")

    g.map_dataframe(plot_with_order)

    g.add_legend()
    g._legend.set_bbox_to_anchor((0.01, 0.99))
    g._legend.set_loc("upper left")

    g.fig.suptitle(f"{group_name}", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # leave space for title
    if save_to:
        plt.savefig(save_as)
    plt.show()


def normalize_task_type(s: str) -> str:
    if pd.isna(s):
        return "missing"

    s = str(s).strip().lower()

    for k, v in PREFIXES.items():
        if s.startswith(k):
            return v

    return s


def get_gold(my_gold):
    my_gold = my_gold[my_gold["annotator"] == "gold"]
    my_gold = my_gold.drop(columns=["annotator"])

    # extract Year and Venue from PaperID
    def extract_year_conf(paper_id):
        if pd.isna(paper_id):
            return "missing", "missing"
        parts = str(paper_id).split(".")
        if len(parts) >= 3:
            year = parts[0]
            conf = parts[1]
            return year, conf
        return "missing", "missing"

    my_gold["year"], my_gold["venue"] = zip(*my_gold["PaperID"].apply(extract_year_conf))

    my_gold["venue"] = (
        my_gold["venue"]
        .str.replace(r'-(long|short|main)$', '', regex=True)
        .replace({
            'nips': 'neurips',
            'findings-emnlp': 'emnlp'
        })
    )

    return my_gold


def normalize_venue_to_filter(df, filter_df):
    # --- YEAR ---
    # min_year = filter_df["year"].min()
    # max_year = filter_df["year"].max()
    #
    # def map_year(y):
    #     if y < min_year:
    #         return f"<{min_year}"
    #     elif y >= max_year:
    #         return f"{max_year}+"
    #     return y
    #
    # df["year"] = df["year"].apply(map_year)

    # --- VENUE ---
    allowed_venues = set(filter_df["venue"].str.lower())

    df["venue"] = df["venue"].str.lower().apply(
        lambda v: v if v in allowed_venues else "other"
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', help="", default='data/original/double+gold.tsv')
    parser.add_argument('--filter', help="", default='data/original/clean_1603_llm-annotated.tsv')
    # parser.add_argument('--random', help="", default='data/original/random_sample_llm_annotated.tsv')
    # parser.add_argument('--random', help="", default='data/original/Updated Large Scale Exp Random - Sheet1.tsv')
    parser.add_argument('--random', help="", default='data/original/Updated Large Scale Exp Random.tsv')
    parser.add_argument("--sanity", action="store_true", help="run exploratory data analysis")
    # parser.add_argument('--res', default='res/samples/')
    parser.add_argument('--pics', default='res/pics/report_pct/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.pics], args=args)  # args.res,

    save_filter = "data/filter.tsv"
    save_random = "data/random.tsv"
    save_gold = "data/gold.tsv"
    if os.path.exists(save_filter) and os.path.exists(save_gold) and os.path.exists(save_random):
        print('Gold, Filter and Random exist. Loading.. ')
        gold_df = load_df(save_gold)
        filter_df = load_df(save_filter)
        random_df = load_df(save_random)
    else:
        filter_df = load_df(args.filter)
        filter_df["payment_rate"] = filter_df["payment_rate"].replace(
            {"not reported": "na", "no": "na"})
        filter_df.to_csv(save_filter, sep="\t", index=False)

        gold_df = load_df(args.gold)
        # Slice double+gold, drop Annotator, add Year and Conf to gold and save gold_df for later use
        gold_df = get_gold(gold_df)
        gold_df = normalize_venue_to_filter(gold_df, filter_df)
        gold_df.to_csv(save_gold, sep="\t", index=False)

        random_df = load_df(args.random)
        random_df["payment_rate"] = random_df["payment_rate"].replace(
            {"not reported": "na", "no": "na"}
        )

        print(random_df["year"].unique())
        print(random_df["venue"].unique())

        random_df = normalize_venue_to_filter(random_df, filter_df)

        # filter_df["year"] = filter_df["year"].replace({"2025": "2025+"})
        random_df.to_csv(save_random, sep="\t", index=False)

    dfs = {
        "gold": gold_df,
        "filter": filter_df,
        "random": random_df,
    }

    # general exploration and sanity checks
    if args.sanity:
        compare_columns(dfs)  # do we have the same columns?

        for name, df in dfs.items():
            if name != "random":
                continue
            print(f"\n == {name.upper()} == ")
            input("\nPress Enter to continue...")
            print(f"\tUnique papers: {df['PaperID'].nunique()}, total rows: {len(df)}")

            for col in ALL_DROP_DOWNS:
                input_sanity_check_column(df, col)

    for name, df in dfs.items():
        print(f"\n == {name.upper()} == ")
        print(", ".join(f"{y}: {n}" for y, n in sorted(Counter(df["venue"]).items())))
        print(", ".join(f"{y}: {n}" for y, n in sorted(Counter(df["year"]).items())))

    # FIELD_GROUPS = {
    #     g: [f for f in fields if f not in TAX['string_digits']]
    #     for g, fields in FIELD_GROUPS.items()
    # }

    # binarisation inside! Jetzt geht es für jedes Feld, das in string_digits ist.
    long_df = build_long_df(
        dfs,
        FIELD_GROUPS,
        multi_label_fields=TAX['multilabel']
    )

    assert (long_df["pct"] <= 1).all()

    for group in FIELD_GROUPS:
        plot_group_facets(long_df, group, save_to=args.pics)

    end = time.time()
    elapsed_time = (end - start) / 60
    if elapsed_time < 60:
        print(f"\nTotal time: {elapsed_time:.1f} min")
    else:
        print(f"\nTotal time: {int(elapsed_time // 60)} h {int(elapsed_time % 60)} min")
