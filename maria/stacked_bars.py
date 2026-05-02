"""
USAGE:
python3 stacked_bars.py --root_dir papers/crawled_and_filtered_papers
"""


import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
import sys
from datetime import datetime

ANNOTATED_NOT_IN_CRAWL = ['2024.acl-long.754-christian.pdf',
                          '2023.acl-long.795-sebastian.pdf', '2023.acl-short.140-sebastian.pdf',
                          '2025.emnlp-main.1275-lisa-chris-yufang.pdf', '2024.naacl-long.362-subhadeep-steffen-chris.pdf',
                          '2025.emnlp-main.1635-arslan-chris.pdf', '2025.emnlp-main.1678-lisa-subhadeep-yufang-mihn.pdf',
                          '2021.acl-long.247-subhadeep-lisa.pdf',
                          '2023.acl-long.711-subhadeep-yanran-lisa-maria.pdf', '2024.naacl-long.245-chris-lotta.pdf',
                          '2025.emnlp-main.1264-subhadeep-arslan.pdf',
                          '2025.emnlp-main.908-subhadeep-chris-yufang.pdf']

# '2026.iclr-main.TikZilla-christian-steffen.pdf', '2026.tacl-main.1-chris-subhadeep.pdf',
# '2024.iclr-main.AutomaTikZ-christian.pdf', '2025.iccv-main.TikZero-christian.pdf', '2024.nips-main.DeTikZify-christian.pdf',


def make_dirs(logs=None, make_them=None):
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
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def parse_extra_filename(f):
    name = f.replace(".pdf", "")
    parts = name.split(".")

    if len(parts) < 2:
        return None

    year = parts[0]
    conf_proc = parts[1]   # e.g. acl-long, nips-main

    if "-" in conf_proc:
        conf, proc_type = conf_proc.split("-", 1)
    else:
        conf, proc_type = conf_proc, "unknown"

    return year, conf, proc_type


def collect_counts(root_dir, extra=None):
    """
    Collect PDFs with structure:
    root_dir/conf/year/proc_type/*.pdf
    Returns counts[year][conf][proc_type] and a flat dataframe
    """
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    rows = []

    for conf in os.listdir(root_dir):
        conf_path = os.path.join(root_dir, conf)
        if not os.path.isdir(conf_path):
            continue

        for year in os.listdir(conf_path):
            year_path = os.path.join(conf_path, year)
            if not os.path.isdir(year_path):
                continue

            for proc_type in os.listdir(year_path):
                proc_path = os.path.join(year_path, proc_type)
                if not os.path.isdir(proc_path):
                    continue

                pdf_files = [f for f in os.listdir(proc_path) if f.lower().endswith(".pdf")]
                n_pdf = len(pdf_files)
                if n_pdf > 0:
                    counts[year][conf][proc_type] += n_pdf
                    rows.append({
                        "conf": conf,
                        "year": year,
                        "proc_type": proc_type,
                        "pdf_count": n_pdf,
                        "path": proc_path
                    })
                    print(f"Found {n_pdf} PDFs in {conf}/{year}/{proc_type}")
    for f in extra:
        year, conf, proc_type = parse_extra_filename(f)
        if year is None:
            continue

        counts[year][conf][proc_type] += 1

    df = pd.DataFrame(rows)

    rows_extra = []
    out_of_scope = []
    for f in extra:
        if any(x in f for x in ['2026', 'iclr', 'iccv', 'nips']):
            out_of_scope.append(f)
            continue

        year, conf, proc_type = parse_extra_filename(f)
        if year is None:
            continue

        # update nested counts
        counts[year][conf][proc_type] += 1

        # add to DataFrame rows
        # proc_path = os.path.join(root_dir, conf, year, proc_type)
        rows_extra.append({
            "conf": conf,
            "year": year,
            "proc_type": proc_type,
            "pdf_count": 1,
            "path": pd.NA
        })

    # append new rows to your existing DataFrame
    df_extra = pd.DataFrame(rows_extra)
    df = pd.concat([df, df_extra], ignore_index=True)
    df = (
        df.groupby(["year", "conf", "proc_type"], as_index=False)
        .agg({"pdf_count": "sum", "path": "first"})  # keep path from first occurrence
    )
    return counts, df


def plot_stacked_confs(counts, save_as=None):
    """
    X-axis: conferences
    Stacks: years
    counts: nested dict counts[year][conf][proc_type] -> pdf count
    """
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={"grid.linestyle": "--", "grid.linewidth": 0.7, "grid.alpha": 0.8}
    )

    years = sorted(counts.keys())
    confs = sorted({c for y in counts for c in counts[y]})

    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = [0] * len(confs)
    for year in years:
        values = [sum(counts[year][conf].values()) for conf in confs]
        if any(values):
            ax.bar(confs, values, bottom=bottom, label=year)
            bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("")
    ax.set_ylabel("Number of PDFs")
    # ax.set_title("Papers per Conference")
    if any(bottom):
        ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.xticks(rotation=0)

    sns.despine(ax=ax, top=True, bottom=False, right=True, left=False)

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)

    plt.show()
    plt.close(fig)


def plot_stacked_years(counts, save_as=None):
    """
    X-axis: years
    Stacks: conferences
    counts: nested dict counts[year][conf][proc_type] -> pdf count
    """
    # Set theme once
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={"grid.linestyle": "--", "grid.linewidth": 0.7, "grid.alpha": 0.8}
    )

    years = sorted(counts.keys())
    confs = sorted({c for y in counts for c in counts[y]})

    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = [0] * len(years)
    plotted_any = False  # Track if at least one bar is plotted

    for conf in confs:
        values = [sum(counts[y].get(conf, {}).values()) for y in years]
        if any(values):  # Only plot non-zero series
            ax.bar(years, values, bottom=bottom, label=conf)
            bottom = [b + v for b, v in zip(bottom, values)]
            plotted_any = True

    # Labels and title
    ax.set_xlabel("")
    ax.set_ylabel("Number of PDFs")
    # ax.set_title("Papers per Year")

    # Legend only if something was plotted
    if plotted_any:
        ax.legend()

    # Y-grid
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    sns.despine(ax=ax, top=True, bottom=False, right=True, left=False)

    # X-ticks rotation
    plt.xticks(rotation=0)

    # Layout & save
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot stacked bar chart of PDFs in nested conference/year folders"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory (e.g., papers/crawled_and_filtered_papers)"
    )
    parser.add_argument('--pics', default='res/pics/')
    parser.add_argument('--stats', default='res/stats/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()

    make_dirs(logs=args.logs, make_them=[args.stats, args.pics])

    counts, my_table = collect_counts(args.root_dir, extra=ANNOTATED_NOT_IN_CRAWL)

    # Print the DataFrame
    if my_table.empty:
        print("No PDFs found.")
    else:
        print("\nNumber of PDFs by conference, year, and proc_type:\n")
        print(my_table.tail())
        missing_paths = my_table[pd.isna(my_table["path"])]

        print(missing_paths)
        my_table.to_csv(f"{args.stats}pdfs_conf_year.tsv", sep='\t')

    # Print total papers
    total_papers = my_table['pdf_count'].sum()
    print(f"\nTotal number of papers: {total_papers}\n")

    # Make a sorted list of all conferences and years
    all_confs = sorted({c for y in counts for c in counts[y]})
    all_years = sorted(counts.keys())

    for bar_type in ["stacked_years", "stacked_confs"]:
        pic_name = f'{args.pics}{bar_type}.png'
        if bar_type == "stacked_confs":
            plot_stacked_confs(counts, save_as=pic_name)
        elif bar_type == "stacked_years":
            plot_stacked_years(counts, save_as=pic_name)

    print('The End')

