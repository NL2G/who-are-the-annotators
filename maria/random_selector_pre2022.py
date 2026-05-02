"""
extract 10 random papers from all conf folders before 2022, and rename them YEAR.CONF.25characters_of_title-tewodros-maria.pdf
USAGE:
python3 random_selector_pre2022.py --root_dir papers/crawled_and_filtered_papers
"""

import os
import random
import re
import shutil
import argparse
import sys
from datetime import datetime


def clean_title(filename, title_char=25):
    name = os.path.splitext(filename)[0]

    # split "57.Some Title" → ("57.", "Some Title")
    m = re.match(r'^(\d+\.)\s*(.*)', name)
    if m:
        prefix, rest = m.groups()
    else:
        prefix, rest = "", name

    # clean title part
    rest = re.sub(r'[^a-zA-Z0-9 ]+', '', rest)
    rest = re.sub(r'\s+', '_', rest.strip())

    # truncate to 25 chars and remove trailing underscore
    short = rest[:title_char].rstrip('_')

    return f"{prefix}{short}"


def make_dirs(logs=None, make_them=None):
    for i in make_them:
        if i:
            os.makedirs(i, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    formatted_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logs}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime}")
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
    parser.add_argument('--out', default='papers/PDFs_Human_Annotation/new_arrivals2/')
    parser.add_argument('--new_raters', default='three_anno')
    parser.add_argument('--title_char', type=int, default=15)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()

    make_dirs(logs=args.logs, make_them=[args.out])

    root_dir = args.root_dir

    all_pdfs = []

    # 1. Collect all PDFs with metadata
    for conf in os.listdir(root_dir):
        conf_path = os.path.join(root_dir, conf)
        if not os.path.isdir(conf_path):
            continue
        valid_years = {"2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"}
        # valid_years = {"2018", "2019", "2020", "2021"}
        year_prefix_pattern = re.compile(r'^(2018|2019|2020|2021|2022|2023|2024|2025)\.')

        for year in os.listdir(conf_path):
            if year not in valid_years:
                continue
            year_path = os.path.join(conf_path, year)
            if not os.path.isdir(year_path):
                continue

            for proc_type in os.listdir(year_path):
                proc_path = os.path.join(year_path, proc_type)
                if not os.path.isdir(proc_path):
                    continue

                for f in os.listdir(proc_path):
                    if not f.lower().endswith(".pdf"):
                        continue

                    # skip already processed files
                    if year_prefix_pattern.match(f):
                        continue

                    all_pdfs.append({
                        "path": os.path.join(proc_path, f),
                        "conf": conf,
                        "year": year,
                        "filename": f
                    })

    rng = random.Random(args.seed)
    sample = rng.sample(all_pdfs, args.k)

    for item in sample:
        short_title = clean_title(item["filename"], title_char=args.title_char)
        new_name = f"{item['year']}.{item['conf']}.{short_title}-{args.new_raters}.pdf"

        src = item["path"]
        dir_path = os.path.dirname(src)
        renamed_path = os.path.join(dir_path, new_name)

        os.rename(src, renamed_path)

        dst = os.path.join(args.out, new_name)
        shutil.copy(renamed_path, dst)

        print(f"Renamed: {src} -> {renamed_path}")
        print(f"Copied to: {dst}")

    print('The End')

