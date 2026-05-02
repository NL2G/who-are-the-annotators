"""
are there duplicates? -- papers with the same number and title, due to double quotes and apostrophe.
and are manually annotated papers in the crawled and filtered sample?

USAGE:
python3 clean_collection.py --root_dir papers/crawled_and_filtered_papers
"""

import os
import random
import re
import pdfplumber
import argparse
import sys
from collections import defaultdict
from datetime import datetime
from rapidfuzz import process, fuzz

from pdfminer.high_level import extract_text


BAD_PATTERNS = [
    r"proceedings",
    r"association for computational linguistics",
    r"acl",
    r"emnlp",
    r"naacl",
    r"eacl",
    r"pages?\s+\d",
    r"\d{4}",
]

def is_bad_line(line):
    l = line.lower()
    return any(re.search(p, l) for p in BAD_PATTERNS)


def score_line(line):
    """
    Heuristic scoring for title-likeness
    """
    score = 0

    # length: titles are medium length
    if 20 < len(line) < 200:
        score += 2

    # penalize boilerplate
    if is_bad_line(line):
        score -= 5

    # titles often have capitalized words
    caps = sum(1 for w in line.split() if w[:1].isupper())
    score += caps * 0.3

    # avoid all-caps lines
    if line.isupper():
        score -= 2

    return score


def extract_title(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]

        words = page.extract_words(extra_attrs=["size"])

        # group by line (y coordinate)
        lines = {}
        for w in words:
            key = round(w["top"], 1)
            lines.setdefault(key, []).append(w)

        candidates = []
        for line_words in lines.values():
            text = " ".join(w["text"] for w in line_words)
            avg_size = sum(w["size"] for w in line_words) / len(line_words)
            candidates.append((text, avg_size))

        # pick largest font line
        best = max(candidates, key=lambda x: x[1])

        return best[0]


def normalize(text):
    if not text:
        return ""

    text = text.lower()

    # remove quotes, punctuation
    text = re.sub(r"[\"'“”]", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # collapse whitespace
    text = " ".join(text.split())

    return text


def get_main_titles(proc_path):
    titles = []

    for f in os.listdir(proc_path):
        if not f.lower().endswith(".pdf"):
            continue

        name = os.path.splitext(f)[0]

        # remove leading number
        parts = name.split(".", 1)
        if len(parts) == 2:
            _, title = parts
        else:
            title = name

        titles.append((f, normalize(title)))

    return titles


def find_best_match(query, candidates):
    # candidates = list of (filename, normalized_title)
    choices = {f: t for f, t in candidates}

    match, score, _ = process.extractOne(
        query,
        choices,
        scorer=fuzz.token_set_ratio
    )

    return match, score


def parse_secondary_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split(".")
    if len(parts) < 3:
        return None, None
    return parts[0], parts[1]


def split_filename(filename):
    name = os.path.splitext(filename)[0]

    # match: number + dot + rest
    m = re.match(r'^(\d+)\.(.+)$', name)
    if not m:
        return None, None

    num, title = m.groups()
    return num, title


def normalize_title(title):
    return (title
            .replace('"', '')
            .replace("'", "")
            .replace("’", "")
            .replace("“", "")
            .replace("”", "")
            .strip()
            .lower())


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
    parser.add_argument('--check_subf', default=['papers/PDFs_Human_Annotation/double', 'papers/PDFs_Human_Annotation'])
    parser.add_argument('--fuzzy_thres', type=int, default=85)
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()

    make_dirs(logs=args.logs, make_them=[])

    root_dir = args.root_dir

    subf_with_dups = set()

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

                groups = defaultdict(list)

                for f in pdf_files:
                    num, title = split_filename(f)
                    if num is None:
                        continue

                    norm = (num, normalize_title(title))
                    groups[norm].append(f)

                # report duplicates within same number
                for (num, norm_title), files in groups.items():
                    if len(files) > 1:
                        has_quotes = any(('"' in f or "'" in f or "’" in f or "“" in f or "”" in f) for f in files)

                        if has_quotes:
                            print(f"\n[Duplicate #{num} in {proc_path}]")
                            print(f"Normalized title: {norm_title}")
                            for f in files:
                                print(f"  - {f}")
                            subf_with_dups.add(proc_path)

    if not subf_with_dups:
        print(f"\nNO quoted duplicates found in any subfolder.\n")
    exit()
    THRESHOLD = args.fuzzy_thres

    for subf in args.check_subf:
        not_matched = []
        matched = []
        for f in os.listdir(subf):
            if not f.lower().endswith(".pdf"):
                continue

            year, conf = parse_secondary_filename(f)
            if not year:
                continue
            if conf.startswith("findings"):
                conf_name = conf.split("-")[1]
            else:
                conf_name = conf.split("-")[0]

            proc_path = os.path.join(root_dir, conf_name, year, f"{year}{conf}")
            if not os.path.isdir(proc_path):
                print(f"[Missing folder] {proc_path}")
                not_matched.append(f)
                continue

            main_titles = get_main_titles(proc_path)

            pdf_path = os.path.join(subf, f)
            extracted_title = extract_title(pdf_path)

            if not extracted_title:
                print(f"[No title] {f}")
                continue

            norm_query = normalize(extracted_title)

            match, score = find_best_match(norm_query, main_titles)

            if score < THRESHOLD:
                print(f"\n[LOW MATCH] {f}")
                print(f"  Extracted: {extracted_title}")
                print(f"  Best: {match}")
                print(f"  Score: {score}")
                not_matched.append(f)
            else:
                print(f"[OK {score}] {f} -> Matched title: {match}")
                matched.append(f)

        print(f'\nSubfolders with annotated files: {subf}')
        print(f'\nAnnotated papers not found in the crawled collection: {len(not_matched)}')
        for f in not_matched:
            print(f"  - {f}")

        print(f'\nAnnotated papers in the crawled collection: {len(matched)}')
        for f in matched:
            print(f"  - {f}")

    print('\nThe End')

