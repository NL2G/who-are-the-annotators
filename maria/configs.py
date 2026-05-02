"""
28 April 2026
suggested uniform configuration for all scripts in this project. 
This file is imported by all scripts, so any changes here will affect all scripts that import it.

I also have my logging setup here
"""

# columns and values
# internal field names to presentable labels for display in tables, graphs, if necessary
# IMHO internal names look neater

import os
import sys
from datetime import datetime

_HEADER_MAP = {
    "Paper's topic": "paper_topic",
    "Annotation task type": "task_type",
    # "Subtask name": "subtask_name",  # free-generation field, not in agreement analysis, but needed for semantic matching
    "Intended use": "intended_use",
    "Annotation guidelines released": "guidelines_released",
    "IAA metric name": "iaa_metric",
    "IAA value": "iaa_value",
    "Number of annotators per item": "annotators_per_item",
    "Total number of annotators": "total_annotators",
    "Total number of annotated items": "total_items",
    "Number of items per annotator": "items_per_annotator",
    # "Human-evaluated sample": "human_evaluated_sample",  # largely duplicated existing fields, and often inconsistently reported,
    # so excluded from agreement analysis
    "Recruitment": "recruitment",
    "Crowd qualifications & screening": "crowd_screening",
    "Annotator training": "training",
    "Annotation task language proficiency": "language_proficiency",
    "Level of expertise": "expertise_level",
    "Reported compensation": "compensation",
    "Payment rate": "payment_rate",
    "Age": "age",
    "Gender": "gender",
    "Nation of origin": "nation_origin",
    "Nation of residence": "nation_residence",
    "Education": "education",
    "Political orientation": "political_orientation",
    "Post-filtering": "post_filtering",
    "Adjudication": "adjudication",
}

FIELDS = list(_HEADER_MAP.values())

FIELD_LABELS = {
    "paper_topic": "Paper's topic",  # Multiple selection possible
    "task_type": "Task type",
    # "subtask_name": "Task Name",
    "intended_use": "Intended use",  # Multiple selection possible
    "guidelines_released": "Guidelines released",
    "iaa_metric": "IAA metric",
    "iaa_value": "IAA value",
    "annotators_per_item": "Annotators/item",
    "total_annotators": "Total annotators",
    "total_items": "Total items",
    "items_per_annotator": "Items/annotator",
    # "human_evaluated_sample": "Human-evaluated Sample",
    "recruitment": "Recruitment",
    "crowd_screening": "Crowd screening",
    "training": "Training",
    "language_proficiency": "Language proficiency",
    "expertise_level": "Expertise level",
    "compensation": "Compensation",
    "payment_rate": "Payment rate",
    "age": "Age",
    "gender": "Gender",
    "nation_origin": "Nation origin",
    "nation_residence": "Nation residence",
    "education": "Education",
    "political_orientation": "Political orient.",
    "post_filtering": "Post-filtering",
    "adjudication": "Adjudication",
}

TAX = {
    # DROP_DOWNS (3-10 values), total: 11
    "multiclass": ["paper_topic", "task_type", "intended_use", "recruitment", "crowd_screening",
                   "training", "language_proficiency", "expertise_level", "compensation",
                   "payment_rate", "adjudication"],
    # a subset of multiclass, which required special handlish: comma, semicolon, or pipe separated fields, total: 3
    "multilabel": ["paper_topic", 'intended_use', "iaa_metric"],
    # _YES_NO_FIELDS, total: 9
    "binary": ["guidelines_released",
               "training",
               "age",
               "gender",
               "nation_origin",
               "nation_residence",
               "education",
               "political_orientation",
               "post_filtering"],
    # total: 5
    'string_digits': ["iaa_value", "annotators_per_item", "total_annotators",
                      "total_items", "items_per_annotator"]
}

ALL_DROP_DOWNS = list(set(FIELDS) - {"items_per_annotator", "total_items", "total_annotators",
                                     "annotators_per_item", "iaa_value"})
FIELD_GROUPS = {
    "Year_and_Venue": ["year", "venue"],
    "General Description": [
        "paper_topic", "task_type",  # "subtask_name",
        "intended_use", "guidelines_released",
    ],
    "Agreement Level": [
        "iaa_metric", "iaa_value",
    ],
    "Workload": [
        "annotators_per_item", "total_annotators",
        "total_items", "items_per_annotator",
        #"human_evaluated_sample",
    ],
    "Recruitment and Qualifications": [
        "recruitment", "crowd_screening",
        "training", "language_proficiency",
        "expertise_level",
    ],
    "Compensation": [
        "compensation", "payment_rate",
    ],
    "Demographics": [
        "age", "gender",
        "nation_origin", "nation_residence",
        "education", "political_orientation",
    ],
    "Quality Control": ["post_filtering", "adjudication"],
}

# unrealistically long nominal values
PREFIXES = {
    "closed-label classification": "closed-label clf",
    "ordinal / scalar rating": "ordinal/scalar rating",
    "span-level or structural annotation": "span-level/structural",
    "comparative / preference judgments": "comparative/preference",
    "open-ended generation or transformation": "open-ended generation",
    # in "expertise_level"
    "general language/knowledge task": "general task",
    "soft labels / distributional labels": "soft labels",
    "other / mixed": "other/mixed"
}

SAMPLES_PALETTE = {
    "gold": "#E6B800",
    "filter": "#1F8A3B",
    "random": "#2A6FDC"
}

COLLAPSABLE_FIELDS = []


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