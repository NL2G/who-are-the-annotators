"""
Field value validation module for human annotation extraction.
Provides deterministic validation and normalization of field values
to ensure they match the allowed vocabulary.
"""

from typing import Any, Optional, List, Union
import re


# =============================================================================
# ALLOWED VALUES FOR EACH FIELD
# =============================================================================

ALLOWED_NLP_TASK_TYPES = [
    "Closed-label classification",
    "Ordinal / scalar rating",
    "Span-level or structural annotation",
    "Comparative / preference judgments",
    "Open-ended generation or transformation",
    "Other",
]

ALLOWED_PAPERS_TOPIC = [
    "Linguistics",
    "Discourse & Pragmatics",
    "Information Extraction",
    "QA & information access",
    "Semantics, inference & similarity",
    "Subjective language & social meaning",
    "Open-ended generation",
    "Compression & reformulation tasks",
    "Multilingual & translation",
    "Multimodal tasks",
]

ALLOWED_INTENDED_USE = [
    "Resource creation",
    "Human performance",
    "Model output evaluation",
    "None of the above",
]

ALLOWED_YES_NO = ["Yes", "No"]

ALLOWED_YES_NO_NA = ["Yes", "No", "NA"]

ALLOWED_IAA_METRICS = [
    "Fleiss' κ",
    "Cohen's κ",
    "Krippendorff's α",
    "F1 agreement",
    "Pearson",
    "Spearman",
    "Majority agreement",
    "Kendall's tau",
    "Total agreement",
    "Other",
    "Not reported",
]

ALLOWED_RECRUITMENT = [
    "Crowdsourcing",
    "Authors",
    "Other",
    "Mixed",
    "Not reported",
]

ALLOWED_LANGUAGE_PROFICIENCY = [
    "Native",
    "Non-Native",
    "Mixed",
    "Medium",
    "Not reported",
]

ALLOWED_EXPERTISE_LEVEL = [
    "High",
    "Medium",
    "Mixed",
    "Not reported",
    "General Language/Knowledge task",
]

ALLOWED_PAYMENT_CURRENCY = [
    "USD",
    "EUR",
    "GBP",
    "INR",
    "CNY",
    "CAD",
    "Other",
    "NA",
    "Not reported",
]

ALLOWED_REPORTED_COMPENSATION = [
    "Paid",
    "Free",
    "NA",
    "Not reported",
]

ALLOWED_PAYMENT_RATE = [
    "Specific numeric rate",
    "General mention",
    "No",
    "NA",
]

ALLOWED_ADJUDICATION = [
    "Majority vote",
    "Expert adjudication",
    "Third annotator",
    "Consensus discussion",
    "Soft labels / Distributional labels",
    "Weighted voting",
    "Other / mixed",
    "None",
]


# =============================================================================
# FIELD CONFIGURATION
# =============================================================================

FIELD_CONFIG = {
    "papers_topic": {
        "allowed": ALLOWED_PAPERS_TOPIC,
        "fallback": "Linguistics",
        "type": "list",
    },
    "nlp_task_type": {
        "allowed": ALLOWED_NLP_TASK_TYPES,
        "fallback": "Other",
        "type": "single",
    },
    "intended_use": {
        "allowed": ALLOWED_INTENDED_USE,
        "fallback": "None of the above",
        "type": "list",
    },
    "guidelines_released": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "iaa_metric_name": {
        "allowed": ALLOWED_IAA_METRICS,
        "fallback": "Not reported",
        "type": "list",
    },
    "iaa_value": {
        "allowed": None,  # Numeric or "NA" or "Not reported"
        "fallback": "Not reported",
        "type": "list",
    },
    "annotators_per_item": {
        "allowed": None,  # Integer string or "Not reported"
        "fallback": "Not reported",
        "type": "numeric_or_not_reported",
    },
    "total_annotators": {
        "allowed": None,
        "fallback": "Not reported",
        "type": "numeric_or_not_reported",
    },
    "total_annotated_items": {
        "allowed": None,
        "fallback": "Not reported",
        "type": "numeric_or_not_reported",
    },
    "items_per_annotator": {
        "allowed": None,
        "fallback": "Not reported",
        "type": "numeric_or_not_reported",
    },
    "human_evaluated_sample": {
        "allowed": None,
        "fallback": "Not reported",
        "type": "free_text",
    },
    "recruitment": {
        "allowed": ALLOWED_RECRUITMENT,
        "fallback": "Not reported",
        "type": "single",
    },
    "crowd_qualifications_screening": {
        "allowed": ALLOWED_YES_NO_NA,
        "fallback": "No",
        "type": "single",
    },
    "annotator_training": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "language_proficiency": {
        "allowed": ALLOWED_LANGUAGE_PROFICIENCY,
        "fallback": "Not reported",
        "type": "single",
    },
    "expertise_level": {
        "allowed": ALLOWED_EXPERTISE_LEVEL,
        "fallback": "Not reported",
        "type": "single",
    },
    "reported_compensation": {
        "allowed": ALLOWED_REPORTED_COMPENSATION,
        "fallback": "Not reported",
        "type": "single",
    },
    "payment_rate": {
        "allowed": ALLOWED_PAYMENT_RATE,
        "fallback": "Not reported",
        "type": "single",
    },
    "age_reported": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "gender_reported": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "nation_origin_reported": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "nation_residence_reported": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "education_reported": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "political_orientation_reported": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "quality_control_filtering": {
        "allowed": ALLOWED_YES_NO,
        "fallback": "No",
        "type": "single",
    },
    "disagreement_resolution_adjudication": {
        "allowed": ALLOWED_ADJUDICATION,
        "fallback": "None",
        "type": "single",
    },
}

# All field names in order
FIELDS = list(FIELD_CONFIG.keys())


# =============================================================================
# NORMALIZATION MAPPINGS
# =============================================================================

# Common variants that should be normalized to allowed values
NORMALIZATION_MAP = {
    # NLP task type variants
    "closed-label classification": "Closed-label classification",
    "classification": "Closed-label classification",
    "binary classification": "Closed-label classification",
    "multi-class classification": "Closed-label classification",
    "ordinal / scalar rating": "Ordinal / scalar rating",
    "ordinal/scalar rating": "Ordinal / scalar rating",
    "rating": "Ordinal / scalar rating",
    "likert": "Ordinal / scalar rating",
    "scale": "Ordinal / scalar rating",
    "span-level or structural annotation": "Span-level or structural annotation",
    "span annotation": "Span-level or structural annotation",
    "ner": "Span-level or structural annotation",
    "sequence labeling": "Span-level or structural annotation",
    "comparative / preference judgments": "Comparative / preference judgments",
    "comparative/preference judgments": "Comparative / preference judgments",
    "preference": "Comparative / preference judgments",
    "ranking": "Comparative / preference judgments",
    "pairwise": "Comparative / preference judgments",
    "open-ended generation or transformation": "Open-ended generation or transformation",
    "generation": "Open-ended generation or transformation",
    "free-text": "Open-ended generation or transformation",
    # Recruitment variants
    "mturk": "Crowdsourcing",
    "amazon mechanical turk": "Crowdsourcing",
    "mechanical turk": "Crowdsourcing",
    "amt": "Crowdsourcing",
    "amazon turks": "Crowdsourcing",
    "prolific": "Crowdsourcing",
    "profilic": "Crowdsourcing",
    "toloka": "Crowdsourcing",
    "yahoo! crowdsourcing": "Crowdsourcing",
    "crowdsourcing": "Crowdsourcing",
    "crowd": "Crowdsourcing",
    "other crowdsourcing platform": "Crowdsourcing",
    "crowdsource": "Crowdsourcing",
    "internal": "Other",
    "in-house": "Other",
    "lab": "Other",
    "recruiting not via crowdsourcing platform": "Other",
    "volunteer": "Other",
    "hired": "Other",
    "authors": "Authors",
    "co-authors": "Authors",
    "any authors as annotators": "Authors",
    # IAA metric variants
    "fleiss' kappa": "Fleiss' κ",
    "fleiss kappa": "Fleiss' κ",
    "fleiss": "Fleiss' κ",
    "cohen's kappa": "Cohen's κ",
    "cohen kappa": "Cohen's κ",
    "cohen": "Cohen's κ",
    "krippendorff's alpha": "Krippendorff's α",
    "krippendorff alpha": "Krippendorff's α",
    "krippendorff": "Krippendorff's α",
    "f1": "F1 agreement",
    "f1-score": "F1 agreement",
    "pearson correlation": "Pearson",
    "pearson's r": "Pearson",
    "spearman correlation": "Spearman",
    "spearman's rho": "Spearman",
    "spearman rho": "Spearman",
    "spearman's ρ": "Spearman",
    "kendall tau": "Kendall's tau",
    "kendall's tau-b": "Kendall's tau",
    "kendall": "Kendall's tau",
    "majority": "Majority agreement",
    "majority vote": "Majority agreement",
    "percent agreement": "Total agreement",
    "percentage agreement": "Total agreement",
    "exact agreement": "Total agreement",
    "agreement": "Total agreement",
    "none": "Not reported",
    # Expertise level variants
    "expert": "High",
    "high": "High",
    "professional": "High",
    "high (e.g. expert)": "High",
    "student": "Medium",
    "medium": "Medium",
    "graduate student": "Medium",
    "medium (e.g. student in the relevant field)": "Medium",
    "mixed": "Mixed",
    "low (e.g. crowd)": "Not reported",
    "low": "Not reported",
    "crowdworker": "Not reported",
    "non-expert": "Not reported",
    "general": "General Language/Knowledge task",
    "general language": "General Language/Knowledge task",
    "general knowledge": "General Language/Knowledge task",
    "general language/knowledge task": "General Language/Knowledge task",
    "general task": "General Language/Knowledge task",
    # Language proficiency
    "native speaker": "Native",
    "native speakers": "Native",
    "l1": "Native",
    "non-native speaker": "Non-Native",
    "non-native": "Non-Native",
    "l2": "Non-Native",
    "medium": "Medium",
    "mixed": "Mixed",
    "both": "Mixed",
    # Reported compensation variants
    "paid": "Paid",
    "free": "Free",
    "voluntary": "Free",
    "volunteer": "Free",
    "unpaid": "Free",
    # Payment rate variants
    "specific numeric rate": "Specific numeric rate",
    "specific rate": "Specific numeric rate",
    "specific": "Specific numeric rate",
    "general mention": "General mention",
    "general": "General mention",
    # Adjudication variants
    "majority vote": "Majority vote",
    "majority voting": "Majority vote",
    "expert adjudication": "Expert adjudication",
    "expert": "Expert adjudication",
    "third annotator": "Third annotator",
    "consensus discussion": "Consensus discussion",
    "consensus": "Consensus discussion",
    "soft labels": "Soft labels / Distributional labels",
    "soft labels / distributional labels": "Soft labels / Distributional labels",
    "distributional labels": "Soft labels / Distributional labels",
    "weighted voting": "Weighted voting",
    "other / mixed": "Other / mixed",
    "none": "None",
    # Yes/No variants
    "yes": "Yes",
    "true": "Yes",
    "1": "Yes",
    "no": "No",
    "false": "No",
    "0": "No",
    "n/a": "NA",
    "na": "NA",
    "not applicable": "NA",
    "not reported": "Not reported",
    "nr": "Not reported",
    "unknown": "Not reported",
    "-": "Not reported",
    "": "Not reported",
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def is_numeric_string(s: str) -> bool:
    """Check if string represents a valid integer."""
    if not s:
        return False
    s = s.strip()
    # Handle negative numbers
    if s.startswith("-"):
        s = s[1:]
    return s.isdigit()


def is_float_string(s: str) -> bool:
    """Check if string represents a valid float."""
    if not s:
        return False
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


def normalize_value(value: Any) -> str:
    """Convert any value to a normalized string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if len(value) == 1 else ""
    return str(value).strip()


def validate_single_field(field: str, value: Any) -> str:
    """
    Validate and normalize a single-value field.

    Args:
        field: Field name
        value: Raw value

    Returns:
        Validated and normalized value
    """
    config = FIELD_CONFIG.get(field, {})
    allowed = config.get("allowed", [])
    fallback = config.get("fallback", "Not reported")

    # Normalize input
    str_value = normalize_value(value)

    # Handle empty/null values
    if not str_value or str_value.lower() in ["", "nan", "none", "null"]:
        return fallback

    # Check exact match (case-sensitive)
    if allowed and str_value in allowed:
        return str_value

    # Check case-insensitive match
    str_lower = str_value.lower()
    if allowed:
        for a in allowed:
            if str_lower == a.lower():
                return a

    # Check normalization map
    if str_lower in NORMALIZATION_MAP:
        normalized = NORMALIZATION_MAP[str_lower]
        if not allowed or normalized in allowed:
            return normalized

    # Check partial matches for longer strings
    if allowed:
        for a in allowed:
            if str_lower in a.lower() or a.lower() in str_lower:
                return a

    # Return fallback if no match found
    return fallback


def validate_numeric_field(field: str, value: Any) -> str:
    """
    Validate a numeric field (integer string or 'Not reported').

    Args:
        field: Field name
        value: Raw value

    Returns:
        Integer string or 'Not reported'
    """
    str_value = normalize_value(value)

    if not str_value or str_value.lower() in ["", "nan", "none", "null", "not reported", "nr", "-"]:
        return "Not reported"

    # Try to extract a clean integer
    str_value = str_value.strip()

    # Handle 'K' suffix (e.g., "27.9K" -> ambiguous, use Not reported)
    if "k" in str_value.lower() or "K" in str_value:
        # Try to convert if it's an exact K value
        match = re.match(r"^(\d+(?:\.\d+)?)\s*[kK]$", str_value)
        if match:
            try:
                num = float(match.group(1)) * 1000
                if num == int(num):
                    return str(int(num))
            except ValueError:
                pass
        return "Not reported"

    # Handle ranges like "2-3" or "2 to 3"
    if "-" in str_value or " to " in str_value.lower():
        return "Not reported"

    # Handle comma-separated numbers (e.g., "1,000")
    clean_value = str_value.replace(",", "")

    # Check if it's a valid integer
    if is_numeric_string(clean_value):
        return clean_value

    # Check if it's a float that can be converted to int
    if is_float_string(clean_value):
        try:
            f = float(clean_value)
            if f == int(f):
                return str(int(f))
        except ValueError:
            pass

    return "Not reported"


def validate_iaa_value(value: Any) -> str:
    """
    Validate an IAA value (numeric string, "NA", or "Not reported").

    Args:
        value: Raw value

    Returns:
        Validated IAA value
    """
    str_value = normalize_value(value)

    if not str_value or str_value.lower() in ["", "nan", "none", "null"]:
        return "Not reported"

    # Strip surrounding quotation marks (e.g. '"No Majority"' -> 'No Majority')
    if len(str_value) >= 2 and str_value[0] == '"' and str_value[-1] == '"':
        str_value = str_value[1:-1].strip()

    if str_value.upper() == "NA":
        return "NA"

    if str_value.lower() == "not reported":
        return "Not reported"

    # Normalize European decimal comma: "0,874" -> "0.874"
    # Only when the value matches a simple decimal-number pattern (e.g. "0,874").
    # Thousands-separator commas (e.g. "1,000") are intentionally left for the
    # caller to handle, but in practice IAA values are in [0, 1].
    if re.match(r"^\d+,\d+$", str_value):
        str_value = str_value.replace(",", ".")

    # Keep percentage values as-is (e.g. "88%", "70%")
    if str_value.endswith("%"):
        return str_value

    # Check if it's a valid float
    if is_float_string(str_value):
        try:
            f = float(str_value)
            # Round to 2 decimal places
            return f"{f:.2f}".rstrip("0").rstrip(".")
        except ValueError:
            pass

    # Handle ranges with dash/en-dash or "to" keyword (e.g. "0.09 to 0.23")
    if "-" in str_value or "–" in str_value or " to " in str_value.lower():
        return str_value

    # Keep any other non-empty descriptive string as-is
    if str_value:
        return str_value

    return "NA"


def _split_csv_list(value: str) -> List[str]:
    """Split a comma-separated value respecting CSV-style quoting.

    Handles cases like ``"Semantics, inference & similarity"`` where a
    single value contains a comma inside quotes.
    """
    import csv
    import io
    reader = csv.reader(io.StringIO(value))
    parts = next(reader)
    return [p.strip() for p in parts if p.strip()]


def validate_list_field(field: str, value: Any) -> List[str]:
    """
    Validate a list field (iaa_metric_name, iaa_value, papers_topic, intended_use).

    Args:
        field: Field name
        value: Raw value (can be list, string with separators, or single value)

    Returns:
        List of validated values
    """
    config = FIELD_CONFIG.get(field, {})
    fallback = config.get("fallback", "Not reported")

    # Convert to list
    if value is None:
        return [fallback]

    if isinstance(value, str):
        # Split on common separators
        if "|" in value:
            items = [v.strip() for v in value.split("|")]
        elif "," in value and field in ("iaa_metric_name", "papers_topic", "intended_use"):
            items = _split_csv_list(value)
        else:
            items = [value.strip()]
    elif isinstance(value, list):
        items = value
    else:
        items = [str(value)]

    # Strip stray quotes from individual items
    items = [v.strip().strip('"') for v in items if v is not None]
    items = [v for v in items if v]

    # Validate each item
    if field == "iaa_metric_name":
        validated = []
        for item in items:
            v = validate_single_field(field, item)
            validated.append(v)
        return validated if validated else [fallback]
    elif field == "iaa_value":
        validated = [validate_iaa_value(item) for item in items]
        return validated if validated else [fallback]
    elif field in ("papers_topic", "intended_use"):
        validated = []
        for item in items:
            v = validate_single_field(field, item)
            validated.append(v)
        return validated if validated else [fallback]

    return items


def validate_compensation(value: Any) -> str:
    """
    Validate compensation field.

    Args:
        value: Raw value

    Returns:
        Validated compensation (numeric string for hourly rate, descriptive string, or "Not reported")
    """
    str_value = normalize_value(value)

    if not str_value or str_value.lower() in ["", "nan", "none", "null", "not reported", "nr", "-"]:
        return "Not reported"

    if str_value.upper() == "NA":
        return "NA"

    # If it's just a number, assume hourly rate
    if is_float_string(str_value):
        try:
            f = float(str_value)
            if f == int(f):
                return str(int(f))
            return str(f)
        except ValueError:
            pass

    # Keep descriptive strings as-is (e.g., "10 yen per 15 items")
    return str_value


def validate_field(field: str, value: Any) -> Union[str, List[str]]:
    """
    Main validation function - routes to appropriate validator based on field type.

    Args:
        field: Field name
        value: Raw value

    Returns:
        Validated value (string or list depending on field type)
    """
    if field not in FIELD_CONFIG:
        return str(value) if value else "Not reported"

    config = FIELD_CONFIG[field]
    field_type = config.get("type", "single")

    if field_type == "single":
        return validate_single_field(field, value)
    elif field_type == "list":
        return validate_list_field(field, value)
    elif field_type == "numeric_or_not_reported":
        return validate_numeric_field(field, value)
    elif field_type == "compensation":
        return validate_compensation(value)
    else:
        return validate_single_field(field, value)


def validate_extraction(extraction: dict) -> dict:
    """
    Validate all fields in an extraction dictionary.

    Args:
        extraction: Raw extraction dictionary

    Returns:
        Validated extraction dictionary
    """
    validated = {}
    for field in FIELDS:
        raw = extraction.get(field)
        validated[field] = validate_field(field, raw)

    # Copy over non-standard fields
    for key in extraction:
        if key not in FIELDS and key not in validated:
            validated[key] = extraction[key]

    return validated


def validate_experiment(experiment: dict) -> dict:
    """
    Validate a single experiment from an extraction.

    Args:
        experiment: Raw experiment dictionary

    Returns:
        Validated experiment dictionary
    """
    validated = {"paper_experiment_id": experiment.get("paper_experiment_id", "")}

    # Preserve subtask_name — it is not in FIELDS (no validation needed) but must
    # be carried through for downstream matching and GPT-judge comparison.
    if "subtask_name" in experiment:
        validated["subtask_name"] = experiment["subtask_name"]

    for field in FIELDS:
        raw = experiment.get(field)
        validated[field] = validate_field(field, raw)

    # Ensure iaa_value aligns with iaa_metric_name
    metric_list = validated.get("iaa_metric_name", ["Not reported"])
    value_list = validated.get("iaa_value", ["Not reported"])

    # Pad or truncate iaa_value to match iaa_metric_name length
    if len(value_list) < len(metric_list):
        value_list = value_list + ["NA"] * (len(metric_list) - len(value_list))
    elif len(value_list) > len(metric_list):
        value_list = value_list[:len(metric_list)]

    # Set value to NA if metric is "Other" or "Not reported"
    for i, metric in enumerate(metric_list):
        if metric in ["Other", "Not reported"]:
            value_list[i] = "NA"

    validated["iaa_value"] = value_list

    return validated


def validate_full_extraction(extraction_json) -> dict:
    """
    Validate a full extraction JSON (with experiments array).

    Args:
        extraction_json: Full extraction JSON with paper_id and experiments

    Returns:
        Validated extraction JSON
    """
    # Handle case where the LLM returns a bare list of experiments
    if isinstance(extraction_json, list):
        extraction_json = {"paper_id": "", "skip": False, "experiments": extraction_json}

    result = {
        "paper_id": extraction_json.get("paper_id", ""),
        "skip": extraction_json.get("skip", False),
        "experiments": [],
    }

    if extraction_json.get("skip", False):
        result["reason"] = extraction_json.get("reason", "")
        return result

    experiments = extraction_json.get("experiments", [])
    for exp in experiments:
        validated_exp = validate_experiment(exp)
        result["experiments"].append(validated_exp)

    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_allowed_values(field: str) -> Optional[List[str]]:
    """Get the list of allowed values for a field."""
    config = FIELD_CONFIG.get(field, {})
    return config.get("allowed")


def is_valid_value(field: str, value: str) -> bool:
    """Check if a value is valid for a given field."""
    allowed = get_allowed_values(field)
    if allowed is None:
        return True  # No restrictions
    return value in allowed


def get_field_type(field: str) -> str:
    """Get the type of a field."""
    config = FIELD_CONFIG.get(field, {})
    return config.get("type", "single")


def get_fallback(field: str) -> str:
    """Get the fallback value for a field."""
    config = FIELD_CONFIG.get(field, {})
    return config.get("fallback", "Not reported")
