"""
Prompts Module (v2) — Engineered for high agreement with human annotators.
Modular prompt construction for human annotation extraction.

Prompt-engineering focus:
- Exact canonical output values (Unicode metrics, exact expertise strings, etc.)
- Explicit decision trees for every Yes/No/NA/Not reported field
- Default-value emphasis (e.g. demographics default to "No")
- Common-mistake warnings to prevent LLM over-reporting / hallucination
"""

import re
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# =============================================================================
# TAXONOMY SCHEMA — Field Definitions with EXACT allowed values
# =============================================================================

TAXONOMY_SCHEMA = """
## Annotation Taxonomy — EXACT Allowed Values

You MUST output values EXACTLY as listed below — no paraphrasing, no synonyms, no extra words.

### 1. paper_experiment_id
Format: {filename}-1, {filename}-2, etc.

### 2. paper_topic — LIST of one or more, each EXACTLY one of:
  • "Linguistics"
  • "Discourse & Pragmatics"
  • "Information Extraction"
  • "QA & information access"
  • "Semantics, inference & similarity"
  • "Subjective language & social meaning"
  • "Open-ended generation"
  • "Compression & reformulation tasks"
  • "Multilingual & translation"
  • "Multimodal tasks"
  Classify the paper's topic(s) based on its main contributions. Use a list with one or more values.

### 3. human_annotation_type — PICK ONE exactly:
  • "Closed-label classification"
  • "Ordinal / scalar rating"
  • "Span-level or structural annotation"
  • "Comparative / preference judgments"
  • "Open-ended generation or transformation"
  • "Other"
Decision rules:
  - Binary / multi-class / multi-label / tagging with fixed schema → "Closed-label classification"
  - Likert scales, 1–5 ratings, quality scores, graded judgments → "Ordinal / scalar rating"
  - NER spans, coreference, constituency/dependency trees, sequence labeling, alignment → "Span-level or structural annotation"
  - A/B choice, ranking, pairwise preference, best–worst scaling → "Comparative / preference judgments"
  - Summarization, paraphrasing, translation, simplification, free-text generation → "Open-ended generation or transformation"
  - Anything that does not fit the above five → "Other"

### 4. subtask_name
Short descriptive name of the HUMAN ANNOTATION subtask — what the annotators were asked to label, rate, or judge.
If one experiment covers multiple subtasks that share the same annotator pool and annotation round, combine them with " | " separator.
  ⚠ Name the ANNOTATION task, NOT the downstream NLP modelling task:
      • Write "hate speech annotation", NOT "hate speech classifier"
      • Write "stance labeling", NOT "stance detection model"
      • Write "named entity tagging", NOT "NER model training"
  The name must reflect what humans were asked to DO, not the downstream prediction objective.

  ⚠ Be SPECIFIC — mirror the paper's own wording, not a generic category:
      CORRECT (specific):   "binary online attack classification"
      WRONG (too generic):  "attack annotation"
      CORRECT (specific):   "clinical named entity recognition (NER)"
      WRONG (too generic):  "NER annotation"
      CORRECT (specific):   "Pairwise Comparison of Model-Generated Summaries"
      WRONG (too generic):  "evaluation"
      CORRECT (specific):   "rating arguments on a Likert scale across 5 dimensions"
      WRONG (too generic):  "argument rating"
      CORRECT (specific):   "evaluate generated headlines for relevance, attractiveness and language fluency"
      WRONG (too generic):  "headline evaluation"

  ⚠ Use " | " ONLY for subtasks sharing the SAME annotator pool and annotation round.
     If tasks differ in annotator group, annotation unit, instructions, or objective → create SEPARATE experiments.
     Example of correct " | " use: "Misogyny detection - Abusive language detection | Misogyny detection - Target identification | Misogyny detection - Hate speech categorization"
       (all three done by the same annotators in the same study)

  ⚠ Papers routinely contain 2–3 distinct annotation experiments. Common patterns:
      • Different label spaces on the same data (e.g. hate speech → then target type → then severity)
      • Same task repeated by two different annotator groups (e.g. expert gold-standard AND crowd annotation)
      • A generation task PLUS a separate rating/evaluation task on the output
      • Multiple rounds: pilot annotation + main annotation with different guidelines or annotators, if IAA is reported separately
      Each of these MUST be a separate experiment with its own row.

### 5. intended_use — LIST of one or more, each EXACTLY one of:
  • "Resource creation"    (the annotation is primarily for building a dataset/resource)
  • "Human performance"    (the annotation measures human performance on a task)
  • "Model output evaluation" (the annotators evaluate model-generated outputs)
  • "None of the above"    (does not fit any of the above categories)
  Use a list with one or more values. Multiple uses are common (e.g. ["Resource creation", "Model output evaluation"]).
  - DEFAULT: ["None of the above"]

### 6. guidelines_released — "Yes" or "No"
  - "Yes" IF any of the following are true:
      • The paper includes a dedicated appendix section with detailed annotation instructions (not just a brief task description in the main text)
      • A URL / hyperlink to guidelines, an annotation manual, or a codebook is provided (even if the link may now be broken)
      • The paper cites another paper whose guidelines were re-used and those guidelines are thus accessible via that citation
      • Supplementary material, a shared GitHub/OSF repository, or a dataset release specifically includes the annotation guidelines or schema
      • The paper explicitly states that guidelines were released or made available
  - "No" if guidelines are only described informally in the main text without a dedicated document, appendix, cited source, or link; OR if no guidelines information is given
  - DEFAULT: "No"
  ⚠ Common mistake: Do NOT mark "Yes" just because the paper describes the annotation task in a paragraph. There must be an accessible, standalone guidelines document or appendix section, an annotator interface screenshot, allowing reproduction.

### 7. iaa_metric_name — LIST of one or more, each EXACTLY one of:
  • "Fleiss' κ"          (use Greek kappa: κ)
  • "Cohen's κ"          (use Greek kappa: κ)
  • "Krippendorff's α"   (use Greek alpha: α)
  • "F1 agreement"
  • "Pearson"
  • "Spearman"
  • "Majority agreement"
  • "Kendall's tau"
  • "Total agreement"     (also called percent agreement, percentage agreement, observed agreement, raw agreement, exact agreement)
  • "Other"               (any explicit metric not in the list above, e.g. ICC, Scott's pi, Gwet's AC1, BLEU, Kappa)
  • "Not reported"        (either not applicable because there is only one annotator, or no IAA metric is given in the paper)
IMPORTANT: If a metric is stated but you cannot identify which one, use "Other".
If no metric is mentioned, use "Not reported".
If multiple metrics are reported, list them in the order they appear in the paper.
⚠ INTERDEPENDENCY: If annotators_per_item == "1" OR the task is open-ended free-text generation → force ["Not reported"].

### 8. iaa_value — LIST aligned 1-to-1 with iaa_metric_name
  - Numeric values as strings (e.g. "0.58", "85.3%", "0.09 to 0.23") for any real metric, or "NA" if the paper does not directly state the value. Refrain from inferring values.
  - If ranges are reported or any non-single-value form is used, enter as a string (e.g. "0.09 to 0.23").
  - "NA" if iaa_metric_name[i] == "Other" (regardless of whether a number is given)
  - "Not reported" if iaa_metric_name[i] == "Not reported"
  
  - Keep the same number of elements as iaa_metric_name
⚠ INTERDEPENDENCY: iaa_value[i] is FULLY DETERMINED by iaa_metric_name[i] — see Field Interdependencies rule A above.
⚠ IMPORTANT: If you identified an IAA metric, ACTIVELY search for its numeric value in: (a) the main text near where the metric is mentioned, (b) ALL tables (especially agreement/statistics tables), (c) the appendix. IAA values are very often reported in tables rather than prose. Only output "NA" after checking tables.

### 9–12. Workload fields — integer string or "Not reported"
  - **annotators_per_item**: annotators who labeled EACH item. Integer or "Not reported". Ranges like "2-3" are acceptable.
  - **total_annotators**: total number of distinct annotators. Integer or "Not reported".
  - **total_annotated_items**: total number of items/instances annotated IN THIS SPECIFIC EXPERIMENT. Integer or "Not reported". If given with K suffix (e.g. "27.9K"), keep as-is. If given with commas keep as-is (e.g. "2,864").
    ⚠ Extract only the item count for the annotation experiment described. An annotated item is a sample under evaluation, not the number of annotated spans or the total tokens/lines across annotated items (e.g. the number of poems, sentences, documents), NOT the total corpus/dataset size that may be larger. If the paper provides multiple numbers (e.g. a larger pre-filtered pool and a smaller annotated subset), use the smaller annotated subset count.
  - **items_per_annotator**: how many items each annotator processed. Integer or "Not reported".
  - DEFAULT: "Not reported" — do NOT guess or calculate from other fields.

### 13. human_evaluated_sample — free-text string
  - Describe the size/scope of the human-evaluated sample (e.g. "100", "all", "200 randomly chosen tasks from the training data", "500", "random 50")
  - If the paper describes a specific number of items evaluated by humans, report that number or description
  - report cross-annotated sample, if different to the total number of annotated items
  - "Not reported" if no human-evaluated sample is described or the size is not specified
  - "all" if all items were human-evaluated, especially if the intended use of the human labour is to create a resource
  - DEFAULT: "Not reported"

### 14. recruitment — PICK ONE exactly:
  • "Crowdsourcing"  (annotators recruited via a crowdsourcing platform or described as crowdworkers)
  • "Authors"        (one or more authors performed the annotation)
  • "Other"          (annotators are a specified non-crowd group, e.g. students, experts, hired assistants, colleagues)
  • "Mixed"          (annotators include people from different specified groups)
  • "Not reported"   (the source of annotators is not specified as a specific group)
  - DEFAULT: "Not reported"

### 15. crowd_qualifications_screening — "Yes", "No", or "NA"
  - "Yes" if any pre-screening, qualification test, or performance record is described for crowdworkers, typically: 95% acceptance rate in previous jobs (not  socio-demographic filter)
  - "No" if crowdsourcing was used but no screening mentioned
  - "NA" if NOT a crowdsourcing study
  - DEFAULT: "NO" when not crowdsourcing; "No" when crowdsourcing 
⚠ INTERDEPENDENCY (set before checking paper content):
  - recruitment == "Crowdsourcing"              → "Yes" or "No" only
  - recruitment == "Authors" / "Other" / "Mixed" / "Not reported" → MUST be "NO"

### 16. annotator_training — "Yes" or "No"
  - "Yes" if training, practice rounds, trial tasks, calibration sessions, or hands-on prior experience with the annotation tool are mentioned
  - "No" if no training is described, or if there is an explicit mention of no training provided
  - DEFAULT: "No"

### 17. language_proficiency — PICK ONE exactly:
  • "Native"          (native speakers explicitly stated; OR annotators' ethnicity is reported and it aligns with the language of the annotation task)
  • "Non-Native"      (authors report some level of proficiency, e.g. fluent, B2, C1, but not native)
  • "Mixed"           (both native and non-native speakers were employed)
  • "Medium"          (annotators have moderate language proficiency, e.g. advanced learners)
  • "Not reported"    (language proficiency not directly reported)
  - Do NOT infer from dataset language or annotator country of residence alone.
  - DEFAULT: "Not reported"

### 18. expertise_level — PICK ONE exactly:
  • "High"         (annotators have specialised domain knowledge relevant to the task, e.g. medical professionals, legal experts, trained linguists, NLP researchers, annotating linguistic phenomena)
  • "Medium"       (annotators have some relevant background but are not deeply specialised, e.g. students in a related field, people with self-reported relevant experience)
  • "Mixed"        (reported levels of expertise vary, e.g. experts and students, or students and crowd workers used together)
  • "Not reported" (the task COULD require specialised knowledge, but the paper does not specify annotators' expertise level)
  • "General Language/Knowledge task" (the task is straightforward and DOES NOT REQUIRE any specialised knowledge — e.g. rating text fluency/relevance/acceptability using only common language sense, evaluating whether a headline is attractive, binary preference judgments on everyday content)
  - DEFAULT: "Not reported"

### 19. reported_compensation — PICK ONE exactly:
  • "Paid"         (any compensation is mentioned: hourly rate, per-task payment, total amount, gift cards, course credit used as incentive, being described as a paid/professional annotator or employee)
  • "Free"         (annotation was done without payment: authors annotated the data themselves, annotators were volunteers or labelled as unpaid, students completed it as a course assignment with no monetary reward)

  • "Not reported" (the paper says nothing about how or whether annotators were compensated)
  - DEFAULT: "Not reported"

### 20. payment_rate — PICK ONE exactly:
  • "Specific numeric rate" (a concrete payment is given: hourly rate like "$12/hr", per-task like "$0.05/HIT", total amount like "paid $500 total", or any explicit monetary figure)
  • "General mention"       (compensation is confirmed but no exact amount: "annotators were paid", "workers received payment", "full-time employees", "compensated for their time")

  • "NA"                    (reported_compensation is "Free" or "Not reported" — payment information not available)
  - DEFAULT: "NA"
  ⚠ Check tables and appendices for payment figures — amounts are often reported there, not in the main text.

### 21–26. Socio-Demographics — ALL are "Yes" or "No" ONLY
  - **age_reported**: "Yes" ONLY if actual age data (mean age, age range, age distribution) is reported WITH values. "No" otherwise.
  - **gender_reported**: "Yes" ONLY if actual gender data (counts, percentages, distribution) is reported WITH values. "No" otherwise.
  - **nation_origin_reported**: "Yes" ONLY if annotators' nationality/country of origin is reported WITH values. "No" otherwise.
  - **nation_residence_reported**: "Yes" ONLY if annotators' country of residence is reported WITH values. "No" otherwise. Indirect references to the residence of participants (e.g. poets from a local club, faculty employed at an American university department of X, English-speaking country) should be treated as Nation of residence = Yes.
  - **education_reported**: "Yes" ONLY if annotators' education level or field of study is reported WITH values. "No" otherwise.
    ⚠ INTERDEPENDENCY: If expertise_level == "High", set education_reported = "Yes" (explicit expert labelling implies education was assessed).
  - **political_orientation_reported**: "Yes" ONLY if political orientation is reported WITH values. "No" otherwise.
  *** CRITICAL DEFAULT: "No" for ALL demographic fields. ***
  *** "No" does NOT mean the paper says "we did not collect demographics". ***
  *** "No" simply means the paper does NOT report that specific demographic data. ***
  *** NEVER use "Not reported" for these fields — only "Yes" or "No". ***

### 27. quality_control_filtering — "Yes" or "No" ONLY
  - "Yes" if: quality control measures are described (related to discarding annotations or deboarding annotators: attention checks, gold or repeat questions, filtering bad annotators, removing low-quality annotations)
  - "No" in ALL other cases — including when experts were employed and their judgments trusted without filtering, or when no QC details are given
  - *** NEVER output "NA" for this field — "NA" is treated as "No", use "No" directly ***
  - DEFAULT: "No"

### 28. disagreement_resolution_adjudication — PICK ONE exactly:
  • "Majority vote"      (final label determined by majority voting among annotators)
  • "Expert adjudication" (an expert resolves disagreements)
  • "Third annotator"     (a third annotator breaks ties)
  • "Consensus discussion" (annotators discuss to reach consensus)
  • "Soft labels / Distributional labels" (all annotations kept as a distribution, no single label chosen)
  • "Weighted voting"     (votes weighted by annotator reliability/expertise)
  • "Other / mixed"       (other resolution method or combination of methods)
  • "None"               (no disagreement resolution described: all opinions used as-is or averaged)
  - DEFAULT: "None"
"""


# =============================================================================
# EXTRACTION RULES — Strict decision-making guidance
# =============================================================================

EXTRACTION_RULES = """
## Extraction Rules — READ CAREFULLY

### Experiment Identification

⚠ **ANNOTATION EXPERIMENTS ONLY** — Only extract experiments where humans actively create, label, rate, rank, or judge data. Do NOT extract model training runs, automatic evaluation steps, or system-based experiments that do not involve human annotators as a core part of the process.

**SPLIT into separate experiments when ANY of the following differ between annotation tasks:**
  1. **Different annotation objective** — distinct label spaces or annotation goals (e.g. hate speech detection vs. stance detection)
  2. **Different annotator pools** — different groups used for different tasks (e.g. experts for one subtask, crowd for another)
  3. **Different recruitment / compensation** — one task paid, another unpaid; or recruited via different channels
  4. **Different annotation units** — sentence-level vs document-level; word-level vs sentence-level
  5. **Separate dedicated annotation rounds** — the paper describes multiple distinct rounds, each with its own guidelines, annotators, or IAA

Real-paper SPLIT examples:
    • binary attack classification / language classification / attack target classification → 3 experiments (different label spaces)
    • summarization generation / summarization rating → 2 experiments (generation task + evaluation task)
    • Expert Gold-Standard NER / Crowd-Based NER → 2 experiments (different annotator pools)
    • poetry generation by crowd / poetry generation by experts → 2 experiments (different recruitment)
    • Is label Acceptable/Unacceptable / If Acceptable, is it Precise/Imprecise → 2 experiments (different annotation objectives)

**MERGE into one experiment when:**
  - The same annotation was applied to multiple datasets with the same label scheme and annotators
  - Multiple IAA metrics are reported for the same annotation task
  - Minor sub-tasks that share annotators, and annotated items, type of human judgment, and process differ only in topic or domain

⚠ DEFAULT BIAS — **ALWAYS err toward SPLITTING, not merging.** Under-splitting is the most common error. A paper that runs hate speech labeling and then target-type labeling on the same dataset has TWO experiments, not one. Different label spaces = different experiments.

⚠ **SELF-AUDIT before writing JSON**: After identifying your first experiment, explicitly ask:
  1. Does the paper describe any OTHER annotation task (different labels, different annotators, different purpose in a different paper section)?
  2. Are there separate sections titled "Experiment 2", "Human Evaluation", "Quality Annotation", "Pilot Study"?
  3. Did any OTHER group of humans label, rate, or judge data in this paper?
  4. Is there a text production task AND a separate evaluation/rating task performed by humans?
  If the answer to ANY of these is yes → add another experiment.

### CRITICAL: Avoid These Common Mistakes
1. **Do NOT use "Not reported" for demographic fields** — use "No". These fields ONLY accept "Yes" or "No".
2. **disagreement_resolution_adjudication must be one of the specific methods** — "Majority vote", "Expert adjudication", "Third annotator", "Consensus discussion", "Soft labels / Distributional labels", "Weighted voting", "Other / mixed", or "None".
3. **Do NOT spell out "kappa" or "alpha"** in iaa_metric_name — use the Unicode symbols κ and α.
4. **Do NOT infer language proficiency** from the dataset language or annotator country alone (exception: ethnicity matching task language → "Native").
5. **Do NOT infer compensation amounts** — but DO infer "Free" when authors annotate their own data, or mention voluntary effort.
6. **Do NOT assume guidelines are released** just because the paper describes the task in prose — there must be a dedicated document, appendix, URL, or cited source.
7. **Do NOT calculate workload fields** from other numbers — only extract explicitly stated values.
8. **For recruitment**, use the broad categories: "Crowdsourcing", "Authors", "Other", "Mixed", "Not reported" — do NOT use platform-specific names.
9. **For compensation, do NOT use "NA"** — use "Not reported" if compensation is unmentioned.
10. **Do NOT confuse expertise_level "General Language/Knowledge task" and "Not reported"**: "General Language/Knowledge task" = task needs no specialization at all; "Not reported" = task could need specialization but paper doesn't specify annotator expertise.
11. **For IAA values, check tables** — numeric values are very often in tables, not the prose. Do not output "NA" for a real metric until you have checked all tables.
12. **For total_annotated_items, use the annotated subset size**, not the total corpus/dataset size — these can differ substantially.
13. **subtask_name must name the ANNOTATION task**, not the downstream NLP modelling task — "sentiment rating" not "sentiment classifier"; "NER tagging" not "NER model".
14. **For every numerical value, record evidence** in the `numerical_evidence` object — do NOT leave it empty when you extracted a non-"Not reported" value.

### Field Interdependencies — APPLY THESE BEFORE FINALISING OUTPUT
These conditional rules take PRECEDENCE over all other guidance. Check each one after filling every field.

**A. iaa_value is fully determined by iaa_metric_name (apply per list position):**
  - iaa_metric_name[i] == "Other"        → iaa_value[i] = "NA"
  - iaa_metric_name[i] == "Not reported" → iaa_value[i] = "Not reported"
  - iaa_metric_name[i] is any real metric → iaa_value[i] = the numeric value reported (or "NA" if value is absent)

**B. iaa_metric_name / iaa_value ← annotators_per_item or task type:**
  - annotators_per_item == "1" (single annotator)                           → iaa_metric_name = ["Not reported"], iaa_value = ["Not reported"]
  - task is open-ended text generation (humans produce free text, no fixed label set) → iaa_metric_name = ["Not reported"], iaa_value = ["Not reported"]

**C. crowd_qualifications_screening ← recruitment:**
  - recruitment == "Crowdsourcing"   → crowd_qualifications_screening = "Yes" OR "No" (based on paper content)
  - recruitment == "Authors"         → crowd_qualifications_screening = "NO"
  - recruitment == "Other"           → crowd_qualifications_screening = "NO"
  - recruitment == "Mixed"           → crowd_qualifications_screening = "NO" (unless paper explicitly mentions crowd screening for the crowd sub-group)
  - recruitment == "Not reported"    → crowd_qualifications_screening = "NO"

**D. education_reported ← expertise_level:**
  - expertise_level == "High" (experts)  → education_reported = "Yes"
    (Rationale: explicitly labelling annotators as domain experts implies their educational/professional qualification was assessed and reported.)
  - All other expertise_level values     → apply standard "Yes"/"No" rule

### Default values summary (when information is absent from the paper):
  - guidelines_released → "No"
  - iaa_metric_name → ["Not reported"]
  - iaa_value → ["Not reported"]
  - workload fields → "Not reported"
  - recruitment → "Not reported"
  - crowd_qualifications_screening → "NA" (non-crowd) or "No" (crowd)
  - annotator_training → "No"
  - language_proficiency → "Not reported"
  - expertise_level → "Not reported"
  - reported_compensation → "Not reported"
  - payment_rate → "NA"
  - human_evaluated_sample → "Not reported"
  - ALL demographics → "No"
  - quality_control_filtering → "No" (NEVER "NA")
  - disagreement_resolution_adjudication → "None"

### Numerical Evidence
For every numerical value you extract, you MUST populate the `numerical_evidence` object with the verbatim text (sentence fragment, table cell, or figure caption) from the paper that supports that number.
  - **Fields that require evidence**: `annotators_per_item`, `total_annotators`, `total_annotated_items`, `items_per_annotator`, each element of `iaa_value` (keyed as `iaa_value_0`, `iaa_value_1`, etc.), and `payment_rate` when it is "Specific numeric rate".
  - If a value was inferred rather than directly stated, write: `"field_name": "inferred from: [brief reason]"`
  - Omit a key from `numerical_evidence` only when the corresponding field value is "Not reported" or "NA".
"""


# =============================================================================
# JSON SCHEMA
# =============================================================================

# =============================================================================
# PYDANTIC MODELS — used for response_json_schema structured output
# =============================================================================

class NumericalEvidence(BaseModel):
    annotators_per_item: Optional[str] = Field(None, description="Exact quote or 'inferred from: ...' supporting the annotators_per_item value.")
    total_annotators: Optional[str] = Field(None, description="Exact quote or 'inferred from: ...' supporting the total_annotators value.")
    total_annotated_items: Optional[str] = Field(None, description="Exact quote or 'inferred from: ...' supporting the total_annotated_items value.")
    items_per_annotator: Optional[str] = Field(None, description="Exact quote or 'inferred from: ...' supporting the items_per_annotator value.")
    iaa_value_0: Optional[str] = Field(None, description="Exact quote or table entry for iaa_value[0].")
    iaa_value_1: Optional[str] = Field(None, description="Exact quote or table entry for iaa_value[1], if present.")
    iaa_value_2: Optional[str] = Field(None, description="Exact quote or table entry for iaa_value[2], if present.")
    payment_rate: Optional[str] = Field(None, description="Exact quote supporting a 'Specific numeric rate' payment_rate value.")


class AnnotationExperiment(BaseModel):
    paper_experiment_id: str = Field(description="Paper experiment identifier, e.g. '{filename}-1'.")
    papers_topic: List[Literal[
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
    ]] = Field(description="Topic area(s) of the paper's annotation task. List of one or more values.")
    human_annotation_type: Literal[
        "Closed-label classification",
        "Ordinal / scalar rating",
        "Span-level or structural annotation",
        "Comparative / preference judgments",
        "Open-ended generation or transformation",
        "Other",
    ] = Field(description="Type of NLP annotation task.")
    subtask_name: str = Field(description="Specific name of the human annotation subtask, mirroring the paper's own wording.")
    intended_use: List[Literal[
        "Resource creation",
        "Human performance",
        "Model output evaluation",
        "None of the above",
    ]] = Field(description="Intended use(s) of the annotation. List of one or more values.")
    guidelines_released: Literal["Yes", "No"] = Field(description="Whether annotation guidelines were released as a standalone document, appendix, URL, or cited source.")
    iaa_metric_name: List[Literal[
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
    ]] = Field(description="List of IAA metric names reported. Use Unicode κ and α. Use ['Not reported'] if none.")
    iaa_value: List[str] = Field(description="List of IAA values aligned 1-to-1 with iaa_metric_name. Use 'Not reported' for 'Not reported' metrics, 'NA' for 'Other' metrics.")
    annotators_per_item: str = Field(description="Number of annotators per item. Integer string, range like '2-3', or 'Not reported'.")
    total_annotators: str = Field(description="Total distinct annotators. Integer string or 'Not reported'.")
    total_annotated_items: str = Field(description="Total annotated items in this experiment. Integer string (may use K suffix) or 'Not reported'.")
    items_per_annotator: str = Field(description="Items processed per annotator. Integer string or 'Not reported'.")
    human_evaluated_sample: str = Field(description="Description of the human-evaluated sample size/scope (e.g. '100', 'all', '200 randomly chosen tasks', 'Not reported'). Default 'Not reported'.")
    recruitment: Literal["Crowdsourcing", "Authors", "Other", "Mixed", "Not reported"] = Field(description="How annotators were recruited.")
    crowd_qualifications_screening: Literal["Yes", "No", "NA"] = Field(description="'NA' if not crowdsourcing. 'Yes' or 'No' if crowdsourcing.")
    annotator_training: Literal["Yes", "No"] = Field(description="Whether annotators received training or practice rounds.")
    language_proficiency: Literal["Native", "Non-Native", "Mixed", "Medium", "Not reported"] = Field(description="Reported language proficiency of annotators.")
    expertise_level: Literal["High", "Medium", "Mixed", "Not reported", "General Language/Knowledge task"] = Field(description="Annotator expertise level. 'General Language/Knowledge task' if task needs no specialisation. 'Not reported' if specialisation could be relevant but paper doesn't specify.")
    reported_compensation: Literal["Paid", "Free", "Not reported"] = Field(description="Annotator compensation. 'Free' if authors annotated or volunteers. 'Not reported' if unmentioned.")
    payment_rate: Literal["Specific numeric rate", "General mention", "NA"] = Field(description="'NA' if reported_compensation is 'Free' or 'Not reported'.")
    age_reported: Literal["Yes", "No"] = Field(description="Whether actual age data with values is reported. Default 'No'.")
    gender_reported: Literal["Yes", "No"] = Field(description="Whether actual gender data with values is reported. Default 'No'.")
    nation_origin_reported: Literal["Yes", "No"] = Field(description="Whether annotators' nationality/country of origin is reported with values. Default 'No'.")
    nation_residence_reported: Literal["Yes", "No"] = Field(description="Whether annotators' country of residence is reported with values. Default 'No'.")
    education_reported: Literal["Yes", "No"] = Field(description="Whether annotators' education level is reported with values. Default 'No'. Set 'Yes' if expertise_level is 'High'.")
    political_orientation_reported: Literal["Yes", "No"] = Field(description="Whether political orientation data with values is reported. Default 'No'.")
    quality_control_filtering: Literal["Yes", "No"] = Field(description="Whether quality control or annotation filtering is described. NEVER 'NA'. Default 'No'.")
    disagreement_resolution_adjudication: Literal[
        "Majority vote",
        "Expert adjudication",
        "Third annotator",
        "Consensus discussion",
        "Soft labels / Distributional labels",
        "Weighted voting",
        "Other / mixed",
        "None",
    ] = Field(description="Specific method of disagreement resolution. 'None' if not described. Default 'None'.")
    numerical_evidence: NumericalEvidence = Field(description="Verbatim quotes or table entries supporting numerical values extracted.")


class PaperExtraction(BaseModel):
    paper_id: str = Field(description="Paper filename/identifier.")
    skip: bool = Field(default=False, description="True only if the paper has no human annotation experiments at all.")
    analysis: str = Field(description="1–2 sentence summary: number of experiments, annotator type, IAA metric+value, compensation.")
    experiments: List[AnnotationExperiment] = Field(description="One entry per distinct human annotation experiment found in the paper.")


def get_extraction_json_schema() -> dict:
    """Return the JSON schema for PaperExtraction, for use with response_json_schema."""
    return PaperExtraction.model_json_schema()


JSON_SCHEMA_TEMPLATE = """
{{
  "paper_id": "{filename}",
  "skip": false,
  "experiments": [
    {{
      "paper_experiment_id": "{filename}-1",
      "papers_topic": ["Linguistics|Discourse & Pragmatics|Information Extraction|QA & information access|Semantics, inference & similarity|Subjective language & social meaning|Open-ended generation|Compression & reformulation tasks|Multilingual & translation|Multimodal tasks"],
      "human_annotation_type": "Closed-label classification|Ordinal / scalar rating|Span-level or structural annotation|Comparative / preference judgments|Open-ended generation or transformation|Other",
      "subtask_name": "task label from the paper (e.g. 'hate speech detection', 'stance annotation', 'summary rating')",
      "intended_use": ["Resource creation|Human performance|Model output evaluation|None of the above"]tive / preference judgments|Open-ended generation or transformation|Other",
      "subtask_name": "task label from the paper (e.g. 'hate speech detection', 'stance annotation', 'summary rating')",
      "intended_use": ["Resource creation|Human performance|Model output evaluation|None of the above"],
      "guidelines_released": "Yes|No",
      "iaa_metric_name": ["Fleiss' κ|Cohen's κ|Krippendorff's α|F1 agreement|Pearson|Spearman|Majority agreement|Kendall's tau|Total agreement|Other|Not reported"],
      "iaa_value": ["numeric string|Not reported|NA"],
      "annotators_per_item": "integer or range string|Not reported",
      "total_annotators": "integer string|Not reported",
      "human_evaluated_sample": "free-text: number, description, 'all', or 'Not reported'",
      "recruitment": "Crowdsourcing|Authors|Other|Mixed|Not reported",
      "crowd_qualifications_screening": "Yes|No|NA",
      "annotator_training": "Yes|No",
      "language_proficiency": "Native|Non-Native|Mixed|Medium|Not reported",
      "expertise_level": "High|Medium|Mixed|Not reported|General Language/Knowledge task",
      "reported_compensation": "Paid|Free|NA|Not reported",
      "payment_rate": "Specific numeric rate|General mention|Nom|Not reported",
      "expertise_level": "High|Medium|Mixed|Not reported|General Language/Knowledge task",
      "reported_compensation": "Paid|Free|NA|Not reported",
      "payment_rate": "Specific numeric rate|General mention|No|NA",
      "age_reported": "Yes|No",
      "gender_reported": "Yes|No",
      "nation_origin_reported": "Yes|No",
      "nation_residence_reported": "Yes|No",
      "education_reported": "Yes|No",
      "political_orientation_reported": "Yes|No",
      "quality_control_filtering": "Yes|No",
      "disagreement_resolution_adjudication": "Majority vote|Expert adjudication|Third annotator|Consensus discussion|Soft labels / Distributional labels|Weighted voting|Other / mixed|None",
      "numerical_evidence": {{
        "annotators_per_item": "exact quote or 'inferred from: ...' ",
        "total_annotators": "exact quote or 'inferred from: ...' ",
        "total_annotated_items": "exact quote or 'inferred from: ...' ",
        "items_per_annotator": "exact quote or 'inferred from: ...' ",
        "iaa_value_0": "exact quote or table entry for iaa_value[0]"
      }}
    }}
  ]
}}
"""


# =============================================================================
# MAIN EXTRACTION PROMPT
# =============================================================================

def build_extraction_prompt(paper_content: str, filename: str) -> str:
    """
    Build the extraction prompt for a paper.

    The prompt instructs the model to produce JSON conforming to PaperExtraction.
    Pass get_extraction_json_schema() as response_json_schema in the Gemini API call
    so the model output is constrained to the schema (structured output mode).

    Args:
        paper_content: Full text content of the paper
        filename: Paper filename/ID

    Returns:
        Complete extraction prompt (to be used with response_json_schema structured output)
    """
    json_schema = JSON_SCHEMA_TEMPLATE.format(filename=filename)
    
    return f"""You are extracting structured data about HUMAN ANNOTATION experiments from a scientific paper.
Your task is to fill in a rigid taxonomy. You MUST use ONLY the exact allowed values specified below.
Do NOT paraphrase, do NOT add explanations, do NOT use synonyms.

{TAXONOMY_SCHEMA}

{EXTRACTION_RULES}

## Approach — Two-step extraction

**STEP 1 — Locate annotation sections BEFORE filling any field.**
Annotation metadata is typically found in sections titled: "Annotation", "Data Collection",
"Annotators", "Human Evaluation", "Experimental Setup", "Crowdsourcing", "Dataset Construction",
"Annotation Procedure", "Annotation Scheme", or appendix sections.
⚠ Only extract HUMAN ANNOTATION experiments (humans labeling/rating/judging data). Ignore model training runs, automatic evaluation steps, or any experiment that does not involve human annotators.
Scan those sections and mentally note:
  • How many DISTINCT annotation experiments? (different tasks or different annotation units = separate experiments; otherwise merge)
  • Who are the annotators — crowd workers, authors, domain experts, students?
  • How many annotators per item? Total annotators count?
  • What IAA metric(s) and numeric value(s) are reported?
  • How were annotators recruited and compensated?
  • Any demographics WITH VALUES, training, guidelines, or quality control described?
  • For each numerical value found, note the EXACT sentence or table entry it came from (for `numerical_evidence`).
**Multi-experiment scan**: Papers with 2–3 distinct annotation experiments are COMMON. After finding your first experiment, continue scanning:
  - Other sections / appendices for additional annotation tasks
  - Whether the paper reports a "pilot" or "validation" annotation with different setup
  - Whether different tasks/label-spaces were used (e.g., first hate-speech, then target classification)
  - Whether two different annotator groups (crowd + expert) independently annotated the same or related data
  Do NOT stop at the first annotation section you find.

**STEP 2 — Fill the experiments array** using only what you found in the paper.
When information is absent, apply the defaults in TAXONOMY_SCHEMA (typically "Not reported", "No", or "NA").

## WORKED EXAMPLE

Hypothetical paper: MTurk hate speech annotation with Fleiss' κ, residence filter, majority-vote adjudication.

```json
{{
  "paper_id": "example-paper",
  "skip": false,
  "experiments": [
    {{
      "paper_experiment_id": "example-paper-1",
      "papers_topic": ["Subjective language & social meaning"],
      "human_annotation_type": "Closed-label classification",
      "subtask_name": "hate speech detection",
      "intended_use": ["Resource creation"],
      "guidelines_released": "No",
      "iaa_metric_name": ["Fleiss' κ"],
      "iaa_value": ["0.62"],
      "annotators_per_item": "3",
      "total_annotators": "50",
      "total_annotated_items": "10000",
      "items_per_annotator": "Not reported",
      "human_evaluated_sample": "Not reported",
      "recruitment": "Crowdsourcing",
      "crowd_qualifications_screening": "Yes",
      "annotator_training": "No",
      "language_proficiency": "Native",
      "expertise_level": "General Language/Knowledge task",
      "reported_compensation": "Paid",
      "payment_rate": "Specific numeric rate",
      "age_reported": "No",
      "gender_reported": "No",
      "nation_origin_reported": "No",
      "nation_residence_reported": "Yes",
      "education_reported": "No",
      "political_orientation_reported": "No",
      "quality_control_filtering": "Yes",
      "disagreement_resolution_adjudication": "Majority vote",
      "numerical_evidence": {{
        "annotators_per_item": "3/item",
        "total_annotators": "50 annotators from MTurk",
        "total_annotated_items": "collected 10,000 posts",
        "iaa_value_0": "Fleiss' κ = 0.62"
      }}
    }}
  ]
}}
```

## Output Format
Output ONLY valid JSON matching this schema:
{json_schema}

## Paper Content
{paper_content}

## REMINDER before you write your JSON:
- **analysis**: write this FIRST — 1–2 sentences: N experiments, annotator type, IAA metric+value, compensation
- Demographics (age/gender/nation_origin/nation_residence/education/political_orientation): default is "No", ONLY "Yes" if actual values WITH NUMBERS are reported
- disagreement_resolution_adjudication: use specific method name ("Majority vote", "Expert adjudication", "Third annotator", "Consensus discussion", "Soft labels / Distributional labels", "Weighted voting", "Other / mixed", "None")
- IAA metrics: use Unicode κ and α (NOT "kappa" or "alpha")
- **IAA values**: check ALL tables — values are very often in a table, not the prose
- recruitment: "Crowdsourcing" only if platform named or "crowd workers" used; "Authors" if authors themselves annotated; "Other" for students/experts/assistants; else "Not reported"
- **compensation**: authors annotating = reported_compensation "Free"; any pay mentioned = "Paid"; nothing mentioned = "Not reported"; "NA" if not applicable
- **payment_rate**: "Specific numeric rate" / "General mention" / "No" (if Free) / "NA" (if Not reported or NA) — check tables for exact amounts
- **expertise_level**: "General Language/Knowledge task" if task needs no specialization (rating fluency/attractiveness/preference); "Not reported" if specialization COULD be relevant but paper doesn't specify
- **guidelines_released**: "Yes" if appendix with detailed instructions, URL, cited guidelines paper, or supplementary material — NOT just because the paper describes the task
- **total_annotated_items**: use the annotated subset count, NOT the full corpus/dataset size
- **subtask_name**: use SPECIFIC wording mirroring the paper (e.g. "binary online attack classification", NOT "attack annotation"; "Pairwise Comparison of Model-Generated Summaries", NOT "evaluation")
- **papers_topic**: LIST of one or more topic areas
- **intended_use**: LIST of one or more intended uses
- **human_evaluated_sample**: free-text describing sample size/scope that was selected for human inspection or cross-annotated, or "Not reported"
- **Multiple experiments**: papers with 2–3 distinct annotation experiments are common — re-read the FULL paper before settling on 1 experiment. Different label spaces, different annotator groups, generation + rating, pilot + main → each is a SEPARATE experiment row

"""


def build_extraction_prompt_compact(paper_content: str, filename: str) -> str:
    """
    Build a more compact extraction prompt (for token-limited models).
    
    Args:
        paper_content: Full text content of the paper  
        filename: Paper filename/ID
        
    Returns:
        Compact extraction prompt
    """
    return f"""Extract human annotation experiments from this paper as JSON.

STRICT RULES:
- Output ONLY valid JSON, no markdown
- Use EXACT canonical values below — no paraphrasing

EXACT ALLOWED VALUES:
- papers_topic (list): "Linguistics" | "Discourse & Pragmatics" | "Information Extraction" | "QA & information access" | "Semantics, inference & similarity" | "Subjective language & social meaning" | "Open-ended generation" | "Compression & reformulation tasks" | "Multilingual & translation" | "Multimodal tasks"
- human_annotation_type: "Closed-label classification" | "Ordinal / scalar rating" | "Span-level or structural annotation" | "Comparative / preference judgments" | "Open-ended generation or transformation" | "Other"
- intended_use (list): "Resource creation" | "Human performance" | "Model output evaluation" | "None of the above"
- iaa_metric_name (list): "Fleiss' κ" | "Cohen's κ" | "Krippendorff's α" | "F1 agreement" | "Pearson" | "Spearman" | "Majority agreement" | "Kendall's tau" | "Total agreement" | "Other" | "Not reported"  (USE Unicode κ α)
- human_evaluated_sample: free-text (number, description, "all", or "Not reported")
- recruitment: "Crowdsourcing" | "Authors" | "Other" | "Mixed" | "Not reported"
- expertise_level: "High" | "Medium" | "Mixed" | "Not reported" | "General Language/Knowledge task"
- language_proficiency: "Native" | "Non-Native" | "Mixed" | "Medium" | "Not reported"
- guidelines_released / annotator_training: "Yes" | "No"
- Demographics (age/gender/nation_origin/nation_residence/education/political_orientation)_reported: "Yes" | "No" ONLY (default "No")
- disagreement_resolution_adjudication: "Majority vote" | "Expert adjudication" | "Third annotator" | "Consensus discussion" | "Soft labels / Distributional labels" | "Weighted voting" | "Other / mixed" | "None"
- quality_control_filtering: "Yes" | "No"
- crowd_qualifications_screening: "Yes" | "No" | "NA"
- reported_compensation: "Paid" | "Free" | "NA" | "Not reported"
- payment_rate: "Specific numeric rate" | "General mention" | "No" | "NA"

*** Demographics default to "No". disagreement_resolution defaults to "None". ***

Paper ID: {filename}

Paper:
{extract_annotation_chunks(paper_content, max_tokens=8000)}

Output JSON with paper_id, skip (false), analysis, and experiments array:
"""


# =============================================================================
# JSON FIX PROMPT
# =============================================================================

def build_json_fix_prompt(broken_json: str, error_message: str) -> str:
    """
    Build a prompt to fix broken JSON.
    
    Args:
        broken_json: The malformed JSON string
        error_message: The error from json.loads()
        
    Returns:
        JSON fix prompt
    """
    return f"""Fix this JSON so Python's json.loads() can parse it.

ERROR: {error_message}

BROKEN JSON:
{broken_json}

RULES:
- Output ONLY the corrected JSON
- Use double quotes for all keys and strings
- Escape backslashes properly (\\\\times not \\times)
- No trailing commas
- Preserve the original content/meaning

FIXED JSON:
"""


# =============================================================================
# EVIDENCE EXTRACTION PROMPT
# =============================================================================

def build_evidence_extraction_prompt(paper_content: str, field: str, value: str) -> str:
    """
    Build a prompt to extract evidence for a specific field value.
    
    Args:
        paper_content: Full text content of the paper
        field: Field name to find evidence for
        value: Extracted value to verify
        
    Returns:
        Evidence extraction prompt
    """
    return f"""Find the quote in this paper that supports this extraction.

Field: {field}
Extracted value: {value}

Paper content:
{paper_content[:8000]}

Output JSON:
{{"field": "{field}", "quote": "exact quote from paper (max 50 words)", "page": page_number_or_null}}

If no supporting quote found, output:
{{"field": "{field}", "quote": null, "page": null}}

JSON:
"""


# =============================================================================
# VALIDATION PROMPT (for uncertain values)
# =============================================================================

def build_validation_prompt(
    paper_content: str, 
    field: str, 
    extracted_value: str, 
    allowed_values: list
) -> str:
    """
    Build a prompt to validate an extracted value against allowed options.
    
    Args:
        paper_content: Paper text for context
        field: Field being validated
        extracted_value: Current extracted value
        allowed_values: List of allowed values for this field
        
    Returns:
        Validation prompt
    """
    return f"""Given this paper content and extracted value, select the BEST matching option.

Field: {field}
Extracted value: "{extracted_value}"

Allowed options:
{chr(10).join(f'- {v}' for v in allowed_values)}

Paper excerpt (for context):
{paper_content[:3000]}

Output ONLY the best matching option from the allowed list (exact text):
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def truncate_paper(paper_content: str, max_tokens: int = 20000) -> str:
    """
    Truncate paper content to approximate token limit.
    Rough estimate: 1 token ≈ 4 characters.
    
    Args:
        paper_content: Full paper text
        max_tokens: Target maximum tokens
        
    Returns:
        Truncated paper content
    """
    max_chars = max_tokens * 4
    if len(paper_content) <= max_chars:
        return paper_content
    
    # Keep beginning and end for context
    half = max_chars // 2
    return paper_content[:half] + "\n\n[... content truncated ...]\n\n" + paper_content[-half:]


# Keywords that indicate annotation-relevant content.
# Used by extract_annotation_chunks() to score paragraphs.
# ── Group 1: Core annotation roles & actions ──────────────────────────────────
# ── Group 2: Inter-annotator agreement metrics ────────────────────────────────
# ── Group 3: Crowdsourcing platforms & workers ────────────────────────────────
# ── Group 4: Annotation tools & interfaces ────────────────────────────────────
# ── Group 5: Recruitment, expertise & language ────────────────────────────────
# ── Group 6: Payment & compensation ───────────────────────────────────────────
# ── Group 7: Quality control & filtering ──────────────────────────────────────
# ── Group 8: Annotation process & setup ───────────────────────────────────────
# ── Group 9: Demographics ─────────────────────────────────────────────────────
_ANNOTATION_KW_RE = re.compile(
    # Group 1 — roles & actions
    r"annotat|labeler|labelling|labeling|\brater\b|\bjudge\b|\bassessor\b"
    r"|\bevaluator\b|human.{0,10}judge|human.{0,10}eval|human.{0,10}rating"
    r"|human.{0,10}assessment|human.{0,10}label|human.{0,10}annot"
    r"|subjective.{0,10}annot|ground.{0,5}truth|reference.{0,10}annot"
    r"|double.{0,10}annot|triple.{0,10}annot|re.{0,5}annot|single.{0,10}annot"
    # Group 2 — IAA metrics
    r"|inter.{0,15}annotator|inter.{0,15}rater|\biaa\b|\birr\b"
    r"|inter.{0,10}rater.{0,10}reliab|inter.{0,10}annot.{0,10}reliab"
    r"|\bkappa\b|krippendorff|fleiss|cohen.{0,10}kappa|weighted.{0,10}kappa"
    r"|scott.{0,5}pi|gwet|brennan.{0,10}prediger|\bicc\b"
    r"|intraclass.{0,10}corr|pearson|spearman|kendall"
    r"|percent.{0,10}agreement|observed.{0,10}agreement|exact.{0,10}agreement"
    r"|raw.{0,10}agreement|pairwise.{0,10}agreement|majority.{0,10}agreement"
    r"|f1.{0,10}agreement|reliability.{0,10}score|agreement.{0,10}rate"
    # Group 3 — platforms & crowd workers
    r"|crowdsourc|crowd.{0,5}worker|\bamt\b|mturk|mechanical.{0,5}turk"
    r"|prolific\b|appen\b|figure.{0,5}eight|crowdflower|clickworker"
    r"|toloka|yandex.{0,10}toloka|surge.{0,5}ai|scale.{0,5}ai"
    r"|lionbridge|transperfect|imerit|remotask|dataannotat"
    r"|turker|\bhit\b.{0,10}approv|\bhit\b.{0,10}reject|\bhit\b.{0,30}crowd"
    r"|crowd.{0,30}\bhit\b|qualification.{0,10}test|master.{0,10}worker"
    r"|\bworker\b.{0,20}paid|\bworker\b.{0,20}hired|hired.{0,10}annot"
    # Group 4 — annotation tools & interfaces
    r"|\bbrat\b|label.{0,10}studio|prodigy.{0,10}annot|doccano|webanno"
    r"|inception.{0,10}annot|annotation.{0,10}interface|annotation.{0,10}tool"
    r"|annotation.{0,10}platform|labeling.{0,10}platform|labeling.{0,10}tool"
    r"|amazon.{0,10}turk|mechanical.{0,5}turk.{0,10}interface"
    # Group 5 — recruitment, expertise & language
    r"|recruitment|annotator.{0,15}recruit|recruit.{0,15}annotator"
    r"|screeni|pre.{0,5}screen|eligibility.{0,10}criteri"
    r"|native.{0,10}speaker|non.{0,5}native|proficien|bilingual|monolingual"
    r"|mother.{0,10}tongue|first.{0,10}language|l1.{0,10}speaker"
    r"|language.{0,10}proficien|fluent.{0,10}speaker"
    r"|domain.{0,10}expert|subject.{0,10}matter.{0,10}expert|sme\b"
    r"|linguist|nlp.{0,10}expert|medical.{0,10}expert|legal.{0,10}expert"
    r"|graduate.{0,10}student|undergraduate.{0,10}student|university.{0,10}student"
    r"|lay.{0,10}annot|lay.{0,10}rater|non.{0,5}expert|crowd.{0,10}labeler"
    r"|guideline|annotation.{0,10}instruction|labeling.{0,10}instruction"
    r"|annotator.{0,10}training|pilot.{0,10}annot|trial.{0,10}task"
    r"|practice.{0,10}round|calibration.{0,10}session"
    # Group 6 — compensation & payment
    r"|compensat|hourly.{0,5}rate|hourly.{0,5}pay|per.{0,10}task.{0,10}pay"
    r"|per.{0,10}annot.{0,10}pay|payment.{0,10}rate|minimum.{0,10}wage"
    r"|fair.{0,10}wage|volunteer|unpaid|paid.{0,10}annot|paid.{0,10}worker"
    r"|annotator.{0,10}paid|participants.{0,10}paid|workers.{0,10}paid"
    r"|remuner|stipend|\$[0-9]|usd.{0,10}per|eur.{0,10}per|gbp.{0,10}per"
    # Group 7 — quality control & filtering
    r"|quality.{0,10}control|quality.{0,10}filter|attention.{0,10}check"
    r"|trap.{0,10}question|honeypot|gold.{0,10}question|gold.{0,10}standard"
    r"|spam.{0,10}detect|outlier.{0,10}remov|low.{0,10}quality.{0,10}annot"
    r"|disqualif|reject.{0,10}annot|filtering.{0,10}annot|flagging.{0,10}annot"
    r"|accept.{0,10}rate|approval.{0,10}rate|rejection.{0,10}rate"
    r"|adjudicat|reconcili|adjudicator|consensus|majority.{0,10}vot"
    r"|tie.{0,10}break|tiebreak|final.{0,10}label|final.{0,10}annot"
    # Group 8 — process & setup
    r"|annotation.{0,10}setup|annotation.{0,10}procedure|annotation.{0,10}process"
    r"|annotation.{0,10}task|annotation.{0,10}scheme|annotation.{0,10}guideline"
    r"|annotation.{0,10}round|annotation.{0,10}batch|annotation.{0,10}phase"
    r"|data.{0,10}collect|corpus.{0,10}creat|dataset.{0,10}collect"
    r"|dataset.{0,10}construct|corpus.{0,10}construct|data.{0,10}acqui"
    r"|labeling.{0,10}process|labeling.{0,10}setup|labeling.{0,10}scheme"
    r"|crowdsourc.{0,10}task|human.{0,10}annot.{0,10}stud"
    # Group 9 — demographics
    r"|demographi|age.{0,10}report|gender.{0,10}report|nation.{0,10}origin"
    r"|country.{0,10}origin|country.{0,10}resid|education.{0,10}level"
    r"|political.{0,10}orient|ethnicit|race.{0,10}annot|background.{0,10}annot",
    re.IGNORECASE,
)


def extract_annotation_chunks(paper_content: str, max_tokens: int = 8000) -> str:
    """
    Return only annotation-relevant paragraphs from *paper_content*, staying
    within *max_tokens* (≈ 4 chars/token).

    Strategy:
    1. Score each paragraph by annotation keyword hits.
    2. Detect section headers (short paragraphs ≤ 120 chars matching annotation
       keywords) and boost those plus their next 4 paragraphs (+500 score bonus)
       so entire annotation sections are pulled in as a unit.
    3. Seed with the first paragraph only (abstract/title, capped at 10% of budget)
       to orient the model without wasting budget on unrelated intro text.
    4. Greedily add highest-scoring (boosted) paragraphs until budget is full.
    5. Falls back to truncate_paper() only when NO annotation keywords are found.

    Args:
        paper_content: Full text of the paper.
        max_tokens: Target token budget for the returned content.

    Returns:
        Focused paper excerpt containing annotation-relevant paragraphs.
    """
    max_chars = max_tokens * 4

    # Primary split: double-newline (standard paragraph boundary).
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", paper_content) if p.strip()]

    # Fallback: if fewer than 5 paragraphs found, the paper uses single newlines.
    if len(paragraphs) < 5:
        paragraphs = [p.strip() for p in paper_content.splitlines() if p.strip()]

    if not paragraphs:
        return truncate_paper(paper_content, max_tokens)

    # Raw keyword scores.
    raw_scores = [len(_ANNOTATION_KW_RE.findall(p)) for p in paragraphs]

    if not any(s > 0 for s in raw_scores):
        # No annotation-related content detected — send full (truncated) paper.
        return truncate_paper(paper_content, max_tokens)

    # Section-header detection: short paragraphs (≤ 120 chars) that match annotation
    # keywords are likely section headings (e.g. "3.2 Annotation Setup", "Annotators").
    # Pull them in with their next 4 paragraphs as a unit (the section body).
    _HEADER_MAX = 120
    section_seeded: set[int] = set()
    for i, (score, para) in enumerate(zip(raw_scores, paragraphs)):
        if len(para) <= _HEADER_MAX and score > 0:
            for j in range(i, min(i + 5, len(paragraphs))):
                section_seeded.add(j)

    # Build boosted score list.  Section-seeded paragraphs get +500 bonus so they
    # always win the greedy selection ahead of scattered keyword hits elsewhere.
    # min-score of 1 ensures zero-score section body paragraphs are not skipped.
    scored: list[tuple[int, int, str]] = []
    for i, (raw_score, para) in enumerate(zip(raw_scores, paragraphs)):
        if i in section_seeded:
            boosted = max(raw_score, 1) + 500
        else:
            boosted = raw_score
        scored.append((i, boosted, para))

    # Seed: include only the first paragraph (abstract/title), capped at 10% of
    # the budget — just enough to orient the model to the paper topic without
    # wasting tokens on unrelated introductory text.
    seed_cap = max_chars // 10
    selected: list[tuple[int, str]] = []
    used_chars = 0
    selected_indices: set[int] = set()
    if paragraphs:
        para = paragraphs[0][:seed_cap]
        selected.append((0, para))
        selected_indices.add(0)
        used_chars += len(para) + 2  # +2 for \n\n separator

    # Greedy fill: sort by boosted score DESC, position ASC as tie-breaker.
    # Truncate oversized paragraphs rather than skipping them entirely.
    for i, score, para in sorted(scored, key=lambda x: (-x[1], x[0])):
        if score == 0:
            break
        if i in selected_indices:
            continue
        remaining = max_chars - used_chars
        if remaining <= 200:
            break
        if len(para) > remaining:
            para = para[:remaining - 2]
        selected.append((i, para))
        selected_indices.add(i)
        used_chars += len(para) + 2

    # Restore original document order.
    selected.sort(key=lambda x: x[0])
    return "\n\n".join(text for _, text in selected)


# =============================================================================
# EMBEDDING-BASED ANNOTATION CHUNK RETRIEVAL
# =============================================================================

# Annotation-focused query strings — one per major taxonomy dimension.
# The embedding model scores each paragraph against ALL queries and takes the
# max, so a paragraph relevant to ANY field will be surfaced.
_ANNOTATION_QUERIES = [
    "inter-annotator agreement kappa alpha reliability score value",
    "crowdsourcing platform MTurk Prolific Appen Toloka crowd workers recruited",
    "annotation guideline instruction codebook released appendix URL",
    "annotator training practice session calibration pilot task",
    "quality control filtering attention check gold question honeypot",
    "disagreement resolution majority voting adjudication consensus",
    "number of annotators per item total annotators dataset size",
    "total number of annotated items sentences documents labeled",
    "items per annotator assigned workload annotation batch",
    "annotator payment compensation hourly rate per task paid",
    "annotator demographics age gender nationality education background",
    "expertise level domain expert linguist student crowd layperson",
    "language proficiency native speaker fluency annotation task",
    "recruitment authors students research assistants hired annotators",
    "annotation procedure setup task description experiment design",
]

# Module-level cache so the model is loaded only once per process.
# Override _EMBEDDING_MODEL_NAME before importing if you want a different model.
# Any sentence-transformers-compatible model works here.  Good alternatives:
#   "BAAI/bge-base-en-v1.5"          (larger, better, slower)
#   "sentence-transformers/all-MiniLM-L6-v2"  (very fast, slightly weaker)
#   "colbert-ir/colbertv2.0"          (via ragatouille, late-interaction)
_EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
_embedding_model_cache = None


def _get_embedding_model():
    """Lazy-load and cache the SentenceTransformer model."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding-based retrieval. "
                "Install with: pip install sentence-transformers"
            ) from exc
        _embedding_model_cache = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedding_model_cache


def extract_annotation_chunks_embedding(
    paper_content: str,
    max_tokens: int = 8000,
) -> str:
    """
    Return annotation-relevant paragraphs from *paper_content* using dense
    vector retrieval rather than keyword matching.

    Strategy
    --------
    1. Split paper into paragraphs (double-newline boundaries; fallback to
       single-newline if fewer than 5 paragraphs are found).
    2. Pre-filter candidate paragraphs with the existing keyword regex so we
       don't embed thousands of irrelevant passages on very long papers.  If
       NO keyword hits are found the full paragraph list is used (so the
       embeddings alone drive selection).
    3. Embed all candidate paragraphs and 15 annotation-focused query strings
       with a bi-encoder model (default: BAAI/bge-small-en-v1.5).
    4. Score each paragraph as MAX cosine similarity across all query
       embeddings — a paragraph is relevant if it covers *any* taxonomy field.
    5. Seed with the first paragraph (title/abstract), capped at 10 % of the
       token budget, to orient the model to the paper without wasting space.
    6. Greedily add highest-scoring paragraphs (truncating if needed) until
       the token budget is exhausted.
    7. Restore original document order before returning.

    Falls back to the regex-based :func:`extract_annotation_chunks` if
    ``sentence-transformers`` or ``numpy`` are not installed.

    Parameters
    ----------
    paper_content:
        Full text of the paper.
    max_tokens:
        Approximate token budget for the returned content (1 token ≈ 4 chars).

    Returns
    -------
    str
        Focused paper excerpt containing annotation-relevant paragraphs.
    """
    try:
        import numpy as np
        model = _get_embedding_model()
    except ImportError:
        return extract_annotation_chunks(paper_content, max_tokens)

    max_chars = max_tokens * 4

    # ── 1. Split into paragraphs ──────────────────────────────────────────────
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", paper_content) if p.strip()]
    if len(paragraphs) < 5:
        paragraphs = [p.strip() for p in paper_content.splitlines() if p.strip()]
    if not paragraphs:
        return truncate_paper(paper_content, max_tokens)

    # ── 2. Keyword pre-filter ─────────────────────────────────────────────────
    raw_hits = [len(_ANNOTATION_KW_RE.findall(p)) for p in paragraphs]
    candidate_indices = [i for i, h in enumerate(raw_hits) if h > 0]
    if not candidate_indices:
        # No keyword hits — embed everything and let vectors decide.
        candidate_indices = list(range(len(paragraphs)))

    candidate_paragraphs = [paragraphs[i] for i in candidate_indices]

    # ── 3. Embed paragraphs and queries ───────────────────────────────────────
    # BGE models work best with this instruction prefix on the query side.
    query_prefix = "Represent this sentence for searching relevant passages: "
    query_embeddings = model.encode(
        [query_prefix + q for q in _ANNOTATION_QUERIES],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    para_embeddings = model.encode(
        candidate_paragraphs,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )

    # ── 4. Score: max cosine similarity across all queries ────────────────────
    # Both embeddings are L2-normalised so dot product == cosine similarity.
    # sim_matrix shape: (n_candidates, n_queries)
    sim_matrix = np.dot(para_embeddings, query_embeddings.T)
    scores = sim_matrix.max(axis=1)  # (n_candidates,)

    indexed_scores = [
        (candidate_indices[i], float(scores[i]), paragraphs[candidate_indices[i]])
        for i in range(len(candidate_indices))
    ]

    # ── 5. Seed with title / abstract paragraph ───────────────────────────────
    seed_cap = max_chars // 10
    selected: list[tuple[int, str]] = []
    selected_set: set[int] = set()
    used_chars = 0
    if paragraphs:
        seed_para = paragraphs[0][:seed_cap]
        selected.append((0, seed_para))
        selected_set.add(0)
        used_chars += len(seed_para) + 2

    # ── 6. Greedy fill ────────────────────────────────────────────────────────
    for orig_idx, score, para in sorted(indexed_scores, key=lambda x: (-x[1], x[0])):
        if orig_idx in selected_set:
            continue
        remaining = max_chars - used_chars
        if remaining <= 200:
            break
        if len(para) > remaining:
            para = para[:remaining - 2]
        selected.append((orig_idx, para))
        selected_set.add(orig_idx)
        used_chars += len(para) + 2

    # ── 7. Restore document order ─────────────────────────────────────────────
    selected.sort(key=lambda x: x[0])
    return "\n\n".join(text for _, text in selected)


def get_system_prompt() -> str:
    """
    Get the system prompt for extraction tasks.
    
    Returns:
        System prompt string
    """
    return """You are an expert at extracting structured information from scientific papers about human annotation experiments.
You follow instructions PRECISELY and output ONLY valid JSON — no markdown fences, no commentary.

CRITICAL: Always write the top-level "analysis" field FIRST in your JSON, before the "experiments" array.
The analysis should be 1–2 sentences summarising: how many experiments you found, who the annotators are,
what IAA metrics/values are reported, and compensation type. Writing this first grounds your extraction
and reduces field-level errors.

Field-level defaults when information is NOT found in the paper:
- guidelines_released → "No"
- annotator_training → "No"
- ALL *_reported demographics fields (age, gender, nation_origin, nation_residence, education, political_orientation) → "No"
- quality_control_filtering → "No"
- disagreement_resolution_adjudication → "None"
- crowd_qualifications_screening → "NA" (if not crowdsourcing) or "No" (if crowdsourcing)
- recruitment, language_proficiency, expertise_level → "Not reported"
- reported_compensation → "Not reported"
- human_evaluated_sample → "Not reported"
- workload fields (annotators_per_item, total_annotators, total_annotated_items, items_per_annotator) → "Not reported"
- iaa_metric_name → ["Not reported"]
- iaa_value → ["Not reported"]
- papers_topic → pick the most relevant topic(s)
- intended_use → ["None of the above"]

Use broad recruitment categories: "Crowdsourcing", "Authors", "Other", "Mixed", or "Not reported" — do NOT use platform-specific names.
NEVER invent or calculate values — only extract what the paper explicitly states.
Demographics fields are "Yes" ONLY when actual values (numbers, percentages, distributions) are explicitly reported."""
