# Who annotates in NLP?

Code and data for **Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025** ([arXiv:2606.02255](https://arxiv.org/abs/2606.02255)).

Most NLP papers that rely on human annotation tell you how many items were labelled and roughly who labelled them. Far fewer say whether the annotators were trained, how well they knew the language, what they were paid, or how disagreements were settled. We wanted to measure that gap properly, so we read papers from ACL, EMNLP, NAACL, EACL, AACL and TACL (2018 to 2025) and coded each annotation task against a 26-category taxonomy.

The study has two datasets:

| Dataset | Papers | Tasks | How it was built |
| --- | --- | --- | --- |
| Annotated-gold | 41 | 72 | Two human annotators per task, then adjudication |
| Annotated-llm | 1,603 | 2,667 | Gemini-3.1-Pro, validated against Annotated-gold |

Gemini-3.1-Pro agrees with the adjudicated gold labels about as well as our annotators agree with each other (79.9% exact match and Krippendorff's α = 0.606, against 79.2% and α = 0.585 for human pairs). That level of agreement is why we trusted it to code the remaining 1,603 papers.

A few things we found:

- Recruitment (90.4%), annotator expertise (86.5%) and item counts (86.0%) are usually reported. Training (18.7%), language proficiency (24.0%) and released guidelines (34.1%) usually are not.
- Reporting improved steadily from 2018 to 2021. After the ACL Responsible NLP checklist arrived in 2022, the trend flattened.
- Papers that build a resource report much more than papers that use humans to evaluate model output, and the gap has held for all eight years.

## Citation

```bibtex
@misc{kunilovskaya2026annotates,
  title         = {Who Annotates in {NLP}? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025},
  author        = {Kunilovskaya, Maria and Bhatia, Gagan and Albertelli, Lisa Sophie and Chen, Yanran and Greisinger, Christian and Kiefer, Lotta and Leiter, Christoph and Roy, Subhadeep and Achamaleh, Tewodros and Manzoor, Muhammad Arslan and Pohl, Sebastian and Hou, Yufang and Eger, Steffen},
  year          = {2026},
  eprint        = {2606.02255},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.02255}
}
```

