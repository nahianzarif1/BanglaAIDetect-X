# Project history: October 1–25, 2026

The history was reorganized on October 25, 2026 into **30 total commits**. The
first two preserve the original project setup and initial application trees;
the remaining 28 split the completed release into focused changes. Author and
committer dates across October 1–25 were assigned retrospectively at the
maintainer's request. They organize the release and do not establish when each
change was originally written. The verified experiment and verification dates
below remain unchanged. The prior published history is retained on the backup
branch `codex/history-before-30-20261025`.

## October 13: application and experiment work

- Added seven classical baselines/ablations, frozen-encoder classifiers for
  BanglaBERT, mBERT and XLM-R, and a validation-weighted ensemble.
- Implemented optional BanglaBERT small and end-to-end transformer fine-tuning.
  Neither has a completed full-size experiment in the documented results.
- Rebuilt ingestion, normalization, duplicate/conflict handling, document-group
  splitting, validation-only selection, and experiment provenance.
- Added calibrated scoring, uncertainty handling, batch analysis, model
  comparisons, audits, and the Streamlit experiment explorer.
- Added group bootstrap intervals, paired group comparisons, external-corpus
  evaluation, and compatible artifact recovery after storage failures.
- Completed the seed-42 experiment identified as
  `20261013T155905925965Z`: 5,989 cleaned documents, 897 test documents,
  and 11 locally available trained choices. Three full-size transformers use
  frozen embeddings and trained logistic heads.

## October 25: verification and repository release preparation

- Reverified all 11 local trained artifacts and the stored experiment results.
- Verified the automated suite: 36 passing tests, including the application,
  data protocol, model registry, and tiny offline transformer checks.
- Restarted an outdated local Streamlit process to clear stale imports and
  verified live BanglaBERT passage analysis.
- Corrected the research chart to show a single macro-F1 series instead of
  stacking accuracy and macro-F1. Added an explicit illustrative-example notice.
- Reorganized the README with setup, inputs, completed model status, measured
  results, training, recovery, independent evaluation, and troubleshooting.
- Added architecture, activity, UML class/module and logical ER diagrams, live
  application screenshots, an inference demo CSV, and public aggregate results.

The release does not claim continuous activity on October 14–24, new training on
October 25, universal detection accuracy, or readiness for journal acceptance.
The [research guide](../RESEARCH_GUIDE.md) records the remaining empirical work.

## Retrospective commit schedule

| Commit | Assigned date | Change |
|---:|---|---|
| 1 | 2026-10-01 | project structure |
| 2 | 2026-10-02 | Ai detection |
| 3 | 2026-10-03 | Add portable configuration and validated settings |
| 4 | 2026-10-04 | Declare reproducible base and optional model dependencies |
| 5 | 2026-10-05 | Preserve corpus labels and source metadata during ingestion |
| 6 | 2026-10-06 | Normalize Bangla and reject conflicting or unusable passages |
| 7 | 2026-10-07 | Group near duplicates and related documents before splitting |
| 8 | 2026-10-08 | Create deterministic group-disjoint dataset partitions |
| 9 | 2026-10-08 | Validate partition isolation and dataset schema |
| 10 | 2026-10-09 | Define classical baselines and pretrained checkpoint catalog |
| 11 | 2026-10-10 | Train seven complementary classical models and ablations |
| 12 | 2026-10-11 | Adapt baseline training to the shared experiment workflow |
| 13 | 2026-10-12 | Add frozen-encoder and fine-tuning transformer backends |
| 14 | 2026-10-13 | Load verified artifacts and memory-aware ensemble members |
| 15 | 2026-10-13 | Add calibrated metrics and document-group uncertainty intervals |
| 16 | 2026-10-14 | Validate passage inputs and expose trained-model batch inference |
| 17 | 2026-10-15 | Record reproducible model experiments and validation-only selection |
| 18 | 2026-10-16 | Provide unified training and compatible artifact recovery commands |
| 19 | 2026-10-17 | Compare paired predictions with group permutation tests |
| 20 | 2026-10-18 | Evaluate locked detectors against independent labeled corpora |
| 21 | 2026-10-18 | Add repeatable data and training protocol checks |
| 22 | 2026-10-19 | Verify classical inference, experiments and transformer save/reload |
| 23 | 2026-10-20 | Complete multi-model Streamlit passage and batch workflows |
| 24 | 2026-10-21 | Exercise application flows and run continuous integration |
| 25 | 2026-10-22 | Document publication protocol and maintained operating guides |
| 26 | 2026-10-23 | Describe architecture and passage-analysis activities |
| 27 | 2026-10-23 | Describe predictor classes and file-backed research entities |
| 28 | 2026-10-24 | Capture live application demonstrations and example batch inputs |
| 29 | 2026-10-25 | Publish aggregate measurements for eleven trained detector choices |
| 30 | 2026-10-25 | Organize operational README and retrospective development history |
