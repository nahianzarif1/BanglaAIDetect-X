# Development record: October 13–25, 2026

This release collects the completed local work from this development window. The
entries below describe verified milestones, not a daily activity record. Commits
use their actual creation timestamps; no backdated daily commits are implied.

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
