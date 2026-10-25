# Research protocol and current limits

## What this implementation establishes

The system provides seven complementary classical baselines, feature ablations,
four supported pretrained transformer checkpoints, a validation-weighted
ensemble, and a common trained-inference interface. Transformer experiments have
explicitly different frozen-encoder and end-to-end fine-tuning modes.

Train-only group folds calibrate SVM and fusion probabilities. Validation data
selects each operating threshold, ensemble weights, and the default model.
The independent test partition provides accuracy, balanced accuracy, AI precision,
recall/F1, macro-F1, MCC, ROC-AUC, PR-AUC, Brier score, log loss, ECE, human false
positive rate, AI false negative rate, and fixed-order confusion matrices.
95% intervals resample document groups with replacement, not individual rows.
Single-class subgroup ranking metrics are undefined and exported as JSON null.

Run records preserve configuration, Python-package versions, encoder checkpoint
revision, source-file hashes, split hashes, per-document predictions, and errors.
All experiments use the same normalization and inference length policy as the app.
Exported error tables reference IDs; join them to the split CSV to inspect text.

## Dataset audit

The supplied workbook contains 5,999 samples with source and generator names,
but no semantic topic, prompt-pair, author, collection date, or verified demographic
annotations. The previous importer invented `topic_<row modulo 100>` labels and
discarded generator identities. Those labels were removed.

The current importer preserves actual source names, records the workbook hash,
and rejects unknown labels. Cleaning logs short/unsupported passages, removes
exact duplicates, and excludes every version of a document with conflicting
labels. Related topic/parent/prompt IDs, when supplied, form common split groups.
Token-trigram MinHash LSH candidates with verified Jaccard overlap of at least
0.9 are also grouped. LSH is approximate and does not detect all paraphrases.
The split searches deterministic allocations for class coverage and approximate
70/15/15 balance; this does not involve model predictions.

When real topic IDs are absent, the evaluation is **document/near-duplicate
held out**. It must not be described as topic-held-out. Missing writer subgroup
values remain unknown; the current corpus cannot support demographic fairness
claims. News/Wikipedia human sources can be confounded with AI generation style.
High accuracy on this corpus can reflect that source separation rather than
robust detection across genres and new generators.

## Experiments needed for a publication claim

1. Verify the original dataset's ownership, annotation process, collection dates,
   generator versions, prompts, sampling procedure, and redistribution terms.
   The local filename alone does not establish IEEE DataPort provenance.
2. Collect human and AI passages matched by topic, genre, length, prompt, and
   source period. Preserve parent/prompt/topic IDs before splitting.
3. Run full fine-tuning and frozen-encoder methods as distinct comparisons on
   identical partitions. Repeat training with at least three declared seeds.
   Archive each run; use identical predeclared settings and avoid repeatedly
   modifying the system after inspecting test errors.
4. Reserve an independent external corpus and a generator completely unseen
   during training/validation. Report each unseen generator separately. Current
   generator subgroup scores are in-distribution diagnostics, not this experiment.
5. Test human-edited AI, paraphrases, spelling changes, different passage lengths,
   and shifted domains. Retain all edits of a parent passage in one group.
6. Compare paired predictions, report uncertainty in model differences, and
   document compute/memory costs. A higher single-run F1 is insufficient evidence
   of a meaningful improvement. Ablations should include lexical features,
   stylometry, individual encoders, and ensemble components.
7. Report human false positives and realistic class prevalence. Define any
   deployment threshold/FPR target using validation data before external testing.
8. Write the contribution around a tested hypothesis and demonstrated robustness.
   A model collection and application alone do not guarantee Q1 publication.

Useful commands:

```bash
python run_pipeline.py --all-models --transformer-mode finetune --seed 42
# Use separate config files with distinct paths for independent-seed protocols.
python -m src.evaluate_external --dataset external.csv --model banglabert
```

Changing split seed changes partitions. For a controlled training-seed comparison,
keep the split files fixed with `--reuse-splits --seed <training_seed>`. Do not use
that option to pretend a different split was regenerated. Configurations and split
hashes in the experiment record resolve this distinction.

## Primary references

- [BanglaBERT model card and required normalizer](https://huggingface.co/csebuetnlp/banglabert)
- [BanglaBERT paper](https://aclanthology.org/2022.findings-naacl.98/)
- [BUET normalization implementation](https://github.com/csebuetnlp/normalizer)
- [mBERT model card](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [XLM-R model card](https://huggingface.co/FacebookAI/xlm-roberta-base)

Pretrained language models are not authorship detectors until the supervised head
or full classifier has been trained. Model-specific licenses apply to downloaded
weights; record them when preparing a public research release.

## Selective decisions and paired comparisons

Primary test metrics use a forced binary decision at the locked operating
threshold. The app labels scores within the configured margin as uncertain.
Selective metrics disclose coverage, uncertain counts, and accuracy among
confident decisions; do not report these without their coverage denominator.

`python -m src.paired_comparison` compares saved test predictions against the
single model selected by validation macro-F1. It reports accuracy/macro-F1
changes, paired document-group bootstrap intervals, and group sign-permutation
p-values with Holm adjustment. The test assumes exchangeability under the null;
its results remain corpus-specific. Predeclare the comparator and hypothesis for
a confirmatory paper rather than selecting them after viewing test results.

External evaluation rejects known training/validation exact-text overlap by
default. `--allow-overlap` is for disclosed diagnostics, not independent testing.
Paraphrase and near-duplicate overlap require an additional audit.
