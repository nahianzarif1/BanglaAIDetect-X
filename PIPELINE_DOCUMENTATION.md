# Pipeline

1. `load_custom_dataset.py` imports labeled Excel/CSV, retaining source/generator
   identities and recording the source hash.
2. `src/data/cleaner.py` normalizes Bangla, validates IDs/labels, and logs excluded
   short, unsupported, duplicate, and conflicting-label documents.
3. `src/data/grouping.py` groups known topics, parent/prompt IDs, duplicates, and
   high-overlap token-shingle candidates; `splitter.py` makes disjoint partitions.
4. `validator.py` validates the actual three split files and fails on hard errors.
5. `src/experiment.py` trains selected classical or transformer models, selects
   thresholds/weights/default on validation, and evaluates independent test rows.
6. `models/registry.json` declares trained artifacts. `src/inference.py` validates
   inputs and loads the selected model. `app/app.py` exposes analysis and research.

No topics are invented. Missing real topics are disclosed as a limitation.
Generator subgroup reporting is distinct from unseen-generator evaluation.
The default transformer method is a frozen encoder plus supervised linear head;
full encoder fine-tuning is explicitly selected with `--transformer-mode finetune`.

See [README.md](README.md) for commands and [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)
for evaluation definitions, provenance, and research limitations.
