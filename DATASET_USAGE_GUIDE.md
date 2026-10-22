# Dataset usage

Provide CSV columns `text,label`, with labels `human` and `ai`. Recommended
metadata: `id,generator,source,genre,topic_id,parent_id,prompt_id,writer_subgroup`.
The supplied Excel columns `source_text,Source,Source Name` are also supported.

```bash
python run_pipeline.py --dataset /path/to/labeled.csv
```

All scripts used for a research run should enter through `run_pipeline.py`.
The older synthetic expansion and block-addition scripts are exploratory utilities
and are excluded from the maintained research pipeline. Do not use augmentation
before grouping/splitting or mix derivatives of test texts into training.
Unlabeled instruction/response data such as `ai_dataset.csv` must not be inferred
to be an authorship-labeled corpus solely from its filename.

See [README.md](README.md) and [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md).
