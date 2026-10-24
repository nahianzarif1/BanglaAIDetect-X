# Live application demonstration

The screenshots in [images](images) were captured from the actual local Streamlit
application on October 25, 2026. They show its responsive layout in the available
browser panel. They are screenshots, not UI mockups. The displayed examples are
illustrative passages without independently verified authorship labels.

## Reproduce the walkthrough

1. Follow the [README setup](../README.md#quick-start), train the desired models,
   then launch `streamlit run app/app.py`.
2. In **Analyze text**, select BanglaBERT and click **ChatGPT-style · শিক্ষা**,
   then **Analyze text**. The documented run returned an AI score of 99.9% for
   the 55-word example. This is an observed prediction, not proof that the example
   is AI-authored or a validated accuracy test. A fresh experiment can differ.
3. Select the calibrated linear SVM for fast batch inference. In **Batch
   analysis**, upload [examples/batch-demo.csv](examples/batch-demo.csv), click
   **Analyze batch**, and download the prediction CSV. Two illustrative passages
   are valid; the deliberately short third passage must receive a row error.
4. In **Research results**, inspect the 11-model comparison and macro-F1 chart.
   Select a model below the chart to inspect confusion, calibration, bootstrap
   intervals, and subgroup results. The reported scores concern the fixed local
   test partition, not the example passages.
5. In **Models & data**, check trained status and open the preparation audits.
   BanglaBERT small remains marked as requiring training.

| Screenshot | Captured operation |
|---|---|
| [01-text-analysis.jpg](images/01-text-analysis.jpg) | Real BanglaBERT result, input passage, model threshold and score caveat |
| [02-batch-analysis.jpg](images/02-batch-analysis.jpg) | Uploaded demo CSV, actual inference output and prediction download |
| [03-research-results.jpg](images/03-research-results.jpg) | Stored test metrics and corrected macro-F1 comparison |
| [04-model-inventory.jpg](images/04-model-inventory.jpg) | Supported/trained model inventory and data audits |

The [aggregate experiment summary](results/experiment-summary.json) records the
experiment identity, counts, split hashes and measured results. Trained weights
are stored locally and excluded from Git; a new clone must train models before
reproducing inference. No unrelated personal documents or raw passage predictions
are included in the documentation update.
