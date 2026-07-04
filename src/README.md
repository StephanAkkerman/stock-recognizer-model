# Src Layout

This directory contains the model-training pipeline plus the supporting scripts
used to inspect, repair, and benchmark the labeled data.

## Intended grouping

```text
src/
  analysis/
    error_analysis.py
    threshold_sweep.py
    validate_descriptions.py

  maintenance/
    audit_train_labels.py
    backfill_metadata.py
    fix_label_policy.py
    patch_labeled_gaps.py
    patch_test_labels.py
    split_test_set.py

  core/
    benchmark.py
    dataset_builder.py
    results_store.py
    test.py
    train.py
```

The real implementations live under `src/core/`, `src/analysis/`, and
`src/maintenance/`.