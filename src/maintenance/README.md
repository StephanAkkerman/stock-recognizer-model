# Maintenance Scripts

Place scripts here that repair data, enforce annotation policy, or patch
generated artifacts.

Current scripts:

- `audit_train_labels.py`
- `backfill_metadata.py`
- `fix_label_policy.py`
- `patch_labeled_gaps.py`
- `patch_test_labels.py`
- `split_test_set.py`

These scripts are intentionally operational and should stay separate from the
training/evaluation core.