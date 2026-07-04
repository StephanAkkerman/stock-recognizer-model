# Utils Layout

This directory groups the repository's standalone helper scripts by task.

## Current layout

```text
utils/
  augmentation/
    augment_data.py

  hf/
    push_model_to_hf.py
    push_to_hf.py

  labeling/
    check_labeled_duplicates.py
    cleaner.py
    deduplicate_csv.py
    fix_annotations.py
    scan_missing_entities.py

  scraping/
    scraper.py
    mine_hard_negatives.py
    mine_abbrev_negatives.py

  synthetic/
    auto_label.py
```

The top-level `utils/` directory is intentionally empty of scripts now. New
helpers should go into the category that matches what they do rather than
sitting at the root.
