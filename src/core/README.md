# Core Training Files

These are the modules that directly control dataset construction, benchmark
caching, training, and evaluation helpers.

Current core files:

- `benchmark.py`
- `dataset_builder.py`
- `results_store.py`
- `train.py`

Keeping the core surface small makes it easier to find the code that actually
controls training and evaluation.