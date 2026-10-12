"""Train a GLiNER2 LoRA adapter on the Label Studio exports in data/labeled.

    python -m src.core.train                                   # default base (large)
    python -m src.core.train --base-model gliner2.5-small-v1
    python -m src.core.train --base-model all                  # every registered base

Refreshes the held-out test split first, then for each base model trains an
adapter, writes ``models/reddit_adapter_<base>_vN/{best,final}`` plus
``training_metadata.json`` and ``recognizer_config.json``, and benchmarks the
result against the test set. All bases share the same data, split and seed so
their scores are comparable.
"""

import argparse
import gc
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from gliner2.processor import WhitespaceTokenSplitter
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from rich.console import Console

from src.core.models import (
    ADAPTER_PREFIX,
    BASE_MODELS,
    DEFAULT_BASE,
    TRAIN_OVERRIDES,
    adapter_dir_name,
    adapter_label,
    load_extractor,
    parse_adapter_dir,
    resolve_bases,
)

console = Console()

SEED = 42
EARLY_STOPPING = True
EPOCHS = 10
VAL_FRACTION = 0.15
DEFAULT_THRESHOLD = 0.75
# Word-token cap per training sample (punctuation counts). 256 cut the tail off
# 150-word chunks, so entities there could not be aligned and collation raised.
MAX_LEN = 384

ENTITY_DESCRIPTIONS = {
    "ticker": (
        "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign "
        "(e.g., $AAPL, TSLA). MUST NOT be option expiry months (JAN, FEB, MAR, APR, MAY, JUN, "
        "JUL, AUG, SEP, OCT, NOV, DEC), Reddit slang acronyms (DYOR, NFA, DD, YOLO, FOMO, "
        "TLDR, IMO, AMA, NGL, AH), or financial regulatory bodies (SEC, FINRA, DTC, CSRC, RSA)."
    ),
    "company": (
        "The name of a corporation, hedge fund, or business entity. MUST NOT be an uppercase "
        "ticker symbol, an index, or generic finance terms."
    ),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_next_version(models_dir="./models", base=DEFAULT_BASE):
    """Next adapter version number for `base` (versions count per base model).

    Legacy ``reddit_adapter_vN`` folders count as the large base's versions.
    """
    os.makedirs(models_dir, exist_ok=True)
    max_v = 0
    for adapter in glob.glob(os.path.join(models_dir, f"{ADAPTER_PREFIX}*")):
        parsed = parse_adapter_dir(os.path.basename(adapter))
        if parsed and parsed[0] == base:
            max_v = max(max_v, parsed[1])
    return max_v + 1


def chunk_text_with_overlap(text, chunk_word_size=150, overlap_words=40):
    """Splits text into overlapping chunks without splitting words."""
    words = text.split()
    chunks = []
    if len(words) <= chunk_word_size:
        return [text]
    step_size = chunk_word_size - overlap_words
    for i in range(0, len(words), step_size):
        chunk_text = " ".join(words[i : i + chunk_word_size])
        chunks.append(chunk_text)
        if i + chunk_word_size >= len(words):
            break
    return chunks


def entity_in_chunk(entity_text, chunk):
    """Token-aware containment check: matches whole tokens only, $-prefixed tickers included."""
    if re.search(rf"(?<!\w){re.escape(entity_text)}(?!\w)", chunk) is None:
        return False
    return _contains_tokens(chunk, entity_text)


def _word_tokens(text):
    return [tok for tok, _, _ in WhitespaceTokenSplitter()(text)]


def _contains_tokens(chunk, entity_text):
    """Whether `entity_text` is a run of whole splitter tokens in `chunk`.

    The boundary-architecture bases (GLiNER2.5) raise on a labeled entity they
    cannot find as whole tokens, so such mentions must be dropped before
    training, as do entities past the MAX_LEN truncation point. Cashtags are fine: ``$AAPL`` tokenizes as ``$`` + ``aapl``.
    """
    needle = _word_tokens(entity_text)
    hay = _word_tokens(chunk)[:MAX_LEN]  # the trainer truncates to MAX_LEN tokens
    n = len(needle)
    return n > 0 and any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def task_to_samples(task):
    """Converts one Label Studio task into one or more chunked training samples."""
    full_text = task["data"]["text"]
    results = task["annotations"][0].get("result", [])

    doc_entities = {}
    for r in results:
        if r.get("type") != "labels":
            continue
        val = r["value"]
        label = val["labels"][0]
        entity_text = full_text[val["start"] : val["end"]]
        doc_entities.setdefault(label, set()).add(entity_text)

    chunks = chunk_text_with_overlap(full_text, chunk_word_size=150, overlap_words=40)

    samples = []
    for chunk in chunks:
        chunk_entities_dict = {}
        for label, entity_set in doc_entities.items():
            valid_ents = sorted(e for e in entity_set if entity_in_chunk(e, chunk))
            if valid_ents:
                chunk_entities_dict[label] = valid_ents

        samples.append(
            {
                "input": chunk,
                "output": {
                    "entities": chunk_entities_dict,
                    "entity_descriptions": ENTITY_DESCRIPTIONS,
                    # Keeps empty-entity chunks from being dropped by the trainer
                    # so the model sees real negatives. See discussion of the
                    # `valid: yes` degeneracy in the train.py review.
                    "classifications": [
                        {
                            "task": "valid",
                            "labels": ["yes"],
                            "true_label": ["yes"],
                        }
                    ],
                },
            }
        )
    return samples


def _load_tasks(folder_path):
    """Yields (task, source_file) for every usable annotation in a folder."""
    if not folder_path or not os.path.isdir(folder_path):
        return
    for fp in glob.glob(os.path.join(folder_path, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            ls_data = json.load(f)
        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue
            yield task, fp


def parse_all_labeled_data(
    labeled_folder,
    augmented_folder=None,
    test_folder=None,
    val_fraction=VAL_FRACTION,
    seed=SEED,
):
    """Loads originals from `labeled_folder` and (optionally) augmented variants
    from `augmented_folder`, splitting by source task id so augmented copies of
    validation tasks never leak into training.

    If `test_folder` is given, every task id present there is excluded from
    train and val — including augmented duplicates that carry the same id —
    so the held-out test set never contaminates training.
    """
    test_ids = set()
    if test_folder and os.path.isdir(test_folder):
        for task, _ in _load_tasks(test_folder):
            test_ids.add(task["id"])

    originals = [
        (t, fp) for t, fp in _load_tasks(labeled_folder) if t["id"] not in test_ids
    ]

    # Split task ids deterministically.
    rng = random.Random(seed)
    ids = sorted({task["id"] for task, _ in originals})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    train_samples, val_samples = [], []

    for task, _ in originals:
        tid = task["id"]
        if tid in val_ids:
            val_samples.extend(task_to_samples(task))
        elif tid in train_ids:
            train_samples.extend(task_to_samples(task))

    # Augmented: training only, and only for tasks whose source is in train_ids.
    # The `test_ids` guard catches augmented variants of test tasks that still
    # live on disk (e.g. `augmented_m*_labeled_300.json` after the 300-file move).
    for task, _ in _load_tasks(augmented_folder):
        tid = task["id"]
        if tid in test_ids:
            continue
        if tid in train_ids:
            train_samples.extend(task_to_samples(task))

    return train_samples, val_samples


def _git_commit():
    """Best-effort: return current git HEAD, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        )
        return out.decode().strip()
    except Exception:
        return None


def _summarise_labeled_file(fp):
    """Read a Label Studio export, return tasks/entities/sha1 summary."""
    with open(fp, "rb") as f:
        raw = f.read()
    sha1 = hashlib.sha1(raw).hexdigest()[:12]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    n_tasks = 0
    n_ents = 0
    for t in data if isinstance(data, list) else []:
        if not t.get("annotations"):
            continue
        if t["annotations"][0].get("was_cancelled"):
            continue
        n_tasks += 1
        n_ents += sum(
            1
            for r in t["annotations"][0].get("result", [])
            if r.get("type") == "labels"
        )
    return {
        "name": os.path.basename(fp),
        "tasks": n_tasks,
        "entities": n_ents,
        "sha1": sha1,
    }


def gather_training_metadata(
    base,
    config,
    train_count,
    val_count,
    effective_batch_size,
    labeled_folder,
    test_folder,
    augmented_folder,
):
    """Snapshot everything needed to reproduce this training run.

    Written next to the adapter as `training_metadata.json` so benchmark.py
    can later merge it into the per-adapter params it persists — keeping
    full training context attached to each version even when re-benchmarked
    months later.
    """
    labeled_files = []
    for fp in sorted(glob.glob(os.path.join(labeled_folder, "*.json"))):
        name = os.path.basename(fp)
        if name.endswith(".bak"):
            continue
        if "negatives" in name:
            continue
        info = _summarise_labeled_file(fp)
        if info:
            labeled_files.append(info)

    test_files = []
    test_held_out = 0
    if os.path.isdir(test_folder):
        for fp in sorted(glob.glob(os.path.join(test_folder, "*.json"))):
            info = _summarise_labeled_file(fp)
            if info:
                test_files.append(info)
                test_held_out += info["tasks"]

    # Pull augmentation knobs at training time. These are module-level
    # constants in utils.augmentation.augment_data; if the user mutated them between
    # `python utils/augmentation/augment_data.py` and `python src/core/train.py`, the
    # captured values reflect the latter — best-effort, not authoritative.
    aug = {}
    try:
        from utils.augmentation.augment_data import (
            CASHTAG_FORMAT_PROB,
            EXPANDED_POOL_WEIGHT,
        )

        aug["expanded_pool_weight"] = EXPANDED_POOL_WEIGHT
        aug["cashtag_format_prob"] = CASHTAG_FORMAT_PROB
    except ImportError:
        pass
    aug["augmented_files"] = (
        len(glob.glob(os.path.join(augmented_folder, "*.json")))
        if augmented_folder and os.path.isdir(augmented_folder)
        else 0
    )

    negatives = None
    for fp in sorted(glob.glob(os.path.join(labeled_folder, "negatives*.json"))):
        info = _summarise_labeled_file(fp)
        if info:
            negatives = info
            break

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": SEED,
        "base_model": BASE_MODELS[base]["hf_id"],
        "config": {
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "effective_batch_size": effective_batch_size,
            "encoder_lr": config.encoder_lr,
            "task_lr": config.task_lr,
            "max_grad_norm": config.max_grad_norm,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "early_stopping": config.early_stopping,
            "early_stopping_patience": getattr(config, "early_stopping_patience", None),
            "val_fraction": VAL_FRACTION,
        },
        "data": {
            "labeled_files": labeled_files,
            "test_files": test_files,
            "test_held_out_tasks": test_held_out,
            "train_samples": train_count,
            "val_samples": val_count,
        },
        "augmentation": aug,
        "negatives": negatives,
    }


def _finalize_adapter_files(output_dir, base, version):
    """Post-process every saved checkpoint so it can be published as-is.

    Sets ``task_type`` to TOKEN_CLS in ``adapter_config.json`` (Hub compliance)
    and drops a ``recognizer_config.json`` recording the base model and default
    threshold, enough for a consumer to load the adapter without this repo.
    """
    for sub in ("best", "final"):
        folder = os.path.join(output_dir, sub)
        if not os.path.isdir(folder):
            continue
        adapter_config_path = os.path.join(folder, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            adapter_config["task_type"] = "TOKEN_CLS"
            with open(adapter_config_path, "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2)
        with open(
            os.path.join(folder, "recognizer_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "base_model": BASE_MODELS[base]["hf_id"],
                    "adapter_version": version,
                    "labels": list(ENTITY_DESCRIPTIONS),
                    "entity_descriptions": ENTITY_DESCRIPTIONS,
                    "threshold": DEFAULT_THRESHOLD,
                },
                f,
                indent=2,
            )


def train_one(base, compile_model=True):
    """Train, finalize and benchmark one adapter for `base`."""
    from src.core.benchmark import locate_adapter_weights

    next_version = get_next_version(base=base)
    output_dir = f"./models/{adapter_dir_name(base, next_version)}"
    console.print(
        f"[bold cyan]Training {BASE_MODELS[base]['hf_id']} -> v{next_version}[/bold cyan]"
    )

    labeled_folder = "data/labeled"
    augmented_folder = "data/augmented"
    test_folder = "data/test"
    train_data, val_data = parse_all_labeled_data(
        labeled_folder, augmented_folder, test_folder=test_folder
    )
    console.print(
        f"[bold green]Train: {len(train_data)} samples | Val: {len(val_data)} samples[/bold green]"
    )

    hp = {
        "batch_size": 4,
        "lora_rank": 32,
        "encoder_lr": 2e-5,
        "task_lr": 5e-4,
        "epochs": EPOCHS,
        **TRAIN_OVERRIDES[base],
    }
    effective_batch_size = hp["batch_size"] * 2

    base_model = load_extractor(base)
    model = torch.compile(base_model) if compile_model else base_model

    config = TrainingConfig(
        output_dir=output_dir,
        experiment_name=f"fintwit_lora_{base}_v{next_version}",
        num_epochs=hp["epochs"],
        batch_size=hp["batch_size"],
        max_len=MAX_LEN,
        gradient_accumulation_steps=effective_batch_size // hp["batch_size"],
        encoder_lr=hp["encoder_lr"],
        task_lr=hp["task_lr"],
        max_grad_norm=1.0,
        use_lora=True,
        lora_r=hp["lora_rank"],
        lora_alpha=hp["lora_rank"] * 2,
        lora_dropout=0.1,
        lora_target_modules=["encoder"],
        save_adapter_only=True,
        fp16=False,
        bf16=torch.cuda.is_available(),
        seed=SEED,
        # Early stopping IS necessary here. v9 (25 epochs, no early stop, w/
        # 55 hard negatives) collapsed to "predict nothing" by step ~6000 —
        # all intermediate checkpoints from 6000 onward score 0% F1. The
        # negatives dominate the gradient once training runs long enough.
        early_stopping=EARLY_STOPPING,
        early_stopping_patience=5,
        # Windows spawns DataLoader workers, which cannot pickle the tokenizer.
        **({"num_workers": 0} if os.name == "nt" else {}),
    )

    trainer = GLiNER2Trainer(model=model, config=config)
    trainer.train(train_data=train_data, eval_data=val_data)
    console.print(f"[bold green]Adapter saved to {output_dir}/final/[/bold green]")

    _finalize_adapter_files(output_dir, base, next_version)

    # Snapshot the full training context next to the adapter so re-benchmark
    # runs months later still know which data + config produced this version.
    metadata = gather_training_metadata(
        base,
        config,
        train_count=len(train_data),
        val_count=len(val_data),
        effective_batch_size=effective_batch_size,
        labeled_folder=labeled_folder,
        test_folder=test_folder,
        augmented_folder=augmented_folder,
    )
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    console.print(f"[cyan]Wrote training metadata to {metadata_path}[/cyan]")

    if locate_adapter_weights(output_dir) is None:
        console.print(
            f"[yellow]No adapter weights found under {output_dir}; "
            "skipping post-train benchmark.[/yellow]"
        )
        return

    # Free the training copy before the benchmark loads its own.
    del trainer, model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # A fresh process rather than benchmark_adapter(): state left behind by
    # training (torch.compile, trainer hooks) has been seen to poison inference.
    console.print(f"[cyan]Benchmarking {adapter_label(base, next_version)}...[/cyan]")
    result = subprocess.run(
        [sys.executable, "-m", "src.core.benchmark", "--base-model", base],
        check=False,
    )
    if result.returncode != 0:
        console.print(
            "[bold red]Benchmark failed; rerun "
            f"`python -m src.core.benchmark --base-model {base}`.[/bold red]"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE,
        help="Slug, Hub id, comma-separated list, or 'all' (default: %(default)s).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (e.g. on CPU or if compilation fails).",
    )
    args = parser.parse_args()
    bases = resolve_bases(args.base_model)

    if not torch.cuda.is_available():
        console.print("[bold red]CUDA is not available. Exiting.[/bold red]")
        return

    # Refresh the held-out test split before loading training data so any
    # newly added labeled_*.json gets its ~15% stratified slice held out
    # automatically. Deterministic via the same SEED, so every base trains on
    # the same data and is scored on the same test set.
    from src.maintenance.split_test_set import run as refresh_test_split

    console.print("[bold cyan]Refreshing stratified test split...[/bold cyan]")
    refresh_test_split(seed=SEED)

    for base in bases:
        set_seed(SEED)
        train_one(base, compile_model=not args.no_compile)


if __name__ == "__main__":
    main()
