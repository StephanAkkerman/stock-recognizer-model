import argparse
import glob
import json
import os
import random
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from rich.console import Console

console = Console()

# Valid entity labels for validation
VALID_LABELS = {"ticker", "company"}


def validate_entity(text, entity):
    """Validate that entity boundaries match the text."""
    start, end = entity["start"], entity["end"]
    if start < 0 or end > len(text) or start >= end:
        return False, f"Invalid boundaries: [{start}:{end}] for text length {len(text)}"
    return True, None


def load_clean_gold_dataset(folder_path, verbose=False):
    """
    Parses all original human-labeled JSON files, stripping Label Studio metadata
    and keeping only clean text and entities. Skips augmented files.

    Args:
        folder_path: Path to the labeled data folder
        verbose: Print warnings about skipped/invalid records

    Returns:
        List of clean records with id, text, and entities
    """
    files = glob.glob(os.path.join(folder_path, "*.json"))
    # Strictly isolate human-labeled data; exclude automated augmentations
    original_files = sorted([f for f in files if "augmented_" not in os.path.basename(f)])

    clean_records = []
    skipped = {"cancelled": 0, "no_annotations": 0, "validation_error": 0}

    for file_path in original_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                ls_data = json.load(f)
            except json.JSONDecodeError:
                console.print(f"[warning]⚠️  Could not parse {Path(file_path).name}, skipping.[/warning]")
                continue

        for task in ls_data:
            # Skip empty or cancelled annotations
            if not task.get("annotations"):
                skipped["no_annotations"] += 1
                continue
            if task["annotations"][0].get("was_cancelled"):
                skipped["cancelled"] += 1
                continue

            task_id = str(task.get("id", len(clean_records)))
            text = task["data"]["text"]

            # Extract clean entity spans
            entities = []
            results = task["annotations"][0].get("result", [])

            for r in results:
                if r.get("type") == "labels":
                    val = r["value"]
                    label = str(val["labels"][0])

                    # Validate label
                    if label not in VALID_LABELS:
                        if verbose:
                            console.print(f"[dim]Invalid label '{label}' in task {task_id}, skipping entity[/dim]")
                        continue

                    entity = {
                        "start": int(val["start"]),
                        "end": int(val["end"]),
                        "label": label,
                    }

                    # Validate entity boundaries
                    is_valid, error = validate_entity(text, entity)
                    if not is_valid:
                        if verbose:
                            console.print(f"[dim]Task {task_id}: {error}[/dim]")
                        skipped["validation_error"] += 1
                        continue

                    entities.append(entity)

            # Sort entities by start position for neatness
            entities.sort(key=lambda x: x["start"])

            clean_records.append({"id": task_id, "text": text, "entities": entities})

    if verbose:
        console.print("\n[dim]Skipped records:[/dim]")
        console.print(f"  Cancelled: {skipped['cancelled']}")
        console.print(f"  No annotations: {skipped['no_annotations']}")
        console.print(f"  Validation errors: {skipped['validation_error']}")

    return clean_records


def split_dataset(records, val_fraction=0.1, test_fraction=0.0, seed=42):
    """
    Split records into train/validation/test sets.

    Args:
        records: List of records to split
        val_fraction: Fraction of data for validation (0.0-1.0)
        test_fraction: Fraction of data for test (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Dict with 'train', 'validation', and optionally 'test' splits
    """
    random.seed(seed)
    shuffled = list(records)
    random.shuffle(shuffled)

    total = len(shuffled)
    val_size = int(total * val_fraction)
    test_size = int(total * test_fraction)
    train_size = total - val_size - test_size

    splits = {
        "train": shuffled[:train_size],
    }

    if val_size > 0:
        splits["validation"] = shuffled[train_size : train_size + val_size]

    if test_size > 0:
        splits["test"] = shuffled[train_size + val_size :]

    return splits


def records_to_hf_format(records):
    """Convert records to Hugging Face columnar format."""
    hf_data = {
        "id": [r["id"] for r in records],
        "text": [r["text"] for r in records],
        "entities": [
            {
                "start": [e["start"] for e in r["entities"]],
                "end": [e["end"] for e in r["entities"]],
                "label": [e["label"] for e in r["entities"]],
            }
            for r in records
        ],
    }
    return hf_data


def push_dataset_to_hub(records, repo_id, private=True, create_splits=False, val_fraction=0.1):
    """
    Converts records to a Hugging Face Dataset with a strict schema and pushes it.

    Args:
        records: List of clean records
        repo_id: Hugging Face repo ID (e.g., "username/dataset-name")
        private: Whether to make the dataset private
        create_splits: Whether to create train/val splits (vs single split)
        val_fraction: If creating splits, fraction for validation
    """
    # Define schema for GLiNER/NER tasks
    features = Features(
        {
            "id": Value("string"),
            "text": Value("string"),
            "entities": Sequence(
                {
                    "start": Value("int32"),
                    "end": Value("int32"),
                    "label": Value("string"),
                }
            ),
        }
    )

    if create_splits:
        splits = split_dataset(records, val_fraction=val_fraction)

        console.print("\n📊 [bold]Dataset Splits:[/bold]")
        for split_name, split_records in splits.items():
            console.print(f"  {split_name}: {len(split_records)} samples")

        console.print("\n📦 Creating datasets and pushing to Hugging Face Hub...")
        for split_name, split_records in splits.items():
            hf_data = records_to_hf_format(split_records)
            dataset = Dataset.from_dict(hf_data, features=features)

            console.print(f"  🚀 Pushing {split_name} split ({len(split_records)} samples)...")
            dataset.push_to_hub(
                repo_id,
                split=split_name,
                private=private,
            )
    else:
        hf_data = records_to_hf_format(records)
        console.print(f"\n📦 Creating Hugging Face Dataset with {len(records)} samples...")
        dataset = Dataset.from_dict(hf_data, features=features)

        console.print(f"🚀 Pushing to Hugging Face Hub: https://huggingface.co/datasets/{repo_id}")
        dataset.push_to_hub(repo_id, private=private)

    visibility = "Private" if private else "Public"
    console.print(f"\n[bold green]✅ Dataset successfully published as {visibility}![/bold green]")
    console.print(f"📖 View at: https://huggingface.co/datasets/{repo_id}")


def print_dataset_stats(records):
    """Print statistics about the dataset."""
    total_samples = len(records)
    total_entities = sum(len(r["entities"]) for r in records)

    label_counts = {"ticker": 0, "company": 0}
    char_counts = [len(r["text"]) for r in records]

    for r in records:
        for e in r["entities"]:
            label_counts[e["label"]] += 1

    console.print("\n[bold]📊 Dataset Statistics:[/bold]")
    console.print(f"  Samples: {total_samples}")
    console.print(f"  Total entities: {total_entities} ({total_entities/total_samples:.1f} per sample avg)")
    console.print(f"  Tickers: {label_counts['ticker']}")
    console.print(f"  Companies: {label_counts['company']}")
    console.print(f"  Avg text length: {sum(char_counts)/len(char_counts):.0f} chars")
    console.print(f"  Total text size: {sum(char_counts)/1024/1024:.1f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push NER dataset to Hugging Face Hub")
    parser.add_argument(
        "--folder",
        default="data/labeled",
        help="Path to labeled data folder (default: data/labeled)"
    )
    parser.add_argument(
        "--repo-id",
        default="StephanAkkerman/wallstreetbets-ner",
        help="Hugging Face repo ID (default: StephanAkkerman/wallstreetbets-ner)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public (default: private)"
    )
    parser.add_argument(
        "--splits",
        action="store_true",
        help="Create train/validation splits (default: single split)"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction if --splits is used (default: 0.1)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed warnings"
    )

    args = parser.parse_args()

    # 1. Parse and validate data
    console.print(f"[bold]Loading dataset from {args.folder}...[/bold]")
    gold_records = load_clean_gold_dataset(args.folder, verbose=args.verbose)

    if not gold_records:
        console.print("[bold red]❌ No valid human-labeled records found to upload.[/bold red]")
        exit(1)

    # 2. Print statistics
    print_dataset_stats(gold_records)

    # 3. Upload to Hugging Face
    push_dataset_to_hub(
        gold_records,
        args.repo_id,
        private=not args.public,
        create_splits=args.splits,
        val_fraction=args.val_fraction,
    )
