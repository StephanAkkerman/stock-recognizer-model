"""Publish a trained adapter to the Hugging Face Hub and tag the commit.

One repo per base model: the default (large) base keeps
``<owner>/stock-recognizer-model``, the others get
``<owner>/stock-recognizer-model-<base>`` (e.g. ``...-gliner2.5-small``). Each
upload is tagged ``v<N>`` so the engine repo can pin a ``revision``.

    python -m utils.hf.push_model_to_hf models/reddit_adapter_v18/final
    python -m utils.hf.push_model_to_hf models/reddit_adapter_gliner2-base-v1_v1
    python -m utils.hf.push_model_to_hf --best          # best-F1 adapter of every base

Run ``python -m src.core.benchmark`` first so ``--best`` has numbers to rank by.
"""

import argparse
import os

from huggingface_hub import HfApi

from src.core.benchmark import get_all_adapters, locate_adapter_weights
from src.core.models import default_repo_id, parse_adapter_dir
from src.core.results_store import load_store

DEFAULT_OWNER = "StephanAkkerman"


def resolve_adapter_folder(path):
    """``(weights_folder, slug, version)`` for an adapter dir or its best/final subfolder."""
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Adapter folder not found: {path}")
    if os.path.basename(path) in ("best", "final"):
        weights, adapter_dir = path, os.path.dirname(path)
    else:
        weights, adapter_dir = locate_adapter_weights(path) or path, path
    parsed = parse_adapter_dir(os.path.basename(adapter_dir))
    if parsed is None:
        raise ValueError(
            "Folder name must look like reddit_adapter_<base-slug>_v<N> "
            f"(or legacy reddit_adapter_v<N>), got {adapter_dir!r}"
        )
    return (weights, *parsed)


def best_adapter_folders(store, models_dir="models"):
    """Highest overall-F1 weights folder per base model, from the benchmark store."""
    best = {}
    for a in get_all_adapters(models_dir):
        runs = store["results"].get(a["name"], {})
        if not runs:
            continue
        latest = max(runs.values(), key=lambda r: r["evaluated_at"])
        f1 = latest["metrics"]["overall"]["f1"]
        if a["base"] not in best or f1 > best[a["base"]][0]:
            best[a["base"]] = (f1, a["path"])
    return [path for _, path in best.values()]


def push_model_to_hub(folder_path, repo_id=None, owner=DEFAULT_OWNER, private=True):
    """Upload an adapter folder and tag the commit ``v<N>``; returns the repo URL."""
    weights, slug, version = resolve_adapter_folder(folder_path)
    repo_id = repo_id or default_repo_id(owner, slug)
    tag = f"v{version}"

    api = HfApi()
    print(f"Creating/verifying repo: https://huggingface.co/{repo_id}")
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"Uploading {weights} to {repo_id}...")
    commit_info = api.upload_folder(
        folder_path=weights,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload adapter ({tag})",
    )
    api.create_tag(
        repo_id, tag=tag, repo_type="model", revision=commit_info.oid, exist_ok=True
    )
    print(f"Published and tagged {tag}: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder_path",
        nargs="?",
        help="Adapter folder, e.g. models/reddit_adapter_v18/final",
    )
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="HF user/org.")
    parser.add_argument("--repo-id", help="Explicit repo id (overrides --owner).")
    parser.add_argument(
        "--best",
        action="store_true",
        help="Instead of one folder, push the best-F1 adapter of every base model.",
    )
    parser.add_argument(
        "--public", action="store_true", help="Make the repo public (default: private)."
    )
    args = parser.parse_args()

    if args.best:
        folders = best_adapter_folders(load_store())
    elif args.folder_path:
        folders = [args.folder_path]
    else:
        parser.error("Pass an adapter folder or --best.")

    for folder in folders:
        push_model_to_hub(
            folder,
            repo_id=None if args.best else args.repo_id,
            owner=args.owner,
            private=not args.public,
        )


if __name__ == "__main__":
    main()
