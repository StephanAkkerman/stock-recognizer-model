"""Publish trained adapters as Hugging Face model repos, one per base model.

Every base goes to ``<owner>/stock-recognizer-<base>`` (e.g. ``...-gliner2-large``).
Repos are public unless ``--private`` is given. Each upload is tagged ``v<N>`` so the engine repo can
pin a ``revision``. The uploaded folder is self-describing: adapter weights,
``recognizer_config.json`` (base model, labels, threshold) and a generated model
card with the benchmark numbers and real example outputs.

    python -m utils.hf.push_model_to_hf models/reddit_adapter_v18/final
    python -m utils.hf.push_model_to_hf models/reddit_adapter_gliner2-base-v1_v1 --threshold 0.7
    python -m utils.hf.push_model_to_hf --best          # best-F1 adapter of every base

Run ``python -m src.core.benchmark`` first so the card (and ``--best``) has
numbers, and ``python -m src.analysis.threshold_sweep`` to pick ``--threshold``.
"""

import argparse
import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi

from src.core.benchmark import get_all_adapters, locate_adapter_weights
from src.core.models import (
    BASE_MODELS,
    adapter_label,
    default_repo_id,
    parse_adapter_dir,
)
from src.core.results_store import load_store

DEFAULT_OWNER = "StephanAkkerman"
DATASET_NAME = "wallstreetbets-ner"
GITHUB_URL = "https://github.com/StephanAkkerman/stock-recognizer-model"
ENGINE_URL = "https://github.com/StephanAkkerman/stock-recognizer"

# Shown on every model card with the model's real output (see `run_examples`).
EXAMPLE_TEXTS = [
    "$GME is mooning but Apple Inc. might crash tomorrow",
    "Bought more nvda and AMC calls, Goldman Sachs says Meta is a buy",
    "YOLO'd my savings into $tsla, DD says PT 400. NFA",
    "The SEC is investigating, but CNBC says Microsoft earnings were fine.",
    "Just a normal post about my day, nothing to see here.",
]


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


def latest_benchmark(store, name):
    """Most recently evaluated ``(test_hash, result)`` for `name`, or ``(None, None)``."""
    runs = store["results"].get(name, {})
    if not runs:
        return None, None
    test_hash = max(runs, key=lambda h: runs[h]["evaluated_at"])
    return test_hash, runs[test_hash]


def best_results(store, models_dir="models"):
    """``{base: (f1, weights_folder, result)}`` for the highest overall-F1 adapter."""
    best = {}
    for a in get_all_adapters(models_dir):
        _, result = latest_benchmark(store, a["name"])
        if result is None:
            continue
        f1 = result["metrics"]["overall"]["f1"]
        if a["base"] not in best or f1 > best[a["base"]][0]:
            best[a["base"]] = (f1, a["path"], result)
    return best


def best_adapter_folders(store, models_dir="models"):
    """Highest overall-F1 weights folder per base model, from the benchmark store."""
    return [path for _, path, _ in best_results(store, models_dir).values()]


def sibling_rows(owner, store, models_dir="models"):
    """One row per base model with its best adapter's score, for the card's size table."""
    best = best_results(store, models_dir)
    rows = []
    for slug, info in BASE_MODELS.items():
        if slug not in best:
            continue
        _, _, result = best[slug]
        rows.append(
            {
                "repo_id": default_repo_id(owner, slug),
                "display": info["display"],
                "params_m": info["params_m"],
                "f1": result["metrics"]["overall"]["f1"],
                "ms": (result["metrics"].get("speed") or {}).get("ms_per_doc"),
            }
        )
    return rows


def run_examples(adapter_dir, slug, texts):
    """Run the staged adapter on `texts`; returns the non-empty entities per text.

    Runs the real model so the card shows what users get.
    """
    from src.core.models import load_adapted, load_extractor

    with open(
        os.path.join(adapter_dir, "recognizer_config.json"), "r", encoding="utf-8"
    ) as f:
        cfg = json.load(f)
    model = load_adapted(load_extractor(slug), adapter_dir)
    outputs = []
    for text in texts:
        entities = model.extract_entities(
            text, cfg["entity_descriptions"], threshold=cfg["threshold"]
        )["entities"]
        outputs.append({label: vals for label, vals in entities.items() if vals})
    return outputs


def build_model_card(
    slug,
    version,
    repo_id,
    threshold,
    entity_descriptions,
    result=None,
    test_hash=None,
    test_docs=None,
    examples=None,
    siblings=None,
):
    """Render the model card (README.md) for one adapter.

    `examples` is a list of ``(text, entities)`` pairs and `siblings` the rows
    from `sibling_rows`; either may be omitted.
    """
    info = BASE_MODELS[slug]
    name = repo_id.split("/")[-1]
    owner = repo_id.split("/")[0]
    dataset_id = f"{owner}/{DATASET_NAME}"
    params = (result or {}).get("params") or {}
    lines = [
        "---",
        f"base_model: {info['hf_id']}",
        "library_name: peft",
        "pipeline_tag: token-classification",
        "language:",
        "- en",
        "license: mit",
        "datasets:",
        f"- {dataset_id}",
        "metrics:",
        "- f1",
        "- precision",
        "- recall",
        "tags:",
        f"- base_model:adapter:{info['hf_id']}",
        "- gliner2",
        "- lora",
        "- finance",
        "- stocks",
        "- wallstreetbets",
        "- reddit",
        "- ner",
        "- named-entity-recognition",
        "- information-extraction",
        "---",
        "",
        f"# {name}",
        "",
        f"LoRA adapter (v{version}) for [`{info['hf_id']}`](https://huggingface.co/{info['hf_id']}) "
        f"({info['params_m']}M parameters) that finds stock tickers and company "
        "names in informal financial text such as Reddit and X posts.",
        "",
        "Smaller bases are faster; the table below compares accuracy and speed "
        "across the sibling `stock-recognizer-*` repos.",
        "",
        f"Code, data pipeline and benchmark: [StephanAkkerman/stock-recognizer-model]({GITHUB_URL}). "
        f"It is the NER backbone of the [stock-recognizer]({ENGINE_URL}) engine, "
        "which resolves the extracted entities to actual tickers.",
        "",
        "## Intended uses",
        "",
        "Retail investors write about stocks in slang, with cashtags (`$GME`), "
        "bare symbols (`AMC`), lowercase symbols (`nvda`) and company names "
        "(`Goldman Sachs`) mixed together. Regexes and general-purpose NER models "
        "handle this poorly. This model tags those mentions, which you can use to:",
        "",
        "- track which stocks a community is discussing over time,",
        "- link posts, comments or news replies to tickers,",
        "- feed a sentiment or trend pipeline with the right symbols,",
        "- pre-label more data for further training.",
        "",
        "It is built for English, informal, post-length text. It only extracts "
        "mentions; it does not resolve ambiguous symbols (`AA` can be Alcoa or "
        "American Airlines) and it is not financial advice.",
        "",
        "## Entities",
        "",
        "| label | what it captures |",
        "|---|---|",
    ]
    for label, desc in entity_descriptions.items():
        lines.append(f"| `{label}` | {desc} |")
    lines += [
        "",
        "Labeling follows one policy: a span that is a valid ticker symbol is a "
        "`ticker` (`$AMC`, `AMC`, `gme`), a name that resolves to a company is a "
        "`company` (`AMC Theatres`, `Nvidia`). Government agencies, financial "
        "metrics and media outlets are not labeled.",
        "",
    ]

    if examples:
        lines += [
            "## Examples",
            "",
            f"Actual output of this model (threshold {threshold}); empty labels are omitted.",
            "",
        ]
        for text, entities in examples:
            lines += [
                "```text",
                f"Input:  {text}",
                f"Output: {json.dumps(entities, ensure_ascii=False)}",
                "```",
                "",
            ]

    if result:
        m = result["metrics"]
        speed = (m.get("speed") or {}).get("ms_per_doc")
        device = params.get("device", "")
        docs = f", {test_docs} documents" if test_docs else ""
        lines += [
            f"## Results (held-out test set `{test_hash}`{docs})",
            "",
            "Scoring is set-based: an entity counts as found if the model finds it "
            "at least once in a document, however often it is repeated. This "
            "mirrors the engine, which returns a deduplicated set of tickers per "
            "text. Case, whitespace and a leading `$` are ignored when comparing.",
            "",
            "| label | precision | recall | F1 |",
            "|---|---|---|---|",
        ]
        for label, v in m.items():
            if label == "speed":
                continue
            lines.append(f"| {label} | {v['p']:.1%} | {v['r']:.1%} | {v['f1']:.1%} |")
        if speed is not None:
            lines += [
                "",
                f"Inference: {speed} ms/doc (chunked, {device or 'unknown device'}).",
            ]
        lines.append("")

    if siblings:
        lines += [
            "## Choosing a model",
            "",
            "All sizes are trained on the same data and scored on the same test set.",
            "",
            "| model | base params | overall F1 | ms/doc |",
            "|---|---|---|---|",
        ]
        for row in siblings:
            ms = f"{row['ms']}" if row["ms"] is not None else "n/a"
            label = f"[{row['display']}](https://huggingface.co/{row['repo_id']})"
            if row["repo_id"] == repo_id:
                label = f"**{row['display']} (this repo)**"
            lines.append(f"| {label} | {row['params_m']}M | {row['f1']:.1%} | {ms} |")
        lines.append("")

    lines += [
        "## Usage",
        "",
        'Needs `pip install "gliner2>=2" peft huggingface_hub`.',
        "",
        "```python",
        "import json",
        "",
        "from gliner2 import AutoExtractor",
        "from huggingface_hub import snapshot_download",
        "from peft import PeftConfig, PeftModel",
        "",
        f'adapter_dir = snapshot_download("{repo_id}", revision="v{version}")',
        'cfg = json.load(open(f"{adapter_dir}/recognizer_config.json"))',
        'model = AutoExtractor.from_pretrained(cfg["base_model"])',
        "",
        "# The adapter holds LoRA weights only; clearing task_type stops PEFT from",
        "# expecting a classifier head. Merging gives a plain extractor.",
        "peft_config = PeftConfig.from_pretrained(adapter_dir)",
        "peft_config.task_type = None",
        "model = PeftModel.from_pretrained(model, adapter_dir, config=peft_config)",
        "model = model.merge_and_unload()",
        "",
        "result = model.extract_entities(",
        '    "$GME is mooning but Apple Inc. might crash tomorrow",',
        '    cfg["entity_descriptions"],',
        '    threshold=cfg["threshold"],',
        ")",
        "print(result)",
        "```",
        "",
        f"Recommended threshold: **{threshold}**. Lower it to find more entities "
        "(higher recall), raise it for fewer, more certain ones.",
        "",
        "The engine loads this adapter for you through `recognize_ai()`.",
        "",
        "## Training data",
        "",
        f"[{dataset_id}](https://huggingface.co/datasets/{dataset_id}): posts from "
        "r/wallstreetbets and related communities, pre-labeled and reviewed by "
        "hand in Label Studio. Train/validation split is 90/10 by post, with "
        "150-word chunks overlapping by 40 words; chunks without entities are "
        "kept as negative examples.",
        "",
    ]
    if params.get("train_samples"):
        lines += [
            f"This adapter saw {params['train_samples']} training chunks and "
            f"{params.get('val_samples', 'n/a')} validation chunks.",
            "",
        ]
    lines += [
        "| parameter | value |",
        "|---|---|",
        f"| LoRA rank / alpha / dropout | {params.get('lora_r', params.get('r', 32))} / "
        f"{params.get('lora_alpha', 64)} / {params.get('lora_dropout', 0.1)} |",
        f"| Target modules | {', '.join(params.get('target_modules') or ['key_proj', 'value_proj', 'query_proj', 'dense'])} |",
        f"| Epochs | up to {params.get('num_epochs', 10)} (early stopping) |",
        f"| Batch size | {params.get('batch_size', 4)} "
        f"(effective {params.get('effective_batch_size', 8)}) |",
        f"| Encoder / task LR | {params.get('encoder_lr', 2e-5)} / {params.get('task_lr', 5e-4)} |",
        "| Max sequence length | 256 |",
        "",
        "## Limitations",
        "",
        "- Trained on Reddit; performance on news, research or formal text may differ.",
        "- Occasional boundary errors on multi-word company names.",
        "- Low-frequency or newly listed tickers may be missed, and unseen company "
        "names may be mislabeled.",
        "- Extracts mentions only; ambiguous symbols are not resolved.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{akkerman_stock_recognizer,",
        "  author = {Stephan Akkerman},",
        "  title  = {stock-recognizer-model},",
        "  year   = {2026},",
        f"  url    = {{{GITHUB_URL}}}",
        "}",
        "```",
        "",
    ]
    return "\n".join(lines)


def push_model_to_hub(
    folder_path,
    repo_id=None,
    owner=DEFAULT_OWNER,
    threshold=None,
    private=False,
    run_card_examples=True,
):
    """Stage and upload an adapter; tags the commit ``v<N>`` and returns the repo URL."""
    weights, slug, version = resolve_adapter_folder(folder_path)
    repo_id = repo_id or default_repo_id(owner, slug)
    tag = f"v{version}"

    store = load_store()
    test_hash, result = latest_benchmark(store, adapter_label(slug, version))
    if result is None:
        print("No benchmark result found; the model card will have no metrics.")
    test_docs = (store["test_sets"].get(test_hash) or {}).get("num_tasks")

    with tempfile.TemporaryDirectory() as staging:
        shutil.copytree(weights, staging, dirs_exist_ok=True)
        cfg_path = os.path.join(staging, "recognizer_config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        if threshold is not None:
            cfg["threshold"] = threshold
        cfg.setdefault("base_model", BASE_MODELS[slug]["hf_id"])
        # Older training runs saved these empty; the model card's usage needs them.
        if not cfg.get("entity_descriptions"):
            from src.core.train import ENTITY_DESCRIPTIONS

            cfg["entity_descriptions"] = ENTITY_DESCRIPTIONS
        if not cfg.get("labels"):
            cfg["labels"] = list(cfg["entity_descriptions"])
        cfg.setdefault("threshold", 0.75)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        examples = None
        if run_card_examples:
            print("Running the adapter on the card examples...")
            examples = list(
                zip(EXAMPLE_TEXTS, run_examples(staging, slug, EXAMPLE_TEXTS))
            )
        with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                build_model_card(
                    slug,
                    version,
                    repo_id,
                    cfg["threshold"],
                    cfg["entity_descriptions"],
                    result,
                    test_hash,
                    test_docs,
                    examples=examples,
                    siblings=sibling_rows(repo_id.split("/")[0], store),
                )
            )

        api = HfApi()
        print(f"Creating/verifying repo: https://huggingface.co/{repo_id}")
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.update_repo_settings(repo_id, repo_type="model", private=private)
        print(f"Uploading {weights} to {repo_id}...")
        commit = api.upload_folder(
            folder_path=staging,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload adapter ({tag})",
        )
        api.create_tag(
            repo_id, tag=tag, repo_type="model", revision=commit.oid, exist_ok=True
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
        "--threshold",
        type=float,
        help="Decision threshold to ship (from threshold_sweep). Default: keep the trained config's.",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Instead of one folder, push the best-F1 adapter of every base model.",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repos private (default: public)."
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip running the model for the card's examples.",
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
            threshold=args.threshold,
            private=args.private,
            run_card_examples=not args.no_examples,
        )


if __name__ == "__main__":
    main()
