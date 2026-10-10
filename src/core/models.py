"""Registry of the GLiNER2 base models we train an adapter for.

One adapter (and one Hub repo) is published per base model, so users can trade
accuracy for speed by picking a smaller base. A slug is the model name without
the org prefix and doubles as the adapter-folder / Hub-repo suffix.

Adapter folders are ``reddit_adapter_<slug>_v<N>``. Versions count per base.
The original single-model folders (``reddit_adapter`` and ``reddit_adapter_vN``,
all trained on ``gliner2-large-v1``) are still recognised as that base's early
versions, so nothing has to be renamed.
"""

import copy
import re

ADAPTER_PREFIX = "reddit_adapter"
HUB_REPO_PREFIX = "stock-recognizer"

# params_m is the base-model parameter count in millions (from the Hub
# metadata); it is shown next to accuracy so the size/accuracy trade-off is
# visible in the benchmark table and model cards.
BASE_MODELS = {
    "gliner2.5-small-v1": {
        "hf_id": "fastino/gliner2.5-small-v1",
        "display": "GLiNER2.5 Small",
        "params_m": 74,
    },
    "gliner2.5-base-v1": {
        "hf_id": "fastino/gliner2.5-base-v1",
        "display": "GLiNER2.5 Base",
        "params_m": 194,
    },
    "gliner2-base-v1": {
        "hf_id": "fastino/gliner2-base-v1",
        "display": "GLiNER2 Base",
        "params_m": 208,
    },
    "gliner2-large-v1": {
        "hf_id": "fastino/gliner2-large-v1",
        "display": "GLiNER2 Large",
        "params_m": 486,
    },
}
DEFAULT_BASE = "gliner2-large-v1"

# Per-base training overrides, merged over the defaults in `train.py`
# (keys: batch_size, lora_rank, encoder_lr, task_lr, epochs). Empty until a
# base is shown to need different settings; smaller models can often take a
# larger batch size or a higher learning rate.
TRAIN_OVERRIDES = {slug: {} for slug in BASE_MODELS}

# Pre-registry folders: ``reddit_adapter`` (v1) and ``reddit_adapter_vN``.
_LEGACY_DIR = re.compile(rf"{ADAPTER_PREFIX}(?:_v(?P<v>\d+))?")
_DIR = re.compile(rf"{ADAPTER_PREFIX}_(?P<slug>.+)_v(?P<v>\d+)")


def resolve_base(spec):
    """Map a slug (``gliner2-large-v1``) or Hub id (``fastino/...``) to a slug."""
    spec = (spec or DEFAULT_BASE).strip()
    if spec in BASE_MODELS:
        return spec
    for slug, info in BASE_MODELS.items():
        if spec == info["hf_id"]:
            return slug
    raise ValueError(
        f"Unknown base model {spec!r}. Choose from: {', '.join(BASE_MODELS)}"
    )


def resolve_bases(spec):
    """Like `resolve_base` but also accepts ``all`` and comma-separated lists."""
    if spec == "all":
        return list(BASE_MODELS)
    return [resolve_base(s) for s in spec.split(",") if s.strip()]


def adapter_dir_name(slug, version):
    return f"{ADAPTER_PREFIX}_{slug}_v{version}"


def parse_adapter_dir(folder_name):
    """Adapter folder name -> ``(slug, version)``; ``None`` if not ours.

    Legacy ``reddit_adapter`` / ``reddit_adapter_vN`` folders map to the
    default (large) base, which is what they were trained on.
    """
    match = _DIR.fullmatch(folder_name)
    if match and match["slug"] in BASE_MODELS:
        return match["slug"], int(match["v"])
    legacy = _LEGACY_DIR.fullmatch(folder_name)
    if legacy:
        return DEFAULT_BASE, int(legacy["v"] or 1)
    return None


def adapter_label(slug, version):
    """Name used in the benchmark store and tables.

    For the default base this equals the pre-registry label
    (``GLiNER2 Large + Adapter v7``), so cached benchmark results stay valid.
    """
    return f"{BASE_MODELS[slug]['display']} + Adapter v{version}"


def base_label(slug):
    return f"{BASE_MODELS[slug]['display']} (no adapter)"


def default_repo_id(owner, slug):
    """Hub repo for a base. The default base keeps the original repo, which the
    engine already pins; the others get a ``-<base>`` suffix (their own ``-v1``
    dropped) so a better adapter simply overwrites that repo's contents."""
    if slug == DEFAULT_BASE:
        return f"{owner}/{HUB_REPO_PREFIX}-model"
    return f"{owner}/{HUB_REPO_PREFIX}-model-{re.sub(r'-v\d+$', '', slug)}"


def load_extractor(slug, **kwargs):
    """Load a base model with `AutoExtractor`.

    GLiNER2.5 checkpoints use the boundary architecture and cannot be loaded
    with ``GLiNER2.from_pretrained`` (the legacy span loader); `AutoExtractor`
    reads the checkpoint's ``architecture`` field and dispatches to the right
    class for both generations. Needs ``gliner2>=2``.
    """
    from gliner2 import AutoExtractor

    return AutoExtractor.from_pretrained(BASE_MODELS[slug]["hf_id"], **kwargs)


def load_adapted(base_model, adapter_path):
    """Copy of `base_model` with the adapter loaded."""
    model = copy.deepcopy(base_model)
    model.load_adapter(adapter_path)
    return model
