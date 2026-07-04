# Stock Recognizer Model 🧠

Training pipeline and labeled data for the GLiNER2 adapter behind [`stock-recognizer`](https://github.com/StephanAkkerman/stock-recognizer)'s `recognize_ai()` — scraping, cleaning, labeling, augmentation, training, benchmarking, and publishing to the Hugging Face Hub.

## Relationship to `stock-recognizer`

- The trained adapter is used **by** the engine repo at inference time.
- Several scripts here import `stock_recognizer` directly (its regex + market-data logic) to build/evaluate datasets. Install it as an editable sibling dependency:

```bash
pip install -e ../stock-recognizer
pip install -r requirements.txt
```

Install torch with CUDA separately if you want to train on GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

- Trained adapters are published to [`StephanAkkerman/stock-recognizer-model`](https://huggingface.co/StephanAkkerman/stock-recognizer-model) on the Hugging Face Hub, tagged per version — not committed to git.

## Pipeline

```
scraper.py → CSV → cleaner.py → preds/ → Label Studio (manual review) → data/labeled/
                                                                                ↓
                                                            augment_data.py → data/augmented/
                                                                                ↓
                                                                          train.py → models/
                                                                                ↓
                                                          push_model_to_hf.py → HF Hub
```

## Commands

```bash
# Run all tests
pytest

# Benchmark all adapter versions under models/
python src/core/benchmark.py

# Fix label-policy violations in annotated data
python src/maintenance/fix_label_policy.py          # preview changes
python src/maintenance/fix_label_policy.py --apply  # apply fixes

# Publish a trained adapter to the Hugging Face Hub
python utils/hf/push_model_to_hf.py models/reddit_adapter_v18/final --version v18
```

## Trainer organization

The src code is organized by responsibility rather than as a single flat list of scripts. See [src/README.md](src/README.md) for the current
layout guide.

## Utils organization

The helper scripts are grouped under [utils/README.md](utils/README.md) by task: labeling, scraping, augmentation, synthetic labeling, and Hugging Face
publishing.

## Label Guidelines 🏷️

When annotating training data in Label Studio, use exactly two labels: `ticker` and `company`. The distinction is based on the **form of the text**, not the author's intent — this makes annotation consistent and removes judgment calls.

### Decision tree

**1. Cashtag (`$` prefix) → always `ticker`**
```
$AMC  $TSLA  $gme  $EUV  $DRAM
```

**2. ALL-CAPS, resolves to a known ticker → `ticker`**
```
AMC   META   NVDA   SOFI   BP
```

**3. ALL-CAPS, but the ticker symbol differs → `company`**
```
NVIDIA  (ticker is NVDA)
TSMC    (ticker is TSM)
APPLE   (ticker is AAPL)
```

**4. Written / mixed-case name → `company`**
```
Meta   Nvidia   Micron   AMC Theatres   Goldman Sachs
```

**5. Informal lowercase ticker (Reddit shorthand) → `ticker`**
```
gme   amc   tsla   spy
```

### What not to label

| Category | Examples |
|---|---|
| Government / regulatory bodies | `CSRC`, `SEC`, `NASA`, `FINRA`, `NDAA` |
| Financial metric acronyms | `PDT`, `EV` (enterprise value), `SG&A`, `RSUs`, `PT` (price target), `ATM`, `IV` |
| Memory / chip technology terms | `DRAM`, `HBM`, `EUV`, `NAND` (without `$` prefix) |
| Media outlets with no public stock | `CNBC`, `Bloomberg`, `HBO`, `MSNBC` |

After fixing `data/labeled/`, regenerate `data/augmented/` before retraining.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
