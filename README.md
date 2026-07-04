# Stock Recognizer Model 🧠

Training pipeline and labeled data for the GLiNER2 adapter behind [`stock-recognizer`](https://github.com/StephanAkkerman/stock-recognizer)'s `recognize_ai()` — scraping, cleaning, labeling, augmentation, training, benchmarking, and publishing to the Hugging Face Hub.

> [!WARNING]
> This project is still in its alpha stage. Expect breaking changes as the labeling policy and training pipeline evolve.

## Relationship to `stock-recognizer`

- The trained adapter is used **by** the engine repo at inference time.
- Several scripts here import `stock_recognizer` directly (its regex + market-data logic) to build/evaluate datasets. Install it as an editable sibling dependency:

```bash
pip install -e ../stock-recognizer
pip install -r requirements.txt
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
python trainer/benchmark.py

# Publish a trained adapter to the Hugging Face Hub
python utils/push_model_to_hf.py models/reddit_adapter_v18/final --version v18
```

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

### Keeping data consistent

```bash
python trainer/fix_label_policy.py          # preview changes
python trainer/fix_label_policy.py --apply  # apply fixes
```

After fixing `data/labeled/`, regenerate `data/augmented/` before retraining.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
