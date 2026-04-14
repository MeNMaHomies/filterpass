# Reproducibility Manual: Filterpass Models

To ensure that researchers can reproduce the performance of our custom deepfake detection models (SAP, MP, and SP architectures), this project follows a strict versioning and environment protocol.

## 1. Hugging Face Model Pinning

While some researchers use `fairseq` commit hashes to pin their Wav2Vec 2.0 backbones, we use Hugging Face `transformers` and its built-in Git revision support.

### 1.1 Pinning the Backbone Revision
Each model on the Hugging Face Hub (like `facebook/wav2vec2-base`) is a Git repository. You can find the specific commit hash in the "History" tab of the model page.

We manage this directly in the configuration file:

1. Open `jaylou_scripts/config.py`.
2. Set the `model_revision` in the `CONFIG` dictionary:

```python
# jaylou_scripts/config.py
CONFIG = {
    ...
    "model_name": "facebook/wav2vec2-base",
    "model_revision": "9596634d1945f096756812833cc175e11f185790", # Lock to a specific commit
    ...
}
```

This revision is automatically passed to the model constructor during training.

---

## 2. Environment Management

### 2.1 Dependencies (`requirements.txt`)
Reproducibility can break if the underlying library (e.g., `transformers`) changes its internal implementation. Our project pins all core libraries in `requirements.txt`:

```text
# CORE LIBRARIES
torch==2.1.0
torchaudio==2.1.0
transformers==4.36.2
librosa==0.10.1
webrtcvad==2.0.10
numpy==1.24.3
scikit-learn==1.3.0
```

To recreate the environment:
```bash
pip install -r requirements.txt
```

### 2.2 Randomness and Seeds
To ensure identical weight initialization and data shuffling during training, we use a global seed function in `jaylou_scripts/train.py`. The seed value itself is controlled by the `SEED` constant in `jaylou_scripts/config.py`.

```python
# jaylou_scripts/config.py
SEED = 42
```

---

## 3. Saving and Loading the Entire Model

### 3.1 Checkpoints
Because our custom models (like `SAPClassifier` in `jaylou_scripts/model.py`) are `nn.Module` subclasses and not standard `PreTrainedModel` instances, we save the entire state dictionary (backbone + custom layers) into a single `.pt` file.

```python
# To load for evaluation (example from jaylou_scripts/train.py):
from jaylou_scripts.model import SAPClassifier
from jaylou_scripts.config import CONFIG, DEVICE

model = SAPClassifier(
    model_name=CONFIG["model_name"],
    revision=CONFIG["model_revision"]
).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_model_SAP.pt", map_location=DEVICE))
```

---

## 4. Benchmarking Reproducibility

When using the `scripts/benchmarking/` CLI, the configuration used for a run is automatically logged.

- **Check `results/<model>-<dataset>/summary.txt`**: This file contains the exact `chunk_ms`, `VAD mode`, and `overlap` used.
- **Check `results/metrics_summary.csv`**: This CSV serves as a permanent record of all experiment results across the project history.

### Reproduction Command Example
If a paper reports results using `filterpass-sap-v4` on `asvspoof2021-la` with 500ms chunks, you can reproduce it exactly using:

```bash
python -m scripts.benchmarking \
    --model filterpass-sap-v4 \
    --dataset asvspoof2021-la \
    --eval_dir data/ASVspoof2021_LA_eval/flac \
    --keys_dir data/keys/LA \
    --chunk_ms 500 \
    --num_workers 8
```

---

## 5. Summary Checklist for Researchers

1. **Clone the repo** and use the exact `requirements.txt`.
2. **Download the weights** from the provided checkpoint directory (not just the backbone).
3. **Verify the `model_revision`** in `jaylou_scripts/config.py` matches the one documented in the paper.
4. **Run the benchmarking CLI** with the specific configuration flags documented in the results summary.
