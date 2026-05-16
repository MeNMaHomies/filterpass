# filterpass

> **Experimental research repo.** Nothing here is production-ready. Expect rough edges, dead code, and leftover experiments.

Research project targeting real-time deepfake audio detection for phone calls and live meetings. The core constraint is RTF < 0.1 - inference must complete in under 50ms for a 500ms audio chunk.

---

## Structure

This repo is a bit of a mess by design - it's an active research workspace shared across the team.

### Model Scripts

| Directory | Description |
|-----------|-------------|
| `duy_scripts/` | Primary model scripts - most up to date |
| `jaylou_scripts/` | Older scripts - largely superseded by `duy_scripts/` |

**`duy_scripts/`**
```
classifiers/
  model_SAP.py          - Wav2Vec2 + Self-Attention Pooling (current architecture)
  model_SP.py           - Simple pooling variant
  model_MP.py           - Mean pooling variant
  model_AASIST.py       - AASIST-based classifier
dataset.py / datasets.py
training_setup.py
train.ipynb
pred_single_file.py
```

**`jaylou_scripts/`** (legacy, kept for reference)
```
model.py, train.py, evaluate.py
dataset.py, cache_dataset.py, config.py
legacy/                 - even older versions
```

### Benchmarking Pipeline (Legacy)

`scripts/benchmarking/` contains the original benchmarking pipeline - model-agnostic, dataset-agnostic CLI for running SOTA models against ASVspoof 2021 LA.

> This has been extracted into its own repo: [filterpass-benchmark](https://github.com/MeNMaHomies/filterpass-benchmark). The copy here is kept for reference but is no longer the source of truth.

```
scripts/benchmarking/
  models/               - adapters for XLSR-Mamba, Wav2Vec2-AASIST, Nes2Net, TCM, filterpass-SAP variants
  datasets/             - ASVspoof 2021 LA adapter
  base/                 - BenchmarkModel and DatasetAdapter base classes
  audio_loader.py       - chunked VAD loader
  inference.py          - batched inference loop
  metrics.py            - EER, RTF, AUC-ROC
  reporter.py           - results output
```

### Data Augmentation Pipeline

`scripts/data_augmentation/` - reusable augmentation library (TensorFlow ImageDataGenerator-style API) covering noise, channel, and temporal augmentations.

### Other Scripts

| File | Description |
|------|-------------|
| `scripts/microphone.py` | Live mic ingestion via VAD, yields 500ms PCM chunks |
| `scripts/ANIRA.py` | ANIRA real-time inference exploration |
| `scripts/oversampling.py` | Dataset oversampling utility |
| `scripts/legacy/` | Old standalone scripts, kept for reference |

### Notebooks

| File | Description |
|------|-------------|
| `main.ipynb` | General scratchpad |
| `filterpass_colab.ipynb` | Colab training notebook |
| `notebooks/analysis.ipynb` | Results analysis and plots |

### Results

`results/` contains benchmark outputs (scores, summaries, plots) for all evaluated models including XLSR-Mamba, Wav2Vec2-AASIST, Nes2Net, TCM, and filterpass-SAP v1-v4.

> **These results are outdated.** For up-to-date benchmark results, see [filterpass-benchmark](https://github.com/MeNMaHomies/filterpass-benchmark).

### Context

`context/research_papers/` - reference PDFs (ASVspoof datasets, SOTA model papers).

---

## Models on HuggingFace

Trained filterpass-SAP checkpoints: [Menmahomies/SAP_Classifier](https://huggingface.co/Menmahomies/SAP_Classifier)

---

## Team

Jaylou Rasonabe - Ardashes Manougian - Khanh Duy Nguyen
