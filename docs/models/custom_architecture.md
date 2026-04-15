# Custom Architecture Candidates

Experimental architectures under consideration for the filterpass custom model.
None of these are finalised — this document tracks the rationale and design details
for evaluation before committing to a training run.

---

## 1. AASIST-Raw (raw waveform, no SSL backbone)

### Motivation
Eliminates the 300M-param SSL backbone entirely. One unified model trained
end-to-end on labeled data. RTF trivially satisfies the < 0.1 constraint.
Target: ~85K–1.5M params.

### Architecture
```
Raw PCM (16kHz, 500ms = 8000 samples)
  → SincConv  (learnable bandpass filters, replaces fixed mel-filterbank)
  → RawNet2-style residual conv blocks  (short-term artifact detection)
  → Heterogeneous Graph Attention Network (HGAT)
       — spectral nodes: which frequency bands carry artifacts
       — temporal nodes: when artifacts appear
       — edges: spectral × temporal co-occurrence patterns
  → graph readout → FC → 2-class logit
```

Reference: AASIST (Jung et al., 2022) — https://arxiv.org/abs/2110.01200

### Robustness strategy

**RawBoost** (waveform-level augmentation during training)
- Linear convolutive noise — simulates channel/room effects
- Non-linear convolutive noise — simulates clipping, saturation
- Impulsive noise — simulates packet loss, transient interference
- Applied stochastically; model never sees the same waveform twice

**Codec multi-condition training**
- Training utterances processed through a-law, µ-law, MP3, Opus at varied bitrates
- Directly targets the C2/C3 phone call deployment conditions
- Prevents false positives on codec-degraded genuine speech

**Spectral masking**
- Randomly zero out frequency bands during training (SpecAugment-style)
- Forces model not to rely on any single frequency band
- Improves generalisation to unseen codec and channel conditions

### Trade-offs
| | Value |
|---|---|
| Params | ~85K (AASIST) – 1.5M (RawNet2) |
| RTF | << 0.01 |
| ASVspoof EER (reported) | ~0.83% (AASIST-L) |
| Out-of-domain risk | High without augmentation; mitigated by RawBoost + codec training |
| Training data needed | More than SSL-based (no transfer knowledge) |

---

## 2. WavLM + Classification Head

### Motivation
Drop-in replacement for the current wav2vec2-base backbone. WavLM adds a
denoising pre-training objective — predicts clean speech from corrupted input —
which directly addresses codec-degradation false positives without changing the
head architecture.

### Architecture
```
Raw PCM
  → WavLM-Large encoder (316M params, frozen or lightly fine-tuned)
       pre-training: masked prediction + denoising on corrupted speech
  → frame embeddings (1024-dim per ~20ms frame)
  → classification head (SAP or AASIST-style)
  → 2-class logit
```

Reference: WavLM (Chen et al., 2022) — https://arxiv.org/abs/2110.13900
HuggingFace: `microsoft/wavlm-large`

### Why WavLM over XLS-R
XLS-R pre-training objective: predict masked frames from clean speech only.
WavLM pre-training objective: predict masked frames from *corrupted* speech.
The denoising objective means WavLM representations are inherently more robust
to noise, codec quantisation artifacts, and channel effects — the exact failure
mode of XLS-R-based detectors on CodecFake and phone call conditions.

SUPERB benchmark: WavLM-Large outperforms XLS-R on every speech understanding
task, including those requiring fine acoustic discrimination.

### Trade-offs
| | Value |
|---|---|
| Params | 316M (backbone) + head |
| RTF | Similar to XLS-R (~0.08–0.12) |
| Out-of-domain | Better than XLS-R due to denoising pre-training |
| Training data needed | Low (transfer from pre-training) |
| Codec robustness | Better than wav2vec2-base and XLS-R out of the box |

---

## 3. Whisper Encoder + Classification Head

### Motivation
Whisper was trained supervised on 680K hours of real-world audio — including
phone calls, meetings, compressed streams, and multiple accents. Unlike SSL
models trained on clean read speech, Whisper has seen codec-degraded genuine
speech during pre-training. The encoder may already represent the boundary
between natural degradation and synthesis artifacts more accurately.

### Architecture
```
Raw PCM
  → log-mel spectrogram (80 bins, 25ms window, 10ms hop)
  → Whisper encoder (transformer, 512 or 1024 dim depending on model size)
       pre-training: supervised ASR on 680K hrs diverse real-world audio
  → encoder output (sequence of frame embeddings)
  → classification head (mean pool / SAP / attention)
  → 2-class logit
```

Reference: Whisper (Radford et al., 2022) — https://arxiv.org/abs/2212.04356
HuggingFace: `openai/whisper-medium` / `openai/whisper-large-v3`

### Key difference from wav2vec2 / WavLM
Input is mel-spectrogram, not raw waveform. Representations are frequency-domain
rather than raw-sample-level. This means low-level phase artifacts may be less
visible to the encoder — a potential weakness for detecting certain vocoders.
Compensated by training data diversity: Whisper has seen real degraded speech
at scale, reducing false positive rate on phone call conditions.

### Model size options
| Variant | Params | Notes |
|---|---|---|
| whisper-small | 244M | Fastest, less expressive |
| whisper-medium | 307M | Good balance |
| whisper-large-v3 | 1.5B | Best representations, high RTF risk |

Recommended starting point: `whisper-medium` encoder only (no decoder needed).

### Trade-offs
| | Value |
|---|---|
| Params | 307M (medium encoder) |
| RTF | Similar to WavLM; large-v3 may exceed 0.1 |
| Out-of-domain | Potentially best — training data includes real-world degraded speech |
| Phase artifact sensitivity | Lower than raw waveform models |
| Training data needed | Low (transfer from pre-training) |

---

## Comparison Summary

| Architecture | Params | RTF | Codec robustness | Training data | Key risk |
|---|---|---|---|---|---|
| AASIST-Raw + RawBoost | ~85K | << 0.01 | Needs augmentation | High | Brittle without codec training |
| WavLM-Large + head | ~316M | ~0.08–0.12 | Good (denoising PT) | Low | May still hit RTF limit on CPU |
| Whisper-medium + head | ~307M | ~0.08–0.12 | Good (diverse PT data) | Low | Phase artifacts less visible |

---

## Next Steps

1. Establish SOTA benchmark results (wav2vec2-aasist, xlsr-sls, xlsr-mamba) as baseline EER/RTF reference.
2. Run ablation: WavLM + SAP head vs. current wav2vec2-base + SAP head on same training data.
3. Run ablation: Whisper-medium encoder + SAP head.
4. If RTF constraint tightens (edge/mobile deployment): prototype AASIST-Raw with RawBoost + codec augmentation.
