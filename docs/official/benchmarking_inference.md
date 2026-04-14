# Benchmarking Inference — Findings & Analysis

**Dataset:** ASVspoof 2021 LA evaluation partition (148,176 utterances)
**Hardware:** NVIDIA RTX 4050 (6 GB VRAM), CUDA 11.6
**Default protocol:** 500 ms non-overlapping chunked inference

---

## 1. Evaluation Protocol

### 1.1 Chunked Inference (Default)

Each utterance is segmented into 500 ms (8,000 sample) non-overlapping chunks. Each chunk
is scored independently by the model. Scores are mean-pooled across all chunks from the
same utterance to produce a single utterance-level score, which is then submitted to the
EER and min-tDCF calculation.

This protocol directly reflects the deployment scenario for filterpass — real-time
detection on a live audio stream where no full-utterance context is available. The model
must make a reliable decision from a 500 ms window alone.

### 1.2 Static Inference (`--static`)

The full utterance is processed as a single forward pass without chunking, optionally
padded to a fixed length depending on the model. This matches the evaluation protocol
used in the original papers and serves as a reference point. Static evaluation is only
used in this benchmark when chunked inference is architecturally infeasible.

### 1.3 Real-Time Factor (RTF)

RTF measures inference efficiency relative to audio duration:

```
RTF = inference_time_seconds / audio_duration_seconds
```

For a 500 ms chunk processed in 3 ms:
```
RTF = 0.003 / 0.5 = 0.006
```

An RTF of 1.0 means the model processes audio exactly as fast as it arrives. The
filterpass deployment constraint is **RTF < 0.1** — inference must complete within 10%
of the chunk duration (50 ms for a 500 ms chunk), leaving the remaining 90% of compute
time for VAD, audio capture, and pipeline overhead.

### 1.4 Why Chunked EER Differs from Published Numbers

Published EER figures from model papers are measured using full-utterance static
evaluation on clean, studio-quality audio. Our chunked EER numbers reflect a
fundamentally different operating condition:

- Shorter context window — 500 ms vs. 4–10 second utterances
- No cross-chunk temporal context
- Codec-degraded conditions (ASVspoof 2021 LA includes C2 a-law and C3 µ-law conditions)

Our numbers are not reproductions of the original results and should not be interpreted
as such. They answer a different question: *how well does this model detect deepfakes
under streaming deployment conditions?*

---

## 2. Model Results

### 2.1 XLSR-Mamba

**Reference:** AustinXiao/XLSR-Mamba-LA (HuggingFace)
**Architecture:** XLS-R 300M (fairseq) frontend + Mamba state-space model backend

#### Results — Chunked (500 ms)

| Metric | Value |
|---|---|
| Parameters | 319,327,542 |
| VRAM load | 1.19 GB |
| VRAM peak | 1.51 GB |
| EER | 6.96% |
| min-tDCF | 0.3365 |
| AUC-ROC | 0.9783 |
| Accuracy (@ EER threshold) | 93.04% |
| RTF mean | 0.0405 |
| RTF median | 0.0300 |
| RTF p95 | **0.1043** |
| Latency mean | 20.2 ms |
| Latency p95 | 52.1 ms |

#### Analysis

XLSR-Mamba is the only model in this benchmark that **violates the RTF < 0.1 constraint
at p95** (0.1043). While the mean RTF (0.0405) appears acceptable, the tail latency
indicates that roughly 5% of chunks — likely those processed during GPU warm-up or on
longer utterances — exceed the 50 ms budget. This makes XLSR-Mamba unreliable for
hard real-time deployment without latency guarantees.

EER of 6.96% is the weakest among SOTA models benchmarked, suggesting that the Mamba
backend does not retain meaningful temporal context across chunk boundaries. Mamba is
a selective state-space model designed to propagate state across long sequences —
its key advantage is handling long-range dependencies more efficiently than a transformer.
When evaluated one 500 ms chunk at a time with no state carry-over, this advantage
disappears entirely. Each chunk is evaluated in isolation, which reduces Mamba to a
simple sequence aggregator, no more expressive than mean pooling for this task.

The unexpectedly high RTF relative to Wav2Vec2-AASIST (0.0405 vs. 0.0037 mean), despite
both using the same XLS-R frontend, suggests the Mamba backend itself is the bottleneck.
Mamba's recurrent computation does not parallelize as efficiently on GPU for short
sequences as attention-based or graph-based backends do.

---

### 2.2 Wav2Vec2-AASIST

**Reference:** Tak et al., Speaker Odyssey 2022 (https://arxiv.org/abs/2202.12233)
**Architecture:** XLS-R 300M (fairseq) frontend + AASIST heterogeneous graph attention backend

#### Results — Chunked (500 ms)

| Metric | Value |
|---|---|
| Parameters | 317,837,834 |
| VRAM load | 2.37 GB |
| VRAM peak | 2.37 GB |
| EER | **5.81%** |
| min-tDCF | **0.3098** |
| AUC-ROC | **0.9854** |
| Accuracy (@ EER threshold) | 94.19% |
| RTF mean | 0.0037 |
| RTF median | 0.0031 |
| RTF p95 | 0.0077 |
| Latency mean | 1.9 ms |
| Latency p95 | 3.9 ms |

#### Analysis

Wav2Vec2-AASIST achieves the best EER (5.81%), best min-tDCF (0.3098), and best
AUC-ROC (0.9854) among all SOTA models evaluated under chunked conditions. RTF is
well within the deployment constraint at all percentiles (p95 = 0.0077), providing
over 10× headroom against the 0.1 limit.

The AASIST backend is the distinguishing factor. Unlike simple pooling or recurrent
backends, AASIST constructs a heterogeneous graph over spectral and temporal features
extracted from each chunk. Graph nodes represent frequency bands and time positions;
edges encode co-occurrence patterns between spectral and temporal artifacts. This
explicit relational modeling allows AASIST to detect structured synthesis artifacts —
such as vocoder periodicity or spectral smearing from neural TTS — even within a
500 ms window, where utterance-level context is unavailable.

The RTF advantage over XLSR-Mamba (0.0037 vs. 0.0405 mean) is substantial despite
identical frontends, confirming that the Mamba recurrent backend is the RTF bottleneck
in that model. AASIST's graph operations parallelize efficiently on GPU for the small
graphs produced by 500 ms chunks.

**Wav2Vec2-AASIST is the strongest open-source baseline under streaming conditions and
the primary reference point for filterpass custom model development.**

---

### 2.3 XLSR-SLS

**Reference:** Zhang et al., ACM MM 2024 (https://openreview.net/pdf?id=acJMIXJg2u)
**Architecture:** XLS-R 300M (fairseq) frontend + SLS (Selective Layer Scoring) classifier

#### Published Results (Static, Full Utterance)

| Dataset | EER |
|---|---|
| ASVspoof 2021 LA | 2.87% |
| ASVspoof 2021 DF | 1.92% |
| In-the-Wild | 7.46% |

#### Chunked Inference: Architecturally Incompatible

XLSR-SLS **cannot be evaluated under the 500 ms chunked protocol** using the released
pretrained weights. This is not a software issue — it is a fundamental architectural
constraint.

**Root cause:** The SLS classifier contains a fully-connected layer `fc1` with a
hardcoded input dimension of 22,847. This dimension is derived from the flattened output
of a fixed-kernel `F.max_pool2d` applied to the XLS-R frame sequence. When the input
sequence length changes, the flattened size changes, and `fc1` fails with a shape
mismatch.

The model was trained with all utterances tile-padded to exactly **64,600 samples
(~4 seconds)**. At 16 kHz with XLS-R's 320-sample stride, this produces T = 201 frames:

```
max_pool2d((201, 1024), kernel=(3,3)) → (67, 341)
flatten → 67 × 341 = 22,847   ✓ matches fc1

max_pool2d((25, 1024), kernel=(3,3)) → (8, 341)   ← 500ms chunk
flatten → 8 × 341 = 2,728     ✗ shape mismatch
```

Other models in this benchmark avoid this problem by using **adaptive pooling** or
**graph readout** before their FC layers, which collapse the time dimension to a
fixed-size vector regardless of input length. XLSR-SLS uses a fixed-kernel pool
followed by flatten — a design choice that implicitly encodes sequence length into
the weight matrix.

**Workaround attempted:** Tile-padding each 500 ms chunk to 64,600 samples before
inference. This resolves the shape error but is methodologically unsound — 87.6% of
every chunk's input is repeated padding, not real audio. The model would be scoring
the padding, not the chunk. Additionally, RTF becomes equivalent to full-utterance
static inference, making it unusable for real-time deployment. Results under this
workaround are not reported.

#### Analysis

XLSR-SLS achieves the best published EER (2.87% LA) of any open-source model in this
benchmark, indicating that the Selective Layer Scoring mechanism — attention-weighted
aggregation across all 24 XLS-R transformer layers — extracts more discriminative
features than single-layer or mean-pooled representations. The SLS contribution is
well-motivated and the static results validate it.

However, the fixed-length architecture makes XLSR-SLS incompatible with any streaming
or variable-length evaluation scenario without architectural modification and full
retraining. This is a significant practical limitation. The model cannot be used for
real-time phone call detection in its current form, regardless of available compute.

**Verdict: Excluded from the chunked benchmark. Listed for static reference only.**

---

### 2.4 Filterpass-SAP

**Architecture:** Wav2Vec2-base (94M) frontend + Self-Attention Pooling classification head
**Weights:** Menmahomies/SAP_Classifier (HuggingFace)

#### Results — Chunked (500 ms)

| Metric | filterpass-sap (v1) | filterpass-sap (v4) |
|---|---|---|
| Parameters | 94,766,211 | 96,932,994 |
| VRAM load | 0.35 GB | 0.36 GB |
| VRAM peak | 0.61 GB | 2.34 GB |
| EER | 13.88% | 14.70% |
| min-tDCF | 0.4305 | 0.4369 |
| AUC-ROC | 0.9310 | 0.8922 |
| Accuracy (@ EER threshold) | 86.12% | 85.30% |
| RTF mean | 0.0021 | 0.0002 |
| RTF p95 | 0.0031 | 0.0003 |
| Latency mean | 1.1 ms | 1.1 ms |
| Latency p95 | 1.5 ms | — |

#### Analysis

Filterpass-SAP has the lowest RTF of all benchmarked models — 10–20× faster than
Wav2Vec2-AASIST and over 100× faster than XLSR-Mamba at the mean. The VRAM footprint
is also the smallest, making it the only model deployable on low-spec hardware or
alongside other GPU workloads. It satisfies the RTF < 0.1 constraint with substantial
margin across all percentiles.

The EER gap relative to Wav2Vec2-AASIST (13.88% vs. 5.81%) is significant and
attributable to two compounding factors:

1. **Backbone capacity.** Wav2vec2-base (94M params, English-only, ~1,000 hours
   pretraining) produces substantially less expressive frame embeddings than XLS-R 300M
   (300M params, 128 languages, 436,000 hours). The frontend representations fed to the
   classifier head carry less acoustic detail about subtle synthesis artifacts.

2. **Classifier head design.** Self-Attention Pooling computes a weighted average over
   the frame sequence and produces a single embedding before classification. Within a
   500 ms chunk, this collapses spatial and temporal structure into one vector — any
   positional or relational information about where and when the artifact occurs is
   discarded. AASIST's graph attention, by contrast, explicitly models relationships
   between spectral and temporal positions, preserving structure that is predictive of
   synthesis artifacts.

The v4 weights show a slight EER regression relative to v1 (14.70% vs. 13.88%) despite
more parameters, suggesting potential overfitting or a training distribution mismatch.
v1 remains the stronger checkpoint.

#### Priority for Improvement

The EER gap to SOTA is the primary concern. Two upgrade paths are identified:

- **Backbone upgrade:** Replace Wav2Vec2-base with WavLM-Large or XLS-R. WavLM's
  denoising pretraining objective is particularly relevant for codec-degraded phone call
  conditions. See `docs/models/custom_architecture.md`.
- **Head upgrade:** Replace SAP with an AASIST-style graph attention head. This
  addresses the loss of spatial-temporal structure during pooling.

Either change alone is expected to reduce EER substantially. Both together would bring
filterpass-SAP into the competitive range of SOTA open-source models.

---

## 3. Summary

| Model | EER | min-tDCF | RTF mean | RTF p95 | Params | RTF < 0.1 |
|---|---|---|---|---|---|---|
| Wav2Vec2-AASIST | **5.81%** | **0.3098** | 0.0037 | 0.0077 | 318M | Yes |
| XLSR-Mamba | 6.96% | 0.3365 | 0.0405 | 0.1043 | 319M | Borderline |
| XLSR-SLS | N/A (static-only) | — | — | — | 341M | No |
| Filterpass-SAP (v1) | 13.88% | 0.4305 | **0.0021** | **0.0031** | 95M | Yes |

### Key Findings

**Wav2Vec2-AASIST** is the best open-source model under streaming conditions. It leads
on all accuracy metrics and satisfies RTF with large headroom. The AASIST graph attention
backend is well-suited to 500 ms chunk-level artifact detection. This is the primary
SOTA reference for filterpass development.

**XLSR-Mamba** fails the RTF constraint at p95 and has the weakest EER among SOTA models.
Mamba's sequential modeling advantage is negated by stateless chunk-level evaluation.
Not recommended for real-time deployment without architectural changes.

**XLSR-SLS** has the best published static EER but is architecturally incompatible with
streaming inference due to a hardcoded sequence-length dependency in the classifier head.
It demonstrates what SLS attention across XLS-R layers is capable of under ideal
conditions, but cannot be used in a real-time pipeline without full retraining on
variable-length inputs.

**Filterpass-SAP** is the fastest model by a large margin and the only one viable on
constrained hardware. The EER gap to SOTA (8+ percentage points) is significant and
motivates backbone and head upgrades in the next training iteration.
