# SOTA Model Candidates: Audio Deepfake & Anti-Spoofing Detection

Survey of open-source SOTA models for audio deepfake detection and anti-spoofing, based on the [Speech DF Arena leaderboard](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena), ASVspoof challenge results, and peer-reviewed publications as of early 2026.

---

## 1. Fake-Mamba

**Architecture:** XLS-R (300M) front-end + bidirectional Mamba back-end (three encoder variants: TransBiMamba, ConBiMamba, PN-BiMamba). Replaces self-attention with state-space-based sequence modeling for more efficient long-range artifact capture.

**Trained on:** ASVspoof 2019 LA

**Performance:**
| Dataset | EER |
| :--- | :--- |
| ASVspoof 2021 LA | 0.97% |
| ASVspoof 2021 DF | 1.74% |
| In-the-Wild | 5.85% |

**Parameters:** ~300M+

**Links:**
- GitHub: https://github.com/xuanxixi/Fake-Mamba
- Paper (arXiv): https://arxiv.org/abs/2508.09294
- HuggingFace paper page: https://huggingface.co/papers/2508.09294

---

## 2. XLSR-Mamba

**Architecture:** XLS-R (300M) front-end + dual-column bidirectional Mamba back-end. Two independent Mamba columns process forward and backward sequences separately, then merge to capture both local and global dependencies. Ranked 2nd on Speech DF Arena.

**Trained on:** ASVspoof 2019 LA

**Performance:**
| Dataset | EER | min t-DCF |
| :--- | :--- | :--- |
| ASVspoof 2021 LA | 0.93% | 0.208 |
| ASVspoof 2021 DF | 1.88% | — |
| In-the-Wild | 6.71% | — |
| Speech DF Arena avg (14 datasets) | 14.21% | — |

**Parameters:** 319M

**Links:**
- GitHub: https://github.com/swagshaw/XLSR-Mamba
- HuggingFace (LA weights): https://huggingface.co/AustinXiao/XLSR-Mamba-LA
- HuggingFace (DF weights): https://huggingface.co/AustinXiao/XLSR-Mamba-DF
- Paper (arXiv): https://arxiv.org/abs/2411.10027

---

## 3. XLSR + SLS (Sensitive Layer Selection)

**Architecture:** XLS-R (300M) front-end + MLP with Sensitive Layer Selection. Treats the transformer as a feature pyramid and learns scalar weights to adaptively fuse representations from all layers, capturing artifacts at different abstraction levels. Ranked 1st on Speech DF Arena generalization average.

**Trained on:** ASVspoof 2019 LA

**Performance:**
| Dataset | EER |
| :--- | :--- |
| ASVspoof 2021 LA | 2.87% |
| ASVspoof 2021 DF | 1.92% |
| In-the-Wild | 7.46% |
| Speech DF Arena avg (14 datasets) | 13.84% |

**Parameters:** ~340M

**Links:**
- GitHub: https://github.com/QiShanZhang/SLSforASVspoof-2021-DF
- Paper (ACM MM 2024): https://dl.acm.org/doi/abs/10.1145/3664647.3681345

---

## 4. Wav2Vec2-AASIST (SSL + Graph Attention)

**Architecture:** wav2vec 2.0 XLSR (300M) self-supervised front-end feeding into the AASIST graph attention back-end. The SSL model acts as a feature extractor; AASIST models spectro-temporal artifacts through heterogeneous graph attention networks. The canonical SSL + AASIST combination.

**Trained on:** ASVspoof 2019 LA

**Performance:**
| Dataset | EER | min t-DCF |
| :--- | :--- | :--- |
| ASVspoof 2021 LA | 0.82% | 0.2066 |
| ASVspoof 2021 DF | 2.85% | — |
| Speech DF Arena avg | 18.02% | — |

**Parameters:** ~318M (300M XLS-R + ~0.3M AASIST head)

**Links:**
- GitHub: https://github.com/TakHemlata/SSL_Anti-spoofing
- Paper (arXiv): https://arxiv.org/abs/2202.12233

---

## 5. TCM (Temporal-Channel Modeling)

**Architecture:** XLS-R (300M) front-end + Conformer-based back-end with Temporal-Channel Modeling in multi-head self-attention. Incorporates a "Head Token" design from DHVT (Dual-Head Vision Transformer) for improved temporal and channel-level artifact modeling. Published at Interspeech 2024.

**Trained on:** ASVspoof 2019/2021 LA and DF

**Performance:**
| Dataset | EER |
| :--- | :--- |
| Speech DF Arena avg | 15.77% |

**Parameters:** ~319M

**Links:**
- GitHub: https://github.com/ductuantruong/tcm_add

---

## 6. DF Arena 1B (RAPTOR Framework)

**Architecture:** ~1B parameter universal antispoofing model using the RAPTOR (Robust Audio Pre-Training for cOmpact Representations) framework. Trained on the broadest dataset collection of any model here, covering speech, singing, and environmental deepfakes. Available via HuggingFace `transformers` pipeline with `task="antispoofing"`.

**Trained on:** ASVspoof 2019, ASVspoof 2024, Codecfake, LibriSeVoc, DFADD, CtrSVDD, SpoofCeleb, MLAAD, EnvSDD

**Performance:**
| Dataset | EER | Accuracy |
| :--- | :--- | :--- |
| DFADD | 0.00% | 99.97% |
| LibriSeVoc | 0.15% | 99.84% |
| In-the-Wild | 0.91% | 99.10% |
| ASVspoof 2019 | 1.14% | 98.86% |
| ASVspoof 2021 LA | 4.66% | 95.34% |
| **Arena avg (14 datasets)** | **5.92%** | **94.08%** |

**Parameters:** ~1B

**Links:**
- HuggingFace: https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1
- Paper (arXiv): https://arxiv.org/abs/2603.06164
- License: Non-commercial only

---

## 7. AASIST / AASIST-L

**Architecture:** End-to-end raw waveform model. SincConv-based encoder followed by a heterogeneous stacking graph attention layer (RawGAT-inspired) that jointly models spectral and temporal artifacts. No SSL backbone — processes raw audio directly. AASIST-L is a stripped-down variant at 85K parameters, viable for real-time on CPU.

**Trained on:** ASVspoof 2019 LA

**Performance (ASVspoof 2019 LA eval):**
| Model | EER | min t-DCF |
| :--- | :--- | :--- |
| AASIST | 0.83% | 0.0275 |
| AASIST-L | 0.99% | 0.0309 |
| Speech DF Arena avg | 34.49% | — |

**Parameters:** 300K (AASIST) / 85K (AASIST-L)

**Links:**
- GitHub: https://github.com/clovaai/aasist
- Paper (arXiv): https://arxiv.org/abs/2110.01200

---

## 8. AASIST3 (KAN-Enhanced AASIST)

**Architecture:** Upgraded AASIST replacing linear layers with Kolmogorov-Arnold Networks (KAN). Adds a Wav2Vec2 encoder for SSL features, a KAN Bridge layer, residual encoder blocks, dual Graph Attention Networks (GAT-S for spatial, GAT-T for temporal), multi-branch inference with master tokens, and a KAN output layer. Trained across multiple datasets simultaneously.

**Trained on:** ASVspoof 2019 LA, ASVspoof 2024 (ASVspoof5), MLAAD, M-AILABS

**Performance:** Submitted to ASVspoof 2024 challenge; specific EER figures vary by configuration.

**Parameters:** Undisclosed (larger than base AASIST due to Wav2Vec2 integration)

**Links:**
- HuggingFace: https://huggingface.co/MTUCI/AASIST3
- Paper (ASVspoof 2024 Workshop): https://doi.org/10.21437/ASVspoof.2024-8
- License: CC BY-NC-ND 4.0 (non-commercial)

---

## 9. RawGAT-ST (Spectro-Temporal Graph Attention)

**Architecture:** End-to-end raw waveform system. SincConv front-end followed by a Spectro-Temporal Graph Attention Network (ST-GAT) that constructs graph nodes across both spectral and temporal domains, then applies GAT layers with graph pooling. No SSL backbone required.

**Trained on:** ASVspoof 2019 LA

**Performance (ASVspoof 2019 LA eval):**
| Dataset | EER | min t-DCF |
| :--- | :--- | :--- |
| ASVspoof 2019 LA | 1.06% | 0.0335 |
| Speech DF Arena avg | 34.92% | — |

**Parameters:** 0.44M

**Links:**
- GitHub: https://github.com/eurecom-asp/RawGAT-ST-antispoofing
- Paper (arXiv): https://arxiv.org/abs/2107.12710

---

## 10. RawNet2

**Architecture:** End-to-end raw waveform model combining SincNet-style sinc filter bank with residual blocks, filter-wise feature map scaling (FMS) attention, and a GRU layer for sequential modeling. One of the first strong end-to-end anti-spoofing models without hand-crafted features.

**Trained on:** ASVspoof 2019 LA

**Performance:**
| Dataset | EER |
| :--- | :--- |
| ASVspoof 2019 LA | competitive |
| Speech DF Arena avg | 35.75% |

**Parameters:** ~17.6M

**Links:**
- GitHub: https://github.com/eurecom-asp/rawnet2-antispoofing
- Paper (arXiv): https://arxiv.org/abs/2011.01108

---

## 11. Nes2Net / Nes2Net-X

**Architecture:** Lightweight nested Res2Net-inspired back-end that processes high-dimensional SSL features without a dimensionality reduction layer, using WavLM Large or Wav2Vec2 as the foundation front-end. Nes2Net-X adds learnable weighted feature fusion. Particularly strong for singing voice deepfake detection (CtrSVDD 2024 task).

**Trained on:** CtrSVDD 2024 (primary); evaluated across ASVspoof 2019/2021/5, PartialSpoof, In-the-Wild

**Performance:**
| Dataset | EER |
| :--- | :--- |
| CtrSVDD 2024 | 2.02% |
| ASVspoof 2021 avg | 2.51–2.55% |

**Parameters:** ~318M total (dominated by WavLM backbone)

**Links:**
- GitHub: https://github.com/Liu-Tianchi/Nes2Net
- Paper (arXiv): https://arxiv.org/abs/2504.05657

---

## Summary Comparison

| Model | Backbone | Params | ASVspoof 2021 LA EER | ASVspoof 2021 DF EER | In-the-Wild EER | Arena Avg EER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fake-Mamba | XLS-R + BiMamba | ~300M | 0.97% | 1.74% | 5.85% | — |
| XLSR-Mamba | XLS-R + Dual-BiMamba | 319M | 0.93% | 1.88% | 6.71% | 14.21% |
| XLSR + SLS | XLS-R + MLP | ~340M | 2.87% | 1.92% | 7.46% | **13.84%** |
| Wav2Vec2-AASIST | XLS-R + Graph Attn | ~318M | **0.82%** | 2.85% | — | 18.02% |
| TCM | XLS-R + Conformer | ~319M | competitive | competitive | — | 15.77% |
| DF Arena 1B | RAPTOR (~1B) | ~1B | 4.66% | — | **0.91%** | **5.92%** |
| AASIST | Graph Attn (raw) | 300K | — | — | — | 34.49% |
| AASIST-L | Graph Attn (raw) | 85K | — | — | — | 34.49% |
| AASIST3 | KAN + GAT + Wav2Vec2 | N/A | — | — | — | — |
| RawGAT-ST | Spectro-Temporal GAT | 440K | — | — | — | 34.92% |
| RawNet2 | SincConv + GRU | 17.6M | — | — | — | 35.75% |
| Nes2Net-X | WavLM + Nested Res2Net | ~318M | 2.51–2.55% | — | — | — |

*Blank cells = not reported on that dataset. AASIST/RawGAT-ST/RawNet2 EERs above are on ASVspoof 2019 LA eval, not 2021.*

---

## Key Observations

1. **Best ASVspoof 2021 performance:** Fake-Mamba (0.97% LA) and XLSR-Mamba (0.93% LA) push the frontier by pairing XLS-R with bidirectional state-space models.
2. **Best real-world generalization:** DF Arena 1B leads on In-the-Wild (0.91% EER) due to its multi-dataset training regime. XLSR+SLS leads the 14-dataset arena average (13.84%).
3. **Best lightweight options:** AASIST-L (85K params) and RawGAT-ST (440K) are the only models viable for real-time CPU inference without a large SSL backbone.
4. **Critical caveat:** All models trained solely on ASVspoof 2019 degrade significantly on out-of-distribution 2024+ data. DF Arena 1B is currently the most deployment-ready option for real-world audio.
5. **Live leaderboard:** The [Speech DF Arena](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena) on HuggingFace provides standardized cross-dataset comparisons and is the best single reference for up-to-date rankings.
