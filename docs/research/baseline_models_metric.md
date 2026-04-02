# ASVspoof 2021

| Track / Task | Baseline Model | Feature | EER (%) | min t-DCF |
| :--- | :--- | :--- | :--- | :--- |
| Logical Access (LA) | B1 (CQCC-GMM) | CQCC | 15.62 | 0.4974 |
| Logical Access (LA) | B2 (LFCC-GMM) | LFCC | 19.30 | 0.5758 |
| Logical Access (LA) | B3 (LFCC-LCNN) | LFCC | 9.26 | 0.3445 |
| Logical Access (LA) | B4 (RawNet2) | Raw Audio | 9.50 | 0.4257 |

---

# ASVspoof 5

| Track / Task | Baseline Model | Feature | EER / t-EER (%) | minDCF / tDCF / a-DCF |
| :--- | :--- | :--- | :--- | :--- |
| Track 1 (Stand-alone CM) | B01 (RawNet2) | Raw Audio | 36.04 (EER) | 0.8266 (minDCF) |
| Track 1 (Stand-alone CM) | B02 (AASIST) | Raw Audio | 29.12 (EER) | 0.7106 (minDCF) |
| Track 2 (SASV) | B03 (Fusion-based) | Raw Audio / Fusion | 28.78 (t-EER) | 0.6806 (min a-DCF) / 0.9295 (min t-DCF) |
| Track 2 (SASV) | B04 (Single integrated)| Raw Audio | - | 0.5741 (min a-DCF) |

---

# COMPSPOOF

| Track / Task | Baseline Model | Feature | F1-Score |
| :--- | :--- | :--- | :--- |
| Component-level spoofing | XLSR-AASIST | Wav2Vec2.0 XLSR | 0.827 (Eval F1), 0.840 (Dev F1) |

---

# ADD 2023

| Track / Task | Baseline Model | Feature | Metric Score |
| :--- | :--- | :--- | :--- |
| Track 1.2 (Detection) | S01 (LFCC-GMM) | LFCC | 53.04 (WEER %) |
| Track 1.2 (Detection) | S02 (LFCC-LCNN) | LFCC | 66.72 (WEER %) |
| Track 1.2 (Detection) | S03 (Wav2vec2-LCNN) | Wav2Vec2 | 30.05 (WEER %) |
| Track 2 (Region Location) | S04 (LFCC-LCNN) | LFCC | 42.25 (Score %) |
| Track 3 (Algorithm Recog.) | S05 (LFCC-ResNet) | LFCC | 53.50 (F1 %) |
| Track 3 (Algorithm Recog.) | S06 (OpenMax) | LFCC | 54.16 (F1 %) |

---

# "In-the-Wild"

| Track / Task | Baseline Model | Feature | EER (%) |
| :--- | :--- | :--- | :--- |
| ASVspoof19 Eval | RawGAT-ST (Full input) | Raw Audio | 1.229 |
| In-the-Wild Eval | RawPC (Full input) | Raw Audio | 45.715 |

*(Note: On the In-the-wild dataset, RawGAT-ST scored 37.154%, and RawPC scored 45.715%. The previous value of 15.715 was a typo)*

---

# SpeechFake

| Track / Task | Baseline Model | Feature | EER (%) |
| :--- | :--- | :--- | :--- |
| Bilingual Dataset (BD) | AASIST (trained on BD) | Raw Audio | 3.48 (Test on BD) |
| Bilingual Dataset (BD) | W2V+AASIST (trained on BD) | Wav2Vec2.0 XLSR | 3.54 (Test on BD) / 2.83 (Test on BD-CN) |

*(Note: Prior draft recorded 39.07% for AASIST which corresponds to AASIST trained on ASV19 evaluated on BD-CN)*

---

# CD-ADD

| Track / Task | Baseline Model | Feature | EER (%) |
| :--- | :--- | :--- | :--- |
| In-model Evaluation | Wav2Vec2-base | Wav2Vec2 | 0.07 (Libri), 0.12 (TED) |
| Cross-model Evaluation | Wav2Vec2-base | Wav2Vec2 | 7.85 (Libri), 21.40 (TED) |