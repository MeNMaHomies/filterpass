# Deepfake Audio Datasets Summary

This document summarizes the primary datasets used in the development and benchmarking of our deepfake audio detection system.

---

# I. Training Datasets

These datasets are used for model training and development to help the system learn discriminative patterns between bona fide and spoofed speech.

## 1. ASVspoof 2019 (Logical Access)

The ASVspoof 2019 challenge was the first to consider all three major spoofing attack types (TTS, VC, and Replay) within a single challenge. Our research focuses on the **Logical Access (LA)** subset.

### Key Characteristics
*   **Focus**: Synthetic speech (TTS) and voice conversion (VC) attacks injected directly into a system (no acoustic propagation).
*   **Base Corpus**: Derived from the high-quality **VCTK** multi-speaker English corpus (96 kHz downsampled to 16 kHz).
*   **Attack Diversity**: Features 17 different TTS and VC systems.
    *   **Known Attacks (A01–A06)**: Used for training and development sets.
    *   **Unknown Attacks (A07–A19)**: Found only in the evaluation set to test generalization.
*   **Primary Metric**: Introduced the **tandem Detection Cost Function (t-DCF)**, measuring the combined performance of the countermeasure and a fixed ASV system.

### Dataset Partitions
| Partition | Speakers | Bona Fide Utterances | Spoofed Utterances |
| :--- | :--- | :--- | :--- |
| Training | 20 | 2,580 | 22,800 |
| Development | 20 | 2,548 | 22,296 |
| Evaluation | 67 | 7,355 | 63,882 |

---

## 2. WaveFake

Originally created by Joel Frank and Lea Schönherr, this dataset expands the diversity of synthetic speech architectures for benchmarking detection robustness.

### Key Characteristics
*   **Volume**: 104,885 generated audio clips (16-bit PCM WAV).
*   **Base Data**: Synthetic samples are trained on **LJSpeech** (English).
*   **Architectures**: Includes a wide variety of modern neural vocoders and end-to-end pipelines.

### Included Generative Models
*   ✅ **MelGAN**
*   ✅ **Parallel WaveGAN**
*   ✅ **Multi-Band MelGAN**
*   ✅ **Full-Band MelGAN**
*   ✅ **HiFi-GAN**
*   ✅ **WaveGlow**
*   ✅ **Full TTS Pipeline** (Conformer + Parallel WaveGAN)

### Significance
WaveFake is essential for training models that are resilient against "unseen" generative architectures, as it provides a broad spectrum of artifacts distinct from those found in the ASVspoof series.

---

# II. Benchmark Datasets

These datasets are used strictly for evaluation to measure the model's performance on unseen channel variability and telephony artifacts.

## 1. ASVspoof 2021 (Logical Access)

The 2021 edition significantly increased the difficulty of the LA task by introducing **channel and compression variability** to simulate real-world telephony scenarios.

### Key Characteristics
*   **Scenario**: Simulates deepfake detection over **VoIP** and **PSTN** networks.
*   **Challenges**: Audio contains encoding artifacts, packet loss effects, and bandwidth limitations (8 kHz vs 16 kHz).
*   **Generalization Test**: No matched training/development data was provided; models must generalize from 2019 data to the degraded 2021 conditions.

### Evaluation Conditions (C1–C7)
| Condition | Codec | Bandwidth | Transmission |
| :--- | :--- | :--- | :--- |
| **LA-C1** | None | 16 kHz | Clean (Baseline) |
| **LA-C2** | a-law | 8 kHz | VoIP |
| **LA-C3** | µ-law / Mixed | 8 kHz | PSTN + VoIP |
| **LA-C4** | G.722 | 16 kHz | VoIP |
| **LA-C5** | TBA (Unknown) | 8 kHz | VoIP |
| **LA-C6** | TBA (Unknown) | 8 kHz | VoIP |
| **LA-C7** | TBA (Unknown) | 16 kHz | VoIP |

---

## 2. ASVspoof 5 (2024)

Consolidates previous LA and DF tasks into a single challenge with two tracks, focusing on crowdsourced data and adversarial attacks.

### Key Characteristics
*   **Corpus**: Transitioned to **Multilingual LibriSpeech (MLS)**, featuring 2,000+ speakers in diverse real-world acoustic environments.
*   **Adversarial Attacks**: Features attacks designed specifically to bypass the detector's internal mechanisms.
*   **Tracks**:
    *   **Track 1**: Stand-alone deepfake detection (comparable to previous DF tracks).
    *   **Track 2**: Spoofing-Robust ASV (SASV) which integrates verification and detection.

---

## 3. SpeechFake (English BD)

A massive-scale dataset designed to cover the linguistic gaps of previous benchmarks. Our focus is strictly on the **Bilingual Dataset (BD)** subset for the **English** language.

### Key Characteristics
*   **Focus**: **English language** samples from the Bilingual Dataset.
*   **Diversity**: Employs 40 different synthesis tools (TTS, VC, and Neural Vocoders) to simulate modern English-language deepfakes.
*   **Significance**: Essential for building a system robust enough for complex English dialects and varied synthesis artifacts.

---

## 4. In-the-Wild (ITW - English)

Unlike synthetic research datasets, ITW datasets consist of actual deepfakes sourced from uncontrolled environments. We focus on **English-language** figures.

### Key Characteristics
*   **Sources**: High-profile **English-speaking** figures (politicians, celebrities) collected from YouTube, X, and Facebook.
*   **Significance**: Tests how well the model handles "unseen" distortion like background music and social media compression on native English speech.

---

## 5. CD-ADD (English Cross-Domain)

Focuses on the generalization capability of detectors when moving across different recording environments and zero-shot synthesis models, with a focus on **English** datasets.

### Key Characteristics
*   **Focus**: **English** speech generated by advanced zero-shot TTS models.
*   **Robustness**: Tests model performance against synthesis techniques that only require a very short (<3 second) English voice sample.

---

## 6. ADD Challenges (English Subsets)

The Audio Deep Synthesis Detection (ADD) challenge series. We specialize in the **English-language** tasks and subsets of these challenges.

### Key Characteristics
*   **ADD 2022**: Focus on **English** Low-Quality (LF) detection and Partially Fake (PF) detection.
*   **ADD 2023**: Focus on English Deepfake Algorithm Recognition and manipulation Localization.

---

## 7. CompSpoof (English - 2025)

An emerging benchmark focusing on the latest generation of "low-artifact" compositional spoofing in **English**.

### Key Characteristics
*   **Focus**: **English-language** compositional spoofing (attacks combining multiple AI techniques with post-processing).
*   **Challenge**: Designed to push detectors toward more granular and explainable forensics on modern English generative models.

---
# III. Notes

![alt text](./images/image.png)