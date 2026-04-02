# Pre-trained Model Comparison for Real-Time Deepfake Detection

This document evaluates candidate architectures for our custom custom model, focusing on the trade-off between **detection accuracy (EER)** and **real-time inference latency**.

## 1. Candidate Models Overview

| Model | Params | Input Type | Real-Time Suitability | Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| **AASIST-L** | ~85k | Raw Audio | **Excellent** | Ultra-lightweight; SOTA graph-based tech. |
| **RawNet2** | ~1.4M | Raw Audio | **Good** | Simple end-to-end pipeline; robust on 2019 data. |
| **LCNN** | ~2M | LFCC/CQCC | **Moderate** | Reliable baseline; requires CPU-heavy extraction. |
| **Wav2Vec 2.0 Base** | ~95M | Raw Audio | **Low** | Maximum accuracy; high hardware requirements. |

---

## 2. Detailed Evaluation

### AASIST-L (Integrated Spectral-Temporal Graph)
- **Parameters**: 85,000
- **Generalization**: High. It uses sophisticated graph neural networks to align features across time and frequency.
- **Latency**: Very low. It can easily run on a mid-range CPU (or even mobile) with sub-10ms inference per 1-second chunk.
- **Verdict**: **Primary Recommended Candidate.** It offers the best balance for a 3-person team aiming for real-time deployment.

### RawNet2
- **Parameters**: 1,400,000
- **Pros**: It processes raw waveforms directly using Sinc-convolutions. This eliminates the need for manual feature extraction code, making the pipeline cleaner.
- **Cons**: Less accurate than AASIST-L on modern benchmarks like ASVspoof 2021.
- **Verdict**: **Strong Backup.** A great choice if you find graph networks too complex to implement.

### LCNN (Lightweight CNN)
- **Parameters**: 2,000,000
- **Pros**: The industry standard "classic" lightweight backbone.
- **Cons**: Requires **LFCC or CQCC** as input. This means your final app must include a high-performance library (like `librosa` or `torchaudio`) to convert raw audio to features, which adds significant processing overhead and code complexity.
- **Verdict**: Only use if you are committed to manual feature engineering.

### Wav2Vec 2.0 Base (and Base-960h)
- **Parameters**: 95,000,000
- **Note on Variants**: You might come across **Wav2Vec2-Base-960h** (trained on 960 hours of Librispeech data). It has the **exact same architecture, weight size, and latency** (~95M parameters) as the standard Base model. The only difference is that its weights have been fine-tuned for speech recognition.
- **Pros**: Uses self-supervised pre-training on thousands of hours of speech. It is nearly impossible to beat for accuracy.
- **Cons**: Massive memory footprint (~380MB for weights). Inference is slow on CPU.
- **Verdict**: **Too Heavy for Real-Time on Standard Hardware.** We could potentially use a "frozen" version of this as a feature extractor, but it would likely blow your latency budget for live phone calls.

---

## 3. Recommended Strategy

Given your team's goal of **real-time detection for phone calls**, we recommend focusing and fine-tuning on **AASIST-L**.

### Next Steps for the Team:
1.  **Search Hugging Face**: Look for `"aasist"` or `"aasist-light"`.
2.  **Verify Input Format**: Ensure your Data Augmentation process outputs 16kHz raw audio arrays, as this is the standard input for AASIST.
3.  **Benchmark Early**: Measure the inference time of a single forward pass on your target hardware (e.g., your laptops) to establish your "latency baseline."

> [!TIP]
> Since we are targeting **ASVspoof 2021** (codec artifacts), AASIST-L is particularly well-suited because its graph-based attention can learn to ignore certain frequency-band artifacts introduced by telephony codecs.
