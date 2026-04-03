# Research Methodology: Deepfake Audio Detection & Generalization Evaluation

## 1. Objective
To develop a robust deepfake audio detection model and rigorously evaluate its cross-domain generalization, real-time telephony survival, and resilience against zero-shot text-to-speech (TTS) synthesis. The evaluation will benchmark our custom architecture against both official dataset baselines and modern State-of-the-Art (SOTA) open-source models.

---

## 2. Dataset Strategy

The datasets are strictly partitioned to ensure the model is evaluated on unseen acoustic conditions, unknown generative architectures, and realistic channel degradations.

### 2.1 Training Datasets (Learning Discriminative Patterns)
* **ASVspoof 2019 (Logical Access):** Provides the foundational clean data (TTS and Voice Conversion) for learning baseline synthetic artifacts.
* **WaveFake:** Expands exposure to diverse neural vocoders (MelGAN, HiFi-GAN, WaveGlow) to prevent overfitting to ASVspoof 2019's specific synthesis methods.

### 2.2 Benchmark Datasets (Evaluation & Cross-Domain Testing)
* **ASVspoof 2021 (LA):** Stress-tests the model against telephony artifacts, VoIP degradation, and packet loss.
* **ASVspoof 5 (2024):** Evaluates performance against crowdsourced acoustic environments and modern adversarial attacks.
* **SpeechFake (English BD):** Tests linguistic and algorithmic robustness using 40 different synthesis tools.
* **In-the-Wild (English):** Measures domain-shift vulnerability against uncontrolled social media audio and background noise.
* **CD-ADD (English Cross-Domain):** Benchmarks survival rates against advanced zero-shot TTS models requiring minimal voice samples.
* **ADD Challenges (English):** Tests localization of partial spoofing and low-quality deepfakes.
* **CompSpoof (English 2025):** Evaluates detection of compositional spoofing (combinations of multiple AI techniques and post-processing).

---

## 3. Evaluation Matrix & Baseline Comparison

To establish absolute performance context, our model will be evaluated across a two-tier comparison matrix without retraining external architectures.

### Tier 1: The Official Baselines (The Performance Floor)
For each benchmark dataset, our model's metrics (EER and min t-DCF) will be directly compared to the dataset's official baseline (e.g., LFCC-LCNN for 2021, AASIST for 2024). This proves fundamental competency and architectural improvement over standard methods for the specific task.

### Tier 2: Pre-Trained SOTA Competitors (The SOTA Ceiling)
To determine true competitiveness, we will run inference using 3 to 4 highly cited, open-source SOTA models (e.g., Wav2Vec 2.0 or HuBERT-based detectors).
* **Method:** Download official pre-trained weights (`.pt` or `.pth`).
* **Execution:** Run these pre-trained models exclusively in inference mode across our benchmark datasets.
* **Goal:** Map exactly where our custom architecture outperforms current SOTA models (e.g., inference speed, cross-domain accuracy drop-off).

---

## 4. Execution Pipeline & Engineering Controls

To ensure mathematical consistency and manage compute resources efficiently, the evaluation pipeline will enforce the following technical constraints:

1.  **Isolated Inference Environments:** Each SOTA competitor model will be executed within its own isolated Conda environment or Docker container to prevent PyTorch/torchaudio dependency conflicts.
2.  **Unified Scoring Script:** Raw output probability scores from all models (our custom model, baselines, and SOTA competitors) will be funneled into a single, standardized ASVspoof evaluation script. This guarantees mathematically identical EER and t-DCF calculations across the board.
3.  **Micro-Benchmarking:** Before executing full dataset runs, the end-to-end pipeline will be validated using a balanced micro-dataset (approx. 1,000 files per corpus) to verify data loaders, memory management, and scoring handshakes.