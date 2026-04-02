# Deepfake Audio Detection Metrics

When evaluating models for real-time deepfake audio detection, we assess them across three distinct categories: overarching anti-spoofing performance, real-time feasibility and speed, and standard classification metrics.

## 1. Anti-Spoofing Metrics

These are the specialized metrics used in challenges like ASVspoof to determine how well a model discriminates between genuine (bona fide) speech and synthetic or converted speech (spoofs).

*   **EER (Equal Error Rate)**: The critical metric where the False Acceptance Rate (FAR) and False Rejection Rate (FRR) are identical. A lower EER indicates a more accurate system overall.
*   **min-tDCF (Minimum Tandem Detection Cost Function)**: A metric defined by the ASVspoof challenge that evaluates the cost of errors when the countermeasure (your deepfake detector) operates in tandem with a theoretical Automatic Speaker Verification (ASV) system. Lower is better.
*   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Measures the system's ability to distinguish between the spoof and genuine classes across all possible decision thresholds. 1.0 is perfect, 0.5 is random guessing.
*   **FAR vs FRR (False Acceptance Rate vs False Rejection Rate)**: 
    *   **FAR**: The percentage of spoofed audio incorrectly classified as genuine (false positives).
    *   **FRR**: The percentage of genuine audio incorrectly classified as spoofed (false negatives).

## 2. Real-Time Metrics

For live meetings and phone calls, operational speed and memory constraints are just as important as accuracy.

*   **RTF (Real-Time Factor)**: The ratio of the processing time to the length of the audio being processed (`Processing Time / Audio Length`). 
    *   *Target:* Ideally `< 0.1` (e.g., `< 100ms` to process `1 second` of audio), ensuring no backlog builds up during streaming.
*   **Inference Latency**: The absolute time delay (in milliseconds) it takes for the model to process a single audio chunk (e.g., a 1-second buffer) and return a verdict.
*   **Parameter Count**: The total number of trainable weights within the model's architecture. Directly affects memory holding size and computational overhead.
*   **VRAM / Memory Usage**: The amount of GPU or CPU RAM consumed during active inference by the loaded model and its immediate tensor activations.

## 3. General Classification Metrics

Standard machine learning evaluation metrics used to quantify the correctness of the model's final classifications.

*   **Accuracy**: The overall percentage of correctly classified predictions (both real and fake) out of all predictions made.
*   **Precision (Positive Predictive Value)**: The proportion of positive identifications (flagged as fake) that were actually correct. High precision means very few false alarms.
*   **Recall (Sensitivity / True Positive Rate)**: The proportion of actual fakes that were correctly identified. High recall means very few fakes slip past the system undetected.
*   **F1 Score**: The harmonic mean of Precision and Recall. Useful for finding the optimal balance when there is an uneven class distribution between genuine and spoofed data.
