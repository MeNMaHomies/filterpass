"""
Result reporting: console output and file writing.

Pure I/O layer — receives dicts from metrics.py and writes scores.txt / summary.txt.
No model or dataset assumptions.
"""

import os

_TELEPHONY_CONDITIONS = {"C2", "C3"}  # ASVspoof codec conditions of interest


def print_results(
    detection: dict,
    rtf: dict,
    scored_count: int,
    skipped_count: int,
    param_count: int,
    vram_load_gb: float,
    vram_peak_gb: float,
) -> None:
    print(f"\nScored:  {scored_count} utterances")
    print(f"Skipped: {skipped_count} utterances")

    print(f"\nRTF (per 500ms chunk, no VAD):")
    print(f"  Mean:   {rtf['rtf_mean']:.4f}  (target < 0.1)")
    print(f"  Median: {rtf['rtf_median']:.4f}")
    print(f"  p95:    {rtf['rtf_p95']:.4f}")
    print(f"  Max:    {rtf['rtf_max']:.4f}")
    print(f"\nInference latency per 500ms chunk:")
    print(f"  Mean:   {rtf['latency_mean_ms']:.1f} ms  (target < 50ms)")
    print(f"  Median: {rtf['latency_median_ms']:.1f} ms")
    print(f"  p95:    {rtf['latency_p95_ms']:.1f} ms")

    print(f"\n=== Results ===")
    print(f"EER:       {detection['eer_pct']:.2f}%")
    if detection.get("min_tDCF") is not None:
        print(f"min-tDCF:  {detection['min_tDCF']:.4f}")
    print(f"AUC-ROC:   {detection['auc_roc']:.4f}")
    print(f"FAR:       {100 * detection['far']:.2f}%")
    print(f"FRR:       {100 * detection['frr']:.2f}%")
    print(f"Accuracy:  {100 * detection['accuracy']:.2f}%")
    print(f"Precision: {detection['precision']:.4f}")
    print(f"Recall:    {detection['recall']:.4f}")
    print(f"F1:        {detection['f1']:.4f}")

    if detection.get("condition_eers"):
        print(f"\nPer-condition EER:")
        for cond, ceer in sorted(detection["condition_eers"].items()):
            marker = " <-- telephony" if cond in _TELEPHONY_CONDITIONS else ""
            print(f"  {cond}: {ceer:.2f}%{marker}")


def write_scores(out_dir: str, all_scores: dict[str, float]) -> str:
    path = os.path.join(out_dir, "scores.txt")
    with open(path, "w") as f:
        for utt_id, score in all_scores.items():
            f.write(f"{utt_id} {score:.6f}\n")
    return path


def write_summary(
    out_dir: str,
    model_name: str,
    dataset_name: str,
    phase: str,
    scored_count: int,
    skipped_count: int,
    param_count: int,
    vram_load_gb: float,
    vram_peak_gb: float,
    detection: dict,
    rtf: dict,
) -> str:
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(f"Model:            {model_name}\n")
        f.write(f"Dataset:          {dataset_name} (phase={phase})\n")
        f.write(f"Scored:           {scored_count} utterances\n")
        f.write(f"Skipped:          {skipped_count} utterances\n")
        f.write(f"\n--- Model ---\n")
        f.write(f"Parameters:       {param_count:,}\n")
        f.write(f"VRAM (load):      {vram_load_gb:.2f} GB\n")
        f.write(f"VRAM (peak):      {vram_peak_gb:.2f} GB\n")
        f.write(f"\n--- Anti-Spoofing ---\n")
        f.write(f"EER:              {detection['eer_pct']:.2f}%\n")
        if detection.get("min_tDCF") is not None:
            f.write(f"min-tDCF:         {detection['min_tDCF']:.4f}\n")
        f.write(f"AUC-ROC:          {detection['auc_roc']:.4f}\n")
        f.write(f"FAR (@ EER thr):  {100 * detection['far']:.2f}%\n")
        f.write(f"FRR (@ EER thr):  {100 * detection['frr']:.2f}%\n")
        f.write(f"\n--- Classification (@ EER threshold) ---\n")
        f.write(f"Accuracy:         {100 * detection['accuracy']:.2f}%\n")
        f.write(f"Precision:        {detection['precision']:.4f}\n")
        f.write(f"Recall:           {detection['recall']:.4f}\n")
        f.write(f"F1:               {detection['f1']:.4f}\n")
        if detection.get("condition_eers"):
            f.write(f"\n--- Per-Condition EER ---\n")
            for cond, ceer in sorted(detection["condition_eers"].items()):
                f.write(f"  {cond}: {ceer:.2f}%\n")
        f.write(f"\n--- Real-Time Performance (no VAD, per 500ms chunk) ---\n")
        f.write(f"RTF mean:         {rtf['rtf_mean']:.4f}\n")
        f.write(f"RTF median:       {rtf['rtf_median']:.4f}\n")
        f.write(f"RTF p95:          {rtf['rtf_p95']:.4f}\n")
        f.write(f"RTF max:          {rtf['rtf_max']:.4f}\n")
        f.write(f"Latency mean:     {rtf['latency_mean_ms']:.1f} ms\n")
        f.write(f"Latency median:   {rtf['latency_median_ms']:.1f} ms\n")
        f.write(f"Latency p95:      {rtf['latency_p95_ms']:.1f} ms\n")
    return path
