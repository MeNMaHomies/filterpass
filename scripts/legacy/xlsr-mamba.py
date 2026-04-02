import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Paths ---
FILTERPASS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = os.path.normpath(os.path.join(FILTERPASS_DIR, "..", "XLSR-Mamba"))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, REPO_PATH)
sys.path.insert(0, SCRIPTS_DIR)

try:
    import eval_metrics as em
    from model import Model
except ImportError as e:
    print(f"Import Error: {e}. Is mamba-ssm installed? (Linux/WSL2 + CUDA required)")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="XLSR-Mamba benchmark on ASVspoof 2021 LA eval"
    )
    p.add_argument(
        "--eval_dir",
        default="data/ASVspoof2021_LA_eval/flac",
        help="Path to directory containing .flac eval files",
    )
    p.add_argument(
        "--keys_dir",
        default="data/keys/LA",
        help="Path to keys/LA directory (contains CM/ and ASV/ subdirs)",
    )
    p.add_argument(
        "--phase", default="eval", help="Phase to evaluate: eval, progress, or hidden"
    )
    p.add_argument(
        "--out_dir",
        default="results/xlsr-mamba",
        help="Directory to write scores and summary",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Number of chunks to batch per model forward pass",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader worker processes for parallel audio loading",
    )
    p.add_argument("--emb_size", type=int, default=144)
    p.add_argument("--num_encoders", type=int, default=12)
    return p.parse_args()


def load_model(emb_size, num_encoders, device):
    print("Locating weights from Hugging Face...")
    model_path = hf_hub_download(
        repo_id="AustinXiao/XLSR-Mamba-LA", filename="model.safetensors"
    )
    print(f"Weights found at: {model_path}")

    model_args = argparse.Namespace(emb_size=emb_size, num_encoders=num_encoders)
    print("Instantiating model architecture...")
    model = Model(model_args, device)

    print("Loading safetensors weights...")
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model ready.")
    return model


TARGET_SR = 16000
CHUNK_DURATION_S = 0.5
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION_S)  # 8000

# --- Tuning ---
NUM_WORKERS = 8  # parallel audio loading processes (tune to CPU core count; try 10 if GPU util < 80%)
BATCH_SIZE = 32  # chunks per model forward pass (reduce to 24 if CUDA OOM)


def _load_and_segment_worker(flac_path):
    """
    Runs in DataLoader worker processes (CPU).
    Loads the full utterance, splits into CHUNK_SAMPLES-sized chunks (no VAD),
    and pads the final chunk if needed.
    Returns (utt_id, chunks_tensor) where chunks_tensor is (N, CHUNK_SAMPLES) float32,
    or (utt_id, None) on failure.
    """
    utt_id = os.path.splitext(os.path.basename(flac_path))[0]
    try:
        try:
            audio, sr = sf.read(flac_path, dtype="float32", always_2d=False)
        except Exception as sf_err:
            print(f"[FALLBACK:TORCHAUDIO] {utt_id}: {sf_err}", flush=True)
            import torchaudio
            waveform, sr = torchaudio.load(flac_path)
            audio = waveform[0].numpy()

        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        if len(audio) == 0:
            return utt_id, None

        # Split into fixed-size chunks, pad the last one
        chunks = []
        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i : i + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            chunks.append(chunk)

        return utt_id, torch.from_numpy(np.stack(chunks))  # (N, 8000)
    except Exception as e:
        print(f"[WORKER ERROR] {utt_id}: {type(e).__name__}: {e}", flush=True)
        return utt_id, None


class UtteranceDataset(Dataset):
    """Each item is a path to a FLAC file. Workers handle loading + segmentation."""

    def __init__(self, flac_paths):
        self.paths = flac_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return _load_and_segment_worker(self.paths[idx])


def collate_utterances(batch):
    """Pass utterances through individually — chunk counts differ per file."""
    return batch  # list of (utt_id, tensor_or_None)


def load_cm_metadata(keys_dir, phase):
    """Returns DataFrame filtered to phase with cols: utt_id(1), label(5), codec(6)."""
    cm_file = os.path.join(keys_dir, "CM", "trial_metadata.txt")
    if not os.path.exists(cm_file):
        print(f"ERROR: CM keys not found: {cm_file}")
        sys.exit(1)
    meta = pd.read_csv(cm_file, sep=" ", header=None)
    meta_phase = meta[meta[7] == phase]
    if len(meta_phase) == 0:
        print(f"WARNING: No entries for phase='{phase}'. Using all entries.")
        meta_phase = meta
    print(f"CM trials loaded ({phase}): {len(meta_phase)}")
    return meta_phase


def load_asv_error_rates(keys_dir, phase):
    """
    Load ASV trial metadata and scores, return (Pfa_asv, Pmiss_asv, Pfa_spoof_asv)
    needed for min-tDCF computation.
    """
    asv_key_file = os.path.join(keys_dir, "ASV", "trial_metadata.txt")
    asv_scr_file = os.path.join(keys_dir, "ASV", "ASVTorch_Kaldi", "score.txt")

    if not os.path.exists(asv_key_file) or not os.path.exists(asv_scr_file):
        print("WARNING: ASV keys/scores not found, skipping min-tDCF.")
        return None, None, None

    asv_key = pd.read_csv(asv_key_file, sep=" ", header=None)
    asv_scr = pd.read_csv(asv_scr_file, sep=" ", header=None)

    phase_mask = asv_key[7] == phase
    asv_scr_phase = asv_scr[phase_mask]

    idx_tar = asv_key[phase_mask][5] == "target"
    idx_non = asv_key[phase_mask][5] == "nontarget"
    idx_spoof = asv_key[phase_mask][5] == "spoof"

    tar_asv = asv_scr_phase[2][idx_tar].values
    non_asv = asv_scr_phase[2][idx_non].values
    spoof_asv = asv_scr_phase[2][idx_spoof].values

    _, asv_threshold = em.compute_eer(tar_asv, non_asv)
    Pfa_asv, Pmiss_asv, _, Pfa_spoof_asv = em.obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )
    return Pfa_asv, Pmiss_asv, Pfa_spoof_asv


def compute_all_metrics(all_scores, meta_phase, Pfa_asv, Pmiss_asv, Pfa_spoof_asv):
    """Compute all metrics and return a dict."""
    results = {}

    # Build aligned score/label arrays
    utt_to_label = dict(zip(meta_phase[1], meta_phase[5]))
    utt_to_codec = dict(zip(meta_phase[1], meta_phase[6]))

    scored_ids = list(all_scores.keys())
    scores_arr = np.array([all_scores[u] for u in scored_ids])
    labels_arr = np.array([utt_to_label[u] for u in scored_ids])

    bona_scores = scores_arr[labels_arr == "bonafide"]
    spoof_scores = scores_arr[labels_arr == "spoof"]

    # --- EER ---
    eer, eer_threshold = em.compute_eer(bona_scores, spoof_scores)
    results["eer_pct"] = 100 * eer
    results["eer_threshold"] = eer_threshold

    # --- min-tDCF ---
    if Pfa_asv is not None:
        Pspoof = 0.05
        cost_model = {
            "Pspoof": Pspoof,
            "Ptar": (1 - Pspoof) * 0.99,
            "Pnon": (1 - Pspoof) * 0.01,
            "Cmiss": 1,
            "Cfa": 10,
            "Cfa_spoof": 10,
        }
        tDCF_curve, _ = em.compute_tDCF(
            bona_scores,
            spoof_scores,
            Pfa_asv,
            Pmiss_asv,
            Pfa_spoof_asv,
            cost_model,
            False,
        )
        results["min_tDCF"] = float(np.min(tDCF_curve))
    else:
        results["min_tDCF"] = None

    # --- AUC-ROC ---
    binary_labels = (labels_arr == "bonafide").astype(int)
    results["auc_roc"] = roc_auc_score(binary_labels, scores_arr)

    # --- FAR / FRR at EER threshold ---
    preds = (scores_arr >= eer_threshold).astype(int)
    results["far"] = float(
        np.sum((preds == 1) & (binary_labels == 0)) / max(np.sum(binary_labels == 0), 1)
    )
    results["frr"] = float(
        np.sum((preds == 0) & (binary_labels == 1)) / max(np.sum(binary_labels == 1), 1)
    )

    # --- Accuracy, Precision, Recall, F1 at EER threshold ---
    results["accuracy"] = accuracy_score(binary_labels, preds)
    results["precision"] = precision_score(binary_labels, preds, zero_division=0)
    results["recall"] = recall_score(binary_labels, preds, zero_division=0)
    results["f1"] = f1_score(binary_labels, preds, zero_division=0)

    # --- Per-condition EER (C1–C7) ---
    condition_eers = {}
    codecs = [utt_to_codec[u] for u in scored_ids]
    codecs_arr = np.array(codecs)
    for cond in sorted(set(codecs_arr)):
        mask = codecs_arr == cond
        c_bona = scores_arr[mask & (labels_arr == "bonafide")]
        c_spoof = scores_arr[mask & (labels_arr == "spoof")]
        if len(c_bona) > 0 and len(c_spoof) > 0:
            c_eer, _ = em.compute_eer(c_bona, c_spoof)
            condition_eers[cond] = 100 * c_eer
    results["condition_eers"] = condition_eers

    return results


def main():
    args = parse_args()

    eval_dir = (
        os.path.join(FILTERPASS_DIR, args.eval_dir)
        if not os.path.isabs(args.eval_dir)
        else args.eval_dir
    )
    keys_dir = (
        os.path.join(FILTERPASS_DIR, args.keys_dir)
        if not os.path.isabs(args.keys_dir)
        else args.keys_dir
    )
    out_dir = (
        os.path.join(FILTERPASS_DIR, args.out_dir)
        if not os.path.isabs(args.out_dir)
        else args.out_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = load_model(args.emb_size, args.num_encoders, device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {param_count:,}")

    vram_after_load_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    print(f"VRAM after model load: {vram_after_load_gb:.2f} GB")

    meta_phase = load_cm_metadata(keys_dir, args.phase)
    Pfa_asv, Pmiss_asv, Pfa_spoof_asv = load_asv_error_rates(keys_dir, args.phase)

    os.makedirs(out_dir, exist_ok=True)
    score_path = os.path.join(out_dir, "scores.txt")
    open(score_path, "w").close()

    all_scores = {}
    all_rtf = []
    skipped = 0

    utt_ids = meta_phase[1].tolist()
    existing = [os.path.join(eval_dir, u + ".flac") for u in utt_ids]

    print(
        f"\nRunning inference on {len(existing)} utterances "
        f"({args.num_workers} loader workers, batch_size={args.batch_size})..."
    )

    dataset = UtteranceDataset(existing)
    loader = DataLoader(
        dataset,
        batch_size=1,  # one utterance at a time from the loader
        num_workers=args.num_workers,
        collate_fn=collate_utterances,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    with open(score_path, "a") as fout:
        for batch in tqdm(loader, total=len(existing)):
            utt_id, chunks_tensor = batch[0]  # collate returns list of (id, tensor)

            if chunks_tensor is None:
                skipped += 1
                continue

            chunks_tensor = chunks_tensor.to(device)  # (N, 8000)
            scores = []
            rtf_list = []

            with torch.no_grad():
                for i in range(0, len(chunks_tensor), args.batch_size):
                    x = chunks_tensor[i : i + args.batch_size]
                    n = len(x)
                    t0 = time.perf_counter()
                    out = model(x)  # (n, 2)
                    elapsed = time.perf_counter() - t0
                    scores.extend(out[:, 1].cpu().numpy().tolist())
                    rtf_list.append(elapsed / (n * CHUNK_DURATION_S))

            score = float(np.mean(scores))
            all_scores[utt_id] = score
            all_rtf.extend(rtf_list)
            fout.write(f"{utt_id} {score:.6f}\n")

    vram_peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )

    print(f"\nScored:  {len(all_scores)} utterances")
    print(f"Skipped: {skipped} utterances (missing file or no voiced audio)")

    if not all_scores:
        print("No scores produced. Check --eval_dir and --keys_dir paths.")
        sys.exit(1)

    # --- RTF / Latency ---
    rtf_array = np.array(all_rtf)
    latency_mean_ms = rtf_array.mean() * CHUNK_DURATION_S * 1000
    latency_median_ms = np.median(rtf_array) * CHUNK_DURATION_S * 1000
    latency_p95_ms = np.percentile(rtf_array, 95) * CHUNK_DURATION_S * 1000

    print(f"\nRTF (model inference, per 500ms chunk, no VAD):")
    print(f"  Mean:   {rtf_array.mean():.4f}  (target < 0.1)")
    print(f"  Median: {np.median(rtf_array):.4f}")
    print(f"  p95:    {np.percentile(rtf_array, 95):.4f}")
    print(f"  Max:    {rtf_array.max():.4f}")
    print(f"\nInference latency per 500ms chunk:")
    print(f"  Mean:   {latency_mean_ms:.1f} ms  (target < 50ms)")
    print(f"  Median: {latency_median_ms:.1f} ms")
    print(f"  p95:    {latency_p95_ms:.1f} ms")

    # --- All accuracy metrics ---
    metrics = compute_all_metrics(
        all_scores, meta_phase, Pfa_asv, Pmiss_asv, Pfa_spoof_asv
    )

    print(f"\n=== Results ===")
    print(f"EER:       {metrics['eer_pct']:.2f}%")
    if metrics["min_tDCF"] is not None:
        print(f"min-tDCF:  {metrics['min_tDCF']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"FAR:       {100*metrics['far']:.2f}%")
    print(f"FRR:       {100*metrics['frr']:.2f}%")
    print(f"Accuracy:  {100*metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"\nPer-condition EER:")
    for cond, ceer in sorted(metrics["condition_eers"].items()):
        marker = " <-- telephony" if cond in ("C2", "C3") else ""
        print(f"  {cond}: {ceer:.2f}%{marker}")

    # --- Save summary ---
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Model:            XLSR-Mamba (AustinXiao/XLSR-Mamba-LA)\n")
        f.write(f"Dataset:          ASVspoof 2021 LA eval (phase={args.phase})\n")
        f.write(f"Scored:           {len(all_scores)} utterances\n")
        f.write(f"Skipped:          {skipped} utterances\n")
        f.write(f"\n--- Model ---\n")
        f.write(f"Parameters:       {param_count:,}\n")
        f.write(f"VRAM (load):      {vram_after_load_gb:.2f} GB\n")
        f.write(f"VRAM (peak):      {vram_peak_gb:.2f} GB\n")
        f.write(f"\n--- Anti-Spoofing ---\n")
        f.write(f"EER:              {metrics['eer_pct']:.2f}%\n")
        if metrics["min_tDCF"] is not None:
            f.write(f"min-tDCF:         {metrics['min_tDCF']:.4f}\n")
        f.write(f"AUC-ROC:          {metrics['auc_roc']:.4f}\n")
        f.write(f"FAR (@ EER thr):  {100*metrics['far']:.2f}%\n")
        f.write(f"FRR (@ EER thr):  {100*metrics['frr']:.2f}%\n")
        f.write(f"\n--- Classification (@ EER threshold) ---\n")
        f.write(f"Accuracy:         {100*metrics['accuracy']:.2f}%\n")
        f.write(f"Precision:        {metrics['precision']:.4f}\n")
        f.write(f"Recall:           {metrics['recall']:.4f}\n")
        f.write(f"F1:               {metrics['f1']:.4f}\n")
        f.write(f"\n--- Per-Condition EER ---\n")
        for cond, ceer in sorted(metrics["condition_eers"].items()):
            f.write(f"  {cond}: {ceer:.2f}%\n")
        f.write(f"\n--- Real-Time Performance (no VAD, per 500ms chunk) ---\n")
        f.write(f"RTF mean:         {rtf_array.mean():.4f}\n")
        f.write(f"RTF median:       {np.median(rtf_array):.4f}\n")
        f.write(f"RTF p95:          {np.percentile(rtf_array, 95):.4f}\n")
        f.write(f"RTF max:          {rtf_array.max():.4f}\n")
        f.write(f"Latency mean:     {latency_mean_ms:.1f} ms\n")
        f.write(f"Latency median:   {latency_median_ms:.1f} ms\n")
        f.write(f"Latency p95:      {latency_p95_ms:.1f} ms\n")

    print(f"\nResults saved to {out_dir}/")
    print(f"  scores.txt  — per-utterance scores")
    print(f"  summary.txt — full metrics summary")


if __name__ == "__main__":
    main()
    print(f"\nResults saved to {out_dir}/")
    print(f"  scores.txt  — per-utterance scores")
    print(f"  summary.txt — full metrics summary")


if __name__ == "__main__":
    main()
