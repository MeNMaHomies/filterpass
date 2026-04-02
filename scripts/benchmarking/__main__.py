"""
Model-agnostic, dataset-agnostic benchmark CLI.

Usage:
    python -m scripts.benchmarking \\
        --model   xlsr-mamba \\
        --dataset asvspoof2021-la \\
        --eval_dir  data/ASVspoof2021_LA_eval/flac \\
        --keys_dir  data/keys/LA \\
        --out_dir   results/xlsr-mamba

Adding a new model:
    1. Implement BenchmarkModel in scripts/benchmarking/models/<name>.py
    2. Register it in scripts/benchmarking/models/__init__.py
    3. Pass --model <name> on the CLI

Adding a new dataset:
    1. Implement DatasetAdapter in scripts/benchmarking/datasets/<name>.py
    2. Register it in scripts/benchmarking/datasets/__init__.py
    3. Pass --dataset <name> on the CLI
"""

import argparse
import os
import sys

import torch

from .audio_loader import build_loader
from .config import BenchmarkConfig
from .inference import run_inference
from .metrics import compute_detection_metrics, compute_rtf_stats
from .models import REGISTRY as MODEL_REGISTRY
from .datasets import REGISTRY as DATASET_REGISTRY
from . import reporter


FILTERPASS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(FILTERPASS_DIR, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deepfake audio detection benchmark")

    p.add_argument("--model",   required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))

    # Paths — passed directly to the dataset adapter constructor
    p.add_argument("--eval_dir", required=True,
                   help="Directory containing evaluation audio files")
    p.add_argument("--keys_dir", default=None,
                   help="Dataset key/metadata directory (if required by the dataset adapter)")

    p.add_argument("--out_dir",     default=None,
                   help="Output directory (defaults to results/<model>-<dataset>)")
    p.add_argument("--phase",       default="eval")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)

    # Model repo path (for models that require an external code checkout)
    p.add_argument("--repo_path", default=None,
                   help="Path to the model's external repo (added to sys.path)")

    # XLSR-Mamba specific — ignored by other adapters
    p.add_argument("--emb_size",     type=int, default=144)
    p.add_argument("--num_encoders", type=int, default=12)

    return p


def _build_model(args):
    adapter_cls = MODEL_REGISTRY[args.model]
    kwargs = {}

    if args.model == "xlsr-mamba":
        kwargs["repo_path"] = args.repo_path or os.path.normpath(
            os.path.join(FILTERPASS_DIR, "..", "XLSR-Mamba")
        )
        kwargs["emb_size"]     = args.emb_size
        kwargs["num_encoders"] = args.num_encoders
    elif args.repo_path:
        kwargs["repo_path"] = args.repo_path

    return adapter_cls(**kwargs)


def _build_dataset(args, em=None):
    adapter_cls = DATASET_REGISTRY[args.dataset]
    kwargs = {"eval_dir": _abs(args.eval_dir)}

    if args.keys_dir is not None:
        kwargs["keys_dir"] = _abs(args.keys_dir)

    # Pass eval_metrics only to adapters that declare it (ASVspoof tDCF)
    import inspect
    if "eval_metrics_module" in inspect.signature(adapter_cls.__init__).parameters:
        kwargs["eval_metrics_module"] = em

    return adapter_cls(**kwargs)


def main() -> None:
    args = build_parser().parse_args()

    out_dir = _abs(args.out_dir or f"results/{args.model}-{args.dataset}")

    cfg = BenchmarkConfig(
        out_dir=out_dir,
        phase=args.phase,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _build_model(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.load(device)
    param_count  = model.parameter_count()
    vram_load_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    print(f"Parameters: {param_count:,}")

    # ── eval_metrics — optional, sourced from model repo after load() sets sys.path ──
    em = None
    try:
        import eval_metrics as em  # noqa: PLC0415
    except ImportError:
        pass  # dataset adapter handles the None case gracefully

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = _build_dataset(args, em)
    trials  = dataset.load_trials(cfg.phase)

    # ── Inference ─────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    flac_paths = [dataset.audio_path(t.utt_id) for t in trials]
    print(f"\nRunning inference on {len(flac_paths)} utterances "
          f"({cfg.num_workers} loader workers, batch_size={cfg.batch_size})...")

    loader = build_loader(flac_paths, cfg.num_workers)
    all_scores, all_rtf, skipped = run_inference(model, loader, device, cfg.batch_size)

    vram_peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )

    if not all_scores:
        print("No scores produced. Check --eval_dir and --keys_dir paths.")
        sys.exit(1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    detection = compute_detection_metrics(all_scores, trials)
    detection.update(dataset.extra_metrics(all_scores, trials, cfg.phase))
    rtf_stats = compute_rtf_stats(all_rtf)

    # ── Report ────────────────────────────────────────────────────────────────
    reporter.print_results(
        detection, rtf_stats, len(all_scores), skipped,
        param_count, vram_load_gb, vram_peak_gb,
    )
    reporter.write_scores(out_dir, all_scores)
    reporter.write_summary(
        out_dir, model.name, dataset.name, cfg.phase,
        len(all_scores), skipped, param_count,
        vram_load_gb, vram_peak_gb, detection, rtf_stats,
    )

    print(f"\nResults saved to {out_dir}/")
    print(f"  scores.txt  — per-utterance scores")
    print(f"  summary.txt — full metrics summary")


if __name__ == "__main__":
    main()
