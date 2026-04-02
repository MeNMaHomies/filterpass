import argparse
from pathlib import Path

from .config import AugmentationConfig
from .pipeline import augment_file, augment_directory


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FLAC audio augmentation: noise, RIR, packet-loss simulation."
    )
    p.add_argument("input",  help="Input .flac file OR directory of .flac files")
    p.add_argument("output", help="Output .flac file OR output directory")
    p.add_argument("--n-aug", type=int, default=1,
                   help="Number of augmented copies per file (directory mode only)")
    p.add_argument("--bg-noise-dir", default=None,
                   help="Directory of background noise files (.wav / .flac)")

    p.add_argument("--no-white-noise",    action="store_true")
    p.add_argument("--no-pink-noise",     action="store_true")
    p.add_argument("--add-brown-noise",   action="store_true")
    p.add_argument("--no-rir",            action="store_true")
    p.add_argument("--no-bandpass",       action="store_true")
    p.add_argument("--no-mic-color",      action="store_true")
    p.add_argument("--no-codec-dist",     action="store_true")
    p.add_argument("--no-packet-loss",    action="store_true")
    p.add_argument("--packet-fill",       choices=["silence", "repeat", "noise"], default="silence")
    p.add_argument("--packet-loss-rate",  type=float, default=0.05)
    p.add_argument("--add-time-stretch",  action="store_true")
    p.add_argument("--no-gain",           action="store_true")
    p.add_argument("--seed",              type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = AugmentationConfig(
        add_white_noise        = not args.no_white_noise,
        add_pink_noise         = not args.no_pink_noise,
        add_brown_noise        = args.add_brown_noise,
        background_noise_dir   = args.bg_noise_dir,
        simulate_rir           = not args.no_rir,
        apply_bandpass         = not args.no_bandpass,
        apply_mic_coloration   = not args.no_mic_color,
        apply_codec_distortion = not args.no_codec_dist,
        simulate_packet_loss   = not args.no_packet_loss,
        packet_loss_fill       = args.packet_fill,
        packet_loss_rate       = args.packet_loss_rate,
        apply_time_stretch     = args.add_time_stretch,
        apply_random_gain      = not args.no_gain,
    )

    inp = Path(args.input)
    out = Path(args.output)

    if inp.is_dir():
        augment_directory(str(inp), str(out), cfg, n_augmentations=args.n_aug)
    elif inp.is_file() and inp.suffix.lower() == ".flac":
        augment_file(str(inp), str(out), cfg, seed=args.seed)
    else:
        print(f"[error] '{inp}' is not a .flac file or directory.")


if __name__ == "__main__":
    main()
