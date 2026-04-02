import argparse
from pathlib import Path

from .presets import telephony, full, light
from .pipeline import augment_file, augment_directory

_PRESETS = {
    "telephony": telephony,
    "full": full,
    "light": light,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audio augmentation with composable transforms."
    )
    p.add_argument("input",  help="Input .flac file OR directory of .flac files")
    p.add_argument("output", help="Output .flac file OR output directory")
    p.add_argument("--preset", choices=list(_PRESETS.keys()), default="full",
                   help="Augmentation preset (default: full)")
    p.add_argument("--n-aug", type=int, default=1,
                   help="Number of augmented copies per file (directory mode only)")
    p.add_argument("--seed", type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    transform = _PRESETS[args.preset]()

    inp = Path(args.input)
    out = Path(args.output)

    if inp.is_dir():
        augment_directory(str(inp), str(out), transform, n_augmentations=args.n_aug)
    elif inp.is_file() and inp.suffix.lower() == ".flac":
        augment_file(str(inp), str(out), transform, seed=args.seed)
    else:
        print(f"[error] '{inp}' is not a .flac file or directory.")


if __name__ == "__main__":
    main()
