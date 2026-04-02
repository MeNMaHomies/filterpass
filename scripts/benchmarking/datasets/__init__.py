"""
Dataset registry.

To register a new dataset adapter:
  1. Import its class here.
  2. Add an entry to REGISTRY mapping a CLI name to the class.

The CLI --dataset flag matches against REGISTRY keys.
"""

from .asvspoof2021_la import ASVspoof2021LA

REGISTRY: dict[str, type] = {
    "asvspoof2021-la": ASVspoof2021LA,
}
