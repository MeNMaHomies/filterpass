"""
Model registry.

To register a new model adapter:
  1. Import its class here.
  2. Add an entry to REGISTRY mapping a CLI name to the class.

The CLI --model flag matches against REGISTRY keys.
"""

from .xlsr_mamba import XLSRMamba
from .fatigue_sense_sap import FatigueSenseSAP

REGISTRY: dict[str, type] = {
    "xlsr-mamba": XLSRMamba,
    "fatigue-sense-sap": FatigueSenseSAP,
}
