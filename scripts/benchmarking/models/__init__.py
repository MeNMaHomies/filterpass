"""
Model registry.

To register a new model adapter:
  1. Import its class here.
  2. Add an entry to REGISTRY mapping a CLI name to the class.

The CLI --model flag matches against REGISTRY keys.
"""

from .filterpass_sap import FilterpassSAP
from .xlsr_mamba import XLSRMamba

REGISTRY: dict[str, type] = {
    "xlsr-mamba": XLSRMamba,
    "filterpass-sap": FilterpassSAP,
}
