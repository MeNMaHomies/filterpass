"""
Model registry.

To register a new model adapter:
  1. Import its class here.
  2. Add an entry to REGISTRY mapping a CLI name to the class.

The CLI --model flag matches against REGISTRY keys.
"""

from __future__ import annotations

from .filterpass_sap import FilterpassSAP
from .filterpass_sap_v2 import FilterpassSAPv2
from .ssl_anti_spoofing import SSLAntiSpoofing
from .xlsr_mamba import XLSRMamba

REGISTRY: dict[str, type] = {
    "xlsr-mamba": XLSRMamba,
    "filterpass-sap": FilterpassSAP,
    "filterpass-sap-v2": FilterpassSAPv2,
    "wav2vec2-aasist": SSLAntiSpoofing,
}
