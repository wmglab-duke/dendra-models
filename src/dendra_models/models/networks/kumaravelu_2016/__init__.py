"""Dendra implementations of the Kumaravelu et al. 2016 CTX-BG-TH model."""

from .params import default_params, params
from .fused_net import Kumaravelu2016Fused, kumaravelu_2016_fused
from .matlab_realization import params_from_matlab_validation

# Public model alias uses the fused production builder.
Kumaravelu2016 = kumaravelu_2016_fused

__all__ = [
    "params",
    "default_params",
    "kumaravelu_2016_fused",
    "Kumaravelu2016",
    "Kumaravelu2016Fused",
    "params_from_matlab_validation",
]
