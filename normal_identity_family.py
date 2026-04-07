from __future__ import annotations

"""Compatibility shim for the split family/provider modules."""

from families import family_from_configs
from families.base import BoostingFamily
from families.normal_identity import NormalIdentityFamily
from families.poisson import PoissonMGDFamily, PoissonNGDFamily
from data_providers.base import StreamBatch
from data_providers.gaussian_class_toy import GaussianClassToyStream
from data_providers.poisson_toy import PoissonToyStream

__all__ = [
    "BoostingFamily",
    "StreamBatch",
    "GaussianClassToyStream",
    "PoissonToyStream",
    "NormalIdentityFamily",
    "PoissonMGDFamily",
    "PoissonNGDFamily",
    "family_from_configs",
]
