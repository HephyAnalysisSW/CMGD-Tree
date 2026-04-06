from __future__ import annotations

from families.base import BoostingFamily
from families.normal_identity import NormalIdentityFamily
from families.poisson import PoissonMGDFamily, PoissonNGDFamily


def family_from_configs(tree_config: dict, dataset_config: dict) -> BoostingFamily:
    family_name = tree_config.get("family", "normal_identity")
    if family_name == "normal_identity":
        return NormalIdentityFamily.from_configs(tree_config, dataset_config)
    if family_name in {"poisson", "poisson_mgd"}:
        return PoissonMGDFamily.from_configs(tree_config, dataset_config)
    if family_name == "poisson_ngd":
        return PoissonNGDFamily.from_configs(tree_config, dataset_config)
    raise ValueError("family must be 'normal_identity', 'poisson', 'poisson_mgd', or 'poisson_ngd'.")
