from __future__ import annotations

from families.base import BoostingFamily
from families.heteroskedastic_normal import HeteroskedasticNormalFamily
from families.normal_identity import NormalIdentityFamily
from families.poisson import PoissonMGDFamily, PoissonNGDFamily


def family_class_from_name(family_name: str) -> type[BoostingFamily]:
    if family_name == "normal_identity":
        return NormalIdentityFamily
    if family_name == "heteroskedastic_normal":
        return HeteroskedasticNormalFamily
    if family_name in {"poisson", "poisson_mgd"}:
        return PoissonMGDFamily
    if family_name == "poisson_ngd":
        return PoissonNGDFamily
    raise ValueError("family must be 'normal_identity', 'heteroskedastic_normal', 'poisson', 'poisson_mgd', or 'poisson_ngd'.")


def family_from_configs(tree_config: dict, dataset_config: dict) -> BoostingFamily:
    family_name = tree_config.get("family", "normal_identity")
    return family_class_from_name(family_name).from_configs(tree_config, dataset_config)
