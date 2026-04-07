from __future__ import annotations

from examples.base import ExampleSpec
from examples.gamma import EXAMPLE as GAMMA_EXAMPLE
from examples.heteroskedastic_normal import EXAMPLE as HETERO_EXAMPLE
from examples.heteroskedastic_normal_ngd import EXAMPLE as HETERO_NGD_EXAMPLE
from examples.negative_binomial import EXAMPLE as NEGATIVE_BINOMIAL_EXAMPLE
from examples.normal_identity import EXAMPLE as NORMAL_IDENTITY_EXAMPLE
from examples.poisson import EXAMPLE as POISSON_EXAMPLE
from examples.poisson_ngd import EXAMPLE as POISSON_NGD_EXAMPLE


EXAMPLES_BY_FAMILY: dict[str, ExampleSpec] = {
    NORMAL_IDENTITY_EXAMPLE.family_name: NORMAL_IDENTITY_EXAMPLE,
    POISSON_EXAMPLE.family_name: POISSON_EXAMPLE,
    POISSON_NGD_EXAMPLE.family_name: POISSON_NGD_EXAMPLE,
    GAMMA_EXAMPLE.family_name: GAMMA_EXAMPLE,
    NEGATIVE_BINOMIAL_EXAMPLE.family_name: NEGATIVE_BINOMIAL_EXAMPLE,
    HETERO_EXAMPLE.family_name: HETERO_EXAMPLE,
    HETERO_NGD_EXAMPLE.family_name: HETERO_NGD_EXAMPLE,
}


def example_from_family_name(family_name: str) -> ExampleSpec:
    return EXAMPLES_BY_FAMILY[family_name]

