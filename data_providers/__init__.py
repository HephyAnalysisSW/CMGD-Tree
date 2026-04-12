from data_providers.base import StreamBatch
from data_providers.gamma_toy import GammaToyStream
from data_providers.gaussian_class_toy import GaussianClassToyStream
from data_providers.heteroskedastic_normal_toy import HeteroskedasticNormalToyStream
from data_providers.negative_binomial_toy import NegativeBinomialToyStream
from data_providers.poisson_toy import PoissonToyStream

DATA_PROVIDERS = {
    "gamma_toy": GammaToyStream,
    "gaussian_class_toy": GaussianClassToyStream,
    "heteroskedastic_normal_toy": HeteroskedasticNormalToyStream,
    "negative_binomial_toy": NegativeBinomialToyStream,
    "poisson_toy": PoissonToyStream,
}


def data_provider_class_from_name(name: str):
    if name not in DATA_PROVIDERS:
        raise ValueError(f"Unknown data_provider '{name}'.")
    return DATA_PROVIDERS[name]


def data_provider_kwargs(dataset_config: dict, class_weights=None) -> dict:
    kwargs = {key: value for key, value in dataset_config.items() if key != "data_provider"}
    if class_weights is not None:
        kwargs["class_weights"] = class_weights
    return kwargs


def build_data_provider(dataset_config: dict, class_weights=None):
    provider_class = data_provider_class_from_name(dataset_config.get("data_provider"))
    return provider_class(**data_provider_kwargs(dataset_config, class_weights=class_weights))


def stream_batches(dataset_config: dict, class_weights=None):
    yield from build_data_provider(dataset_config, class_weights=class_weights)

__all__ = [
    "DATA_PROVIDERS",
    "GammaToyStream",
    "GaussianClassToyStream",
    "HeteroskedasticNormalToyStream",
    "NegativeBinomialToyStream",
    "PoissonToyStream",
    "StreamBatch",
    "build_data_provider",
    "data_provider_class_from_name",
    "data_provider_kwargs",
    "stream_batches",
]
