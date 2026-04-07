from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleSpec:
    family_name: str
    tree_defaults: dict
    dataset_defaults: dict
    training_defaults: dict
