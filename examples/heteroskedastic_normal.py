from examples.base import ExampleSpec


EXAMPLE = ExampleSpec(
    family_name="heteroskedastic_normal",
    tree_defaults={
        "max_depth": 2,
        "max_leaves": 4,
    },
    dataset_defaults={
        "n_features": 2,
        "n_classes": 2,
        "n_batches": 24,
    },
    training_defaults={
        "n_boost_rounds": 50,
        "learning_rate": 0.2,
    },
)
