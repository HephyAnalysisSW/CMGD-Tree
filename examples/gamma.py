from examples.base import ExampleSpec


EXAMPLE = ExampleSpec(
    family_name="gamma",
    tree_defaults={
        "max_depth": 3,
        "max_leaves": 8,
    },
    dataset_defaults={
        "n_features": 4,
        "n_classes": 4,
    },
    training_defaults={
        "n_boost_rounds": 50,
        "learning_rate": 0.2,
    },
)
