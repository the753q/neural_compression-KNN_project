from .basic_ae import BasicAE
from .DCAL_2018 import DCAL_2018

MODEL_REGISTRY = {
    "basic": BasicAE,
    "DCAL_2018": DCAL_2018
}


def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name](**kwargs)
