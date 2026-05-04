from .basic_ae import BasicAE, train_model as train_basic
from .DCAL_2018 import DCAL_2018, train_model as train_dcal

MODEL_REGISTRY = {
    "basic": BasicAE,
    "DCAL_2018": DCAL_2018,
}

TRAIN_REGISTRY = {
    "basic": train_basic,
    "DCAL_2018": train_dcal,
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name](**kwargs)

def get_train_function(model_name):
    if model_name not in TRAIN_REGISTRY:
        raise ValueError(f"Train function for model '{model_name}' not found.")
    
    return TRAIN_REGISTRY[model_name]
