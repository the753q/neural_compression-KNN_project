from .DCAL_2018 import DCAL_2018, train_model as train_dcal
from .DCAL_Native import DCAL_Native, train_model as train_dcal_native
from .Balle2017 import Balle2017, train_model as train_balle
from .CustomCompressor import CustomCompressor, train_model as train_custom
from .Hyperprior import Hyperprior, train_model as train_hyperprior

MODEL_REGISTRY = {
    "DCAL_2018": DCAL_2018,
    "DCAL_Native": DCAL_Native,
    "Balle2017": Balle2017,
    "CustomCompressor": CustomCompressor,
    "Hyperprior": Hyperprior
}

TRAIN_REGISTRY = {
    "DCAL_2018": train_dcal,
    "DCAL_Native": train_dcal_native,
    "Balle2017": train_balle,
    "CustomCompressor": train_custom,
    "Hyperprior": train_hyperprior
}


def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name](**kwargs)


def get_train_function(model_name):
    if model_name not in TRAIN_REGISTRY:
        raise ValueError(f"Train function for model '{model_name}' not found.")

    return TRAIN_REGISTRY[model_name]
