from .DCAL_2018 import DCAL_2018, train_model as train_dcal
from .DCAL_Native import DCAL_Native, train_model as train_dcal_native
from .Balle2017 import Balle2017, train_model as train_balle
from .CustomCompressor import CustomCompressor, train_model as train_custom
from .DCAL_LAB import DCAL_LAB, train_model as train_dcal_lab
from .Hyperprior import Hyperprior, train_model as train_hyperprior
from .DCAL_YCbCr_Subsampled import (
    DCAL_YCbCr_Subsampled,
    train_model as train_dcal_ycbcr_subsampled,
)
from .DCAL_YCbCr_Base import (
    DCAL_YCbCr_Base,
    train_model as train_dcal_ycbcr_base,
)
from .DCAL_pool import (
    DCAL_pool,
    train_model as train_dcal_pool
)
from .DCAL_triple import (
    DCAL_triple,
    train_model as train_dcal_triple
)
from .DCAL_simple import (
    DCAL_simple,
    train_model as train_dcal_simple
)
from .DCAL_extended import (
    DCAL_extended,
    train_model as train_dcal_extended
)

MODEL_REGISTRY = {
    "DCAL_2018": DCAL_2018,
    "DCAL_Native": DCAL_Native,
    "Balle2017": Balle2017,
    "CustomCompressor": CustomCompressor,
    "DCAL_LAB": DCAL_LAB,
    "Hyperprior": Hyperprior,
    "DCAL_YCbCr_Subsampled": DCAL_YCbCr_Subsampled,
    "DCAL_YCbCr_Base": DCAL_YCbCr_Base,
    "DCAL_pool": DCAL_pool,
    "DCAL_triple": DCAL_triple,
    "DCAL_simple": DCAL_simple,
    "DCAL_extended": DCAL_extended,
}

TRAIN_REGISTRY = {
    "DCAL_2018": train_dcal,
    "DCAL_Native": train_dcal_native,
    "Balle2017": train_balle,
    "CustomCompressor": train_custom,
    "DCAL_LAB": train_dcal_lab,
    "Hyperprior": train_hyperprior,
    "DCAL_YCbCr_Subsampled": train_dcal_ycbcr_subsampled,
    "DCAL_YCbCr_Base": train_dcal_ycbcr_base,
    "DCAL_pool": DCAL_pool,
    "DCAL_triple": DCAL_triple,
    "DCAL_simple": DCAL_simple,
    "DCAL_extended": DCAL_extended,
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
