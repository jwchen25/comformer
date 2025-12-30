"""ComFormer: Complete and Efficient Graph Transformers for Crystal Material Property Prediction."""

__version__ = "2025.12.30"
__author__ = "Junwu Chen"
__email__ = "junwu.chen@epfl.ch"

# Expose main interfaces for easy import
from comformer.custom_train import train_from_list, train_from_extxyz
from comformer.predict import load_predictor, load_predictor_hf, ComformerPredictor

__all__ = [
    "train_from_list",
    "train_from_extxyz",
    "load_predictor",
    "load_predictor_hf",
    "ComformerPredictor",
    "__version__",
]
