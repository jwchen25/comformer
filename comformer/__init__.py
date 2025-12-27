"""ComFormer: Complete and Efficient Graph Transformers for Crystal Material Property Prediction."""

__version__ = "2025.12.26"
__author__ = "Junwu Chen"
__email__ = "junwu.chen@epfl.ch"

# Expose main interfaces for easy import
from comformer.custom_train import train_custom_icomformer as train_model
from comformer.predict import load_predictor, ComformerPredictor

__all__ = [
    "train_model",
    "load_predictor",
    "ComformerPredictor",
    "__version__",
]
