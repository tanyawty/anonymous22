# models/__init__.py

from .stgnn import STGNN
from .fouriergnn import FourierGNN_LearnOnly

__all__ = [
    "STGNN",
    "FourierGNN_LearnOnly",
]
