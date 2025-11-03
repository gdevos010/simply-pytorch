"""Simply PyTorch: PyTorch port of Simply optimizers with Cautious Weight Decay.

This package provides PyTorch implementations of optimizers from the Simply
codebase, including support for Cautious Weight Decay (CWD) as described in
https://arxiv.org/html/2510.12402v1
"""

from simply_pytorch.optimizers import SGD, Adam, Lion, Muon

__version__ = "0.1.0"
__all__ = ["SGD", "Adam", "Lion", "Muon"]
