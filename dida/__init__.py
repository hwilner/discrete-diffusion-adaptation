"""
Discrete Diffusion Adaptation (DiDA) Implementation

An educational implementation of the DiDA mechanism from:
"Emu3.5: Native Multimodal Models are World Learners" (arXiv:2510.26583)

This package provides:
- DiDACore: Core denoising model with transformer architecture
- DiscreteDiffusionScheduler: Noise scheduling for discrete diffusion
- DiDAAttentionMask: Hybrid attention mask generation
- DiDASampler: Sampling interface for generation
- DiDAInference: High-level inference API
- CTMCUniformizationSampler: exact CTMC sampling via uniformization
- log_gumbel_softmax: log-space Gumbel-Softmax (underflow-safe)
"""

from .core import (
    DiDACore,
    DiscreteDiffusionScheduler,
    DiDAAttentionMask
)
from .sampler import (
    DiDASampler,
    DiDAInference
)
from .ctmc import (
    CTMCUniformizationSampler,
    log_gumbel_softmax
)

__version__ = "0.2.0"

__all__ = [
    "DiDACore",
    "DiscreteDiffusionScheduler",
    "DiDAAttentionMask",
    "DiDASampler",
    "DiDAInference",
    "CTMCUniformizationSampler",
    "log_gumbel_softmax",
]
