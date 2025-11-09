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

__version__ = "0.1.0"

__all__ = [
    "DiDACore",
    "DiscreteDiffusionScheduler",
    "DiDAAttentionMask",
    "DiDASampler",
    "DiDAInference",
]
