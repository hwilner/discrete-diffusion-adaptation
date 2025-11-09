"""
# Discrete Diffusion Adaptation (DiDA)

This repository provides a clean, educational, and from-scratch implementation of **Discrete Diffusion Adaptation (DiDA)**, the innovative inference technique from the paper:

> **Emu3.5: Native Multimodal Models are World Learners**
> *Yufeng Cui, Honghao Chen, Haoge Deng, et al.*
> [arXiv:2510.26583](https://arxiv.org/abs/2510.26583)

DiDA is a lightweight adaptation approach that converts the standard token-by-token autoregressive decoding for image generation into a highly efficient **bidirectional parallel prediction** process. This accelerates per-image inference by approximately **20x** without sacrificing quality, all while leaving text generation capabilities untouched.

This implementation is designed for researchers, students, and engineers who want to understand the core mechanics of DiDA. It is written in PyTorch and follows Google-style docstrings, with a strong emphasis on clarity and correctness.

## Key Features

- **Core DiDA Mechanism:** Full implementation of the discrete diffusion process over visual tokens.
- **Hybrid Attention:** Custom attention masks that enable bidirectional attention for noisy image tokens while preserving causal attention for clean text/image tokens.
- **Complete Sampling Pipeline:** Includes a high-level sampler for both pure image generation and interleaved text-to-image generation.
- **Comprehensive Test Suite:** Includes **18 unit and integration tests** to ensure the correctness of the implementation. All tests pass successfully.
- **Detailed Documentation:** In-depth explanations of the architecture, testing methodology, and usage.
- **Working Examples:** Clear examples demonstrating how to use the DiDA implementation for various tasks.

## Getting Started

### Installation

To install the package, clone the repository and install the required dependencies:

```bash
git clone https://github.com/hwilner/discrete-diffusion-adaptation.git
cd discrete-diffusion-adaptation
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

Here is a simple example of how to use the DiDA implementation to generate image tokens:

```python
import torch
from dida import DiDACore, DiscreteDiffusionScheduler, DiDAInference

# 1. Initialize the core components
model = DiDACore(vocab_size=2000, hidden_dim=256, num_heads=8, num_layers=4)
scheduler = DiscreteDiffusionScheduler(num_steps=10)
inference = DiDAInference(
    model=model,
    scheduler=scheduler,
    vocab_size=2000,
    text_vocab_size=1000,
    image_vocab_size=1000
)

# 2. Generate image tokens
image_tokens = inference.generate_image(
    batch_size=1,
    image_size=256,
    patch_size=16,
    device=torch.device('cpu')
)

print(f"Generated {image_tokens.shape[1]} image tokens.")
```

For more detailed examples, please see the `examples/` directory.

## Project Structure

```
discrete-diffusion-adaptation/
├── dida/              # Source code for the DiDA implementation
├── tests/             # Test suite (18 tests, all passing)
├── examples/          # Usage examples
├── docs/              # Detailed documentation
├── README.md          # This file
├── requirements.txt   # Python dependencies
├── setup.py           # Package setup file
└── LICENSE            # MIT License
```

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md):** A deep dive into the DiDA architecture, including the diffusion process and hybrid attention mechanism.
- **[TESTING.md](docs/TESTING.md):** An explanation of the testing methodology, including the use of synthetic data for validation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Credit to the authors of the Emu3.5 paper for their groundbreaking research and for open-sourcing their work.
"""
