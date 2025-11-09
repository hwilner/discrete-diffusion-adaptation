"""
# Testing Methodology

This document outlines the testing methodology used to validate the DiDA implementation. All tests are located in the `tests/` directory and are implemented using the `pytest` framework.

## Use of Synthetic Data

**Note:** All tests in this repository use **synthetic (random) data** for validation purposes. This is a standard and necessary practice in machine learning research implementations for the following reasons:

1.  **Correctness Verification:** Using synthetic data allows us to verify the correctness of the implementation's logic, shape transformations, and data flow without needing a fully trained model or a specific dataset.
2.  **Unit Testing:** It enables the creation of focused unit tests that can isolate and validate individual components of the implementation, such as the scheduler, attention mask generation, and core model.
3.  **Reproducibility:** Tests based on synthetic data are fully reproducible and do not depend on external data sources.

## Test Suite

The test suite is organized into several classes, each targeting a specific component of the DiDA implementation:

-   **`TestDiscreteDiffusionScheduler`:** Tests the noise scheduler, including noise addition and timestep generation.
-   **`TestDiDAAttentionMask`:** Tests the creation of the hybrid attention masks.
-   **`TestDiDACore`:** Tests the core denoising model, including the forward pass and denoising step.
-   **`TestDiDASampler`:** Tests the sampling pipeline for both pure image generation and interleaved text-image generation.
-   **`TestDiDAInference`:** Tests the high-level inference interface.
-   **`TestIntegration`:** Includes end-to-end tests that validate the entire generation pipeline.

## Running the Tests

To run the tests, first install the required dependencies:

```bash
pip install -r requirements.txt
pip install pytest
```

Then, run the test suite from the root of the repository:

```bash
python -m pytest -v
```

All 18 tests should pass, confirming the correctness of the implementation.
"""
