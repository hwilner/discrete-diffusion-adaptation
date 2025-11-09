"""
Basic Usage Example for DiDA

This script demonstrates how to use the DiDA implementation for basic
image generation and text-to-image generation.
"""

import torch
from dida import (
    DiDACore,
    DiscreteDiffusionScheduler,
    DiDAInference
)


def main():
    """Main function to run the examples."""
    
    print("--- DiDA Basic Usage Example ---")
    
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize the core components
    # Note: These are randomly initialized for demonstration purposes.
    # In a real-world scenario, you would load a pre-trained model.
    model = DiDACore(
        vocab_size=2000,
        hidden_dim=256,
        num_heads=8,
        num_layers=4
    ).to(device)
    
    scheduler = DiscreteDiffusionScheduler(num_steps=10)
    
    inference = DiDAInference(
        model=model,
        scheduler=scheduler,
        vocab_size=2000,
        text_vocab_size=1000,
        image_vocab_size=1000
    )
    
    # --- Example 1: Pure Image Generation ---
    print("\n--- Running Example 1: Pure Image Generation ---")
    
    image_tokens = inference.generate_image(
        batch_size=1,
        image_size=64,  # Smaller size for faster example
        patch_size=16,
        device=device,
        temperature=1.0
    )
    
    print(f"Generated {image_tokens.shape[1]} image tokens.")
    print(f"Shape of generated tokens: {image_tokens.shape}")
    
    # --- Example 2: Text-to-Image Generation ---
    print("\n--- Running Example 2: Text-to-Image Generation ---")
    
    # Create a synthetic text prompt (random tokens)
    text_prompt = torch.randint(0, 1000, (1, 32), device=device)
    
    image_tokens_from_text = inference.text_to_image(
        text_tokens=text_prompt,
        image_size=64,
        patch_size=16,
        temperature=1.0
    )
    
    print(f"Generated {image_tokens_from_text.shape[1]} image tokens from text prompt.")
    print(f"Shape of generated tokens: {image_tokens_from_text.shape}")
    
    print("\n--- Example script finished successfully! ---")


if __name__ == "__main__":
    main()
