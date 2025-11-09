"""
DiDA Sampler Module

Implements the complete sampling pipeline for Discrete Diffusion Adaptation,
including the iterative denoising process and hybrid text-image generation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from .core import DiDACore, DiscreteDiffusionScheduler, DiDAAttentionMask


class DiDASampler:
    """
    Sampler for DiDA-based generation.
    
    Handles the complete sampling process including:
    - Initialization of noisy tokens
    - Iterative denoising
    - Hybrid text (autoregressive) and image (parallel) generation
    
    Attributes:
        model: DiDA core model for denoising.
        scheduler: Diffusion scheduler managing noise levels.
        vocab_size: Size of token vocabulary.
    """
    
    def __init__(
        self,
        model: DiDACore,
        scheduler: DiscreteDiffusionScheduler,
        vocab_size: int
    ):
        """
        Initialize DiDA sampler.
        
        Args:
            model: Trained DiDA core model.
            scheduler: Diffusion scheduler.
            vocab_size: Size of token vocabulary.
        """
        self.model = model
        self.scheduler = scheduler
        self.vocab_size = vocab_size
        
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        temperature: float = 1.0,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Generate tokens using DiDA sampling.
        
        Args:
            batch_size: Number of samples to generate.
            num_tokens: Number of tokens to generate per sample.
            device: Device to generate on.
            temperature: Sampling temperature.
            guidance_scale: Classifier-free guidance scale.
            
        Returns:
            Generated token indices, shape (batch_size, num_tokens).
        """
        # Initialize with random tokens (fully noisy)
        tokens = torch.randint(
            0, self.vocab_size,
            (batch_size, num_tokens),
            device=device
        )
        
        # Get timesteps in reverse order (denoising)
        timesteps = self.scheduler.get_timesteps(reverse=True)
        
        # Iterative denoising
        for t in timesteps:
            # Create attention mask (bidirectional for all tokens during generation)
            attention_mask = DiDAAttentionMask.create_dida_mask(
                seq_len=num_tokens,
                noisy_start_idx=0,
                noisy_end_idx=num_tokens,
                device=device
            )
            
            # Denoise
            tokens = self.model.denoise_step(
                tokens,
                timestep=t.item(),
                attention_mask=attention_mask,
                temperature=temperature
            )
        
        return tokens
    
    @torch.no_grad()
    def sample_interleaved(
        self,
        text_tokens: torch.Tensor,
        num_image_tokens: int,
        device: torch.device,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate image tokens conditioned on text tokens.
        
        This implements the hybrid generation where text tokens are given
        (from autoregressive generation) and image tokens are generated
        in parallel using DiDA.
        
        Args:
            text_tokens: Conditioning text tokens, shape (batch_size, text_len).
            num_image_tokens: Number of image tokens to generate.
            device: Device to generate on.
            temperature: Sampling temperature for image generation.
            
        Returns:
            Tuple of (full_sequence, image_tokens) where:
            - full_sequence: Concatenated text and image tokens
            - image_tokens: Generated image tokens only
        """
        batch_size, text_len = text_tokens.shape
        
        # Initialize image tokens with random noise
        image_tokens = torch.randint(
            0, self.vocab_size,
            (batch_size, num_image_tokens),
            device=device
        )
        
        # Concatenate text and image tokens
        full_sequence = torch.cat([text_tokens, image_tokens], dim=1)
        seq_len = full_sequence.shape[1]
        
        # Get timesteps
        timesteps = self.scheduler.get_timesteps(reverse=True)
        
        # Iterative denoising for image tokens
        for t in timesteps:
            # Create hybrid attention mask
            # Text tokens: causal attention
            # Image tokens: bidirectional attention among themselves,
            #               causal attention to text
            attention_mask = DiDAAttentionMask.create_dida_mask(
                seq_len=seq_len,
                noisy_start_idx=text_len,
                noisy_end_idx=seq_len,
                device=device
            )
            
            # Denoise full sequence
            denoised_sequence = self.model.denoise_step(
                full_sequence,
                timestep=t.item(),
                attention_mask=attention_mask,
                temperature=temperature
            )
            
            # Update only image tokens (keep text tokens fixed)
            full_sequence = torch.cat([
                text_tokens,
                denoised_sequence[:, text_len:]
            ], dim=1)
        
        # Extract final image tokens
        final_image_tokens = full_sequence[:, text_len:]
        
        return full_sequence, final_image_tokens
    
    @torch.no_grad()
    def sample_with_guidance(
        self,
        batch_size: int,
        num_tokens: int,
        condition: Optional[torch.Tensor],
        device: torch.device,
        guidance_scale: float = 7.5,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate tokens with classifier-free guidance.
        
        Args:
            batch_size: Number of samples to generate.
            num_tokens: Number of tokens per sample.
            condition: Optional conditioning tokens.
            device: Device to generate on.
            guidance_scale: Strength of guidance (1.0 = no guidance).
            temperature: Sampling temperature.
            
        Returns:
            Generated tokens with guidance applied.
        """
        # Initialize noisy tokens
        tokens = torch.randint(
            0, self.vocab_size,
            (batch_size, num_tokens),
            device=device
        )
        
        timesteps = self.scheduler.get_timesteps(reverse=True)
        
        for t in timesteps:
            attention_mask = DiDAAttentionMask.create_dida_mask(
                seq_len=num_tokens,
                noisy_start_idx=0,
                noisy_end_idx=num_tokens,
                device=device
            )
            
            # Get conditional logits
            logits_cond = self.model(
                tokens,
                torch.tensor(t.item(), device=device),
                attention_mask
            )
            
            if guidance_scale != 1.0 and condition is not None:
                # Get unconditional logits
                logits_uncond = self.model(
                    tokens,
                    torch.tensor(t.item(), device=device),
                    attention_mask
                )
                
                # Apply classifier-free guidance
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond
            
            # Sample from guided logits
            probs = torch.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1
            ).view(batch_size, num_tokens)
        
        return tokens


class DiDAInference:
    """
    High-level inference interface for DiDA.
    
    Provides convenient methods for different generation scenarios:
    - Pure image generation
    - Text-to-image generation
    - Interleaved text-image generation
    """
    
    def __init__(
        self,
        model: DiDACore,
        scheduler: DiscreteDiffusionScheduler,
        vocab_size: int,
        text_vocab_size: int,
        image_vocab_size: int
    ):
        """
        Initialize DiDA inference interface.
        
        Args:
            model: Trained DiDA model.
            scheduler: Diffusion scheduler.
            vocab_size: Total vocabulary size.
            text_vocab_size: Size of text vocabulary.
            image_vocab_size: Size of image vocabulary.
        """
        self.sampler = DiDASampler(model, scheduler, vocab_size)
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        
    def generate_image(
        self,
        batch_size: int = 1,
        image_size: int = 256,
        patch_size: int = 16,
        device: torch.device = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate image tokens.
        
        Args:
            batch_size: Number of images to generate.
            image_size: Size of image in pixels.
            patch_size: Size of each patch in pixels.
            device: Device to generate on.
            temperature: Sampling temperature.
            
        Returns:
            Generated image tokens.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate number of tokens needed
        num_patches = (image_size // patch_size) ** 2
        
        # Generate tokens
        image_tokens = self.sampler.sample(
            batch_size=batch_size,
            num_tokens=num_patches,
            device=device,
            temperature=temperature
        )
        
        return image_tokens
    
    def text_to_image(
        self,
        text_tokens: torch.Tensor,
        image_size: int = 256,
        patch_size: int = 16,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate image from text prompt.
        
        Args:
            text_tokens: Text prompt tokens, shape (batch_size, text_len).
            image_size: Size of image to generate.
            patch_size: Patch size for tokenization.
            temperature: Sampling temperature.
            
        Returns:
            Generated image tokens.
        """
        device = text_tokens.device
        num_patches = (image_size // patch_size) ** 2
        
        # Generate image tokens conditioned on text
        _, image_tokens = self.sampler.sample_interleaved(
            text_tokens=text_tokens,
            num_image_tokens=num_patches,
            device=device,
            temperature=temperature
        )
        
        return image_tokens


def test_dida_sampler():
    """
    Test DiDA sampler implementation.
    Uses synthetic data for validation.
    """
    print("Testing DiDA Sampler...")
    
    # Test parameters
    batch_size = 2
    num_tokens = 64
    vocab_size = 1000
    hidden_dim = 256
    
    # Initialize components
    model = DiDACore(vocab_size=vocab_size, hidden_dim=hidden_dim, num_heads=8, num_layers=4)
    scheduler = DiscreteDiffusionScheduler(num_steps=10)
    sampler = DiDASampler(model, scheduler, vocab_size)
    
    # Test basic sampling
    device = torch.device('cpu')
    generated = sampler.sample(
        batch_size=batch_size,
        num_tokens=num_tokens,
        device=device,
        temperature=1.0
    )
    print(f"Generated tokens shape: {generated.shape}")
    assert generated.shape == (batch_size, num_tokens)
    
    # Test interleaved sampling
    text_tokens = torch.randint(0, vocab_size, (batch_size, 32))
    full_seq, image_tokens = sampler.sample_interleaved(
        text_tokens=text_tokens,
        num_image_tokens=32,
        device=device,
        temperature=1.0
    )
    print(f"Full sequence shape: {full_seq.shape}")
    print(f"Image tokens shape: {image_tokens.shape}")
    assert full_seq.shape == (batch_size, 64)
    assert image_tokens.shape == (batch_size, 32)
    
    print("DiDA Sampler test passed!")


if __name__ == "__main__":
    test_dida_sampler()
