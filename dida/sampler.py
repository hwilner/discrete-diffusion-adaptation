"""
DiDA Sampler Module

Implements the complete sampling pipeline for Discrete Diffusion Adaptation,
including the iterative denoising process and hybrid text-image generation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from .core import DiDACore, DiscreteDiffusionScheduler, DiDAAttentionMask


def _validated_condition(
    condition: Optional[torch.Tensor],
    batch_size: int,
    vocab_size: int,
    device: torch.device
) -> Optional[torch.Tensor]:
    """Validate the optional conditioning prefix of a guided sampling run.

    Args:
        condition: Conditioning tokens, shape (batch_size, condition_len),
            or None for unconditional generation.
        batch_size: Batch size the caller asked for.
        vocab_size: Size of the token vocabulary.
        device: Device the tokens must live on.

    Returns:
        The conditioning tokens as a ``torch.long`` tensor on ``device``, or
        None when ``condition`` is None.

    Raises:
        ValueError: If ``condition`` is not a 2-D integer tensor with
            ``batch_size`` rows, or if it holds token indices outside
            ``[0, vocab_size)``.
    """
    if condition is None:
        return None

    if not isinstance(condition, torch.Tensor):
        raise ValueError(
            "condition must be a torch.Tensor of token indices or None."
        )
    if condition.dim() != 2:
        raise ValueError(
            "condition must have shape (batch_size, condition_len); got %d "
            "dimension(s)." % condition.dim()
        )
    if condition.shape[0] != batch_size:
        raise ValueError(
            "condition has %d rows but batch_size is %d; the two must agree."
            % (condition.shape[0], batch_size)
        )
    if condition.dtype.is_floating_point or condition.dtype.is_complex:
        raise ValueError(
            "condition must hold integer token indices, got dtype %s."
            % condition.dtype
        )
    if condition.numel() > 0:
        lowest = int(condition.min())
        highest = int(condition.max())
        if lowest < 0 or highest >= vocab_size:
            raise ValueError(
                "condition holds token indices outside [0, %d): got [%d, %d]."
                % (vocab_size, lowest, highest)
            )

    return condition.to(device=device, dtype=torch.long)


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    vocab_size: int
) -> torch.Tensor:
    """Sample token indices from a logit tensor at the given temperature.

    Args:
        logits: Logits of shape (..., vocab_size).
        temperature: Sampling temperature; values at or below zero decode
            greedily instead of sampling.
        vocab_size: Size of the token vocabulary.

    Returns:
        Sampled token indices of shape ``logits.shape[:-1]``.
    """
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        flat = probs.reshape(-1, vocab_size)
        return torch.multinomial(flat, num_samples=1).reshape(logits.shape[:-1])

    # Greedy decoding, matching DiDACore.denoise_step's temperature <= 0 path.
    return torch.argmax(logits, dim=-1)


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

        The run is unconditional: there is no conditioning prefix to drop,
        so ``guidance_scale`` cannot change the result and is forwarded to
        :meth:`sample_with_guidance` for API symmetry only. Use
        :meth:`sample_with_guidance` to condition on a token prefix.

        Args:
            batch_size: Number of samples to generate.
            num_tokens: Number of tokens to generate per sample.
            device: Device to generate on.
            temperature: Sampling temperature.
            guidance_scale: Classifier-free guidance scale. Inert here
                because this method generates unconditionally.

        Returns:
            Generated token indices, shape (batch_size, num_tokens).
        """
        return self.sample_with_guidance(
            batch_size=batch_size,
            num_tokens=num_tokens,
            condition=None,
            device=device,
            guidance_scale=guidance_scale,
            temperature=temperature
        )

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

        The conditioning tokens form a fixed prefix of the sequence and the
        generated tokens are denoised behind it. Every step runs the
        denoiser twice: once where the generated tokens can attend to the
        prefix (the conditional pass) and once where the prefix is hidden
        from them (the unconditional pass). The two logit sets are combined
        as ``uncond + guidance_scale * (cond - uncond)``, which amplifies
        the difference the prefix makes. ``guidance_scale = 1.0`` leaves the
        conditional logits unchanged, ``guidance_scale = 0.0`` leaves the
        unconditional logits unchanged, and without a condition there is
        nothing to hide, so only the conditional pass runs.

        Args:
            batch_size: Number of samples to generate.
            num_tokens: Number of tokens per sample.
            condition: Optional conditioning tokens of shape
                (batch_size, condition_len). None means unconditional
                generation.
            device: Device to generate on.
            guidance_scale: Strength of guidance (1.0 = no guidance).
            temperature: Sampling temperature; values at or below zero
                decode greedily.

        Returns:
            Generated token indices, shape (batch_size, num_tokens).

        Raises:
            ValueError: If ``condition`` is not a 2-D integer tensor with
                ``batch_size`` rows and in-range token indices.
        """
        condition_tokens = _validated_condition(
            condition,
            batch_size=batch_size,
            vocab_size=self.vocab_size,
            device=device
        )
        condition_len = 0 if condition_tokens is None else condition_tokens.shape[1]

        # Initialize noisy tokens
        tokens = torch.randint(
            0, self.vocab_size,
            (batch_size, num_tokens),
            device=device
        )

        timesteps = self.scheduler.get_timesteps(reverse=True)

        # The masks depend only on the sequence layout, so they are built
        # once instead of once per denoising step.
        conditional_mask, unconditional_mask = DiDAAttentionMask.create_guidance_masks(
            seq_len=condition_len + num_tokens,
            condition_len=condition_len,
            device=device
        )

        for t in timesteps:
            # The prefix is re-attached every step: only the generated
            # tokens are resampled, the conditioning tokens stay fixed.
            sequence = tokens if condition_tokens is None else torch.cat(
                [condition_tokens, tokens], dim=1
            )
            timestep_tensor = torch.tensor(t.item(), device=device)

            # Get conditional logits
            logits_cond = self.model(
                sequence,
                timestep_tensor,
                conditional_mask
            )

            if condition_tokens is None or guidance_scale == 1.0:
                # Without a prefix to hide there is no second pass to take,
                # and at scale 1.0 the combination reduces to the
                # conditional logits anyway.
                logits = logits_cond
            else:
                # Get unconditional logits
                logits_uncond = self.model(
                    sequence,
                    timestep_tensor,
                    unconditional_mask
                )

                # Apply classifier-free guidance
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)

            # Only the generated positions are resampled; the prefix logits
            # belong to the fixed conditioning tokens.
            logits = logits[:, condition_len:]

            # Sample from guided logits
            tokens = _sample_from_logits(
                logits,
                temperature=temperature,
                vocab_size=self.vocab_size
            )

        return tokens


class DiDAInference:
    """
    High-level inference interface for DiDA.

    Provides convenient methods for different generation scenarios:
    - Pure image generation (:meth:`generate_image`)
    - Text-to-image generation (:meth:`text_to_image`), which generates the
      image tokens in parallel behind the given text tokens
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
