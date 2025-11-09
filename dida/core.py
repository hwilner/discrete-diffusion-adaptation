"""
Discrete Diffusion Adaptation (DiDA) - Core Implementation

This module implements the core DiDA mechanism as described in:
"Emu3.5: Native Multimodal Models are World Learners" (arXiv:2510.26583)

DiDA converts token-by-token autoregressive decoding into bidirectional parallel
prediction for visual tokens, achieving approximately 20x speedup for image generation
while maintaining text generation capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DiscreteDiffusionScheduler:
    """
    Scheduler for discrete diffusion process.
    
    Manages the noise schedule and timestep progression for the discrete
    diffusion adaptation process.
    
    Attributes:
        num_steps: Total number of diffusion steps.
        beta_start: Starting value for noise schedule.
        beta_end: Ending value for noise schedule.
    """
    
    def __init__(
        self,
        num_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Initialize the discrete diffusion scheduler.
        
        Args:
            num_steps: Number of denoising steps.
            beta_start: Initial noise level.
            beta_end: Final noise level.
        """
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear schedule for discrete diffusion
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(
        self,
        clean_tokens: torch.Tensor,
        timestep: int,
        vocab_size: int
    ) -> torch.Tensor:
        """
        Add noise to clean tokens at a given timestep.
        
        For discrete diffusion, noise is added by randomly replacing tokens
        with uniform samples from the vocabulary.
        
        Args:
            clean_tokens: Clean token indices, shape (batch_size, seq_len).
            timestep: Current diffusion timestep.
            vocab_size: Size of the token vocabulary.
            
        Returns:
            Noisy token indices with same shape as input.
        """
        batch_size, seq_len = clean_tokens.shape
        device = clean_tokens.device
        
        # Get noise level for this timestep
        alpha_t = self.alphas_cumprod[timestep]
        
        # Create mask for which tokens to corrupt
        # Probability of keeping original token is alpha_t
        keep_mask = torch.rand(batch_size, seq_len, device=device) < alpha_t
        
        # Generate random tokens from vocabulary
        random_tokens = torch.randint(
            0, vocab_size,
            (batch_size, seq_len),
            device=device
        )
        
        # Mix clean and random tokens based on mask
        noisy_tokens = torch.where(keep_mask, clean_tokens, random_tokens)
        
        return noisy_tokens
    
    def get_timesteps(self, reverse: bool = True) -> torch.Tensor:
        """
        Get timestep sequence for diffusion process.
        
        Args:
            reverse: If True, return timesteps in reverse order (for denoising).
                    If False, return in forward order (for adding noise).
                    
        Returns:
            Tensor of timestep indices.
        """
        timesteps = torch.arange(self.num_steps)
        if reverse:
            timesteps = torch.flip(timesteps, dims=[0])
        return timesteps


class DiDAAttentionMask:
    """
    Attention mask generator for DiDA.
    
    Creates the hybrid attention pattern where:
    - Noisy image tokens attend causally to preceding clean tokens
    - Noisy image tokens attend bidirectionally to other noisy tokens in same image
    - Clean tokens (text and image) follow causal attention pattern
    """
    
    @staticmethod
    def create_dida_mask(
        seq_len: int,
        noisy_start_idx: int,
        noisy_end_idx: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Create DiDA attention mask.
        
        Args:
            seq_len: Total sequence length.
            noisy_start_idx: Start index of noisy tokens.
            noisy_end_idx: End index of noisy tokens (exclusive).
            device: Device to create mask on.
            
        Returns:
            Attention mask of shape (seq_len, seq_len) where True indicates
            positions that can be attended to.
        """
        if device is None:
            device = torch.device('cpu')
            
        # Start with causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        # For noisy tokens, allow bidirectional attention within noisy region
        if noisy_start_idx < noisy_end_idx:
            mask[noisy_start_idx:noisy_end_idx, noisy_start_idx:noisy_end_idx] = True
        
        return mask
    
    @staticmethod
    def create_interleaved_mask(
        text_positions: torch.Tensor,
        image_positions: torch.Tensor,
        noisy_positions: torch.Tensor,
        seq_len: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Create attention mask for interleaved text-image sequences.
        
        Args:
            text_positions: Boolean tensor indicating text token positions.
            image_positions: Boolean tensor indicating image token positions.
            noisy_positions: Boolean tensor indicating noisy token positions.
            seq_len: Total sequence length.
            device: Device to create mask on.
            
        Returns:
            Attention mask of shape (seq_len, seq_len).
        """
        if device is None:
            device = text_positions.device
            
        # Initialize with causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        # Noisy tokens attend bidirectionally to other noisy tokens
        noisy_indices = torch.where(noisy_positions)[0]
        if len(noisy_indices) > 0:
            mask[noisy_indices[:, None], noisy_indices] = True
        
        return mask


class DiDACore(nn.Module):
    """
    Core DiDA denoising module.
    
    Implements the discrete diffusion denoising process for visual tokens.
    This module predicts clean tokens from noisy tokens using a transformer
    with modified attention patterns.
    
    Attributes:
        vocab_size: Size of the token vocabulary.
        hidden_dim: Dimension of hidden representations.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        """
        Initialize DiDA core module.
        
        Args:
            vocab_size: Size of token vocabulary.
            hidden_dim: Hidden dimension size.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Timestep embedding
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(
        self,
        noisy_tokens: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for denoising.
        
        Args:
            noisy_tokens: Noisy token indices, shape (batch_size, seq_len).
            timestep: Current timestep, shape (batch_size,) or scalar.
            attention_mask: Optional attention mask, shape (seq_len, seq_len).
            
        Returns:
            Logits for predicted clean tokens, shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = noisy_tokens.shape
        device = noisy_tokens.device
        
        # Embed tokens
        token_embeds = self.token_embedding(noisy_tokens)
        
        # Embed timestep
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(batch_size)
        timestep_embeds = self.timestep_embedding(
            timestep.float().unsqueeze(-1)
        )
        
        # Add timestep embedding to token embeddings
        # Broadcast timestep embedding across sequence
        hidden_states = token_embeds + timestep_embeds.unsqueeze(1)
        
        # Convert attention mask to format expected by transformer
        if attention_mask is not None:
            # Convert boolean mask to float mask
            # True (can attend) -> 0.0, False (cannot attend) -> -inf
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Apply transformer
        hidden_states = self.transformer(
            hidden_states,
            mask=attention_mask,
            is_causal=False
        )
        
        # Project to vocabulary
        logits = self.output_proj(hidden_states)
        
        return logits
    
    def denoise_step(
        self,
        noisy_tokens: torch.Tensor,
        timestep: int,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Perform a single denoising step.
        
        Args:
            noisy_tokens: Noisy token indices.
            timestep: Current timestep.
            attention_mask: Optional attention mask.
            temperature: Sampling temperature.
            
        Returns:
            Denoised token indices.
        """
        # Get logits from model
        logits = self.forward(
            noisy_tokens,
            torch.tensor(timestep, device=noisy_tokens.device),
            attention_mask
        )
        
        # Sample from logits with temperature
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            denoised_tokens = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1
            ).view(noisy_tokens.shape)
        else:
            # Greedy decoding
            denoised_tokens = torch.argmax(logits, dim=-1)
        
        return denoised_tokens


def test_dida_core():
    """
    Test the DiDA core implementation.
    Uses synthetic data for validation.
    """
    print("Testing DiDA Core Implementation...")
    
    # Test parameters
    batch_size = 2
    seq_len = 64
    vocab_size = 1000
    hidden_dim = 256
    num_heads = 8
    num_layers = 4
    
    # Create synthetic test data
    clean_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Initialize scheduler
    scheduler = DiscreteDiffusionScheduler(num_steps=10)
    
    # Add noise
    noisy_tokens = scheduler.add_noise(clean_tokens, timestep=5, vocab_size=vocab_size)
    print(f"Clean tokens shape: {clean_tokens.shape}")
    print(f"Noisy tokens shape: {noisy_tokens.shape}")
    
    # Create attention mask
    mask_gen = DiDAAttentionMask()
    attention_mask = mask_gen.create_dida_mask(
        seq_len=seq_len,
        noisy_start_idx=32,
        noisy_end_idx=64
    )
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Initialize DiDA core
    dida = DiDACore(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Forward pass
    logits = dida(noisy_tokens, torch.tensor(5), attention_mask)
    print(f"Output logits shape: {logits.shape}")
    
    # Verify shapes
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    
    # Test denoising step
    denoised = dida.denoise_step(noisy_tokens, timestep=5, attention_mask=attention_mask)
    print(f"Denoised tokens shape: {denoised.shape}")
    assert denoised.shape == noisy_tokens.shape
    
    print("DiDA Core test passed!")


if __name__ == "__main__":
    test_dida_core()
