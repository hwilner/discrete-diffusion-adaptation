"""
Comprehensive test suite for DiDA implementation.

Note: All tests use synthetic/random data for validation purposes only.
This is standard practice in ML research implementations to verify correctness.
"""

import torch
import pytest
from dida import (
    DiDACore,
    DiscreteDiffusionScheduler,
    DiDAAttentionMask,
    DiDASampler,
    DiDAInference
)


class TestDiscreteDiffusionScheduler:
    """Test cases for DiscreteDiffusionScheduler."""
    
    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = DiscreteDiffusionScheduler(num_steps=50)
        assert scheduler.num_steps == 50
        assert len(scheduler.betas) == 50
        assert len(scheduler.alphas) == 50
        assert len(scheduler.alphas_cumprod) == 50
        
    def test_add_noise(self):
        """Test noise addition to clean tokens."""
        scheduler = DiscreteDiffusionScheduler(num_steps=10)
        clean_tokens = torch.randint(0, 1000, (2, 64))
        
        noisy_tokens = scheduler.add_noise(clean_tokens, timestep=5, vocab_size=1000)
        
        assert noisy_tokens.shape == clean_tokens.shape
        assert noisy_tokens.dtype == clean_tokens.dtype
        
    def test_get_timesteps(self):
        """Test timestep generation."""
        scheduler = DiscreteDiffusionScheduler(num_steps=10)
        
        # Forward timesteps
        forward_steps = scheduler.get_timesteps(reverse=False)
        assert len(forward_steps) == 10
        assert forward_steps[0] == 0
        assert forward_steps[-1] == 9
        
        # Reverse timesteps (for denoising)
        reverse_steps = scheduler.get_timesteps(reverse=True)
        assert len(reverse_steps) == 10
        assert reverse_steps[0] == 9
        assert reverse_steps[-1] == 0


class TestDiDAAttentionMask:
    """Test cases for DiDA attention mask generation."""
    
    def test_create_dida_mask(self):
        """Test DiDA attention mask creation."""
        seq_len = 64
        noisy_start = 32
        noisy_end = 64
        
        mask = DiDAAttentionMask.create_dida_mask(
            seq_len=seq_len,
            noisy_start_idx=noisy_start,
            noisy_end_idx=noisy_end
        )
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check causal pattern for clean tokens
        for i in range(noisy_start):
            for j in range(seq_len):
                if j <= i:
                    assert mask[i, j] == True
                elif j >= noisy_start:
                    assert mask[i, j] == False
                    
        # Check bidirectional pattern for noisy tokens
        for i in range(noisy_start, noisy_end):
            for j in range(noisy_start, noisy_end):
                assert mask[i, j] == True
                
    def test_create_interleaved_mask(self):
        """Test interleaved text-image attention mask."""
        seq_len = 64
        text_positions = torch.zeros(seq_len, dtype=torch.bool)
        text_positions[:32] = True
        
        image_positions = torch.zeros(seq_len, dtype=torch.bool)
        image_positions[32:] = True
        
        noisy_positions = torch.zeros(seq_len, dtype=torch.bool)
        noisy_positions[32:] = True
        
        mask = DiDAAttentionMask.create_interleaved_mask(
            text_positions=text_positions,
            image_positions=image_positions,
            noisy_positions=noisy_positions,
            seq_len=seq_len
        )
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool


class TestDiDACore:
    """Test cases for DiDA core model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = DiDACore(
            vocab_size=1000,
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        )
        
        assert model.vocab_size == 1000
        assert model.hidden_dim == 256
        assert model.num_heads == 8
        assert model.num_layers == 4
        
    def test_forward_pass(self):
        """Test forward pass with synthetic data."""
        model = DiDACore(vocab_size=1000, hidden_dim=256, num_heads=8, num_layers=4)
        
        batch_size = 2
        seq_len = 64
        noisy_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        timestep = torch.tensor(5)
        
        logits = model(noisy_tokens, timestep)
        
        assert logits.shape == (batch_size, seq_len, 1000)
        
    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        model = DiDACore(vocab_size=1000, hidden_dim=256, num_heads=8, num_layers=4)
        
        batch_size = 2
        seq_len = 64
        noisy_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        timestep = torch.tensor(5)
        
        attention_mask = DiDAAttentionMask.create_dida_mask(
            seq_len=seq_len,
            noisy_start_idx=32,
            noisy_end_idx=64
        )
        
        logits = model(noisy_tokens, timestep, attention_mask)
        
        assert logits.shape == (batch_size, seq_len, 1000)
        
    def test_denoise_step(self):
        """Test single denoising step."""
        model = DiDACore(vocab_size=1000, hidden_dim=256, num_heads=8, num_layers=4)
        
        batch_size = 2
        seq_len = 64
        noisy_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        
        denoised = model.denoise_step(noisy_tokens, timestep=5, temperature=1.0)
        
        assert denoised.shape == noisy_tokens.shape
        assert denoised.dtype == torch.long
        
    def test_denoise_step_greedy(self):
        """Test greedy denoising (temperature=0)."""
        model = DiDACore(vocab_size=1000, hidden_dim=256, num_heads=8, num_layers=4)
        
        batch_size = 2
        seq_len = 64
        noisy_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        
        denoised = model.denoise_step(noisy_tokens, timestep=5, temperature=0.0)
        
        assert denoised.shape == noisy_tokens.shape


class TestDiDASampler:
    """Test cases for DiDA sampler."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        model = DiDACore(vocab_size=1000, hidden_dim=256, num_heads=8, num_layers=4)
        scheduler = DiscreteDiffusionScheduler(num_steps=10)
        sampler = DiDASampler(model, scheduler, vocab_size=1000)
        
        assert sampler.model == model
        assert sampler.scheduler == scheduler
        assert sampler.vocab_size == 1000
        
    def test_sample(self):
        """Test basic sampling."""
        model = DiDACore(vocab_size=1000, hidden_dim=256, num_heads=8, num_layers=4)
        scheduler = DiscreteDiffusionScheduler(num_steps=5)
        sampler = DiDASampler(model, scheduler, vocab_size=1000)
        
        batch_size = 2
        num_tokens = 64
        device = torch.device('cpu')
        
        generated = sampler.sample(
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=device,
            temperature=1.0
        )
        
        assert generated.shape == (batch_size, num_tokens)
        assert generated.dtype == torch.long
        
    def test_sample_interleaved(self):
        """Test interleaved text-image sampling."""
        model = DiDACore(vocab_size=1000, hidden_dim=256, num_heads=8, num_layers=4)
        scheduler = DiscreteDiffusionScheduler(num_steps=5)
        sampler = DiDASampler(model, scheduler, vocab_size=1000)
        
        batch_size = 2
        text_tokens = torch.randint(0, 1000, (batch_size, 32))
        num_image_tokens = 32
        device = torch.device('cpu')
        
        full_seq, image_tokens = sampler.sample_interleaved(
            text_tokens=text_tokens,
            num_image_tokens=num_image_tokens,
            device=device,
            temperature=1.0
        )
        
        assert full_seq.shape == (batch_size, 64)
        assert image_tokens.shape == (batch_size, num_image_tokens)
        
        # Verify text tokens are preserved
        assert torch.all(full_seq[:, :32] == text_tokens)


class TestDiDAInference:
    """Test cases for DiDA inference interface."""
    
    def test_initialization(self):
        """Test inference interface initialization."""
        model = DiDACore(vocab_size=2000, hidden_dim=256, num_heads=8, num_layers=4)
        scheduler = DiscreteDiffusionScheduler(num_steps=10)
        
        inference = DiDAInference(
            model=model,
            scheduler=scheduler,
            vocab_size=2000,
            text_vocab_size=1000,
            image_vocab_size=1000
        )
        
        assert inference.vocab_size == 2000
        assert inference.text_vocab_size == 1000
        assert inference.image_vocab_size == 1000
        
    def test_generate_image(self):
        """Test image generation."""
        model = DiDACore(vocab_size=2000, hidden_dim=256, num_heads=8, num_layers=4)
        scheduler = DiscreteDiffusionScheduler(num_steps=5)
        
        inference = DiDAInference(
            model=model,
            scheduler=scheduler,
            vocab_size=2000,
            text_vocab_size=1000,
            image_vocab_size=1000
        )
        
        image_tokens = inference.generate_image(
            batch_size=2,
            image_size=256,
            patch_size=16,
            device=torch.device('cpu'),
            temperature=1.0
        )
        
        expected_tokens = (256 // 16) ** 2
        assert image_tokens.shape == (2, expected_tokens)
        
    def test_text_to_image(self):
        """Test text-to-image generation."""
        model = DiDACore(vocab_size=2000, hidden_dim=256, num_heads=8, num_layers=4)
        scheduler = DiscreteDiffusionScheduler(num_steps=5)
        
        inference = DiDAInference(
            model=model,
            scheduler=scheduler,
            vocab_size=2000,
            text_vocab_size=1000,
            image_vocab_size=1000
        )
        
        text_tokens = torch.randint(0, 1000, (2, 32))
        
        image_tokens = inference.text_to_image(
            text_tokens=text_tokens,
            image_size=256,
            patch_size=16,
            temperature=1.0
        )
        
        expected_tokens = (256 // 16) ** 2
        assert image_tokens.shape == (2, expected_tokens)


class TestIntegration:
    """Integration tests for complete DiDA pipeline."""
    
    def test_end_to_end_generation(self):
        """Test complete generation pipeline."""
        # Initialize components
        vocab_size = 1000
        model = DiDACore(vocab_size=vocab_size, hidden_dim=256, num_heads=8, num_layers=4)
        scheduler = DiscreteDiffusionScheduler(num_steps=5)
        sampler = DiDASampler(model, scheduler, vocab_size)
        
        # Generate tokens
        batch_size = 2
        num_tokens = 64
        device = torch.device('cpu')
        
        generated = sampler.sample(
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=device
        )
        
        # Verify output
        assert generated.shape == (batch_size, num_tokens)
        assert torch.all(generated >= 0)
        assert torch.all(generated < vocab_size)
        
    def test_hybrid_generation(self):
        """Test hybrid text-image generation."""
        vocab_size = 1000
        model = DiDACore(vocab_size=vocab_size, hidden_dim=256, num_heads=8, num_layers=4)
        scheduler = DiscreteDiffusionScheduler(num_steps=5)
        sampler = DiDASampler(model, scheduler, vocab_size)
        
        # Create text tokens
        batch_size = 2
        text_len = 32
        image_len = 32
        text_tokens = torch.randint(0, vocab_size, (batch_size, text_len))
        
        # Generate image tokens
        full_seq, image_tokens = sampler.sample_interleaved(
            text_tokens=text_tokens,
            num_image_tokens=image_len,
            device=torch.device('cpu')
        )
        
        # Verify outputs
        assert full_seq.shape == (batch_size, text_len + image_len)
        assert image_tokens.shape == (batch_size, image_len)
        assert torch.all(full_seq[:, :text_len] == text_tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
