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


class _GuidanceStubModel:
    """Model stand-in whose preference depends on the visible prefix.

    For every generated position the stub asks two questions: can this
    position attend to the conditioning prefix, and does the prefix carry
    the trigger token? A position that can see the trigger token prefers
    the conditional token, a position that can see a prefix without the
    trigger prefers the secondary token, and a position that cannot see
    the prefix at all prefers the unconditional token. That is the
    smallest difference the two guidance passes can have, and it is
    enough to tell the passes apart without training anything.

    The sampler calls the model as ``model(tokens, timestep,
    attention_mask)``, the way it calls :class:`torch.nn.Module`, so the
    stub implements ``__call__`` for the same reason.
    """

    def __init__(
        self,
        vocab_size,
        condition_len,
        conditional_token,
        secondary_token,
        unconditional_token,
        trigger_token
    ):
        """Record the vocabulary, the prefix length, and the preferences."""
        self.vocab_size = vocab_size
        self.condition_len = condition_len
        self.conditional_token = conditional_token
        self.secondary_token = secondary_token
        self.unconditional_token = unconditional_token
        self.trigger_token = trigger_token

    def __call__(self, tokens, timestep, attention_mask=None):
        """Call :meth:`forward`, the way ``nn.Module`` does."""
        return self.forward(tokens, timestep, attention_mask)

    def denoise_step(self, tokens, timestep, attention_mask=None, temperature=1.0):
        """Mirror ``DiDACore.denoise_step`` so the stub is a full stand-in.

        Args:
            tokens: Token indices of shape (batch, length).
            timestep: Diffusion timestep.
            attention_mask: Optional boolean mask of shape (length, length).
            temperature: Sampling temperature; values at or below zero
                decode greedily.

        Returns:
            Sampled token indices of shape (batch, length).
        """
        logits = self.forward(tokens, timestep, attention_mask)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(
                probs.reshape(-1, self.vocab_size), num_samples=1
            ).reshape(logits.shape[:-1])
        return torch.argmax(logits, dim=-1)

    def forward(self, tokens, timestep, attention_mask=None):
        """Return logits that favour one token per generated position.

        Args:
            tokens: Token indices of shape (batch, length).
            timestep: Diffusion timestep (unused by the stub).
            attention_mask: Optional boolean mask of shape (length, length).

        Returns:
            Logits of shape (batch, length, vocab_size).
        """
        batch_size, length = tokens.shape
        logits = torch.zeros(batch_size, length, self.vocab_size)
        # The prefix is only present when a condition was supplied; its
        # first token carries the trigger value.
        triggered = (
            self.condition_len > 0
            and bool(tokens[0, 0] == self.trigger_token)
        )
        for position in range(self.condition_len, length):
            # Column 0 holds the first prefix token, so it is the cheapest
            # probe for "may this position see the prefix at all?".
            sees_prefix = True
            if attention_mask is not None:
                sees_prefix = bool(attention_mask[position, 0])
            if not sees_prefix:
                token = self.unconditional_token
            elif triggered:
                token = self.conditional_token
            else:
                token = self.secondary_token
            # A logit this large makes the softmax pick the token
            # deterministically, so the assertions do not depend on seeds.
            logits[:, position, token] = 20.0
        return logits


class TestDiDASamplerGuidance:
    """Test cases for classifier-free guidance in DiDASampler."""

    VOCAB_SIZE = 16
    CONDITION_LEN = 4
    TRIGGER_TOKEN = 5
    CONDITIONAL_TOKEN = 3
    SECONDARY_TOKEN = 11
    UNCONDITIONAL_TOKEN = 7

    def _stub_sampler(self):
        """Build a sampler wired to the prefix-aware stub model."""
        model = _GuidanceStubModel(
            vocab_size=self.VOCAB_SIZE,
            condition_len=self.CONDITION_LEN,
            conditional_token=self.CONDITIONAL_TOKEN,
            secondary_token=self.SECONDARY_TOKEN,
            unconditional_token=self.UNCONDITIONAL_TOKEN,
            trigger_token=self.TRIGGER_TOKEN,
        )
        scheduler = DiscreteDiffusionScheduler(num_steps=3)
        return DiDASampler(model, scheduler, vocab_size=self.VOCAB_SIZE)

    def _trigger_condition(self):
        """Return a condition whose first token carries the trigger value."""
        return torch.full((2, self.CONDITION_LEN), self.TRIGGER_TOKEN, dtype=torch.long)

    def _plain_condition(self):
        """Return a condition without the trigger value."""
        return torch.full((2, self.CONDITION_LEN), 1, dtype=torch.long)

    def test_guidance_scale_zero_selects_the_unconditional_pass(self):
        """A guidance scale of zero must leave the unconditional logits."""
        sampler = self._stub_sampler()

        tokens = sampler.sample_with_guidance(
            batch_size=2,
            num_tokens=8,
            condition=self._trigger_condition(),
            device=torch.device('cpu'),
            guidance_scale=0.0,
        )

        assert (tokens == self.UNCONDITIONAL_TOKEN).all()

    def test_guidance_scale_one_selects_the_conditional_pass(self):
        """A guidance scale of one must leave the conditional logits."""
        sampler = self._stub_sampler()

        tokens = sampler.sample_with_guidance(
            batch_size=2,
            num_tokens=8,
            condition=self._trigger_condition(),
            device=torch.device('cpu'),
            guidance_scale=1.0,
        )

        assert (tokens == self.CONDITIONAL_TOKEN).all()

    def test_guidance_scale_above_one_amplifies_the_conditional_pass(self):
        """A guidance scale above one must keep the conditional direction."""
        sampler = self._stub_sampler()

        tokens = sampler.sample_with_guidance(
            batch_size=2,
            num_tokens=8,
            condition=self._trigger_condition(),
            device=torch.device('cpu'),
            guidance_scale=7.5,
        )

        assert (tokens == self.CONDITIONAL_TOKEN).all()

    def test_condition_changes_the_generated_tokens(self):
        """Different conditioning prefixes must give different tokens."""
        sampler = self._stub_sampler()
        device = torch.device('cpu')

        triggered = sampler.sample_with_guidance(
            batch_size=2,
            num_tokens=8,
            condition=self._trigger_condition(),
            device=device,
        )
        plain = sampler.sample_with_guidance(
            batch_size=2,
            num_tokens=8,
            condition=self._plain_condition(),
            device=device,
        )

        assert not torch.equal(triggered, plain)

    def test_condition_is_not_dropped_from_the_sequence(self):
        """The conditioning prefix must reach the denoiser every step."""
        sampler = self._stub_sampler()
        device = torch.device('cpu')
        condition = self._trigger_condition()

        tokens = sampler.sample_with_guidance(
            batch_size=2,
            num_tokens=8,
            condition=condition,
            device=device,
            guidance_scale=1.0,
        )

        # The stub only produces the conditional token when it can see a
        # trigger token in the prefix, so the trigger reaching the model
        # is what puts the conditional token in the output.
        assert (tokens == self.CONDITIONAL_TOKEN).all()

    def test_guidance_masks_hide_the_prefix_from_generated_tokens(self):
        """The unconditional mask must hide the prefix from the tokens."""
        seq_len = 6
        condition_len = 2
        conditional, unconditional = DiDAAttentionMask.create_guidance_masks(
            seq_len=seq_len, condition_len=condition_len
        )

        # Generated tokens see the prefix in the conditional pass ...
        assert conditional[condition_len:, :condition_len].all()
        # ... and do not in the unconditional pass.
        assert not unconditional[condition_len:, :condition_len].any()
        # Generated tokens still attend to each other in both passes.
        assert unconditional[condition_len:, condition_len:].all()
        # The prefix itself stays causal in both passes.
        for row in range(condition_len):
            assert conditional[row, :row + 1].all()
            assert unconditional[row, :row + 1].all()
            assert not conditional[row, row + 1:].any()

    def test_invalid_condition_raises(self):
        """A malformed conditioning prefix must fail loudly."""
        sampler = self._stub_sampler()
        device = torch.device('cpu')

        with pytest.raises(ValueError):
            sampler.sample_with_guidance(
                batch_size=2,
                num_tokens=8,
                condition=torch.randint(0, self.VOCAB_SIZE, (3, 4)),
                device=device,
            )

        with pytest.raises(ValueError):
            sampler.sample_with_guidance(
                batch_size=2,
                num_tokens=8,
                condition=torch.rand(2, 4),
                device=device,
            )

        with pytest.raises(ValueError):
            sampler.sample_with_guidance(
                batch_size=2,
                num_tokens=8,
                condition=torch.randint(0, 99, (2, 4)),
                device=device,
            )

    def test_unconditional_sample_is_unchanged(self):
        """Sampling without a condition must still work and stay in range."""
        sampler = self._stub_sampler()
        device = torch.device('cpu')

        tokens = sampler.sample(
            batch_size=2,
            num_tokens=8,
            device=device,
            temperature=1.0,
        )

        assert tokens.shape == (2, 8)
        assert tokens.dtype == torch.long
        assert (tokens >= 0).all() and (tokens < self.VOCAB_SIZE).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
