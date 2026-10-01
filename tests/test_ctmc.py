"""Tests for CTMC uniformization sampling and log-space Gumbel-Softmax.

All tests use synthetic random data, consistent with the repository testing
charter. CTMC marginals are validated against the closed-form matrix
exponential ``expm(Q T)`` on small chains, and the log-space Gumbel-Softmax is
validated for underflow resistance and distributional correctness.
"""

import torch
import torch.nn.functional as F

from dida.ctmc import CTMCUniformizationSampler, log_gumbel_softmax


def _two_state_chain(rate=2.0):
    """Symmetric two-state CTMC generator with known closed-form expm.

    Args:
        rate: Transition rate in each direction.

    Returns:
        A ``(2, 2)`` rate matrix.
    """
    return torch.tensor([[-rate, rate], [rate, -rate]])


def _random_generator_matrix(k, seed):
    """Build a random valid CTMC rate matrix.

    Args:
        k: Number of states.
        seed: Random seed.

    Returns:
        A ``(K, K)`` rate matrix with rows summing to zero.
    """
    generator = torch.Generator().manual_seed(seed)
    off_diag = torch.rand(k, k, generator=generator) * 3.0
    off_diag.fill_diagonal_(0.0)
    q = off_diag
    q.fill_diagonal_(0.0)
    q -= torch.diag(q.sum(dim=-1))
    return q


class TestLogGumbelSoftmax:
    """Underflow resistance and correctness of the log-space reparameterization."""

    def test_no_underflow_extreme_logits(self):
        """Large negative logits stay finite in log space (linear softmax = 0)."""
        logits = torch.tensor([[0.0, -500.0, -1000.0]])
        log_sample = log_gumbel_softmax(logits, tau=0.1)
        # Linear-space softmax would be exactly 0 for entries 1 and 2.
        linear = F.softmax(logits / 0.1, dim=-1)
        assert linear[0, 2].item() == 0.0
        assert torch.isfinite(log_sample).all()

    def test_valid_log_distribution(self):
        """Exponentiated samples form a valid probability distribution."""
        torch.manual_seed(0)
        logits = torch.randn(4, 100)
        log_sample = log_gumbel_softmax(logits, tau=1.0)
        probs = log_sample.exp()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert (probs >= 0).all()

    def test_hard_returns_one_hot(self):
        """Hard mode returns a one-hot (log) vector with a single 0 entry."""
        torch.manual_seed(1)
        logits = torch.randn(2, 50)
        log_sample = log_gumbel_softmax(logits, tau=0.5, hard=True)
        assert ((log_sample == 0.0).sum(dim=-1) == 1).all()
        assert (log_sample < -1e30).sum(dim=-1).eq(49).all()

    def test_temperature_sharpens(self):
        """Lower temperature concentrates probability mass."""
        torch.manual_seed(2)
        logits = torch.randn(1000, 10)
        sharp = log_gumbel_softmax(logits, tau=0.1).exp().max(dim=-1).values.mean()
        soft = log_gumbel_softmax(logits, tau=10.0).exp().max(dim=-1).values.mean()
        assert sharp > soft

    def test_gradient_flows(self):
        """Gradients propagate through the log-space sample."""
        logits = torch.randn(3, 20, requires_grad=True)
        log_sample = log_gumbel_softmax(logits, tau=1.0)
        log_sample.exp().sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestCTMCUniformization:
    """Exactness of uniformization against the closed-form matrix exponential."""

    def test_two_state_marginals_closed_form(self):
        """Terminal marginals match the analytic two-state solution."""
        rate = 2.0
        duration = 0.7
        sampler = CTMCUniformizationSampler(_two_state_chain(rate))
        torch.manual_seed(0)
        starts = torch.zeros(20000, dtype=torch.long)
        terminals = sampler.sample_terminal_state(starts, duration)
        empirical = (terminals == 1).float().mean().item()
        analytic = 0.5 * (1.0 - torch.exp(torch.tensor(-2 * rate * duration)).item())
        assert abs(empirical - analytic) < 0.02

    def test_matches_matrix_exponential_random_chain(self):
        """Terminal distribution matches expm(QT) on a random 4-state chain."""
        q = _random_generator_matrix(4, seed=7)
        duration = 0.4
        sampler = CTMCUniformizationSampler(q)
        torch.manual_seed(1)
        starts = torch.full((20000,), 2, dtype=torch.long)
        terminals = sampler.sample_terminal_state(starts, duration)
        empirical = torch.bincount(terminals, minlength=4).float() / len(terminals)
        analytic = torch.linalg.matrix_exp(q * duration)[2]
        assert torch.allclose(empirical, analytic, atol=0.02)

    def test_jump_counts_poisson(self):
        """Jump counts follow Poisson(lambda * T)."""
        sampler = CTMCUniformizationSampler(_two_state_chain(1.0))
        duration = 3.0
        counts = sampler.sample_num_jumps(duration, batch_shape=(5000,))
        expected_mean = sampler.uniformization_rate * duration
        assert abs(counts.float().mean().item() - expected_mean) < 0.15 * expected_mean

    def test_step_compression(self):
        """Mean number of CTMC evaluations is far below a dense step grid."""
        sampler = CTMCUniformizationSampler(_two_state_chain(1.0))
        duration = 1.0
        mean_jumps = sampler.sample_num_jumps(duration, batch_shape=(2000,)).float().mean()
        dense_grid_steps = 50  # typical discrete diffusion step count
        assert mean_jumps.item() < dense_grid_steps / 5

    def test_rejects_invalid_rate_matrix(self):
        """Non-conservative generators are rejected."""
        bad = torch.tensor([[-1.0, 0.5], [0.5, -1.0]])  # rows sum to -0.5
        try:
            CTMCUniformizationSampler(bad)
        except ValueError:
            return
        raise AssertionError("expected ValueError for invalid rate matrix")


class TestReverseGuidedStep:
    """Entropy-regulated guided reverse transition."""

    def test_guidance_suppresses_penalized_tokens(self):
        """Verifier penalties shift samples away from disagreed-upon tokens."""
        torch.manual_seed(3)
        batch, length, vocab = 4, 8, 30
        logits = torch.zeros(batch, length, vocab)  # uniform prior
        penalty = torch.zeros(batch, length, vocab)
        penalty[..., 0] = 20.0  # token 0 heavily penalized by verifier
        sampler = CTMCUniformizationSampler(_two_state_chain(1.0))
        tokens = sampler.reverse_guided_step(
            torch.zeros(batch, length, dtype=torch.long),
            logits,
            guidance_weights=penalty,
            tau=0.5,
        )
        assert (tokens == 0).float().mean().item() < 0.02

    def test_output_shape_and_range(self):
        """Sampled tokens are valid indices of the right shape."""
        torch.manual_seed(4)
        batch, length, vocab = 2, 5, 12
        logits = torch.randn(batch, length, vocab)
        sampler = CTMCUniformizationSampler(_two_state_chain(1.0))
        tokens = sampler.reverse_guided_step(
            torch.zeros(batch, length, dtype=torch.long), logits
        )
        assert tokens.shape == (batch, length)
        assert (tokens >= 0).all() and (tokens < vocab).all()
