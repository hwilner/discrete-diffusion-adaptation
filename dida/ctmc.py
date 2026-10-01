"""Continuous-Time Markov Chain (CTMC) samplers and log-space reparameterization.

Implements the discrete-diffusion sampling upgrades specified in the repository
roadmap (Phase 2: Differentiable Simulation Engines):

* :func:`log_gumbel_softmax` -- Gumbel-Softmax reparameterization computed
  strictly in logarithmic space, preventing underflow over large token
  vocabularies where softmax probabilities fall below the float32 minimum.
* :class:`CTMCUniformizationSampler` -- reformulates the discrete noise
  schedule as a continuous-time Markov chain and samples exact trajectories
  via uniformization. Token transitions are treated as a Poisson jump
  process, so a reverse trajectory over ``t in [0, T]`` is generated with
  ``Poisson(lambda_max * T)`` jump evaluations instead of one categorical
  re-sampling per discretization step, compressing sampling trajectories into
  significantly fewer evaluations.

Mathematical background (uniformization): for a rate matrix ``Q`` with
``lambda >= max_i |Q_ii|``, the transition semigroup satisfies

    exp(Q t) = sum_{n>=0} Poisson(n; lambda t) * P^n,   P = I + Q / lambda,

so an exact CTMC path is simulated by drawing a Poisson number of jumps and
evolving the discrete-time chain ``P`` between jumps.

References:
    Continuous-time discrete diffusion via uniformization, e.g. Campbell et
    al. 2022 ("A Continuous Time Framework for Discrete Denoising Models").
"""

from typing import Optional, Tuple

import math

import torch
import torch.nn.functional as F

__all__ = [
    "log_gumbel_softmax",
    "CTMCUniformizationSampler",
]

_EPS = 1e-45  # smallest positive float32; log-domain floor


def log_gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Gumbel-Softmax sample computed entirely in logarithmic space.

    Standard ``F.gumbel_softmax`` exponentiates normalized log-probabilities
    and underflows to exact zeros once ``logits / tau < -103`` (float32),
    which occurs routinely for large-vocabulary token distributions with low
    temperature. This implementation returns log-probabilities directly and
    never materializes the softmax denominator in linear space.

    Args:
        logits: Unnormalized log-probabilities ``(..., K)``.
        tau: Temperature; lower values sharpen the distribution.
        hard: If True, returns a one-hot sample via the straight-through
            estimator (still expressed as log-probabilities).
        generator: Optional torch random generator for reproducibility.

    Returns:
        Log-probabilities of the relaxed sample, shape ``(..., K)``. Apply
        ``torch.exp`` only if downstream code truly needs linear space.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    # Gumbel noise in log space: g = -log(-log U), U ~ Uniform(0, 1).
    uniform = torch.rand(
        logits.shape, device=logits.device, dtype=logits.dtype, generator=generator
    ).clamp_min(_EPS)
    gumbel = -torch.log((-torch.log(uniform)).clamp_min(_EPS))

    perturbed = (logits + gumbel) / tau
    log_sample = perturbed - torch.logsumexp(perturbed, dim=-1, keepdim=True)

    if hard:
        index = log_sample.argmax(dim=-1, keepdim=True)
        one_hot_log = torch.full_like(log_sample, -float("inf"))
        one_hot_log.scatter_(-1, index, 0.0)
        # Straight-through: forward one-hot, backward relaxed.
        log_sample = one_hot_log + (log_sample - log_sample.detach())

    return log_sample


class CTMCUniformizationSampler:
    """Exact CTMC trajectory sampler via uniformization.

    Treats discrete diffusion token transitions as a continuous Poisson jump
    process. A base rate matrix ``Q`` (vocab x vocab) defines the corruption
    dynamics; reverse-time sampling interleaves exact CTMC holding times with
    model-provided denoising distributions at jump events.

    Attributes:
        vocab_size: Number of categorical states ``K``.
        uniformization_rate: ``lambda >= max_i |Q_ii|``; chosen automatically
            from ``Q`` if not provided.
    """

    def __init__(
        self,
        rate_matrix: torch.Tensor,
        uniformization_rate: Optional[float] = None,
    ):
        """Initialize the sampler from a rate matrix.

        Args:
            rate_matrix: CTMC generator ``Q`` of shape ``(K, K)`` with
                non-negative off-diagonal entries and rows summing to zero.
            uniformization_rate: Optional uniformization rate; defaults to
                ``1.05 * max_i |Q_ii|`` for a small safety margin.
        """
        if rate_matrix.ndim != 2 or rate_matrix.shape[0] != rate_matrix.shape[1]:
            raise ValueError("rate_matrix must be square (K, K)")
        row_sums = rate_matrix.sum(dim=-1)
        if not torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5):
            raise ValueError("rate_matrix rows must sum to zero")

        self.rate_matrix = rate_matrix
        self.vocab_size = rate_matrix.shape[0]

        max_exit = rate_matrix.diagonal().abs().max().item()
        if max_exit <= 0:
            raise ValueError("rate_matrix has no transitions")
        if uniformization_rate is None:
            uniformization_rate = 1.05 * max_exit
        if uniformization_rate < max_exit:
            raise ValueError("uniformization_rate must be >= max |Q_ii|")
        self.uniformization_rate = float(uniformization_rate)

        # Uniformized discrete-time jump kernel (row-stochastic).
        eye = torch.eye(self.vocab_size, device=rate_matrix.device, dtype=rate_matrix.dtype)
        self.jump_kernel = eye + rate_matrix / self.uniformization_rate

    def sample_num_jumps(
        self,
        duration: float,
        batch_shape: Tuple[int, ...] = (),
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw the Poisson number of jump events over an interval.

        Args:
            duration: Interval length ``T``.
            batch_shape: Leading shape of the returned count tensor.
            generator: Optional torch random generator.

        Returns:
            Integer tensor of jump counts with shape ``batch_shape``.
        """
        if duration < 0:
            raise ValueError("duration must be non-negative")
        mean = torch.tensor(self.uniformization_rate * duration)
        poisson = torch.distributions.Poisson(mean)
        counts = poisson.sample(batch_shape)
        return counts.long()

    def sample_terminal_state(
        self,
        start_states: torch.Tensor,
        duration: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample exact CTMC terminal states after ``duration`` time units.

        Rather than iterating per jump, the per-path transition matrix is
        computed via the uniformization series truncated at a quantile of the
        Poisson distribution, which is exact up to tail mass < 1e-12.

        Args:
            start_states: Integer state indices, shape ``(B,)``.
            duration: Time horizon ``T``.
            generator: Optional torch random generator.

        Returns:
            Integer terminal state indices, shape ``(B,)``.
        """
        lam_t = self.uniformization_rate * duration
        # Truncation: upper quantile with tail mass < 1e-12.
        n_max = int(lam_t + 10.0 * (lam_t ** 0.5 + 1.0)) + 20

        eye = torch.eye(
            self.vocab_size,
            device=self.rate_matrix.device,
            dtype=self.rate_matrix.dtype,
        )
        # Uniformization series: exp(QT) = sum_n Poisson(n; lam T) P^n.
        transition = torch.zeros_like(eye)
        power = eye.clone()
        log_coeff = -lam_t  # log Poisson weight at n = 0
        for n in range(n_max + 1):
            if n > 0:
                power = power @ self.jump_kernel
                log_coeff += math.log(lam_t) - math.log(float(n))
            transition = transition + math.exp(log_coeff) * power

        rows = transition[start_states]  # (B, K)
        rows = rows.clamp_min(0)
        rows = rows / rows.sum(dim=-1, keepdim=True)
        return torch.multinomial(rows, 1, generator=generator).squeeze(-1)

    def reverse_guided_step(
        self,
        current_tokens: torch.Tensor,
        model_logits: torch.Tensor,
        guidance_weights: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """One entropy-regulated reverse transition with verifier guidance.

        Modulates the model's reverse-step distribution by a semantic
        verifier score, mirroring the roadmap mechanism

            Q_tilde(i, j) ∝ Q_t(i, j) * exp(-lambda * H(Verifier(x_t = j)))

        implemented additively in log space (never exponentiating
        probabilities), then samples via log-space Gumbel-Softmax.

        Args:
            current_tokens: Current token indices ``(B, L)`` (used for API
                symmetry; the transition prior enters via ``model_logits``).
            model_logits: Denoiser log-probabilities ``(B, L, K)``.
            guidance_weights: Optional verifier penalty ``(B, L, K)``; larger
                values suppress tokens that downstream verifiers disagree on.
            tau: Gumbel-Softmax temperature.
            generator: Optional torch random generator.

        Returns:
            Sampled token indices ``(B, L)``.
        """
        log_probs = F.log_softmax(model_logits, dim=-1)
        if guidance_weights is not None:
            log_probs = log_probs - guidance_weights
            log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_sample = log_gumbel_softmax(
            log_probs, tau=tau, hard=True, generator=generator
        )
        return log_sample.exp().argmax(dim=-1)
