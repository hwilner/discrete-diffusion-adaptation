"""
End-to-End Image-Token Generation Walkthrough (Commented)

This script is a slow-paced, heavily commented companion to
``examples/basic_usage.py``. Where ``basic_usage.py`` shows the shortest path
to a generated sample, this walkthrough opens up each step of the generate
path and explains it in plain language:

1. Masking — how the forward diffusion process corrupts clean tokens.
2. Hybrid attention — how the DiDA attention mask mixes causal and
   bidirectional attention.
3. Confidence — how per-token prediction confidence evolves as denoising
   progresses (the signal that confidence-unmasking schedulers exploit).
4. Generation — the full iterative-denoising path, end to end, on CPU.

For the zero-background conceptual introduction (painting analogy included),
see ``docs/EXTENDED_INTRODUCTION.md``. For the architecture deep dive, see
``docs/ARCHITECTURE.md``.

The model used here is randomly initialized, so the generated grid is
arbitrary token IDs — the point of this walkthrough is the mechanics of the
pipeline, not sample quality.
"""

import torch

from dida import (
    DiDACore,
    DiscreteDiffusionScheduler,
    DiDAAttentionMask,
    DiDAInference,
)


# A fixed brightness ramp used to render token IDs as printable glyphs.
# Ten levels keep the mapping easy to read: token_id % 10 picks the glyph.
GLYPH_RAMP = " .:-=+*#%@"


def render_grid(tokens, grid_width):
    """Render a flat token-ID tensor as rows of integers.

    Args:
        tokens: 1-D tensor (or 2-D with batch size 1) of token IDs whose
            length is a perfect square multiple of ``grid_width``.
        grid_width: Number of tokens per printed row.

    Returns:
        A string with one row of space-separated token IDs per line.
    """
    flat = tokens.reshape(-1).tolist()
    rows = [
        " ".join(f"{token:4d}" for token in flat[start:start + grid_width])
        for start in range(0, len(flat), grid_width)
    ]
    return "\n".join(rows)


def render_glyphs(tokens, grid_width):
    """Render a flat token-ID tensor as brightness glyphs.

    Args:
        tokens: 1-D tensor (or 2-D with batch size 1) of token IDs.
        grid_width: Number of glyphs per printed row.

    Returns:
        A string mapping each token ID to a glyph in ``GLYPH_RAMP`` via
        ``token_id % len(GLYPH_RAMP)``, one row per ``grid_width`` tokens.
    """
    flat = tokens.reshape(-1).tolist()
    rows = [
        "".join(GLYPH_RAMP[token % len(GLYPH_RAMP)] for token in flat[start:start + grid_width])
        for start in range(0, len(flat), grid_width)
    ]
    return "\n".join(rows)


def main():
    """Run the full commented walkthrough; finishes in seconds on CPU."""
    # ------------------------------------------------------------------
    # Step 0: Reproducibility and a tiny CPU-friendly configuration.
    # ------------------------------------------------------------------
    # A fixed seed makes every printed value below reproducible run to run.
    torch.manual_seed(0)
    # The walkthrough is designed for CPU; no accelerator is required.
    device = torch.device("cpu")

    # A small square grid keeps the output readable: 8 x 8 = 64 tokens.
    # ``DiDAInference.generate_image`` derives the token count from
    # ``(image_size // patch_size) ** 2``, so we pick the two to match.
    image_size = 64   # pixels per side of the notional image
    patch_size = 8    # pixels per patch -> 8 patches per side
    grid_width = image_size // patch_size  # 8 tokens per row

    # A small vocabulary and a small transformer keep runtime negligible.
    vocab_size = 512

    # ------------------------------------------------------------------
    # Step 1: Build the three public components of the generate path.
    # ------------------------------------------------------------------
    # ``DiDACore`` is the denoising network: a transformer that reads noisy
    # tokens plus the current timestep and predicts logits over clean tokens.
    # It is randomly initialized here — this walkthrough demonstrates the
    # pipeline mechanics, not trained sample quality.
    model = DiDACore(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,  # deterministic behaviour for a reproducible demo
    ).to(device)
    model.eval()  # generation uses no gradient tracking and no dropout

    # ``DiscreteDiffusionScheduler`` owns the noise schedule: it knows how
    # much of the clean signal survives at each timestep (``alphas_cumprod``)
    # and hands out the denoising timestep order.
    scheduler = DiscreteDiffusionScheduler(num_steps=8)

    # ``DiDAInference`` is the high-level API: it wires the model and
    # scheduler into a ``DiDASampler`` and exposes ``generate_image``.
    inference = DiDAInference(
        model=model,
        scheduler=scheduler,
        vocab_size=vocab_size,
        text_vocab_size=vocab_size,   # unused in this pure-image walkthrough
        image_vocab_size=vocab_size,  # unused in this pure-image walkthrough
    )

    print("=== Step 1: components built (randomly initialized model) ===")
    print(f"grid: {grid_width}x{grid_width} = {grid_width ** 2} tokens, "
          f"vocab_size={vocab_size}, num_steps={scheduler.num_steps}")

    # ------------------------------------------------------------------
    # Step 2: Masking — the forward (noise-adding) direction of diffusion.
    # ------------------------------------------------------------------
    # Discrete diffusion corrupts clean tokens by randomly replacing them
    # with uniform vocabulary draws. The replacement probability grows with
    # the timestep; this schedule is gentle (small betas), so even the last
    # timestep replaces only a small fraction of this 64-token grid.
    # ``add_noise`` takes clean token IDs and returns a corrupted copy at
    # the requested timestep.
    clean_demo = torch.arange(grid_width ** 2).unsqueeze(0) % vocab_size
    print("\n=== Step 2: masking demo (forward process) ===")
    print("clean grid:")
    print(render_grid(clean_demo, grid_width))
    for timestep in (0, scheduler.num_steps // 2, scheduler.num_steps - 1):
        noisy_demo = scheduler.add_noise(clean_demo, timestep, vocab_size)
        replaced = (noisy_demo != clean_demo).float().mean().item()
        print(f"timestep {timestep}: {replaced:.0%} of tokens replaced")

    # ------------------------------------------------------------------
    # Step 3: Hybrid attention — causal for clean, bidirectional for noisy.
    # ------------------------------------------------------------------
    # The DiDA attention mask lets every *noisy* token attend to every other
    # noisy token in the same image (bidirectional), while tokens keep
    # causal attention to everything before them. ``create_dida_mask``
    # returns a boolean (seq_len, seq_len) mask; True means "may attend".
    # We print a tiny 6-token example: 2 clean prefix tokens + 4 noisy ones.
    print("\n=== Step 3: hybrid attention mask (rows attend to columns) ===")
    demo_mask = DiDAAttentionMask.create_dida_mask(
        seq_len=6, noisy_start_idx=2, noisy_end_idx=6, device=device
    )
    for row in demo_mask.tolist():
        print(" ".join("1" if cell else "." for cell in row))
    print("rows 0-1 are causal; rows 2-5 additionally see the whole "
          "noisy block")

    # ------------------------------------------------------------------
    # Step 4: The full generate path in one call.
    # ------------------------------------------------------------------
    # ``generate_image`` runs the complete reverse process: start from pure
    # noise, then repeatedly (a) build the hybrid mask, (b) predict clean
    # tokens with ``DiDACore``, and (c) re-sample — for ``num_steps``
    # iterations. The returned tensor holds the final token IDs.
    print("\n=== Step 4: one-call generation via DiDAInference ===")
    image_tokens = inference.generate_image(
        batch_size=1,
        image_size=image_size,
        patch_size=patch_size,
        device=device,
        temperature=1.0,
    )
    print(f"generated token tensor shape: {tuple(image_tokens.shape)}")
    print("token grid:")
    print(render_grid(image_tokens, grid_width))
    print("glyph rendering (token_id % 10 -> brightness):")
    print(render_glyphs(image_tokens, grid_width))

    # ------------------------------------------------------------------
    # Step 5: The same path opened up, one denoising step at a time.
    # ------------------------------------------------------------------
    # To make the iterative refinement visible, we replay the generate loop
    # manually with the same public pieces ``DiDASampler.sample`` uses:
    # ``scheduler.get_timesteps``, ``DiDAAttentionMask.create_dida_mask``,
    # and ``model.denoise_step``. At each step we print the mean per-token
    # confidence (mean max softmax probability). Confidence-unmasking
    # schedulers — see docs/EXTENDED_INTRODUCTION.md — use exactly this
    # signal to commit the most confident positions first; this educational
    # sampler simply re-samples every position at every step.
    print("\n=== Step 5: step-by-step denoising with confidence trace ===")
    num_tokens = grid_width ** 2
    tokens = torch.randint(0, vocab_size, (1, num_tokens), device=device)
    for t in scheduler.get_timesteps(reverse=True).tolist():
        # All tokens are noisy during generation, so the mask is the full
        # bidirectional block over the grid.
        attention_mask = DiDAAttentionMask.create_dida_mask(
            seq_len=num_tokens,
            noisy_start_idx=0,
            noisy_end_idx=num_tokens,
            device=device,
        )
        with torch.no_grad():
            logits = model(tokens, torch.tensor(t, device=device), attention_mask)
        confidence = logits.softmax(dim=-1).max(dim=-1).values.mean().item()
        tokens = model.denoise_step(
            tokens, timestep=t, attention_mask=attention_mask, temperature=1.0
        )
        print(f"step t={t}: mean confidence {confidence:.3f}")
    print("grid after the manual loop:")
    print(render_glyphs(tokens, grid_width))

    print("\nWalkthrough finished: full generate path ran without errors.")


if __name__ == "__main__":
    main()
