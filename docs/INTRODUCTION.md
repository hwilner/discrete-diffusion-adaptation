# Introduction: Why Discrete Diffusion Adaptation?

This document is a ground-up introduction for a technical newcomer — someone
comfortable with PyTorch and transformers, but new to discrete diffusion and to
DiDA (Discrete Diffusion Adaptation). It explains the problem DiDA solves, the
lineage of ideas it builds on, the surrounding research landscape, and the
honest limits of this educational implementation.

![Concept figure: DiDA decoding — a fully masked grid of image tokens is refined through hybrid attention (bidirectional among noisy tokens, causal over the clean text prefix) and confidence-based parallel unmasking, repeating for a few steps to produce the final image tokens.](figures/concept_figure.svg)

*Figure 1. The DiDA decoding loop: start from all-`[MASK]` image tokens, predict all positions in parallel under a hybrid attention mask, commit the highest-confidence tokens, re-mask the rest, and repeat a handful of times instead of decoding left to right.*

## 1. Why autoregressive decoding is slow for images

The dominant recipe for tokenized image generation is autoregressive (AR)
decoding, as in Emu3 [2]: quantize an image into a grid of discrete tokens,
flatten them, and train a transformer to predict the next token given all
previous ones. At inference time, the model generates tokens **one at a time,
left to right**. A 16×16 token grid needs 256 sequential forward passes; a
32×32 grid needs 1,024.

Two things make this painful:

- **Latency scales linearly with image size.** Each forward pass must wait for
  the previous one, so wall-clock time grows with token count even though the
  passes themselves are cheap.
- **The order is artificial.** Pixels have no inherent left-to-right ordering.
  The model is forced into a causal factorization for training convenience,
  not because images are sequential.

The natural question: can we generate many tokens **in parallel** without
sacrificing quality?

## 2. Discrete diffusion foundations

Discrete diffusion answers that question by replacing "generate left to right"
with "start from noise, denoise everything at once, gradually."

- **D3PM** (Austin et al., 2021) [3] formalized diffusion over discrete
  states: a forward process progressively corrupts tokens (e.g., replacing
  them with a special `[MASK]` token), and a learned reverse process predicts
  the clean tokens. The *absorbing-state* variant — where corruption means
  masking — is the direct ancestor of masked-diffusion decoding.
- **MaskGIT** (Chang et al., 2022) [4] showed that masked-token models can
  generate images in a handful of parallel steps: predict all masked
  positions, keep the highest-confidence predictions, re-mask the rest, and
  repeat. Confidence-based unmasking is DiDA's conceptual template.
- **SEDD** (Lou, Meng, Ermon, 2024) [5] and **MDLM** (Sahoo et al., 2024) [6]
  pushed masked diffusion for text to the point of beating GPT-2-scale AR
  language models, with clean training objectives (score entropy; simple
  masked-diffusion recipes).
- **Diffusion-LM** (Li et al., 2022) [7] explored the alternative route of
  diffusing in continuous embedding space rather than over discrete tokens.
- **LLaDA** (Nie et al., 2025) [8] scaled masked diffusion language models
  from scratch to 8B parameters, showing the paradigm is not limited to small
  models.

The common thread: parallel denoising trades a *small number of full-sequence
passes* for the *many sequential passes* of AR decoding.

## 3. What DiDA does

Training a masked-diffusion model from scratch is expensive. **DiDA**
(Cui, Chen, Deng et al., Emu3.5, 2025) [1] takes a shortcut: start from a
**pretrained AR multimodal model** and adapt it. Three ingredients:

1. **Mask surgery.** Repurpose the AR model's bidirectional attention capacity
   for denoising: feed it a sequence where image tokens are progressively
   masked, and train it (briefly, via self-distillation) to predict the masked
   tokens all at once.
2. **Hybrid attention masks.** Noisy image tokens attend **bidirectionally**
   (each noisy token sees all others, as in MaskGIT), while clean prefix
   tokens — text prompts, already-committed image tokens — keep their original
   **causal** attention. This preserves the AR model's text abilities while
   enabling parallel image decoding.
3. **Self-distillation.** The adapted model is trained to match its own AR
   outputs, so quality is preserved while decoding becomes parallel.

The result reported in Emu3.5: roughly **20× faster per-image inference at
unchanged quality**, with text generation untouched. This repository
re-implements those mechanics from scratch on synthetic image-token grids.

## 4. The AR→diffusion adaptation landscape for text and code

DiDA is part of a broader wave of *adapting* pretrained AR models into
diffusion decoders rather than training diffusers from scratch:

- **DiffuLLaMA** (Gong et al., 2024) [10] gives a recipe for converting AR
  LLMs into masked diffusion LMs.
- **Dream 7B** (Ye et al., 2025) [9] initializes a discrete diffusion LM from
  an AR model, adding context-adaptive, token-level noise rescheduling.
- **Dream-Coder 7B** (2025) [14] applies the same adaptation idea to code
  generation.
- **DiffSound** (Yang et al., 2023) [15] shows masked discrete diffusion over
  audio tokens — though trained from scratch, not adapted.

Inference-time improvements are also active: **Fast-dLLM** (Wu et al., 2025)
[11] adds confidence-thresholded parallel unmasking and KV caching without
training; **ReMDM** (Wang et al., 2025) [12] derives principled remasking
samplers with inference-time scaling.

## 5. Speculative decoding: the competing paradigm

The main rival for accelerating AR inference is **speculative decoding**
(Leviathan, Kalman, Matias, 2023) [13]: a small "draft" model proposes several
tokens, the big model verifies them in one parallel pass, and only verified
tokens are kept. It yields exact-output 2–3× speedups — but it still generates
left to right and requires a second model. Discrete diffusion instead changes
the *decoding structure itself*, and reported speedups are larger (~20×) for
image tokens. The paradigms are not mutually exclusive, but they make
different bets: speculation preserves AR semantics exactly; diffusion rewrites
them and must prove quality is retained.

## 6. Honest limitations of this implementation

This repository is **educational**, and it is important to be clear about what
it is and is not:

- **Synthetic data only.** Models are tiny transformers trained on procedural
  image-token patterns. No pretrained AR model, no real images, no real
  quality claims.
- **No speedup measurements against a real baseline.** The ~20× figure is from
  the Emu3.5 paper, not from this codebase. Measuring forward-pass counts on
  synthetic tasks is future work (see `docs/ROADMAP.md`).
- **Scale.** DiDACore is a small transformer (default: 4 layers, hidden 256).
  Nothing here validates behavior at Emu3.5's scale.
- **Text is out of scope for claims.** Adapting AR text models to diffusion
  already exists in the literature [9, 10, 14]; any text-leg work here is
  educational reproduction, not a novelty claim.
- **18 passing tests verify mechanics** (masks, scheduler, sampling shapes and
  determinism) — not image quality.

What the repo *does* offer: a clean, tested, from-scratch reference for the
core DiDA mechanics — the diffusion scheduler, the hybrid attention mask, and
the sampling loop — that you can read in an afternoon and extend.

## References

1. Cui, Y., Chen, H., Deng, H., et al. (BAAI), 2025. *Emu3.5: Native Multimodal Models are World Learners*. arXiv:2510.26583.
2. Wang, X., et al. (BAAI), 2024. *Emu3: Next-Token Prediction is All You Need*. arXiv:2409.18869.
3. Austin, J., et al., NeurIPS 2021. *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM). arXiv:2107.03006.
4. Chang, H., et al., CVPR 2022. *MaskGIT: Masked Generative Image Transformer*. arXiv:2202.04200.
5. Lou, A., Meng, C., Ermon, S., ICML 2024. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD). arXiv:2310.16834.
6. Sahoo, S., et al., NeurIPS 2024. *Simple and Effective Masked Diffusion Language Models* (MDLM). arXiv:2406.07524.
7. Li, X. L., et al., NeurIPS 2022. *Diffusion-LM Improves Controllable Text Generation*. arXiv:2205.14217.
8. Nie, S., et al., 2025. *Large Language Diffusion Models* (LLaDA). arXiv:2502.09992.
9. Ye, J., et al., 2025. *Dream 7B*. arXiv:2508.15487.
10. Gong, S., et al., 2024. *Scaling Diffusion Language Models via Adaptation from Autoregressive Models* (DiffuLLaMA). arXiv:2410.17891.
11. Wu, C., et al., 2025. *Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding*. arXiv:2505.22618.
12. Wang, J., Schiff, Y., Sahoo, S., Kuleshov, V., 2025. *Remasking Discrete Diffusion Models with Inference-Time Scaling* (ReMDM). arXiv:2503.00307.
13. Leviathan, Y., Kalman, M., Matias, Y., ICML 2023. *Fast Inference from Transformers via Speculative Decoding*. arXiv:2211.17192.
14. Dream-Coder 7B, 2025. arXiv:2509.01142.
15. Yang, D., et al., IEEE/ACM TASLP 2023. *DiffSound: Discrete Diffusion Model for Text-to-Sound Generation*. arXiv:2207.09983.
