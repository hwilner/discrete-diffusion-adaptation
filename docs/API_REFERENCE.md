# DiDA API Reference

This document is the reference for the five public classes exported from
`dida/__init__.py`:

- `DiDACore`, `DiscreteDiffusionScheduler`, `DiDAAttentionMask` (from `dida/core.py`)
- `DiDASampler`, `DiDAInference` (from `dida/sampler.py`)

It covers signatures, parameters, and input/output shapes, with a minimal
runnable snippet per class. Every snippet below runs on CPU exactly as
written. For the conceptual deep dive (the discrete diffusion process, the
hybrid attention rationale, and the sampling pipeline), see
[ARCHITECTURE.md](ARCHITECTURE.md); for a zero-background introduction, see
[EXTENDED_INTRODUCTION.md](EXTENDED_INTRODUCTION.md). Only the public API is
documented here — internal helpers are out of scope.

All models in the snippets are small and randomly initialized; the snippets
demonstrate shapes and call patterns, not trained sample quality.

---

## `DiDACore`

Core DiDA denoising module. A transformer that reads noisy token indices plus
the current diffusion timestep and predicts logits over clean tokens, using
the hybrid attention mask described in
[ARCHITECTURE.md §3](ARCHITECTURE.md#3-dida-core-model).

**Module:** `dida/core.py`

### Constructor

```python
DiDACore(
    vocab_size: int,
    hidden_dim: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    dropout: float = 0.1
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | *(required)* | Size of the token vocabulary. |
| `hidden_dim` | `int` | `768` | Hidden dimension size. |
| `num_heads` | `int` | `12` | Number of attention heads. |
| `num_layers` | `int` | `12` | Number of transformer layers. |
| `dropout` | `float` | `0.1` | Dropout probability. |

Attributes of the same names (`vocab_size`, `hidden_dim`, `num_heads`,
`num_layers`) are stored on the instance.

### `forward`

```python
forward(
    noisy_tokens: torch.Tensor,
    timestep: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor
```

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `noisy_tokens` | `torch.Tensor` | `(batch_size, seq_len)` | Noisy token indices. |
| `timestep` | `torch.Tensor` | `(batch_size,)` or scalar | Current timestep. |
| `attention_mask` | `Optional[torch.Tensor]` | `(seq_len, seq_len)` | Optional boolean mask; `True` = may attend. |

**Returns:** `torch.Tensor` of shape `(batch_size, seq_len, vocab_size)` —
logits for the predicted clean tokens.

### `denoise_step`

```python
denoise_step(
    noisy_tokens: torch.Tensor,
    timestep: int,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `noisy_tokens` | `torch.Tensor` | *(required)* | Noisy token indices, `(batch_size, seq_len)`. |
| `timestep` | `int` | *(required)* | Current timestep. |
| `attention_mask` | `Optional[torch.Tensor]` | `None` | Optional attention mask, `(seq_len, seq_len)`. |
| `temperature` | `float` | `1.0` | Sampling temperature; `0` selects greedy argmax decoding. |

**Returns:** `torch.Tensor` of shape `(batch_size, seq_len)` — denoised token
indices, sampled from the softmax of the logits scaled by `temperature`.

### Snippet

```python
import torch
from dida import DiDACore

model = DiDACore(vocab_size=512, hidden_dim=128, num_heads=4, num_layers=2)
noisy = torch.randint(0, 512, (1, 16))

logits = model(noisy, torch.tensor(3))
print(logits.shape)  # torch.Size([1, 16, 512])

denoised = model.denoise_step(noisy, timestep=3, temperature=1.0)
print(denoised.shape)  # torch.Size([1, 16])
```

---

## `DiscreteDiffusionScheduler`

Noise schedule for the discrete diffusion process: owns the linear beta
schedule and its derived quantities, adds noise to clean tokens in the
forward direction, and hands out the timestep order for denoising. See
[ARCHITECTURE.md §2](ARCHITECTURE.md#2-discrete-diffusion-process).

**Module:** `dida/core.py`

### Constructor

```python
DiscreteDiffusionScheduler(
    num_steps: int = 50,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_steps` | `int` | `50` | Number of denoising steps. |
| `beta_start` | `float` | `0.0001` | Initial noise level. |
| `beta_end` | `float` | `0.02` | Final noise level. |

### Derived attributes

| Attribute | Shape | Description |
|---|---|---|
| `betas` | `(num_steps,)` | Linear schedule from `beta_start` to `beta_end`. |
| `alphas` | `(num_steps,)` | `1.0 - betas`. |
| `alphas_cumprod` | `(num_steps,)` | Cumulative product of `alphas`; entry `t` is the probability a token survives un-replaced at timestep `t`. |

### `add_noise`

```python
add_noise(
    clean_tokens: torch.Tensor,
    timestep: int,
    vocab_size: int
) -> torch.Tensor
```

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `clean_tokens` | `torch.Tensor` | `(batch_size, seq_len)` | Clean token indices. |
| `timestep` | `int` | — | Current diffusion timestep. |
| `vocab_size` | `int` | — | Size of the token vocabulary. |

**Returns:** `torch.Tensor` of shape `(batch_size, seq_len)` — noisy token
indices. Each position keeps its clean token with probability
`alphas_cumprod[timestep]` and is otherwise replaced by a uniform vocabulary
draw.

### `get_timesteps`

```python
get_timesteps(reverse: bool = True) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reverse` | `bool` | `True` | `True` returns timesteps in reverse (denoising) order; `False` in forward (noise-adding) order. |

**Returns:** `torch.Tensor` of shape `(num_steps,)` — timestep indices.

### Snippet

```python
import torch
from dida import DiscreteDiffusionScheduler

torch.manual_seed(0)
scheduler = DiscreteDiffusionScheduler(num_steps=8)

clean = torch.arange(8).unsqueeze(0) % 512
noisy = scheduler.add_noise(clean, timestep=4, vocab_size=512)
print(noisy.shape)  # torch.Size([1, 8])

print(scheduler.get_timesteps(reverse=True).tolist())  # [7, 6, 5, 4, 3, 2, 1, 0]
```

---

## `DiDAAttentionMask`

Attention mask generator for the hybrid DiDA attention pattern: noisy tokens
attend bidirectionally to other noisy tokens in the same image and causally
to preceding clean tokens; clean tokens follow a standard causal pattern. See
[ARCHITECTURE.md §4](ARCHITECTURE.md#4-hybrid-attention-mechanism).

**Module:** `dida/core.py`

The class is used through its static methods; no instance state is needed.

### `create_dida_mask` (static)

```python
DiDAAttentionMask.create_dida_mask(
    seq_len: int,
    noisy_start_idx: int,
    noisy_end_idx: int,
    device: torch.device = None
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seq_len` | `int` | *(required)* | Total sequence length. |
| `noisy_start_idx` | `int` | *(required)* | Start index of the noisy tokens. |
| `noisy_end_idx` | `int` | *(required)* | End index of the noisy tokens (exclusive). |
| `device` | `torch.device` | `None` | Device to create the mask on; defaults to CPU. |

**Returns:** boolean `torch.Tensor` of shape `(seq_len, seq_len)`; entry
`(i, j)` is `True` when position `i` may attend to position `j`.

### `create_interleaved_mask` (static)

```python
DiDAAttentionMask.create_interleaved_mask(
    text_positions: torch.Tensor,
    image_positions: torch.Tensor,
    noisy_positions: torch.Tensor,
    seq_len: int,
    device: torch.device = None
) -> torch.Tensor
```

| Parameter | Type | Description |
|---|---|---|
| `text_positions` | `torch.Tensor` | Boolean tensor marking text token positions. |
| `image_positions` | `torch.Tensor` | Boolean tensor marking image token positions. |
| `noisy_positions` | `torch.Tensor` | Boolean tensor marking noisy token positions. |
| `seq_len` | `int` | Total sequence length. |
| `device` | `torch.device` | Device for the mask; defaults to the device of `text_positions`. |

**Returns:** boolean `torch.Tensor` of shape `(seq_len, seq_len)` — a causal
base mask with the noisy positions opened up to attend bidirectionally to one
another.

### Snippet

```python
import torch
from dida import DiDAAttentionMask

# Two clean prefix tokens (0-1) followed by four noisy tokens (2-5).
mask = DiDAAttentionMask.create_dida_mask(seq_len=6, noisy_start_idx=2, noisy_end_idx=6)
print(mask.shape)       # torch.Size([6, 6])
print(mask.dtype)       # torch.bool
print(int(mask[2, 5]))  # 1 — a noisy token attends to a later noisy token
print(int(mask[0, 5]))  # 0 — a clean prefix token cannot look ahead

# The interleaved variant marks positions explicitly.
text_pos = torch.tensor([True, True, False, False])
image_pos = ~text_pos
noisy_pos = image_pos.clone()
interleaved = DiDAAttentionMask.create_interleaved_mask(
    text_pos, image_pos, noisy_pos, seq_len=4
)
print(interleaved.shape)  # torch.Size([4, 4])
```

---

## `DiDASampler`

Sampling interface for DiDA-based generation: initializes the fully noisy
token sequence and iterates the denoising loop for pure image generation,
text-conditioned generation, and classifier-free guidance. See
[ARCHITECTURE.md §5](ARCHITECTURE.md#5-sampling-and-inference).

**Module:** `dida/sampler.py`

### Constructor

```python
DiDASampler(
    model: DiDACore,
    scheduler: DiscreteDiffusionScheduler,
    vocab_size: int
)
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `DiDACore` | DiDA core model for denoising. |
| `scheduler` | `DiscreteDiffusionScheduler` | Diffusion scheduler. |
| `vocab_size` | `int` | Size of the token vocabulary. |

### `sample`

```python
sample(
    batch_size: int,
    num_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    guidance_scale: float = 1.0
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | *(required)* | Number of samples to generate. |
| `num_tokens` | `int` | *(required)* | Number of tokens to generate per sample. |
| `device` | `torch.device` | *(required)* | Device to generate on. |
| `temperature` | `float` | `1.0` | Sampling temperature. |
| `guidance_scale` | `float` | `1.0` | Classifier-free guidance scale (accepted; guidance is applied by `sample_with_guidance`). |

**Returns:** `torch.Tensor` of shape `(batch_size, num_tokens)` — generated
token indices. Runs `num_steps` denoising iterations under `torch.no_grad()`,
starting from uniform random tokens.

### `sample_interleaved`

```python
sample_interleaved(
    text_tokens: torch.Tensor,
    num_image_tokens: int,
    device: torch.device,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]
```

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `text_tokens` | `torch.Tensor` | `(batch_size, text_len)` | Conditioning text tokens; kept fixed throughout. |
| `num_image_tokens` | `int` | — | Number of image tokens to generate. |
| `device` | `torch.device` | — | Device to generate on. |
| `temperature` | `float` | — | Sampling temperature for image generation (default `1.0`). |

**Returns:** a tuple `(full_sequence, image_tokens)` where `full_sequence`
has shape `(batch_size, text_len + num_image_tokens)` (text concatenated with
the generated image tokens) and `image_tokens` has shape
`(batch_size, num_image_tokens)` (the generated image tokens only). Only the
image positions are denoised; the text prefix is re-asserted unchanged after
every step.

### `sample_with_guidance`

```python
sample_with_guidance(
    batch_size: int,
    num_tokens: int,
    condition: Optional[torch.Tensor],
    device: torch.device,
    guidance_scale: float = 7.5,
    temperature: float = 1.0
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | *(required)* | Number of samples to generate. |
| `num_tokens` | `int` | *(required)* | Number of tokens per sample. |
| `condition` | `Optional[torch.Tensor]` | *(required)* | Optional conditioning tokens. |
| `device` | `torch.device` | *(required)* | Device to generate on. |
| `guidance_scale` | `float` | `7.5` | Strength of guidance (`1.0` = no guidance). |
| `temperature` | `float` | `1.0` | Sampling temperature. |

**Returns:** `torch.Tensor` of shape `(batch_size, num_tokens)` — generated
token indices. When `guidance_scale != 1.0` and `condition` is not `None`,
each step combines conditional and unconditional logits as
`uncond + guidance_scale * (cond - uncond)` before sampling.

### Snippet

```python
import torch
from dida import DiDACore, DiscreteDiffusionScheduler, DiDASampler

torch.manual_seed(0)
model = DiDACore(vocab_size=512, hidden_dim=128, num_heads=4, num_layers=2)
scheduler = DiscreteDiffusionScheduler(num_steps=4)
sampler = DiDASampler(model, scheduler, vocab_size=512)

device = torch.device("cpu")
tokens = sampler.sample(batch_size=1, num_tokens=16, device=device)
print(tokens.shape)  # torch.Size([1, 16])

text = torch.randint(0, 512, (1, 5))
full, image = sampler.sample_interleaved(text, num_image_tokens=8, device=device)
print(full.shape)   # torch.Size([1, 13])
print(image.shape)  # torch.Size([1, 8])
```

---

## `DiDAInference`

High-level inference interface: wires a `DiDACore` model and a
`DiscreteDiffusionScheduler` into an internal `DiDASampler` and exposes the
two common generation scenarios as one call each. See
[ARCHITECTURE.md §5](ARCHITECTURE.md#5-sampling-and-inference).

**Module:** `dida/sampler.py`

### Constructor

```python
DiDAInference(
    model: DiDACore,
    scheduler: DiscreteDiffusionScheduler,
    vocab_size: int,
    text_vocab_size: int,
    image_vocab_size: int
)
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `DiDACore` | DiDA core model. |
| `scheduler` | `DiscreteDiffusionScheduler` | Diffusion scheduler. |
| `vocab_size` | `int` | Total vocabulary size. |
| `text_vocab_size` | `int` | Size of the text vocabulary. |
| `image_vocab_size` | `int` | Size of the image vocabulary. |

The constructor builds an internal `DiDASampler(model, scheduler, vocab_size)`,
stored as `self.sampler`.

### `generate_image`

```python
generate_image(
    batch_size: int = 1,
    image_size: int = 256,
    patch_size: int = 16,
    device: torch.device = None,
    temperature: float = 1.0
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | `1` | Number of images to generate. |
| `image_size` | `int` | `256` | Size of the image in pixels (square). |
| `patch_size` | `int` | `16` | Size of each patch in pixels. |
| `device` | `torch.device` | `None` | Device to generate on; defaults to CUDA when available, else CPU. |
| `temperature` | `float` | `1.0` | Sampling temperature. |

**Returns:** `torch.Tensor` of shape
`(batch_size, (image_size // patch_size) ** 2)` — generated image tokens; the
token count is the number of patches in the image grid.

### `text_to_image`

```python
text_to_image(
    text_tokens: torch.Tensor,
    image_size: int = 256,
    patch_size: int = 16,
    temperature: float = 1.0
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_tokens` | `torch.Tensor` | *(required)* | Text prompt tokens, `(batch_size, text_len)`; the device is taken from this tensor. |
| `image_size` | `int` | `256` | Size of the image to generate. |
| `patch_size` | `int` | `16` | Patch size for tokenization. |
| `temperature` | `float` | `1.0` | Sampling temperature. |

**Returns:** `torch.Tensor` of shape
`(batch_size, (image_size // patch_size) ** 2)` — image tokens generated
conditioned on the text prompt (via `DiDASampler.sample_interleaved`).

### Snippet

```python
import torch
from dida import DiDACore, DiscreteDiffusionScheduler, DiDAInference

torch.manual_seed(0)
model = DiDACore(vocab_size=512, hidden_dim=128, num_heads=4, num_layers=2)
scheduler = DiscreteDiffusionScheduler(num_steps=4)
inference = DiDAInference(
    model=model,
    scheduler=scheduler,
    vocab_size=512,
    text_vocab_size=512,
    image_vocab_size=512,
)

# image_size=32 with patch_size=8 gives a 4 x 4 = 16-token grid.
image_tokens = inference.generate_image(
    batch_size=1, image_size=32, patch_size=8, device=torch.device("cpu")
)
print(image_tokens.shape)  # torch.Size([1, 16])

text_prompt = torch.randint(0, 512, (1, 5))
conditioned = inference.text_to_image(text_prompt, image_size=32, patch_size=8)
print(conditioned.shape)  # torch.Size([1, 16])
```

---

*Scope note: this reference covers the five public classes listed in
`dida/__init__.py`'s export documentation at the time of writing. For method
behavioral detail beyond signatures and shapes, the docstrings in
`dida/core.py` and `dida/sampler.py` are the source of truth.*
