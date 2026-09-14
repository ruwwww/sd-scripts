# Anima low-memory LoRA recipes

This guide documents the opt-in memory and speed paths for
`anima_train_network.py`. The paths target the activation memory of the DiT
MLP while keeping the LoRA parameters, gradients, optimizer, residual stream,
and attention in their normal training dtypes.

These options are intentionally opt-in. Start with the exact BF16 recipe,
then compare the FP8 recipe on a short fixed-seed run before using it for a
long training job.

## What is being optimized

For a frozen-base LoRA MLP,

```text
X -> Linear(base + LoRA) -> GELU -> Linear(base + LoRA)
```

the custom explicit-VJP path avoids retaining the usual collection of
autograd intermediates. It saves only the tensors needed to produce the
same parameter and input vector-Jacobian products. The FP8 mode additionally
stores the wide pre-GELU state as row-wise E4M3 FP8 with one FP16 scale per
row. Backward dequantizes it and computes the GELU derivative in a fused
Triton kernel when available.

This is activation *storage* compression. It is not a request to run every
GEMM in FP8, and it does not quantize Prodigy or Adam states.

The implementation currently targets dropout-free `GPT2FeedForward` MLPs
where both projections have LoRA modules. Other MLP layouts and LoRA dropout
fall back to the normal path.

## Recipe 1: exact BF16 and lowest numerical risk

Use this first to validate that the custom explicit-VJP path has the same
training behavior as the ordinary path. `--fused_mlp_storage=bf16` does not
quantize the saved MLP state; it only removes avoidable autograd intermediates.

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --pretrained_model_name_or_path="/path/to/anima.safetensors" \
  --qwen3="/path/to/Qwen3-0.6B" \
  --vae="/path/to/qwen-image-vae.safetensors" \
  --dataset_config="/path/to/anima.toml" \
  --output_dir="/path/to/output" \
  --output_name="anima_lora_bf16_fused" \
  --network_module=networks.lora_anima \
  --network_dim=16 \
  --learning_rate=1e-4 \
  --optimizer_type="Prodigy" \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --selective_checkpointing=count \
  --checkpoint_blocks=12 \
  --fused_lora \
  --fused_mlp \
  --fused_mlp_storage=bf16 \
  --cache_latents \
  --cache_text_encoder_outputs
```

Use the same seed, dataset order, initial LoRA weights, and optimizer settings
when comparing this recipe with the ordinary implementation. A small floating
point difference is expected from GEMM ordering, but there should be no loss
of finite values or systematic divergence.

## Recipe 2: recommended Triton FP8 activation cache

This is the primary speed/memory candidate when Triton is available. It keeps
the compute path in BF16 and compresses only the wide pre-GELU state retained
for backward. `auto` selects the fused Triton pack/dequantize kernels and
falls back to the eager implementation if the runtime cannot load Triton.

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --pretrained_model_name_or_path="/path/to/anima.safetensors" \
  --qwen3="/path/to/Qwen3-0.6B" \
  --vae="/path/to/qwen-image-vae.safetensors" \
  --dataset_config="/path/to/anima.toml" \
  --output_dir="/path/to/output" \
  --output_name="anima_lora_fp8_cache" \
  --network_module=networks.lora_anima \
  --network_dim=16 \
  --learning_rate=1e-4 \
  --optimizer_type="Prodigy" \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --selective_checkpointing=count \
  --checkpoint_blocks=8 \
  --fused_lora \
  --fused_mlp \
  --fused_mlp_storage=fp8 \
  --fused_mlp_fp8_backend=auto \
  --cache_latents \
  --cache_text_encoder_outputs
```

On the tested 28-block Anima target (batch size 1, 3952 tokens, hidden size
2048, BF16, LoRA rank 16, RTX 5060 Ti), this configuration reached a stable
~1.58 s/step with ~13.0 GiB allocated and ~3.1 GiB reported headroom. The
exact values depend on sequence length, resolution, CUDA version, and the
other training options. If the available VRAM is tighter, keep the same FP8
cache but increase `--checkpoint_blocks` to 10 or 12.

For a more conservative memory comparison, use 12 checkpointed blocks first:

```text
--selective_checkpointing=count --checkpoint_blocks=12
```

The tested 12-block FP8 configuration used ~11.4 GiB allocated and had a
gradient cosine of 0.999995 against the exact BF16 fused path, with relative
gradient L2 error around 0.30%. Treat those numbers as a reference, not a
guarantee for every GPU or model shape.

### Optional R&D: store the MLP input `X` as FP8

The custom MLP VJP also saves its input `X` for the first LoRA down-gradient.
The following additional flag stores that private saved copy as row-wise FP8
and dequantizes it only when computing `dLoRA_down`:

```bash
--fused_mlp_storage=fp8 \
--fused_mlp_fp8_backend=auto \
--fused_mlp_fp8_input
```

This does not change the forward output and does not compress the residual
stream globally. It is an experimental memory extension. On the tested target
with 12 checkpointed blocks it reduced peak allocation from about 11,391 MiB
to 11,281 MiB, with step time changing from about 1,657 ms to 1,660 ms. With
8 checkpointed blocks it reduced peak allocation by about 133 MiB and remained
within the same timing noise. Validate LoRA gradient and image drift on the
target dataset before enabling it for a long run; the additional quantization
only affects the LoRA input-gradient path, but it is still a reduced-precision
backward.

### Optional R&D: direct backward from the FP8 cache

The Triton backend also has an experimental
`--fused_mlp_fp8_direct_backward` path. It computes the LoRA `dDown2` product
directly from the row-wise FP8 pre-GELU cache and writes only the GELU input
gradient. This avoids allocating the full BF16 GELU activation during
backward; it does not use recomputation and it does not change the forward
GEMMs. It requires `--fused_mlp_storage=fp8` and the Triton backend.

```bash
--fused_mlp_storage=fp8 \
--fused_mlp_fp8_backend=triton \
--fused_mlp_fp8_direct_backward
```

On the same target, after 3 warmup steps and 5 measured steps, count 8 was
stable at about 1,566 ms/step and 12,944 MiB peak allocation, with 3.16 GiB
headroom to the benchmark budget. Adding `--fused_mlp_fp8_input` reduced the
peak to about 12,810 MiB and measured 1,567 ms/step, but also quantizes the
private input used by the LoRA `dDown1` gradient. Keep the input flag off for
the conservative recipe; use it only after a fixed-seed trajectory and image
comparison.

A 10-step soak (after 5 warmup steps) measured 1,567 ms/step with unchanged
allocated/reserved memory, finite gradients, and no allocator growth.

### Recipe 2b: fastest safe path with per-block `torch.compile`

Anima already supports per-block `torch.compile`. On the same target, combining
it with the direct Triton FP8 path makes checkpointing much cheaper: compiled
count 1 is faster than the uncompiled count 8 recipe while retaining a usable
memory margin.

```bash
--selective_checkpointing=count \
--checkpoint_blocks=1 \
--compile \
--compile_mode=default \
--compile_cache_size_limit=32 \
--fused_lora \
--fused_mlp \
--fused_mlp_storage=fp8 \
--fused_mlp_fp8_backend=triton \
--fused_mlp_fp8_direct_backward
```

The first pass through each resolution bucket is slower because Inductor
compiles the block. Keep `--compile_mode=default`; `max-autotune` is not a good
fit for this 36-SM GPU. With a multi-resolution dataset, raise the compile
cache limit and expect one compilation per new shape.

| Compiled checkpoint count | Step ms | Peak allocated | Actual GPU headroom | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1,088 | 13,479 MiB | 2.31 GiB | reject |
| **1** | **1,134** | **12,701 MiB** | **3.07 GiB** | **fastest safe** |
| 4 | 1,223 | 11,829 MiB | 3.93 GiB | conservative |
| 8 | 1,354 | 10,666 MiB | 5.06 GiB | maximum margin |

The count-1 and count-8 compiled runs had maximum loss difference `0.00258`
and maximum relative gradient-norm difference `0.15%` over the 10-step probe.
These are short fixed-seed measurements, so validate a real dataset before
making compile count-1 your unattended default. If the margin is not enough,
use count 4.

## Recipe 3: low-rank activation storage (research mode)

This mode stores a randomized rank-`r` approximation of the pre-GELU state.
It can provide a different memory/speed tradeoff, but it is approximate and
adds projection/QR work during the forward pass. Do not use it as the default
until a fixed-seed training comparison is acceptable for the target dataset.

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --pretrained_model_name_or_path="/path/to/anima.safetensors" \
  --qwen3="/path/to/Qwen3-0.6B" \
  --vae="/path/to/qwen-image-vae.safetensors" \
  --dataset_config="/path/to/anima.toml" \
  --output_dir="/path/to/output" \
  --output_name="anima_lora_lowrank64" \
  --network_module=networks.lora_anima \
  --network_dim=16 \
  --learning_rate=1e-4 \
  --optimizer_type="Prodigy" \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --selective_checkpointing=count \
  --checkpoint_blocks=7 \
  --fused_lora \
  --fused_mlp \
  --fused_mlp_storage=lowrank \
  --fused_mlp_rank=64 \
  --cache_latents \
  --cache_text_encoder_outputs
```

The tested rank-64 path had gradient cosine ~0.9996 and relative gradient L2
error ~2.8% against exact BF16. Rank 32 was noticeably less faithful in the
same probe, so rank 64 is the starting point. If this changes the training
trajectory or validation images too much, return to Recipe 2 rather than
raising the optimizer precision or changing Prodigy.

## Recipe 4: controlled backend comparison

Use this short diagnostic matrix before a long run. Keep the seed, one fixed
batch, and initial model/LoRA state identical.

```bash
# Exact custom-VJP reference
--fused_lora --fused_mlp --fused_mlp_storage=bf16 \
--selective_checkpointing=count --checkpoint_blocks=12

# FP8 storage with ordinary PyTorch packing; diagnostic only
--fused_lora --fused_mlp --fused_mlp_storage=fp8 \
--fused_mlp_fp8_backend=eager \
--selective_checkpointing=count --checkpoint_blocks=12

# FP8 storage with the fused Triton kernels
--fused_lora --fused_mlp --fused_mlp_storage=fp8 \
--fused_mlp_fp8_backend=triton \
--selective_checkpointing=count --checkpoint_blocks=12
```

The eager FP8 path is useful for correctness isolation, but can be slower
because it launches separate reductions, casts, scale operations, and
dequantization kernels. Triton helps by fusing those operations. If the
explicit `triton` backend fails on a particular GPU, use `auto` and record
which backend was selected in the log.

## Choosing the checkpoint count

Checkpointing and activation compression solve different parts of the same
memory budget:

| Goal | Starting point |
| --- | --- |
| Match ordinary training dynamics | BF16 fused, 12 blocks |
| More headroom | FP8 Triton, 12 blocks |
| Best tested safe throughput with enough VRAM | FP8 Triton direct backward + `torch.compile`, 1 block |
| More conservative compiled recipe | FP8 Triton direct backward + `torch.compile`, 4 blocks |
| Research tradeoff for extra compression | Low-rank rank 64, 7 blocks |
| OOM after enabling a recipe | Keep the recipe and add 2–4 checkpointed blocks |

The direct-backward checkpoint sweep on the target was:

| Checkpoint blocks | Step ms | Peak allocated | Budget headroom | Decision |
| ---: | ---: | ---: | ---: | --- |
| 6 | 1,528 | 13,751 MiB | 2.37 GiB | too little margin |
| 7 + FP8 input | 1,547 | 13,194 MiB | 2.91 GiB | aggressive, not default |
| 8 | 1,566 | 12,944 MiB | 3.16 GiB | recommended |
| 8 + FP8 input | 1,567 | 12,810 MiB | 3.29 GiB | conservative memory extension |

The safety gate used here is at least 3 GiB headroom to a 15.8 GiB training
budget. Count 7 is likely to run on this exact setup, but the margin is too
small to call it robust against allocator variation, longer sequences, or
other training options.

For a 28-block model, `--checkpoint_blocks=N` selects approximately `N`
evenly spaced blocks. It implies gradient checkpointing. `--selective_checkpointing=full`
is the compatibility mode equivalent to checkpointing all eligible blocks;
use an explicit count when tuning the memory/speed curve.

## Validation gates before a long run

For each candidate, validate all of the following:

1. The first 10–20 steps remain finite.
2. Loss and gradient norms are in the same order as the BF16 reference.
3. No steady-state increase in allocated or reserved CUDA memory occurs.
4. The candidate fits with a safety margin, not merely one successful step.
5. A fixed-seed short trajectory and validation image are acceptable.

For FP8 storage, a practical initial gate is gradient cosine >= 0.9999 and
relative gradient L2 error <= 1%. The low-rank path is expected to need a
looser, task-specific gate because it is intentionally approximate.

## Rollback and unsupported combinations

Remove `--fused_lora`, `--fused_mlp`, `--selective_checkpointing`, and
`--checkpoint_blocks` to return to the ordinary training path. Keep
`--mixed_precision=bf16`; these features do not require CPU offload and do not
change Prodigy state representation.

The custom MLP path currently skips modules with LoRA dropout, rank dropout,
module dropout, or biased projections. It also does not replace attention.
Do not replace every `nn.Linear` blindly: the explicit VJP relies on the
frozen-base LoRA topology and the module names used by Anima LoRA loading.

## Implementation references

* `networks/fused_lora.py` — explicit-VJP LoRA Linear.
* `networks/fused_mlp.py` — BF16, FP8, and low-rank activation storage.
* `networks/fp8_kernels.py` — Triton row-wise pack, unpack, fused GELU backward, and direct dDown2 kernels.
* `library/selective_checkpointing.py` — block checkpoint placement.
* [PyTorch saved-tensor hooks](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)
  — background on activation packing.
* [NVIDIA Transformer Engine low-precision training](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
  — hardware/runtime context for FP8 compute and storage.
