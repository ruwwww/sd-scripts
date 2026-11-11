# Anima local pre-caching: encode VAE latents and Qwen3 text-encoder outputs on the local machine so that
# remote/cloud training can run with --skip_vae --skip_text_encoder (no VAE / Qwen3 download, load, or VRAM).
#
# Usage: pass exactly the same trainer argv as the training job (dataset config, caption options, VAE/Qwen3
# paths), plus the precache options below:
#
#   python anima_precache.py --dataset_config=.../dataset.toml --vae=.../qwen_image_vae.safetensors \
#       --qwen_image_vae_2d --qwen3=.../qwen_3_06b_base.safetensors --shuffle_caption --keep_tokens=1 \
#       --caption_tag_dropout_rate=0.3 --caption_dropout_rate=0.05 --max_train_epochs=14 --te_variants=16
#
# Outputs (next to each image, picked up by the normal sd-scripts caching code paths):
#   <image>_<W>x<H>_anima.npz   latents for every bucket resolution (+flipped latents when flip_aug)
#   <image>_anima_te.npz        Qwen3 hidden states. When shuffle_caption / caption_tag_dropout_rate / wildcards are
#                               active, N independently augmented caption variants are stored ([N, L, D]); the
#                               training dataloader draws one variant at random per sample. Variants are trimmed to
#                               the longest valid sequence and stored as raw bf16 bits; they are re-padded on load
#                               so tensors are identical to online encoding.

import argparse
import copy
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

import library.args as args_util
import library.accelerator_setup as accelerator_setup
from library import anima_train_utils, anima_utils, config_util, strategy_anima, strategy_base
from library.config_util import BlueprintGenerator, ConfigSanitizer
import anima_train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

TE_CACHE_FORMAT = "anima_te_multi_v1"


def _subset_augments_caption(subset) -> bool:
    return bool(subset.shuffle_caption or subset.caption_tag_dropout_rate > 0 or subset.enable_wildcard)


def _build_dataset_groups(args):
    if args.dataset_config is None:
        raise ValueError("anima_precache.py requires --dataset_config (same file as the training job)")
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
    user_config = config_util.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)
    return config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)


def _iter_datasets(train_group, val_group):
    for group in (train_group, val_group):
        if group is None:
            continue
        for ds in group.datasets:
            yield ds


def cache_latents(args, train_group, val_group, accelerator, weight_dtype):
    logger.info("Loading Qwen Image VAE for latent pre-caching...")
    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()
    with torch.no_grad():
        for ds in _iter_datasets(train_group, val_group):
            ds.new_cache_latents(vae, accelerator)
    vae.to("cpu")
    del vae
    clean_memory_on_device(accelerator.device)


def _bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def cache_text_encoder_outputs(args, train_group, val_group, device, weight_dtype):
    tokenize_strategy: strategy_anima.AnimaTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    caching_strategy = strategy_anima.AnimaTextEncoderOutputsCachingStrategy(True, None, False, False)
    if tokenize_strategy.qwen3_tokenizer.padding_side != "right" or tokenize_strategy.t5_tokenizer.padding_side != "right":
        raise RuntimeError("anima_precache.py expects right-padded tokenizers (needed for trimmed variant storage)")

    logger.info("Loading Qwen3 text encoder for TE pre-caching...")
    qwen3, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device=str(device))
    qwen3.eval()
    qwen3.requires_grad_(False)

    rng_seed = args.precache_seed if args.precache_seed is not None else (args.seed if args.seed is not None else 0)
    random.seed(rng_seed)

    n_files = 0
    total_bytes = 0
    for ds in _iter_datasets(train_group, val_group):
        for info in ds.image_data.values():
            subset = ds.image_to_subset[info.image_key]
            if subset.token_warmup_step and subset.token_warmup_step > 0:
                raise ValueError("token_warmup_step is not supported with pre-cached TE outputs")

            # Caption dropout is NOT baked in: it is applied at train time from caption_dropout_rate
            # (drop_cached_text_encoder_outputs), exactly like regular sd-scripts TE caching.
            sub = copy.copy(subset)
            sub.caption_dropout_rate = 0.0
            sub.caption_dropout_every_n_epochs = 0
            n_variants = args.te_variants if _subset_augments_caption(subset) else 1
            captions = [ds.process_caption(sub, info.caption) for _ in range(n_variants)]

            tokens = tokenize_strategy.tokenize(captions)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=weight_dtype):
                prompt_embeds, attn_mask, t5_ids, t5_mask = text_encoding_strategy.encode_tokens(
                    tokenize_strategy, [qwen3], tokens
                )

            q_len = max(1, int(attn_mask.sum(dim=1).max().item()))
            t_len = max(1, int(t5_mask.sum(dim=1).max().item()))
            prompt_embeds = prompt_embeds[:, :q_len]
            payload = dict(
                attn_mask=attn_mask[:, :q_len].cpu().numpy().astype(np.int32),
                t5_input_ids=t5_ids[:, :t_len].cpu().numpy().astype(np.int32),
                t5_attn_mask=t5_mask[:, :t_len].cpu().numpy().astype(np.int32),
                caption_dropout_rate=np.array(info.caption_dropout_rate, dtype=np.float32),
                qwen3_max_length=np.array(tokenize_strategy.qwen3_max_length, dtype=np.int32),
                t5_max_length=np.array(tokenize_strategy.t5_max_length, dtype=np.int32),
                format=np.array(TE_CACHE_FORMAT),
                variant_captions=np.array(captions),
            )
            if prompt_embeds.dtype == torch.bfloat16:
                payload["prompt_embeds"] = _bf16_bits(prompt_embeds)
                payload["prompt_embeds_dtype"] = np.array("bf16")
            else:
                payload["prompt_embeds"] = prompt_embeds.float().cpu().numpy()

            npz_path = caching_strategy.get_outputs_npz_path(info.absolute_path)
            tmp_path = npz_path + ".tmp.npz"
            np.savez_compressed(tmp_path, **payload)
            os.replace(tmp_path, npz_path)
            n_files += 1
            total_bytes += os.path.getsize(npz_path)

    del qwen3
    clean_memory_on_device(device)
    logger.info(
        f"TE pre-cache done: {n_files} files, {total_bytes / 1e6:.1f} MB total "
        f"(variants per augmented caption: {args.te_variants})"
    )


def setup_parser() -> argparse.ArgumentParser:
    parser = anima_train_network.setup_parser()
    parser.add_argument(
        "--te_variants",
        type=int,
        default=16,
        help="number of independently shuffled/tag-dropped caption variants cached per image (default 16)",
    )
    parser.add_argument("--precache_seed", type=int, default=None, help="RNG seed for caption variants (default: --seed)")
    parser.add_argument("--no_latents", action="store_true", help="skip latent pre-caching")
    parser.add_argument("--no_text_encoder", action="store_true", help="skip text encoder pre-caching")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    args = args_util.read_config_from_file(args, parser)
    accelerator_setup.prepare_dataset_args(args, True)
    if args.te_variants < 1:
        raise ValueError("--te_variants must be >= 1")

    # same tokenizer / caching strategies as training, but always to disk
    args.cache_latents = True
    args.cache_latents_to_disk = True
    trainer = anima_train_network.AnimaNetworkTrainer()
    strategy_base.TokenizeStrategy.set_strategy(trainer.get_tokenize_strategy(args))
    strategy_base.LatentsCachingStrategy.set_strategy(
        strategy_anima.AnimaLatentsCachingStrategy(True, args.vae_batch_size, args.skip_cache_check)
    )

    train_group, val_group = _build_dataset_groups(args)
    if len(train_group) == 0:
        raise RuntimeError("dataset is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    accelerator = SimpleNamespace(
        device=device, num_processes=1, process_index=0, is_main_process=True, wait_for_everyone=lambda: None
    )

    t0 = time.time()
    if not args.no_latents:
        cache_latents(args, train_group, val_group, accelerator, weight_dtype)
    if not args.no_text_encoder:
        cache_text_encoder_outputs(args, train_group, val_group, device, weight_dtype)
    logger.info(f"Anima pre-caching finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
