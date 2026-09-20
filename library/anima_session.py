"""Persistent in-VRAM Model Session for Anima DiT LoRA Training (Phase 2).

Maintains a permanently loaded and compiled Anima DiT, Qwen3 text encoder,
Qwen-Image VAE, and pre-hooked LoRA adapter plane across multiple training jobs.
Enables instant job starts (<1s) by re-initializing LoRA weights in-place without
re-compiling TorchInductor blocks or reloading model weights from disk.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional
import torch

logger = logging.getLogger(__name__)


class AnimaModelSession:
    """Holds a persistent base model and compiled DiT graph in VRAM."""

    _active_session: Optional[AnimaModelSession] = None

    def __init__(
        self,
        session_key: dict[str, Any],
        text_encoders: list[Any],
        vae: Any,
        unet: Any,
        network: Any,
    ):
        self.session_key = session_key
        self.text_encoders = text_encoders
        self.vae = vae
        self.unet = unet
        self.network = network
        self.job_count = 0

    @classmethod
    def get_active_session(cls) -> Optional[AnimaModelSession]:
        return cls._active_session

    @classmethod
    def set_active_session(cls, session: Optional[AnimaModelSession]) -> None:
        cls._active_session = session

    @classmethod
    def is_compatible(cls, key: dict[str, Any]) -> bool:
        if cls._active_session is None:
            return False
        current = cls._active_session.session_key
        for k, v in key.items():
            if current.get(k) != v:
                return False
        return True

    @classmethod
    def close_active_session(cls) -> None:
        if cls._active_session is not None:
            logger.info("Closing active AnimaModelSession and freeing VRAM...")
            try:
                cls._active_session.close()
            finally:
                cls._active_session = None

    def reset_for_new_job(self) -> None:
        """In-place re-initialization of LoRA parameters without breaking compiled graph."""
        self.job_count += 1
        logger.info(f"[AnimaModelSession] Resetting LoRA parameters in-place for job #{self.job_count}...")
        if self.network is not None:
            if hasattr(self.network, "reset_parameters"):
                self.network.reset_parameters()
            else:
                for p in self.network.parameters():
                    if p.requires_grad:
                        torch.nn.init.zeros_(p)

    def close(self) -> None:
        del self.network
        del self.unet
        del self.vae
        del self.text_encoders
        self.network = None
        self.unet = None
        self.vae = None
        self.text_encoders = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
