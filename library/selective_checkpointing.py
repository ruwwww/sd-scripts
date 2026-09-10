"""Deterministic per-block gradient-checkpointing selection for Anima.

The Anima block implementation owns one ``gradient_checkpointing`` flag per
block. This module only selects and applies those existing flags; it does not
replace the block forward or checkpoint implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SELECTIVE_CHECKPOINTING_STRATEGIES = ("none", "full", "interleaved", "every4", "count")


@dataclass(frozen=True)
class CheckpointPlan:
    """Resolved checkpoint placement and offload policy for one model."""

    num_blocks: int
    strategy: str
    checkpointed_indices: tuple[int, ...]
    cpu_offload: bool = False
    unsloth_offload: bool = False

    @property
    def checkpointed_blocks(self) -> int:
        return len(self.checkpointed_indices)

    @property
    def checkpoint_fraction(self) -> float:
        return self.checkpointed_blocks / self.num_blocks


def _validate_num_blocks(num_blocks: int) -> None:
    if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks <= 0:
        raise ValueError(f"num_blocks must be a positive integer, got {num_blocks!r}")


def checkpoint_indices(
    num_blocks: int,
    strategy: str,
    checkpoint_blocks: Optional[int] = None,
) -> tuple[int, ...]:
    """Return sorted, unique block indices for a named strategy.

    ``count`` uses ``floor(i * num_blocks / count)``. This produces an exact,
    deterministic count with approximately uniform spacing; in particular,
    14/28 and 7/28 resolve to the requested every-second and every-fourth
    placements.
    """

    _validate_num_blocks(num_blocks)
    if strategy not in SELECTIVE_CHECKPOINTING_STRATEGIES:
        raise ValueError(
            f"unknown selective checkpointing strategy {strategy!r}; "
            f"expected one of {SELECTIVE_CHECKPOINTING_STRATEGIES}"
        )

    if strategy == "none":
        if checkpoint_blocks not in (None, 0):
            raise ValueError("checkpoint_blocks cannot be nonzero with strategy='none'")
        return ()
    if strategy == "full":
        if checkpoint_blocks not in (None, num_blocks):
            raise ValueError("checkpoint_blocks must be omitted or equal to num_blocks with strategy='full'")
        return tuple(range(num_blocks))
    if strategy == "interleaved":
        if checkpoint_blocks is not None:
            raise ValueError("checkpoint_blocks cannot be combined with strategy='interleaved'")
        return tuple(range(0, num_blocks, 2))
    if strategy == "every4":
        if checkpoint_blocks is not None:
            raise ValueError("checkpoint_blocks cannot be combined with strategy='every4'")
        return tuple(range(0, num_blocks, 4))

    if checkpoint_blocks is None:
        raise ValueError("strategy='count' requires checkpoint_blocks")
    if not isinstance(checkpoint_blocks, int) or isinstance(checkpoint_blocks, bool):
        raise ValueError(f"checkpoint_blocks must be an integer, got {checkpoint_blocks!r}")
    if checkpoint_blocks < 0 or checkpoint_blocks > num_blocks:
        raise ValueError(f"checkpoint_blocks must be between 0 and {num_blocks}, got {checkpoint_blocks}")
    if checkpoint_blocks == 0:
        return ()
    if checkpoint_blocks == num_blocks:
        return tuple(range(num_blocks))
    return tuple((index * num_blocks) // checkpoint_blocks for index in range(checkpoint_blocks))


def make_checkpoint_plan(
    num_blocks: int,
    *,
    strategy: Optional[str] = None,
    checkpoint_blocks: Optional[int] = None,
    cpu_offload: bool = False,
    unsloth_offload: bool = False,
) -> CheckpointPlan:
    """Resolve and validate a checkpointing request without mutating a model."""

    _validate_num_blocks(num_blocks)
    if cpu_offload and unsloth_offload:
        raise ValueError("cpu_offload and unsloth_offload cannot both be enabled")
    if strategy is None:
        strategy = "count" if checkpoint_blocks is not None else "full"
    indices = checkpoint_indices(num_blocks, strategy, checkpoint_blocks)
    if not indices and (cpu_offload or unsloth_offload):
        raise ValueError("checkpoint offload requires at least one checkpointed block")
    return CheckpointPlan(
        num_blocks=num_blocks,
        strategy=strategy,
        checkpointed_indices=indices,
        cpu_offload=bool(cpu_offload),
        unsloth_offload=bool(unsloth_offload),
    )


def apply_selective_checkpointing(
    model,
    strategy: Optional[str] = None,
    *,
    checkpoint_blocks: Optional[int] = None,
    cpu_offload: bool = False,
    unsloth_offload: bool = False,
) -> CheckpointPlan:
    """Apply a validated plan to an Anima-like model and return that plan.

    The model must expose ``blocks`` and each block must expose Kohya's
    ``enable_gradient_checkpointing`` and ``disable_gradient_checkpointing``
    methods. Validation completes before any block flag is changed.
    """

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("model must expose a blocks collection")
    plan = make_checkpoint_plan(
        len(blocks),
        strategy=strategy,
        checkpoint_blocks=checkpoint_blocks,
        cpu_offload=cpu_offload,
        unsloth_offload=unsloth_offload,
    )
    for index, block in enumerate(blocks):
        if not callable(getattr(block, "enable_gradient_checkpointing", None)):
            raise TypeError(f"model.blocks[{index}] lacks enable_gradient_checkpointing")
        if not callable(getattr(block, "disable_gradient_checkpointing", None)):
            raise TypeError(f"model.blocks[{index}] lacks disable_gradient_checkpointing")

    selected = set(plan.checkpointed_indices)
    for block in blocks:
        block.disable_gradient_checkpointing()
    for index in plan.checkpointed_indices:
        blocks[index].enable_gradient_checkpointing(
            cpu_offload=plan.cpu_offload,
            unsloth_offload=plan.unsloth_offload,
        )

    # Runtime metadata only; this is not a parameter or state-dict entry.
    model._selective_checkpointing_plan = plan
    model._selective_checkpointing_mask = tuple(index in selected for index in range(plan.num_blocks))
    return plan

