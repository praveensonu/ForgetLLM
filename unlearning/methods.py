"""The unlearning method registry.

The single source of truth for which methods exist, what each one needs from the
dataset, and the hyperparameters it runs with. The API schemas, the orchestrator and
the frontend all derive from this, so a new method is added here only. Trainer
defaults come from open-unlearning's `configs/trainer/*.yaml`. Users can override
beta/delta and supply the shared gamma/alpha strengths.

Kept dependency-free so both the API process and the training subprocess can import it
cheaply.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

from unlearning.rmu.config import resolve_rmu_params


class UnlearningMethodSpec(NamedTuple):
    label: str
    trainer_args: Dict[str, Any]
    requires_retain: bool
    forget_only: bool


UNLEARNING_METHODS: Dict[str, UnlearningMethodSpec] = {
    "rmu": UnlearningMethodSpec(
        label="RMU",
        trainer_args=resolve_rmu_params(),
        requires_retain=False,
        forget_only=False,
    ),
    "grad_ascent": UnlearningMethodSpec(
        label="Gradient Ascent",
        trainer_args={},
        requires_retain=False,
        forget_only=True,
    ),
    "grad_diff": UnlearningMethodSpec(
        label="Gradient Difference",
        trainer_args={"retain_loss_type": "NLL"},
        requires_retain=True,
        forget_only=False,
    ),
    "npo": UnlearningMethodSpec(
        label="NPO",
        trainer_args={"beta": 0.1, "retain_loss_type": "NLL"},
        requires_retain=False,
        forget_only=False,
    ),
    "dpo": UnlearningMethodSpec(
        label="DPO",
        trainer_args={"beta": 0.1, "retain_loss_type": "NLL"},
        requires_retain=False,
        forget_only=False,
    ),
    "simnpo": UnlearningMethodSpec(
        label="SimNPO",
        trainer_args={
            "beta": 4.5,
            "delta": 0.0,
            "retain_loss_type": "NLL",
        },
        requires_retain=False,
        forget_only=False,
    ),
}

DEFAULT_UNLEARNING_METHOD = "simnpo"

UNLEARNING_METHOD_ARGS: Dict[str, Dict[str, Any]] = {
    name: spec.trainer_args for name, spec in UNLEARNING_METHODS.items()
}

RETAIN_REQUIRED_METHODS = frozenset(
    name for name, spec in UNLEARNING_METHODS.items() if spec.requires_retain
)

FORGET_ONLY_METHODS = frozenset(
    name for name, spec in UNLEARNING_METHODS.items() if spec.forget_only
)
