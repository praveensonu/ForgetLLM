"""RMU trainers for forget-only and forget/retain runs."""

import copy
import math
import re

import torch

from unlearning.base import UnlearnTrainer
from unlearning.gd.losses import to_model_inputs
from unlearning.rmu.config import resolve_rmu_params
from unlearning.rmu.rmu_utils import (
    forward_with_cache,
    masked_mse_over_tokens,
    resolve_module,
)


class RMUTrainer(UnlearnTrainer):
    uses_retain = True

    def __init__(self, layers_to_transform=None, module_regex=None,
                 trainable_params_regex=None, steering_coeff=20.0, **kwargs):
        params = resolve_rmu_params({
            **({'layers_to_transform': layers_to_transform} if layers_to_transform is not None else {}),
            **({'module_regex': module_regex} if module_regex is not None else {}),
            'steering_coeff': steering_coeff,
        })
        if not math.isfinite(steering_coeff) or steering_coeff < 0:
            raise ValueError('steering_coeff must be finite and non-negative.')
        args = kwargs.get('args')
        if args is not None:
            args._n_gpu = 1 if args.device.type == 'cuda' else 0
        super().__init__(**kwargs)
        if self.accelerator.num_processes > 1:
            raise ValueError('RMU requires a single training process on one GPU.')
        if self.args.gradient_checkpointing or getattr(self.model, 'is_gradient_checkpointing', False):
            raise ValueError('RMU activation hooks require gradient checkpointing to be disabled.')
        self.layers_to_transform = params['layers_to_transform']
        self.module_regex = params['module_regex']
        self.trainable_params_regex = params['trainable_params_regex']
        self.steering_coeff = steering_coeff
        self._control_vec = None

        # The orchestrator loads onto one device. Consolidate direct Trainer users too.
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(self.model, recurse=True)
        self.model.to(self.accelerator.device)
        if hasattr(self.model, 'hf_device_map'):
            self.model.hf_device_map = {'': str(self.accelerator.device)}
        self.is_model_parallel = False
        self._model_module = resolve_module(self.model, self.module_regex)
        self._select_trainable_parameters()

        # Snapshot before any optimizer step, only for the retain variant.
        self.ref_model = None
        self._ref_module = None
        if self.uses_retain:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()
            self._ref_module = resolve_module(self.ref_model, self.module_regex)

    def _select_trainable_parameters(self):
        matched_layers = set()
        selected = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search(self.trainable_params_regex, name):
                matched_layers.add(int(re.search(r'layers\.(\d+)\.', name).group(1)))
                selected.append(param)
        missing = set(self.layers_to_transform) - matched_layers
        if missing:
            raise ValueError(
                f'No trainable parameters in RMU layers {sorted(missing)}. '
                'Check the model layer indices and the adapter target layers.'
            )
        self.model.requires_grad_(False)
        for param in selected:
            param.requires_grad_(True)

    def _get_control_vec(self, dim, dtype, device):
        if self._control_vec is None:
            # A local generator keeps the direction reproducible across both data modes.
            generator = torch.Generator(device=device).manual_seed(self.args.seed)
            vector = torch.rand(1, 1, dim, generator=generator, device=device)
            self._control_vec = (vector / vector.norm() * self.steering_coeff).detach()
        return self._control_vec.to(dtype=dtype, device=device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.uses_retain:
            forget, retain = (to_model_inputs(batch) for batch in inputs)
        else:
            forget, retain = inputs, None
        activation, outputs = forward_with_cache(model, forget, self._model_module)
        control = self._get_control_vec(activation.shape[-1], activation.dtype, activation.device)
        loss = self.gamma * masked_mse_over_tokens(activation, control, forget['labels'] != -100)
        if retain is not None:
            retained, _ = forward_with_cache(model, retain, self._model_module)
            reference, _ = forward_with_cache(self.ref_model, retain, self._ref_module, no_grad=True)
            loss = loss + self.alpha * masked_mse_over_tokens(
                retained, reference.to(retained), retain['labels'] != -100,
            )
        return (loss, outputs) if return_outputs else loss


class RMUForgetOnlyTrainer(RMUTrainer):
    uses_retain = False
