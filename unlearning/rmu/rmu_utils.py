"""Activation capture and losses for Representation Misdirection for Unlearning."""

import torch


def resolve_module(model, module_regex):
    import re

    matches = [(name, module) for name, module in model.named_modules()
               if re.search(module_regex, name)]
    if len(matches) != 1:
        raise ValueError(
            f"Module regex '{module_regex}' matched {len(matches)} modules; "
            f"expected exactly one. Matches: {[name for name, _ in matches][:8]}"
        )
    return matches[0][1]


def forward_with_cache(model, inputs, module, no_grad=False):
    cache = []

    def hook(_module, _inputs, output):
        cache.append(output[0] if isinstance(output, tuple) else output)

    handle = module.register_forward_hook(hook)
    try:
        # Activation losses need labels only for masking, not a language-model loss.
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**{key: inputs[key] for key in ('input_ids', 'attention_mask')})
    finally:
        handle.remove()
    if not cache:
        raise RuntimeError('RMU hook captured no activation.')
    return cache[0], outputs


def masked_mse_over_tokens(activation, target, mask):
    """Mean squared error per answer token, averaged over examples in float32."""
    squared_error = (activation.float() - target.float()).square().mean(dim=-1)
    valid = mask.to(device=activation.device, dtype=squared_error.dtype)
    return ((squared_error * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)).mean()


def __getattr__(name):
    # Keep the supplied example's import working without a circular trainer import.
    if name == 'RMUTrainer':
        from unlearning.rmu.trainer import RMUTrainer
        return RMUTrainer
    raise AttributeError(name)
