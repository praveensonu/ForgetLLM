"""Dependency-free RMU configuration shared by the API and training worker."""

import re

DEFAULT_LAYERS = [22, 23, 24, 25, 26, 27, 28]


def trainable_params_regex(layers):
    """Select layer parameters; the trainer also preserves PEFT's frozen weights."""
    indices = '|'.join(str(layer) for layer in sorted(layers))
    return rf'.*layers\.({indices})\..*'


def resolve_rmu_params(params=None):
    params = dict(params or {})
    layers = sorted(params.get('layers_to_transform', DEFAULT_LAYERS))
    if not layers or any(type(layer) is not int or layer < 0 for layer in layers):
        raise ValueError('layers_to_transform must contain non-negative integer layer indices.')
    if len(set(layers)) != len(layers):
        raise ValueError('layers_to_transform must not contain duplicates.')
    module_regex = params.get('module_regex', rf'.*layers\.{max(layers)}$')
    if not module_regex or not module_regex.strip():
        raise ValueError('module_regex must not be empty.')
    re.compile(module_regex)
    return {
        'layers_to_transform': layers,
        'module_regex': module_regex,
        'trainable_params_regex': trainable_params_regex(layers),
        'steering_coeff': params.get('steering_coeff', 20.0),
    }
