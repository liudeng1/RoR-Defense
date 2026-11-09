"""Model module helpers."""
from __future__ import annotations

import torch.nn as nn


def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        return model.model.decoder.layers
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        return model.decoder.layers
    raise ValueError(f'Cannot locate Transformer blocks in model type {type(model)!r}')


def find_layers(module: nn.Module, layers=(nn.Linear,), name: str = '') -> dict[str, nn.Module]:
    if isinstance(module, tuple(layers)):
        return {name: module}
    found = {}
    for child_name, child in module.named_children():
        key = f'{name}.{child_name}' if name else child_name
        found.update(find_layers(child, layers=layers, name=key))
    return found


def replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    if '.' in name:
        parent, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), leaf, replacement)
    else:
        setattr(model, name, replacement)


def strict_bind_profile_to_linears(model: nn.Module, profile_layers: dict) -> dict[str, str]:
    eligible = {
        name: mod for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and 'lm_head' not in name
    }
    output = {}
    for raw_key in profile_layers:
        key = str(raw_key).strip('.')
        matched = [name for name in eligible if name == key or name.endswith('.' + key)]
        if len(matched) != 1:
            raise RuntimeError(f'Profile key {key!r} matched {len(matched)} Linears: {matched[:8]}')
        if matched[0] in output:
            raise RuntimeError(f'Duplicate profile binding {matched[0]}')
        output[matched[0]] = key
    missing = sorted(set(eligible) - set(output))
    if missing:
        raise RuntimeError(f'Profile does not cover {len(missing)} model Linear layers: {missing[:8]}')
    return output
