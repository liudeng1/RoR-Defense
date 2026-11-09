"""SPoF positions and INT8 bit injection."""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SpofSpec:
    layer_idx: int
    submodule_name: str
    row: int
    col: int
    bit: int = 7
    module_root: str = 'model.layers'

    @property
    def module_path(self) -> str:
        return f'{self.module_root}.{self.layer_idx}.{self.submodule_name}'


MODEL_PRESETS = {
    'qwen7b': {
        'name': 'Qwen2.5-7B', 'model_type': 'qwen2',
        'beta': 0.30,
        'spof': SpofSpec(layer_idx=1, submodule_name='mlp.gate_proj', row=17094, col=1923),
    },
    'opt125m': {
        'name': 'OPT-125M', 'model_type': 'opt',
        'beta': 0.15,
        'spof': SpofSpec(layer_idx=2, submodule_name='fc1', row=643, col=706,
                         module_root='model.decoder.layers'),
    },
}


def xor_signed_int8_bit(value: int, bit: int) -> int:
    if not 0 <= bit < 8:
        raise ValueError(f'INT8 bit must be 0..7, got {bit}')
    code = (int(value) & 0xFF) ^ (1 << bit)
    return code - 256 if code & 0x80 else code


def flip_one_int8_bit(q_clean: torch.Tensor, *, col: int, bit: int) -> tuple[torch.Tensor, int, int]:
    if q_clean.ndim != 1:
        raise ValueError('q_clean must be one row')
    if not 0 <= col < q_clean.numel():
        raise IndexError(f'column={col}, width={q_clean.numel()}')
    old = int(q_clean[col].item())
    new = xor_signed_int8_bit(old, bit)
    attacked = q_clean.clone()
    attacked[col] = new
    return attacked, old, new
