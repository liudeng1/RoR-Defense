"""Per-output-channel INT8 quantization for the demo."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ror.logging_utils import logger


@dataclass
class QuantizedRow:
    q: torch.Tensor
    scale: float


def quantize_all_linears_(model: nn.Module, *, target_module: str, target_row: int,
                          chunk_rows: int = 256) -> tuple[QuantizedRow, int]:

    if chunk_rows < 1:
        raise ValueError('chunk_rows must be >= 1')
    selected = None
    count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or 'lm_head' in name:
                continue
            count += 1
            weight = module.weight
            for st in range(0, weight.shape[0], int(chunk_rows)):
                ed = min(weight.shape[0], st + int(chunk_rows))
                original = weight.data[st:ed].detach().float()
                maxima = original.abs().amax(dim=1, keepdim=True)
                maxima.clamp_(min=torch.finfo(torch.float32).eps)
                scales = maxima / 127.0
                q = torch.round(original / scales).clamp_(-127, 127).to(torch.int16)
                if name == target_module and st <= target_row < ed:
                    selected = QuantizedRow(
                        q=q[target_row-st].detach().clone().cpu(),
                        scale=float(scales[target_row-st, 0].item()),
                    )
                weight.data[st:ed].copy_((q.float() * scales).to(dtype=weight.dtype))
            if count % 28 == 0:
                logger.info('INT8 quantization: %d Linears', count)
    if selected is None:
        raise RuntimeError(f'SPoF target not found: module={target_module}, row={target_row}')
    return selected, count


def dequantize_row(q: torch.Tensor, scale: float, dtype: torch.dtype, device) -> torch.Tensor:
    return (q.to(device=device, dtype=torch.float32) * float(scale)).to(dtype)
