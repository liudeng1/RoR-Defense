"""RoR rotation and weight fusion."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ror.model_utils import strict_bind_profile_to_linears, replace_module
from ror.plan import AlphaBetaRoRPlan, build_alpha_beta_model_plan
from ror.signs import generate_block_random_signs
from ror.logging_utils import logger

_INV_SQRT2 = 2.0 ** -0.5


def apply_ranked_tree(ranked: torch.Tensor, *, pair_left_by_depth: Sequence[torch.Tensor],
                      pair_right_by_depth: Sequence[torch.Tensor], transpose: bool = False) -> torch.Tensor:
    """Apply B=fine-to-coarse Householder butterfly (Eq.24)."""
    dtype, dev = ranked.dtype, ranked.device
    y = ranked.float().contiguous()
    depths = (range(len(pair_left_by_depth)) if transpose
              else range(len(pair_left_by_depth) - 1, -1, -1))
    for depth in depths:
        left, right = pair_left_by_depth[depth], pair_right_by_depth[depth]
        if left.numel() == 0:
            continue
        if left.device != dev:
            left, right = left.to(dev), right.to(dev)
        a, b = y.index_select(-1, left), y.index_select(-1, right)
        y.index_copy_(-1, left, (a + b) * _INV_SQRT2)
        y.index_copy_(-1, right, (a - b) * _INV_SQRT2)
    return y.to(dtype=dtype)


def apply_pbd(x: torch.Tensor, *, permutation: torch.Tensor,
              ranked_signs: torch.Tensor, pair_left_by_depth: Sequence[torch.Tensor],
              pair_right_by_depth: Sequence[torch.Tensor]) -> torch.Tensor:

    dtype = x.dtype
    perm = permutation.to(device=x.device, dtype=torch.long)
    ranked = x.index_select(-1, perm)
    ranked = apply_ranked_tree(ranked, pair_left_by_depth=pair_left_by_depth,
                               pair_right_by_depth=pair_right_by_depth, transpose=False)
    signs = ranked_signs.to(device=x.device, dtype=ranked.dtype)
    return (ranked * signs).to(dtype=dtype)


def offline_rotate_weight(weight: torch.Tensor, *, permutation: torch.Tensor,
                          ranked_signs: torch.Tensor,
                          pair_left_by_depth: Sequence[torch.Tensor],
                          pair_right_by_depth: Sequence[torch.Tensor],
                          chunk_rows: int = 512) -> torch.Tensor:

    chunk_rows = max(1, int(chunk_rows))
    rotated = torch.empty_like(weight)
    for start in range(0, int(weight.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(weight.shape[0]))
        rotated[start:end] = apply_pbd(
            weight[start:end].detach().float(), permutation=permutation,
            ranked_signs=ranked_signs,
            pair_left_by_depth=pair_left_by_depth,
            pair_right_by_depth=pair_right_by_depth,
        ).to(dtype=weight.dtype)
    return rotated


class RoRLinear(nn.Linear):

    def __init__(self, original_linear: nn.Linear, *, plan: AlphaBetaRoRPlan,
                 layer_seed: int, chunk_rows: int = 512, module_name: str = ''):
        super().__init__(
            original_linear.in_features, original_linear.out_features,
            bias=original_linear.bias is not None,
            device=original_linear.weight.device, dtype=original_linear.weight.dtype,
        )
        if int(plan.d) != int(original_linear.in_features):
            raise RuntimeError(f'{module_name}: plan width={plan.d} vs model width={original_linear.in_features}')
        self.module_name = str(module_name)
        self.alpha_beta_plan = plan
        dev = original_linear.weight.device
        perm = torch.tensor(plan.ranking, dtype=torch.int32, device=dev).contiguous()
        signs_original = generate_block_random_signs(plan.block_sizes, int(layer_seed), device=dev)
        signs_ranked = signs_original.index_select(0, perm.long()).to(torch.int8).contiguous()
        self.register_buffer('ror_permutation', perm, persistent=True)
        self.register_buffer('ror_ranked_signs', signs_ranked, persistent=True)
        self._pair_left_names, self._pair_right_names = [], []
        for depth, (left, right) in enumerate(zip(plan.pair_left_by_depth, plan.pair_right_by_depth)):
            left_name, right_name = f'ror_pair_left_{depth}', f'ror_pair_right_{depth}'
            self.register_buffer(left_name, left.to(device=dev, dtype=torch.int64).contiguous(), persistent=False)
            self.register_buffer(right_name, right.to(device=dev, dtype=torch.int64).contiguous(), persistent=False)
            self._pair_left_names.append(left_name)
            self._pair_right_names.append(right_name)
        with torch.no_grad():
            self.weight.copy_(offline_rotate_weight(
                original_linear.weight.detach(), permutation=self.ror_permutation,
                ranked_signs=self.ror_ranked_signs, pair_left_by_depth=self._left(),
                pair_right_by_depth=self._right(), chunk_rows=chunk_rows,
            ))
            if original_linear.bias is not None:
                self.bias.copy_(original_linear.bias.detach())

    def _left(self):
        return tuple(getattr(self, name) for name in self._pair_left_names)

    def _right(self):
        return tuple(getattr(self, name) for name in self._pair_right_names)

    def transform_row(self, row: torch.Tensor) -> torch.Tensor:
        return apply_pbd(row, permutation=self.ror_permutation,
                         ranked_signs=self.ror_ranked_signs,
                         pair_left_by_depth=self._left(), pair_right_by_depth=self._right())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.transform_row(x), self.weight, self.bias)


def inject_ror(model: nn.Module, *, activation_profile: dict,
               alpha: float = 6.0, beta: float = .30, seed: int = 0,
               chunk_rows: int = 512) -> tuple[nn.Module, dict]:

    if abs(float(alpha) - 6.0) > 1e-12:
        raise ValueError(f'Paper RoR uses alpha=6, got {alpha}')
    mp = build_alpha_beta_model_plan(activation_profile, alpha=alpha, beta=beta)
    bindings = strict_bind_profile_to_linears(model, activation_profile['layers'])
    eligible_names = [name for name, module in model.named_modules()
                      if isinstance(module, nn.Linear) and 'lm_head' not in name]
    for i, name in enumerate(eligible_names, start=1):
        module = model.get_submodule(name)
        plan_key = bindings[name]
        plan = mp.layer_plans[plan_key]
        new = RoRLinear(module, plan=plan, layer_seed=int(seed) + i - 1,
                        chunk_rows=chunk_rows, module_name=name)
        replace_module(model, name, new)
        del module
        if i % 28 == 0 or i == len(eligible_names):
            logger.info('RoR offline fusion: %s/%s Transformer Linears', i, len(eligible_names))
    model.eval()
    summary = {
        'rotation': 'Q=PBD (D in sensitivity-ranked coordinates)',
        'alpha': alpha, 'beta': beta, 'protected_linears': len(eligible_names),
        'total_butterflies_per_token': mp.total_butterflies,
        'full_fwht_butterflies_per_token': mp.total_fwht_butterflies,
        'high_channels': mp.high_count, 'middle_channels': mp.middle_count,
        'low_channels': mp.low_count,
    }
    logger.info('RoR ready: %s layers, %s butterfly pairs/token', len(eligible_names), mp.total_butterflies)
    return model, summary
