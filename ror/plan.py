"""Activation profiling and hierarchical RoR plan construction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from ror.logging_utils import logger

PROFILE_VERSION = "alpha-beta-ror-activation-profile-v1"


def is_power_of_two(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def greedy_power_of_two_blocks(target_dim: int) -> Tuple[int, ...]:
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    blocks: List[int] = []
    remaining = target_dim
    current = 1 << (target_dim.bit_length() - 1)
    while remaining > 0:
        if current <= remaining:
            blocks.append(current)
            remaining -= current
        else:
            current >>= 1
    return tuple(blocks)


def _block_offsets(block_sizes: Sequence[int]) -> Tuple[int, ...]:
    out: List[int] = []
    off = 0
    for n in block_sizes:
        out.append(off)
        off += int(n)
    return tuple(out)


def block_fwht_butterflies(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 0
    if not is_power_of_two(n):
        raise ValueError(f"block must be power-of-two, got {n}")
    return (n // 2) * int(math.log2(n))


def load_activation_profile(path: str | Path) -> Dict[str, object]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(p, map_location="cpu")
    if not isinstance(obj, dict) or not isinstance(obj.get("layers"), dict):
        raise RuntimeError(f"invalid activation profile: {p}")
    if obj.get("version") != PROFILE_VERSION:
        raise RuntimeError(
            f"unexpected activation profile version={obj.get('version')!r}; expected {PROFILE_VERSION!r}"
        )
    return obj


def structural_min_depth(block_size: int) -> int:
    n = int(block_size)
    if n <= 1:
        return 0
    if not is_power_of_two(n):
        raise ValueError(f"power-of-two block required, got {n}")
    L = int(math.log2(n))
    if L <= 1:
        return L
    return max(1, L // 2 - 1)


def original_ror_threshold(
    mu: float,
    sigma: float,
    alpha: float = 6.0,
    rel_floor: float = 2.0,
    abs_floor: float = 1.0,
) -> float:
    return float(max(float(mu) + float(alpha) * float(sigma),
                     float(rel_floor) * float(mu),
                     float(abs_floor)))


def graded_depth_from_activation(
    *, m: float, tau_beta: float, tau_alpha: float, k_min: int, L: int
) -> int:
    m = float(m); tau_beta = float(tau_beta); tau_alpha = float(tau_alpha)
    k_min = int(k_min); L = int(L)
    if L <= 0:
        return 0
    if not (0.0 < tau_beta < tau_alpha):
        raise ValueError((tau_beta, tau_alpha))
    if not (0 <= k_min < L):
        raise ValueError((k_min, L))
    m = min(tau_alpha, max(tau_beta, m))
    u = math.log(m / tau_beta) / math.log(tau_alpha / tau_beta)
    u = min(1.0, max(0.0, u))
    k = k_min + int(math.ceil((L - k_min) * u - 1e-12))
    return int(min(L, max(k_min + 1, k)))


def _active_from_required_depths(required: np.ndarray) -> List[Set[int]]:
    n = int(required.size)
    if n <= 1:
        return []
    L = int(math.log2(n))
    active: List[Set[int]] = [set() for _ in range(L)]
    for t in range(L):
        block_size = n >> t
        block_max = required.reshape(1 << t, block_size).max(axis=1)
        active[t].update(int(v) for v in np.nonzero(block_max > t)[0].tolist())
    return active


def _leaf_depths(n: int, active: Sequence[Set[int]]) -> np.ndarray:
    dep = np.zeros(int(n), dtype=np.int16)
    for t, blocks in enumerate(active):
        block_size = int(n) >> t
        for bi in blocks:
            s = int(bi) * block_size
            dep[s:s + block_size] += 1
    return dep


def _pairs_from_active_local(
    n: int, active: Sequence[Set[int]], *, offset: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    left_all: List[torch.Tensor] = []
    right_all: List[torch.Tensor] = []
    for depth, blocks in enumerate(active):
        block_size = int(n) >> depth
        half = block_size // 2
        left: List[int] = []
        right: List[int] = []
        for bi in sorted(blocks):
            base = int(offset) + int(bi) * block_size
            left.extend(range(base, base + half))
            right.extend(range(base + half, base + block_size))
        left_all.append(torch.tensor(left, dtype=torch.int64))
        right_all.append(torch.tensor(right, dtype=torch.int64))
    return left_all, right_all


def _active_cost(n: int, active: Sequence[Set[int]]) -> int:
    return int(sum(len(active[t]) * ((int(n) >> t) // 2) for t in range(len(active))))


def _hist(values: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        s = str(int(v)); out[s] = out.get(s, 0) + 1
    return out


@dataclass(frozen=True)
class AlphaBetaRoRPlan:
    d: int
    ranking: Tuple[int, ...]
    block_sizes: Tuple[int, ...]
    block_offsets: Tuple[int, ...]
    block_min_depths: Tuple[int, ...]
    block_max_depths: Tuple[int, ...]
    pair_left_by_depth: Tuple[torch.Tensor, ...]
    pair_right_by_depth: Tuple[torch.Tensor, ...]
    leaf_depths: Tuple[int, ...]
    leaf_max_depths: Tuple[int, ...]
    target_leaf_depths: Tuple[int, ...]
    alpha: float
    beta: float
    score_mean: float
    score_std: float
    alpha_threshold: float
    beta_threshold: float
    high_count: int
    middle_count: int
    low_count: int
    target_depth_histogram: Dict[str, int]
    butterfly_count: int
    full_fwht_butterfly_count: int
    full_depth_count: int
    normal_full_depth_count: int
    legacy_strict_count: int
    legacy_strict_full_depth_count: int

@dataclass(frozen=True)
class AlphaBetaRoRModelPlan:
    layer_plans: Dict[str, AlphaBetaRoRPlan]
    alpha: float
    beta: float
    total_butterflies: int
    total_fwht_butterflies: int
    achieved_ratio_vs_fwht: float
    ideal_ratio_vs_fwht: float
    high_count: int
    middle_count: int
    low_count: int
    full_depth_count: int
    normal_full_depth_count: int
    legacy_strict_count: int
    legacy_strict_full_depth_count: int


def build_alpha_beta_layer_plan(
    item: Mapping[str, object], *, alpha: float = 6.0, beta: float = 0.30,
    rel_floor: float = 2.0, abs_floor: float = 1.0,
) -> AlphaBetaRoRPlan:
    alpha = float(alpha); beta = float(beta)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be >0, got {alpha}")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must satisfy 0<beta<1, got {beta}")

    d = int(item["in_features"])
    scores_orig = item["scores"].detach().float().cpu().flatten()
    ranking_t = item["ranking"].detach().to(torch.int64).cpu().flatten()
    if scores_orig.numel() != d or ranking_t.numel() != d:
        raise RuntimeError("activation profile length mismatch")
    if torch.unique(ranking_t).numel() != d:
        raise RuntimeError("profile ranking is not a permutation")
    scores_ranked = scores_orig.index_select(0, ranking_t)
    if bool((scores_ranked[:-1] < scores_ranked[1:]).any()):
        raise RuntimeError("activation profile ranking is not non-increasing")

    mu = float(item.get("score_mean", scores_orig.mean().item()))
    sigma = float(item.get("score_std", scores_orig.std(unbiased=False).item()))
    tau_alpha = original_ror_threshold(mu, sigma, alpha, rel_floor, abs_floor)
    tau_beta = float(beta * tau_alpha)
    if not (0.0 < tau_beta < tau_alpha):
        raise AssertionError((tau_beta, tau_alpha))

    if "threshold" in item and abs(alpha - 6.0) < 1e-12:
        stored = float(item["threshold"])
        if abs(stored - tau_alpha) > max(1e-6, 1e-6 * abs(stored)):
            raise AssertionError((stored, tau_alpha, mu, sigma))

    sr = scores_ranked.numpy().astype(np.float64)
    high_mask = sr > tau_alpha
    low_mask = sr <= tau_beta
    mid_mask = ~(high_mask | low_mask)

    block_sizes = greedy_power_of_two_blocks(d)
    offsets = _block_offsets(block_sizes)
    max_global_L = max((int(math.log2(n)) for n in block_sizes if n > 1), default=0)
    global_left: List[List[torch.Tensor]] = [[] for _ in range(max_global_L)]
    global_right: List[List[torch.Tensor]] = [[] for _ in range(max_global_L)]
    required = np.zeros(d, dtype=np.int16)
    leaf = np.zeros(d, dtype=np.int16)
    leaf_max = np.zeros(d, dtype=np.int16)
    block_kmins: List[int] = []
    block_Ls: List[int] = []
    total_cost = 0
    total_fw = 0

    for n, offset in zip(block_sizes, offsets):
        n = int(n); offset = int(offset)
        if n <= 1:
            block_kmins.append(0); block_Ls.append(0)
            continue
        L = int(math.log2(n))
        k_min = structural_min_depth(n)
        block_kmins.append(k_min); block_Ls.append(L)
        sl = slice(offset, offset + n)
        local_sr = sr[sl]
        local_high = high_mask[sl]
        local_low = low_mask[sl]
        local_mid = mid_mask[sl]
        req = np.empty(n, dtype=np.int16)
        req[local_low] = k_min
        req[local_high] = L
        for i in np.nonzero(local_mid)[0].tolist():
            req[i] = graded_depth_from_activation(
                m=float(local_sr[i]), tau_beta=tau_beta, tau_alpha=tau_alpha,
                k_min=k_min, L=L,
            )
        if bool(np.any(req[:-1] < req[1:])):
            raise AssertionError(f"requested depth is not non-increasing inside block n={n}, offset={offset}")

        active = _active_from_required_depths(req)
        local_leaf = _leaf_depths(n, active)
        if np.any(local_leaf < req):
            raise AssertionError("prefix closure produced a shallower leaf than requested")
        left, right = _pairs_from_active_local(n, active, offset=offset)
        for depth in range(len(left)):
            if left[depth].numel():
                global_left[depth].append(left[depth]); global_right[depth].append(right[depth])
        local_cost = _active_cost(n, active)
        if int(local_leaf.sum()) != 2 * local_cost:
            raise AssertionError("sum(local leaf depths) != 2*butterfly_count")
        required[sl] = req
        leaf[sl] = local_leaf
        leaf_max[sl] = L
        total_cost += int(local_cost)
        total_fw += int(block_fwht_butterflies(n))

    pair_left = tuple(torch.cat(v, dim=0) if v else torch.empty(0, dtype=torch.int64) for v in global_left)
    pair_right = tuple(torch.cat(v, dim=0) if v else torch.empty(0, dtype=torch.int64) for v in global_right)
    if sum(int(x.numel()) for x in pair_left) != total_cost:
        raise AssertionError("global pair count != butterfly count")
    if int(leaf.sum()) != 2 * total_cost:
        raise AssertionError("sum(global leaf depths) != 2*butterfly_count")

    strict_t = item.get("strict_outliers", torch.empty(0, dtype=torch.int64))
    strict_orig = set(int(v) for v in strict_t.detach().to(torch.int64).cpu().flatten().tolist())
    inv_rank = {int(orig): pos for pos, orig in enumerate(ranking_t.tolist())}
    strict_full = sum(
        1 for orig in strict_orig
        if int(leaf[inv_rank[orig]]) == int(leaf_max[inv_rank[orig]])
    )
    if strict_full != len(strict_orig):
        raise AssertionError(f"legacy strict outlier lost local full depth: {strict_full}/{len(strict_orig)}")
    if abs(alpha - 6.0) < 1e-12 and int(high_mask.sum()) != len(strict_orig):
        raise AssertionError(
            f"alpha=6 abnormal set != frozen strict set: {int(high_mask.sum())}/{len(strict_orig)}"
        )

    full_mask = leaf == leaf_max
    full = int(full_mask.sum())
    normal_full = int((full_mask & (~high_mask)).sum())
    return AlphaBetaRoRPlan(
        d=d,
        ranking=tuple(int(v) for v in ranking_t.tolist()),
        block_sizes=tuple(int(v) for v in block_sizes),
        block_offsets=tuple(int(v) for v in offsets),
        block_min_depths=tuple(int(v) for v in block_kmins),
        block_max_depths=tuple(int(v) for v in block_Ls),
        pair_left_by_depth=pair_left,
        pair_right_by_depth=pair_right,
        leaf_depths=tuple(int(v) for v in leaf.tolist()),
        leaf_max_depths=tuple(int(v) for v in leaf_max.tolist()),
        target_leaf_depths=tuple(int(v) for v in required.tolist()),
        alpha=alpha, beta=beta,
        score_mean=mu, score_std=sigma,
        alpha_threshold=tau_alpha, beta_threshold=tau_beta,
        high_count=int(high_mask.sum()), middle_count=int(mid_mask.sum()), low_count=int(low_mask.sum()),
        target_depth_histogram=_hist(required.tolist()),
        butterfly_count=int(total_cost),
        full_fwht_butterfly_count=int(total_fw),
        full_depth_count=full,
        normal_full_depth_count=normal_full,
        legacy_strict_count=len(strict_orig),
        legacy_strict_full_depth_count=strict_full,
    )


def build_alpha_beta_model_plan(
    activation_profile: Mapping[str, object], *, alpha: float = 6.0, beta: float = 0.30,
) -> AlphaBetaRoRModelPlan:
    layers = activation_profile.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise RuntimeError("activation profile has no layers")
    rel_floor = float(activation_profile.get("rel_thresh", 2.0))
    abs_floor = float(activation_profile.get("abs_thresh", 1.0))
    plans = {
        str(k): build_alpha_beta_layer_plan(
            v, alpha=alpha, beta=beta, rel_floor=rel_floor, abs_floor=abs_floor
        ) for k, v in layers.items()
    }
    total = int(sum(p.butterfly_count for p in plans.values()))
    total_fw = int(sum(p.full_fwht_butterfly_count for p in plans.values()))
    ideal_num = int(sum(sum(p.target_leaf_depths) for p in plans.values()))
    ideal_den = int(2 * total_fw)
    return AlphaBetaRoRModelPlan(
        layer_plans=plans,
        alpha=float(alpha), beta=float(beta),
        total_butterflies=total,
        total_fwht_butterflies=total_fw,
        achieved_ratio_vs_fwht=float(total / total_fw) if total_fw else 0.0,
        ideal_ratio_vs_fwht=float(ideal_num / ideal_den) if ideal_den else 0.0,
        high_count=int(sum(p.high_count for p in plans.values())),
        middle_count=int(sum(p.middle_count for p in plans.values())),
        low_count=int(sum(p.low_count for p in plans.values())),
        full_depth_count=int(sum(p.full_depth_count for p in plans.values())),
        normal_full_depth_count=int(sum(p.normal_full_depth_count for p in plans.values())),
        legacy_strict_count=int(sum(p.legacy_strict_count for p in plans.values())),
        legacy_strict_full_depth_count=int(sum(p.legacy_strict_full_depth_count for p in plans.values())),
    )



def _extract_input_ids(sample: Any) -> torch.Tensor:
    if torch.is_tensor(sample):
        x = sample
    elif isinstance(sample, dict):
        x = sample.get("input_ids")
    elif isinstance(sample, (tuple, list)) and sample:
        x = sample[0]
        if isinstance(x, dict):
            x = x.get("input_ids")
    elif hasattr(sample, "input_ids"):
        x = sample.input_ids
    else:
        raise RuntimeError(f"unsupported calibration sample type: {type(sample)!r}")
    if not torch.is_tensor(x):
        raise RuntimeError(f"input_ids is not a tensor: {type(x)!r}")
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise RuntimeError(f"expected input_ids [batch,seqlen], got {tuple(x.shape)}")
    return x


def _pick_backbone(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    return inner if isinstance(inner, nn.Module) else model


def build_activation_profile(
    model: nn.Module,
    calib_loader,
    dev: torch.device,
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
    model_path: str,
    dataset: str,
    sigma_thresh: float = 6.0,
    rel_thresh: float = 2.0,
    abs_thresh: float = 1.0,
) -> dict[str, object]:
    from ror.model_utils import find_layers, get_model_layers

    if nsamples <= 0 or seqlen <= 0:
        raise ValueError("nsamples and seqlen must be positive")
    if not isinstance(calib_loader, (list, tuple)):
        calib_loader = list(calib_loader)
    if len(calib_loader) < nsamples:
        raise RuntimeError(f"calibration loader too short: {len(calib_loader)}/{nsamples}")

    layers = get_model_layers(model)
    entries: dict[str, nn.Linear] = {}
    for i, layer in enumerate(layers):
        local = find_layers(layer, layers=[nn.Linear])
        for name, module in local.items():
            key = f"{i}.{name}"
            if key in entries:
                raise RuntimeError(f"duplicate AB profile key: {key}")
            entries[key] = module
    if not entries:
        raise RuntimeError("no transformer Linear layers found for AB profile")

    running: dict[str, torch.Tensor | None] = {k: None for k in entries}
    handles = []
    cache = {"tensor": None, "amax": None}

    def make_hook(key: str, expected_d: int):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"{key}: Linear pre-hook did not receive tensor activation")
            x = inputs[0]
            if int(x.shape[-1]) != int(expected_d):
                raise RuntimeError(
                    f"{key}: activation dim {x.shape[-1]} != in_features {expected_d}"
                )
            if x is cache["tensor"]:
                cur = cache["amax"]
            else:
                cur = x.detach().abs().reshape(-1, expected_d).amax(dim=0).to(torch.float32)
                cache["tensor"] = x
                cache["amax"] = cur
            old = running[key]
            if old is None:
                running[key] = cur.clone()
            else:
                torch.maximum(old, cur, out=old)
        return hook

    for key, module in entries.items():
        handles.append(module.register_forward_pre_hook(make_hook(key, int(module.in_features))))

    backbone = _pick_backbone(model)
    seen = 0
    try:
        with torch.inference_mode():
            for sample in calib_loader[:nsamples]:
                cache["tensor"] = None
                cache["amax"] = None
                input_ids = _extract_input_ids(sample).to(dev, non_blocking=True)
                try:
                    backbone(input_ids=input_ids, use_cache=False, return_dict=False)
                except TypeError:
                    try:
                        backbone(input_ids=input_ids, use_cache=False)
                    except TypeError:
                        backbone(input_ids)
                seen += 1
                if seen == 1 or seen % 8 == 0 or seen == nsamples:
                    logger.info(f"[AB Profile] calibration {seen}/{nsamples}")
    finally:
        for h in handles:
            h.remove()
        cache["tensor"] = None
        cache["amax"] = None

    if seen != nsamples:
        raise RuntimeError(f"AB calibration ended early: {seen}/{nsamples}")
    missing = [k for k, v in running.items() if v is None]
    if missing:
        raise RuntimeError(f"AB profile hooks missed {len(missing)} layers: {missing[:20]}")

    prof_layers: dict[str, dict[str, object]] = {}
    total_outliers = 0
    for key, module in entries.items():
        scores = running[key].detach().to(device="cpu", dtype=torch.float32).contiguous()  # type: ignore[union-attr]
        if not torch.isfinite(scores).all():
            raise RuntimeError(f"{key}: non-finite AB activation score")
        try:
            ranking = torch.argsort(scores, descending=True, stable=True).to(torch.int64)
        except TypeError:
            ranking = torch.argsort(scores, descending=True).to(torch.int64)
        mu = float(scores.mean().item())
        sigma = float(scores.std(unbiased=False).item())
        threshold = float(max(mu + sigma_thresh * sigma, rel_thresh * mu, abs_thresh))
        strict = torch.nonzero(scores > threshold, as_tuple=False).flatten().to(torch.int64)
        total_outliers += int(strict.numel())
        prof_layers[key] = {
            "in_features": int(module.in_features),
            "scores": scores,
            "ranking": ranking,
            "strict_outliers": strict,
            "score_mean": mu,
            "score_std": sigma,
            "threshold": threshold,
        }

    profile: dict[str, object] = {
        "version": PROFILE_VERSION,
        "sigma_thresh": float(sigma_thresh),
        "rel_thresh": float(rel_thresh),
        "abs_thresh": float(abs_thresh),
        "seqlen": int(seqlen),
        "nsamples": int(nsamples),
        "layers": prof_layers,
        "model_path": str(model_path),
        "dataset": str(dataset),
        "seed": int(seed),
    }
    logger.info(
        f"[AB Profile] complete: layers={len(prof_layers)}, strict_outliers={total_outliers}"
    )
    return profile


def load_or_build_activation_profile(
    model: nn.Module, calib_loader, dev: torch.device, *, path: str | Path,
    nsamples: int, seqlen: int, seed: int, model_path: str,
    dataset: str, force_rebuild: bool = False,
) -> tuple[dict[str, object], bool]:
    path = Path(path)
    if path.exists() and not force_rebuild:
        profile = load_activation_profile(path)
        for key, expected in {
            "nsamples": nsamples, "seqlen": seqlen, "seed": seed,
            "model_path": model_path, "dataset": dataset,
        }.items():
            if profile.get(key) != expected:
                raise ValueError(f"Profile {key} mismatch; run with --rebuild-profile")
        logger.info("Using activation profile: %s", path)
        return profile, True

    profile = build_activation_profile(
        model, calib_loader, dev, nsamples=nsamples, seqlen=seqlen,
        seed=seed, model_path=model_path, dataset=dataset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, path)
    logger.info("Saved activation profile: %s", path)
    return profile, False
