"""Random signs for the ranked rotation."""
import torch


def generate_block_random_signs(block_sizes, seed: int, *, device=None):
    signs = []
    cur = int(seed)
    for n in block_sizes:
        g = torch.Generator(device='cpu')
        g.manual_seed(cur)
        s = (torch.randint(0, 2, (int(n),), generator=g, dtype=torch.int8, device='cpu')
             .mul(2).sub(1))
        signs.append(s)
        cur += 1
    out = torch.cat(signs, dim=0)
    return out.to(device=device) if device is not None else out
