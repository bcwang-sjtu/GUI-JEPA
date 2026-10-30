"""Region-of-interest pooling over the 14x14 grid of ViT patch tokens."""
import torch

GRID = 14


def roi_pool(z: torch.Tensor, bounds, grid: int = GRID) -> torch.Tensor:
    """z: (grid*grid, D); bounds normalised (l, t, r, b) -> mean of the covered patches."""
    l, t, r, b = bounds
    c0, c1 = int(l * grid), min(int(r * grid) + 1, grid)
    r0, r1 = int(t * grid), min(int(b * grid) + 1, grid)
    c0, r0 = max(0, min(c0, grid - 1)), max(0, min(r0, grid - 1))
    c1, r1 = max(c0 + 1, c1), max(r0 + 1, r1)
    idx = [rr * grid + cc for rr in range(r0, r1) for cc in range(c0, c1)]
    return z[idx].mean(0)


def batch_roi(z: torch.Tensor, metas, cand_bounds, grid: int = GRID) -> torch.Tensor:
    """Per-candidate ROI features; global actions get the full-frame mean."""
    g = z.mean(0)
    out = []
    for m in metas:
        if m.get("cand") is not None:
            out.append(roi_pool(z, cand_bounds[m["cand"]].tolist(), grid))
        else:
            out.append(g)
    return torch.stack(out)
