"""Pretrain the residual latent predictor on AndroidControl transitions (policy initialisation).

Given the screen latent z_t and the recorded action (16-d action vector + ROI feature of
the acted-on element), predict the next screen latent:  z_{t+1} ~ z_t + Delta(z_t, a).
The resulting predictor.pt initialises the policy's predictor (scripts/train_policy.py
--init-predictor). The paper shows that a randomly initialised predictor does as well.

  python scripts/pretrain_dynamics.py --cache $CACHE --out $CKPT/dynamics
"""
import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gui_jepa.actions import A_DIM, gt_action_vector  # noqa: E402
from gui_jepa.data.cache import load_manifest, iter_step_records  # noqa: E402
from gui_jepa.models import LatentPredictor  # noqa: E402
from gui_jepa.roi import roi_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pred-depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    train_e = load_manifest(args.cache)["train"]
    predictor = LatentPredictor(embed_dim=args.embed_dim, action_dim=A_DIM + args.embed_dim,
                                num_patches_side=args.grid, depth=args.pred_depth).to(dev)
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    step_n = 0
    for epoch in range(args.epochs):
        buf = []
        for ep, si, step in iter_step_records(train_e, shuffle=True):
            fz = ep["frames_z"]
            if si + 1 >= fz.size(0):
                continue
            gt = step["gt"]
            z_t = fz[si].float()
            roi = torch.zeros(args.embed_dim)
            if gt["gt_bbox"] and gt["type"] in ("click", "long_press"):
                w, h = step["screen_wh"]
                b = gt["gt_bbox"]
                roi = roi_pool(z_t, [b[0] / max(w, 1), b[1] / max(h, 1),
                                     b[2] / max(w, 1), b[3] / max(h, 1)], args.grid)
            a = torch.cat([gt_action_vector(gt, step["screen_wh"]), roi])
            buf.append((z_t, a, fz[si + 1].float()))
            if len(buf) == args.batch_size:
                z = torch.stack([x[0] for x in buf]).to(dev)
                a_ = torch.stack([x[1] for x in buf]).to(dev)
                zn = torch.stack([x[2] for x in buf]).to(dev)
                loss = F.smooth_l1_loss(z + predictor(z, a_), zn)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step_n += 1
                if step_n % 100 == 0:
                    print(f"[dynamics] ep{epoch} step{step_n} loss {loss.item():.4f}", flush=True)
                buf = []
    torch.save(predictor.state_dict(), os.path.join(args.out, "predictor.pt"))
    print(f"[dynamics] saved {args.out}/predictor.pt", flush=True)


if __name__ == "__main__":
    main()
