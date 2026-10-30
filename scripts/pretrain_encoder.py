"""Stage 1: JEPA pretraining of the GUI encoder on unlabelled screenshots.

Paper recipe: ImageNet-initialised ViT-L/16 at 224 px, 189K screenshots
(GUI-Odyssey mobile + VideoCUA desktop frames, one folder tree), random 50% patch
masking, EMA 0.996, batch 96, 2 epochs, peak LR 5e-5 with warm-up. The policy uses
the checkpoint after the second epoch (encoder_1.pt).

  python scripts/pretrain_encoder.py --data $PRETRAIN_SCREENSHOTS --out $CKPT/encoder
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gui_jepa.data.screenshots import ScreenshotDataset  # noqa: E402
from gui_jepa.pretrain import IJEPA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root folder of pretraining screenshots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="vit_large_patch16_224")
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--no-imagenet-init", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    ds = ScreenshotDataset(args.data)
    print(f"[encoder] {len(ds)} screenshots", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    model = IJEPA(backbone_name=args.backbone, pretrained_backbone=not args.no_imagenet_init,
                  embed_dim=args.embed_dim, ema_m=args.ema).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)))
    scaler = torch.amp.GradScaler("cuda")

    step, t0, log = 0, time.time(), []
    for epoch in range(args.epochs):
        for x in dl:
            x = x.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model(x, mask_ratio=args.mask_ratio)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            model.momentum_update()
            step += 1
            if step % args.log_every == 0:
                print(f"[encoder] step={step} epoch={epoch} loss={metrics['loss']:.4f} "
                      f"tgt_std={metrics['tgt_std']:.3f} lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
                log.append({"step": step, "epoch": epoch, **metrics})
        # the student encoder is what the policy uses
        torch.save(model.student.state_dict(), os.path.join(args.out, f"encoder_{epoch}.pt"))
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f)
    print(f"[encoder] done in {time.time() - t0:.0f}s ({step} steps)", flush=True)


if __name__ == "__main__":
    main()
