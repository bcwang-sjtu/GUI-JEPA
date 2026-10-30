"""Encode AndroidControl once with the frozen JEPA encoder and cache per-episode tensors.

Writes <out>/ep_<eid>.pt (see gui_jepa/data/cache.py for the format) and
<out>/manifest.json with the train / validation / test episodes; test episodes carry
their subsplit tag from test_subsplits.json. Policy training never runs the backbone.

  python scripts/preprocess/build_cache.py --shards "$AC/android_control-*" \
      --splits-dir $AC --encoder-ckpt $CKPT/encoder/encoder_1.pt --out $CACHE
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gui_jepa.clip_text import CLIPTextEncoder  # noqa: E402
from gui_jepa.data.androidcontrol import (iter_episodes, episode_steps,  # noqa: E402
                                          extract_candidates, match_point_to_candidate)
from gui_jepa.models import load_encoder, extract_patch_tokens  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(png_or_pil, size=224):
    im = png_or_pil if isinstance(png_or_pil, Image.Image) else Image.open(io.BytesIO(png_or_pil))
    im = im.convert("RGB").resize((size, size))
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def encode_frames(vit, pngs, device, batch=32):
    imgs = [preprocess_image(p) for p in pngs]
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.stack(imgs[i:i + batch]).to(device)
        out.append(extract_patch_tokens(vit, x).half().cpu())
    return torch.cat(out, 0)  # (F, 196, 1024)


def parse_gt(st, cands):
    a = st.action
    at = a.get("action_type")
    gt = {"type": at, "cand_idx": -1, "scroll_dir": None, "text": None,
          "app_name": None, "gt_bbox": None}
    if at in ("click", "long_press"):
        x, y = a.get("x"), a.get("y")
        if x is not None:
            idx = match_point_to_candidate(cands, x, y)
            if idx is not None:
                gt["cand_idx"] = idx
                gt["gt_bbox"] = list(cands[idx].bounds)
    elif at == "scroll":
        gt["scroll_dir"] = a.get("direction")
    elif at == "input_text":
        gt["text"] = a.get("text")
    elif at == "open_app":
        gt["app_name"] = a.get("app_name")
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob of AndroidControl TFRecord shards")
    ap.add_argument("--splits-dir", required=True, help="dir with splits.json + test_subsplits.json")
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-episodes", type=int, default=0, help="0 = all training episodes")
    ap.add_argument("--max-candidates", type=int, default=80)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.splits_dir, "splits.json")) as f:
        split_of = {int(eid): name for name, ids in json.load(f).items() for eid in ids}
    with open(os.path.join(args.splits_dir, "test_subsplits.json")) as f:
        subsplit_of = {int(eid): name for name, ids in json.load(f).items() for eid in ids}

    dev = torch.device(args.device)
    vit = load_encoder(args.encoder_ckpt, device=dev)
    clip = CLIPTextEncoder(device=dev)

    manifest = {"train": [], "validation": [], "test": []}
    n_train = 0
    for sf in sorted(glob.glob(args.shards)):
        for ex in iter_episodes(sf):
            steps = episode_steps(ex)
            if not steps:
                continue
            eid = steps[0].episode_id
            split = split_of.get(eid)
            if split is None:
                continue
            if split == "train" and args.train_episodes and n_train >= args.train_episodes:
                continue
            out_path = os.path.join(args.out, f"ep_{eid}.pt")
            entry = {"eid": eid, "path": out_path, "subsplit": subsplit_of.get(eid)}
            if not os.path.exists(out_path):
                pngs = [st.screenshot_png for st in steps]
                shots = ex.get("screenshots", [])
                if len(shots) > len(steps):
                    pngs.append(shots[len(steps)])  # final frame (dynamics target)
                try:
                    frames_z = encode_frames(vit, pngs, dev)
                except Exception as e:
                    print(f"[warn] encoding failed for episode {eid}: {e}")
                    continue
                goal_emb = clip.encode([steps[0].goal]).half().cpu()[0]
                instr_embs = clip.encode([st.step_instruction or steps[0].goal
                                          for st in steps]).half().cpu()
                step_recs = []
                for st in steps:
                    cands = extract_candidates(st.forest_bytes, st.screen_w, st.screen_h,
                                               max_candidates=args.max_candidates)
                    w, h = max(st.screen_w, 1), max(st.screen_h, 1)
                    if cands:
                        cb = torch.tensor([[c.bounds[0] / w, c.bounds[1] / h,
                                            c.bounds[2] / w, c.bounds[3] / h] for c in cands])
                        cf = torch.tensor([[c.clickable, c.long_clickable, c.editable, c.scrollable]
                                           for c in cands], dtype=torch.float32)
                    else:
                        cb, cf = torch.zeros(0, 4), torch.zeros(0, 4)
                    step_recs.append({"cand_bounds": cb, "cand_flags": cf,
                                      "gt": parse_gt(st, cands),
                                      "screen_wh": (st.screen_w, st.screen_h)})
                torch.save({"eid": eid, "goal": steps[0].goal, "frames_z": frames_z,
                            "goal_emb": goal_emb, "instr_embs": instr_embs,
                            "steps": step_recs}, out_path)
            manifest[split].append(entry)
            n_train += int(split == "train")
            n_done = sum(len(v) for v in manifest.values())
            if n_done % 200 == 0:
                print(f"[cache] {n_done} episodes (train {n_train}, test {len(manifest['test'])})",
                      flush=True)
                with open(os.path.join(args.out, "manifest.json"), "w") as f:
                    json.dump(manifest, f)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(f"[cache] done: train={n_train} validation={len(manifest['validation'])} "
          f"test={len(manifest['test'])}")


if __name__ == "__main__":
    main()
