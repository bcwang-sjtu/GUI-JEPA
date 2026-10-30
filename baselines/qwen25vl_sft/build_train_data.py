"""Build the SFT jsonl (one line per step: image, prompt, target) for the training episodes.

  python baselines/qwen25vl_sft/build_train_data.py --cache $CACHE --prep $SFT/prep \
      --images $IMAGES --mode high --out $SFT/train_high.jsonl
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_prompt, format_target, hist_text_of, load_texts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--mode", choices=("high", "low"), default="high")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    entries = json.load(open(os.path.join(args.prep, "train_entries.json")))
    ct_get, it_get = load_texts(args.cache)
    n_samp = n_skip = 0
    stats = {}
    with open(args.out, "w") as fo:
        for e in entries:
            eid = int(e["eid"])
            try:
                ep = torch.load(e["path"], map_location="cpu", weights_only=False)
            except Exception:
                continue
            ct_e = ct_get(eid)
            goal = it_get(eid).get("goal", ep.get("goal", ""))
            steps = ep["steps"]
            for si, step in enumerate(steps):
                gt = step["gt"]
                imgp = os.path.join(args.images, str(eid), f"s{si}.jpg")
                labels = ct_e[si] if si < len(ct_e) else []
                if not os.path.exists(imgp) or not labels:
                    n_skip += 1
                    continue
                if gt["type"] in ("click", "long_press") and not (0 <= gt["cand_idx"] < len(labels)):
                    n_skip += 1
                    continue
                hist = hist_text_of(steps, ct_e, si)
                if args.mode == "low":
                    instrs = it_get(eid).get("instrs", [])
                    prompt = build_prompt(goal, hist, labels,
                                          step_instr=instrs[si] if si < len(instrs) else "")
                else:
                    prompt = build_prompt(goal, hist, labels)
                fo.write(json.dumps({"image": imgp, "prompt": prompt, "target": format_target(gt),
                                     "eid": eid, "si": si, "gt_type": gt["type"]}) + "\n")
                n_samp += 1
                stats[gt["type"]] = stats.get(gt["type"], 0) + 1
    print(f"[build] mode={args.mode}: {n_samp} samples, {n_skip} skipped -> {args.out}")
    print("[build] per type:", stats)


if __name__ == "__main__":
    main()
