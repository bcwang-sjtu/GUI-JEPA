"""Access to the per-episode feature cache written by scripts/preprocess/build_cache.py.

Each episode file (ep_<eid>.pt) holds:
  frames_z   (F, 196, 1024) fp16  frozen JEPA patch tokens per screenshot
  goal_emb   (512,)         fp16  CLIP embedding of the goal
  instr_embs (S, 512)       fp16  CLIP embeddings of the step instructions
  steps: per step {cand_bounds (Nc, 4) normalised, cand_flags (Nc, 4),
                   gt {type, cand_idx, scroll_dir, text, app_name, gt_bbox}, screen_wh}
"""
import json
import os
import random

import torch

SUBSPLITS = ("IDD", "app_unseen", "task_unseen", "category_unseen")


def load_manifest(cache):
    with open(os.path.join(cache, "manifest.json")) as f:
        return json.load(f)


def iter_step_records(entries, shuffle=False):
    order = list(entries)
    if shuffle:
        random.shuffle(order)
    for e in order:
        try:
            ep = torch.load(e["path"], map_location="cpu", weights_only=False)
        except Exception:
            continue
        for si, step in enumerate(ep["steps"]):
            yield ep, si, step


def select_eval_episodes(manifest, split="test", subsplit="IDD", subsplits_json=None):
    """Episodes to evaluate.

    split="validation" returns the validation split. For split="test",
    subsplit="IDD" uses the IDD tag of the manifest (IDD episodes that also belong
    to an OOD subsplit are counted as OOD), "full" returns the whole test set, and
    the OOD subsplits are read from the official test_subsplits.json.
    """
    if split == "validation":
        return manifest["validation"]
    if subsplit in ("full", "all"):
        return manifest["test"]
    if subsplit == "IDD":
        return [e for e in manifest["test"] if e.get("subsplit") == "IDD"]
    if subsplits_json is None:
        raise ValueError("--subsplits-json is required for OOD subsplits")
    with open(subsplits_json) as f:
        ids = {int(x) for x in json.load(f)[subsplit]}
    return [e for e in manifest["test"] if e["eid"] in ids]


def load_json_int_keys(path):
    with open(path) as f:
        return {k if not k.lstrip("-").isdigit() else int(k): v for k, v in json.load(f).items()}
