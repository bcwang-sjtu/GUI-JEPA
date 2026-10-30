"""Action space: candidate enumeration and the 16-d action vector.

Every step offers one click per enumerated a11y element, one input_text per
editable element, and a fixed set of global actions (four scroll directions,
navigate_back, navigate_home, wait, open_app).
"""
import torch

ACTION_TYPES = ["click", "long_press", "scroll", "open_app", "input_text",
                "navigate_home", "navigate_back", "wait"]
SCROLL_DIRS = ["up", "down", "left", "right"]
A_DIM = 16

# Action kinds that receive a learned bias in the structural head.
KINDS = ["click", "input_text", "scroll", "navigate_back", "navigate_home",
         "wait", "open_app"]


def encode_action(atype: str, x: float = 0., y: float = 0., sdir: str = None):
    """16-d action vector: type one-hot (8) + normalised (x, y) + scroll direction (4)."""
    v = torch.zeros(A_DIM)
    if atype in ACTION_TYPES:
        v[ACTION_TYPES.index(atype)] = 1.
    v[8], v[9] = x, y
    if sdir in SCROLL_DIRS:
        v[10 + SCROLL_DIRS.index(sdir)] = 1.
    return v


def build_candidate_actions(step):
    """Returns (action_vecs (N, 16), metas) covering element and global actions."""
    cb, cf = step["cand_bounds"], step["cand_flags"]
    acts, metas = [], []
    for i in range(cb.size(0)):
        cx = float((cb[i, 0] + cb[i, 2]) / 2)
        cy = float((cb[i, 1] + cb[i, 3]) / 2)
        acts.append(encode_action("click", cx, cy))
        metas.append({"kind": "click", "cand": i, "cx": cx, "cy": cy})
        if cf[i, 2] > 0:  # editable -> input_text
            acts.append(encode_action("input_text", cx, cy))
            metas.append({"kind": "input_text", "cand": i, "cx": cx, "cy": cy})
    for d in SCROLL_DIRS:
        acts.append(encode_action("scroll", 0.5, 0.5, d))
        metas.append({"kind": "scroll", "dir": d})
    for g in ("navigate_back", "navigate_home", "wait", "open_app"):
        acts.append(encode_action(g))
        metas.append({"kind": g})
    return torch.stack(acts), metas


def gt_action_index(step, metas):
    """Index of the ground-truth candidate, or None if it is not enumerated."""
    gt = step["gt"]
    t = gt["type"]
    if t in ("click", "long_press"):
        if gt["cand_idx"] < 0:
            return None
        for j, m in enumerate(metas):
            if m["kind"] == "click" and m.get("cand") == gt["cand_idx"]:
                return j
        return None
    if t == "input_text":
        if gt["cand_idx"] >= 0:
            for j, m in enumerate(metas):
                if m["kind"] == "input_text" and m.get("cand") == gt["cand_idx"]:
                    return j
        js = [j for j, m in enumerate(metas) if m["kind"] == "input_text"]
        return js[0] if js else None
    if t == "scroll":
        for j, m in enumerate(metas):
            if m["kind"] == "scroll" and m.get("dir") == gt["scroll_dir"]:
                return j
        return None
    for j, m in enumerate(metas):
        if m["kind"] == t:
            return j
    return None


def kind_id(m):
    k = m["kind"]
    return KINDS.index(k) if k in KINDS else 0


def gt_action_vector(gt, screen_wh):
    """Action vector of a recorded action (used for the history and for dynamics pretraining)."""
    cx = cy = 0.
    sdir = gt["scroll_dir"] if gt["type"] == "scroll" else None
    if gt["gt_bbox"] and gt["type"] in ("click", "long_press"):
        w, h = screen_wh
        b = gt["gt_bbox"]
        cx, cy = (b[0] + b[2]) / 2 / max(w, 1), (b[1] + b[3]) / 2 / max(h, 1)
    return encode_action(gt["type"], cx, cy, sdir)
