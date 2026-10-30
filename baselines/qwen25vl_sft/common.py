"""Shared helpers for the same-protocol Qwen2.5-VL-7B reference (LoRA SFT).

Prompt, candidates and judge match the GUI-JEPA evaluation (scripts/train_policy.py):
same IDD test episodes and steps, the same enumerated candidates (cand_texts.json),
the same ground-truth action history, and the same step-accuracy rules
(click = chosen candidate centre inside the ground-truth box; input_text = normalised
exact match; scroll = direction; open_app = app name; long_press accepts a click).
"""
import json
import re

IMG_MIN_PIXELS = 64 * 28 * 28
IMG_MAX_PIXELS = 384 * 28 * 28  # at most 384 vision tokens

ACTION_SPEC = (
    'Respond with EXACTLY ONE action as a single-line JSON object, choosing one form:\n'
    '{"action_type":"click","target":<element index>}\n'
    '{"action_type":"input_text","text":"<text to type>"}\n'
    '{"action_type":"scroll","direction":"up|down|left|right"}\n'
    '{"action_type":"open_app","app_name":"<app name>"}\n'
    '{"action_type":"navigate_back"}\n'
    '{"action_type":"navigate_home"}\n'
    '{"action_type":"wait"}'
)


def clean_label(s: str, cap: int = 90) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())[:cap]


def hist_text_of(steps, cand_texts_e, si, k=4):
    """Ground-truth action history text (same as gui_jepa.text_utils.history_text)."""
    out = []
    ct = cand_texts_e or []
    for pi in range(max(0, si - k), si):
        g = steps[pi]["gt"]
        t = g["type"]
        if t in ("click", "long_press") and g["cand_idx"] >= 0 \
                and pi < len(ct) and g["cand_idx"] < len(ct[pi]):
            out.append(f"{t} [{ct[pi][g['cand_idx']].split(' | ')[0]}]")
        elif t == "input_text":
            out.append(f"type [{(g['text'] or '')[:24]}]")
        elif t == "open_app":
            out.append(f"open [{g['app_name']}]")
        else:
            out.append(t)
    return "; ".join(out)


def build_prompt(goal, hist, labels, step_instr=None):
    """User text next to the screenshot. step_instr=None: AC-High; otherwise AC-Low."""
    if step_instr is None:
        lines = [
            "You are an agent controlling an Android phone. Given the goal, the current "
            "screenshot, the actions already done, and the list of interactive UI elements "
            "on screen, output the single best next action.",
            "",
            f"Goal: {goal}",
            f"Actions done so far: {hist if hist else '(none)'}",
        ]
    else:
        lines = [
            "You are an agent controlling an Android phone. Given the goal, the instruction "
            "for the current step, the current screenshot, the actions already done, and the "
            "list of interactive UI elements on screen, output the single best next action.",
            "",
            f"Goal: {goal}",
            f"Current step instruction: {step_instr.strip() if step_instr and step_instr.strip() else '(none)'}",
            f"Actions done so far: {hist if hist else '(none)'}",
        ]
    lines += ["", "Interactive UI elements on the current screen (index: description):"]
    for i, lb in enumerate(labels):
        lines.append(f"{i}: {clean_label(lb)}")
    lines += ["", ACTION_SPEC]
    return "\n".join(lines)


def format_target(gt):
    """Ground-truth action -> canonical JSON target string."""
    t = gt["type"]
    sep = (",", ":")
    if t in ("click", "long_press"):
        return json.dumps({"action_type": "click", "target": int(gt["cand_idx"])}, separators=sep)
    if t == "input_text":
        return json.dumps({"action_type": "input_text", "text": gt.get("text") or ""}, separators=sep)
    if t == "scroll":
        return json.dumps({"action_type": "scroll", "direction": gt.get("scroll_dir") or "down"},
                          separators=sep)
    if t == "open_app":
        return json.dumps({"action_type": "open_app", "app_name": gt.get("app_name") or ""},
                          separators=sep)
    return json.dumps({"action_type": t}, separators=sep)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_action(text):
    """Model output -> action dict (JSON first, regex fallback)."""
    if not text:
        return {"action_type": None}
    m = _JSON_RE.search(text)
    frag = m.group(0) if m else text
    try:
        d = json.loads(frag)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    d = {}
    for key, pat, cast in (("action_type", r'action_type"?\s*[:=]\s*"?(\w+)', str),
                           ("target", r'target"?\s*[:=]\s*"?(\d+)', int),
                           ("direction", r'direction"?\s*[:=]\s*"?(\w+)', str),
                           ("text", r'text"?\s*[:=]\s*"([^"]*)"', str),
                           ("app_name", r'app_name"?\s*[:=]\s*"([^"]*)"', str)):
        mm = re.search(pat, text)
        if mm:
            d[key] = cast(mm.group(1))
    return d


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def cand_center(cand_bounds_row):
    l, t, r, b = [float(x) for x in cand_bounds_row]
    return (l + r) / 2.0, (t + b) / 2.0


def judge(pred, gt, cand_bounds, screen_wh):
    """Returns (correct, type_correct, predicted_kind)."""
    gt_type = gt["type"]
    pred_kind = pred.get("action_type")
    t_ok = (pred_kind == gt_type) or (gt_type == "long_press" and pred_kind == "click")
    if not t_ok:
        return False, False, pred_kind
    if gt_type in ("click", "long_press"):
        if not gt.get("gt_bbox"):
            return False, True, pred_kind
        tgt = pred.get("target")
        if tgt is None or not (0 <= int(tgt) < cand_bounds.shape[0]):
            return False, True, pred_kind
        cx, cy = cand_center(cand_bounds[int(tgt)])
        w, h = screen_wh
        b = gt["gt_bbox"]
        return bool(b[0] <= cx * w <= b[2] and b[1] <= cy * h <= b[3]), True, pred_kind
    if gt_type == "scroll":
        return pred.get("direction") == gt.get("scroll_dir"), True, pred_kind
    if gt_type == "input_text":
        return _norm(pred.get("text")) == _norm(gt.get("text")), True, pred_kind
    if gt_type == "open_app":
        return _norm(pred.get("app_name")) == _norm(gt.get("app_name")), True, pred_kind
    return True, True, pred_kind  # navigate_back / navigate_home / wait


def load_texts(cache, cand_texts_name="cand_texts.json"):
    import os
    cand_texts = json.load(open(os.path.join(cache, cand_texts_name)))
    instr = json.load(open(os.path.join(cache, "instr_texts.json")))
    ct_get = lambda e: cand_texts.get(str(e)) or cand_texts.get(e) or []
    it_get = lambda e: instr.get(str(e)) or instr.get(e) or {}
    return ct_get, it_get
