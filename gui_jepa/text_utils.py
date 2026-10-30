"""Text helpers: argument heuristics, span enumeration, history text and string matching."""
import random
import re


def extract_type_text(instr: str, goal: str) -> str:
    """Rule-based fallback for the typed text (used only when no span is available)."""
    for s in (instr, goal):
        if not s:
            continue
        m = re.search(r"['\"]([^'\"]+)['\"]", s)
        if m:
            return m.group(1)
        m = re.search(r"(?:type|enter|input|search for|search|write|fill in|fill)\s+(?:in\s+)?(.+?)(?:\s+(?:in|into|on|at)\s+the\b.*)?$",
                      s, re.IGNORECASE)
        if m:
            return m.group(1).strip(" .")
    return (instr or goal).strip(" .")


def predict_app(instr: str, goal: str, vocab) -> str:
    """open_app argument: the longest training-vocabulary app name mentioned in the text."""
    src = f" {instr} {goal} ".lower()
    best = None
    for app in vocab:  # vocab sorted by training frequency
        if app.lower() in src:
            if best is None or len(app) > len(best):
                best = app
    return best or (vocab[0] if vocab else "")


def text_match(pred: str, gt: str) -> bool:
    """Whitespace- and case-normalised exact match."""
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return norm(pred) == norm(gt)


def span_candidates(s, max_words=16, cap=800):
    """All word n-gram spans of `s` (up to `max_words` words), shuffled and capped.

    A quoted substring, if present, is placed first.
    """
    s = (s or "").strip()
    words = s.split()
    spans = set()
    for i in range(len(words)):
        for n in range(1, min(max_words, len(words) - i) + 1):
            spans.add(" ".join(words[i:i + n]).strip(" .,'\""))
    m = re.search(r"['\"]([^'\"]+)['\"]", s)
    out = list(spans)
    random.shuffle(out)
    if m:
        out = [m.group(1)] + out
    return out[:cap]


def history_text(steps, step_cand_texts, si, k=4):
    """Textual ground-truth action history of the last `k` steps (matcher query suffix)."""
    out = []
    ct = step_cand_texts or []
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
