"""Type-gated click reranker.

Runs only when the base policy picks a click. It re-scores the click candidates
of the step from the goal text and each candidate's text, flags and box geometry;
the final click score is base_score + beta * zscore(reranker_score), and the pick
changes only when the reranker margin exceeds a confidence gate.
"""
import json
import os

import torch
import torch.nn as nn

AUX_DIM = 11
DEFAULT_RERANKER_LM = "sentence-transformers/all-MiniLM-L12-v2"


def click_cand_aux(cb_row, cf_row):
    """cb_row: normalised [x0, y0, x1, y1]; cf_row: [clickable, long_clickable, editable, scrollable].
    Returns the 11-d feature [cx, cy, w, h, area, x0, y0] + flags."""
    x0, y0, x1, y1 = float(cb_row[0]), float(cb_row[1]), float(cb_row[2]), float(cb_row[3])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
    fl = [float(cf_row[0]), float(cf_row[1]), float(cf_row[2]), float(cf_row[3])]
    return [cx, cy, w, h, w * h, x0, y0] + fl


class ClickReranker(nn.Module):
    """MiniLM cross-encoder over (goal, candidate text) fused with candidate geometry and flags."""

    def __init__(self, device, model_name=DEFAULT_RERANKER_LM, aux_dim=AUX_DIM, dropout=0.1):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModel.from_pretrained(model_name)
        hidden = self.lm.config.hidden_size
        self.aux_dim = aux_dim
        self.head = nn.Sequential(
            nn.Linear(hidden + aux_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.device = device

    def forward(self, goal_text, cand_texts, aux):
        """goal_text: str; cand_texts: list[str] (N); aux: (N, aux_dim). Returns (N,) scores."""
        g = goal_text[:200] if goal_text else "[empty]"
        enc = self.tok([g] * len(cand_texts),
                       [(c or "[empty]")[:100] for c in cand_texts],
                       padding=True, truncation=True, max_length=96,
                       return_tensors="pt").to(self.device)
        cls = self.lm(**enc).last_hidden_state[:, 0]
        return self.head(torch.cat([cls, aux.to(self.device).float()], -1)).squeeze(-1)


def load_reranker(ckpt_dir, device):
    from transformers import AutoModel
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    m = ClickReranker(device, model_name=meta["model_name"],
                      aux_dim=meta.get("aux_dim", AUX_DIM), dropout=0.0)
    m.lm = AutoModel.from_pretrained(os.path.join(ckpt_dir, "reranker_lm")).to(device)
    sd = torch.load(os.path.join(ckpt_dir, "reranker_head.pt"), map_location=device)
    m.head.load_state_dict(sd["head"] if isinstance(sd, dict) and "head" in sd else sd)
    m.to(device).eval()
    return m


@torch.no_grad()
def rerank_click(reranker, goal_text, scores, metas, argmax_idx, step, step_cand_texts,
                 beta=1.0, gate=1.0):
    """Returns the index of the final click (unchanged if the gate does not fire)."""
    cjs = [j for j, mm in enumerate(metas) if mm["kind"] == "click"]
    if len(cjs) < 2 or argmax_idx not in cjs:
        return argmax_idx
    cb, cf = step["cand_bounds"], step["cand_flags"]
    texts, aux = [], []
    for j in cjs:
        ci = metas[j]["cand"]
        texts.append(step_cand_texts[ci] if ci < len(step_cand_texts) else "[empty]")
        aux.append(click_cand_aux(cb[ci], cf[ci]))
    dev = scores.device
    r = reranker(goal_text, texts, torch.tensor(aux, device=dev)).detach().float()
    rz = (r - r.mean()) / (r.std() + 1e-6)
    base = scores.detach().float()[torch.tensor(cjs, device=dev)]
    new_local = int((base + beta * rz).argmax())
    base_local = cjs.index(argmax_idx)
    if float(rz[new_local] - rz[base_local]) > gate:
        return cjs[new_local]
    return argmax_idx
