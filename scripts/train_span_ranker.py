"""Train the goal-source span ranker that fills the input_text argument.

The typed text is almost always a verbatim span of the goal. We enumerate word
n-gram spans of the goal and train a MiniLM-L6 cross-encoder ("goal [SEP] span")
with a multi-positive listwise cross-entropy (positives = spans that match the
recorded text). The ranker reads the goal, which is the text available in AC-High.

  python scripts/train_span_ranker.py --cache $CACHE --out $CKPT/span_ranker
"""
import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gui_jepa.data.cache import load_manifest, iter_step_records  # noqa: E402
from gui_jepa.models import CrossEncoderMatcher, DEFAULT_SPAN_RANKER  # noqa: E402
from gui_jepa.text_utils import extract_type_text, text_match, span_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--texts", default=None, help="default: <cache>/instr_texts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-name", default=DEFAULT_SPAN_RANKER)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-lm", type=float, default=2e-5)
    ap.add_argument("--span-max-words", type=int, default=16)
    ap.add_argument("--span-cap", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    texts_path = args.texts or os.path.join(args.cache, "instr_texts.json")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    man = load_manifest(args.cache)
    with open(texts_path) as f:
        texts = {int(k): v for k, v in json.load(f).items()}

    def gather(entries):
        """(goal, step instruction, recorded text) for every input_text step."""
        data = []
        for ep, si, step in iter_step_records(entries):
            gt = step["gt"]
            if gt["type"] != "input_text" or not gt["text"]:
                continue
            tinfo = texts.get(ep["eid"], {})
            instrs = tinfo.get("instrs", [])
            data.append((tinfo.get("goal", ""), instrs[si] if si < len(instrs) else "", gt["text"]))
        return data

    train_d = gather(man["train"])
    val_d = gather(man["validation"])
    print(f"[span] train {len(train_d)} / validation {len(val_d)} input_text steps", flush=True)

    ranker = CrossEncoderMatcher(dev, args.model_name).to(dev)
    opt = torch.optim.AdamW(ranker.parameters(), lr=args.lr_lm)
    spans = lambda s: span_candidates(s, max_words=args.span_max_words, cap=args.span_cap)

    step_n = 0
    for epoch in range(args.epochs):
        random.shuffle(train_d)
        for goal, _, gt_text in train_d:
            cands = spans(goal)
            pos = [i for i, c in enumerate(cands) if text_match(c, gt_text)]
            if not pos:
                continue
            scores = ranker(goal, cands)
            pos_t = torch.tensor(pos, device=dev)
            loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[pos_t], dim=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_n += 1
            if step_n % 500 == 0:
                print(f"[span] ep{epoch} step{step_n} loss {loss.item():.3f}", flush=True)
    ranker.lm.save_pretrained(os.path.join(args.out, "ranker_lm"))
    torch.save(ranker.head.state_dict(), os.path.join(args.out, "ranker_head.pt"))

    @torch.no_grad()
    def evaluate(use_goal):
        """Argument accuracy on validation input_text steps, with the action type given."""
        ranker.eval()
        heur = learned = ceil = n = 0
        for goal, instr, gt_text in val_d:
            src = goal if use_goal else (instr or goal)
            n += 1
            heur += int(text_match(extract_type_text(src, goal), gt_text))
            cands = spans(src)
            ceil += int(any(text_match(c, gt_text) for c in cands))
            if cands:
                learned += int(text_match(cands[int(ranker(src, cands).argmax())], gt_text))
        return {"n": n, "heuristic_acc": round(100 * heur / max(n, 1), 2),
                "learned_acc": round(100 * learned / max(n, 1), 2),
                "span_ceiling": round(100 * ceil / max(n, 1), 2)}

    res = {"goal": evaluate(True), "step_instruction": evaluate(False)}
    print(json.dumps(res, indent=1), flush=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
