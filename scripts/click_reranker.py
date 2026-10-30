"""Build data for and train the type-gated click reranker.

  python scripts/click_reranker.py build --cache $CACHE --out $CACHE/click_reranker_train.pt
  python scripts/click_reranker.py train --data $CACHE/click_reranker_train.pt \
      --out $CKPT/click_reranker_s1 --seed 1

Training data: every training click step with at least two click candidates. All click
candidates whose centre lies inside the recorded element box are positives (listwise
multi-positive cross-entropy). At inference the reranker sees only the goal, the
candidate texts, flags and boxes; beta and the gate (both 1.0) were set on validation.
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gui_jepa.actions import build_candidate_actions  # noqa: E402
from gui_jepa.data.cache import load_manifest, iter_step_records, load_json_int_keys  # noqa: E402
from gui_jepa.rankers import AUX_DIM, DEFAULT_RERANKER_LM, ClickReranker, click_cand_aux  # noqa: E402


def cmd_build(args):
    man = load_manifest(args.cache)
    entries = man[args.split][:args.max_eps] if args.max_eps else man[args.split]
    texts = load_json_int_keys(args.texts or os.path.join(args.cache, "instr_texts.json"))
    cand_texts = load_json_int_keys(args.cand_texts or os.path.join(args.cache, "cand_texts.json"))
    out_steps = []
    n_click = n_kept = n_nopos = 0
    for ep, si, step in iter_step_records(entries):
        gt = step["gt"]
        if gt["type"] not in ("click", "long_press") or not gt.get("gt_bbox"):
            continue
        n_click += 1
        _, metas = build_candidate_actions(step)
        cb, cf = step["cand_bounds"], step["cand_flags"]
        ctexts = cand_texts.get(ep["eid"], [])
        ct = ctexts[si] if si < len(ctexts) else []
        w, h = step["screen_wh"]
        b = gt["gt_bbox"]
        click_js = [j for j, m in enumerate(metas) if m["kind"] == "click"]
        if len(click_js) < 2:
            continue
        texts_l, aux_l, pos_l = [], [], []
        for j in click_js:
            ci = metas[j]["cand"]
            px, py = metas[j]["cx"] * w, metas[j]["cy"] * h
            texts_l.append(ct[ci] if ci < len(ct) else "[empty]")
            aux_l.append(click_cand_aux(cb[ci], cf[ci]))
            pos_l.append(int(b[0] <= px <= b[2] and b[1] <= py <= b[3]))
        if sum(pos_l) == 0:
            n_nopos += 1
            continue
        goal = texts.get(ep["eid"], {}).get("goal", "")
        out_steps.append({"goal": goal, "texts": texts_l, "aux": aux_l, "pos": pos_l})
        n_kept += 1
    torch.save(out_steps, args.out)
    print(f"[build] split={args.split} click_steps={n_click} kept={n_kept} "
          f"no_positive={n_nopos} -> {args.out}", flush=True)


def cmd_train(args):
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = torch.load(args.data, weights_only=False)
    print(f"[train] {len(data)} click steps, seed={args.seed}", flush=True)
    model = ClickReranker(dev, model_name=args.model_name, dropout=args.dropout).to(dev)
    model.train()
    opt = torch.optim.AdamW([{"params": model.lm.parameters(), "lr": args.lr_lm},
                             {"params": model.head.parameters(), "lr": args.lr}],
                            weight_decay=args.wd)
    os.makedirs(args.out, exist_ok=True)
    step_n = 0
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        run_loss = run_n = 0
        for idx in order:
            d = data[idx]
            N = len(d["texts"])
            if N < 2:
                continue
            # cap the candidates per step for speed, always keeping every positive
            if args.cap and N > args.cap:
                pos_j = [i for i in range(N) if d["pos"][i]]
                neg_j = [i for i in range(N) if not d["pos"][i]]
                keep_neg = args.cap - len(pos_j)
                if keep_neg < 1:
                    sel = pos_j[:args.cap]
                else:
                    random.shuffle(neg_j)
                    sel = pos_j + neg_j[:keep_neg]
                sel.sort()
            else:
                sel = list(range(N))
            texts = [d["texts"][i] for i in sel]
            aux = torch.tensor([d["aux"][i] for i in sel], dtype=torch.float32)
            pos = torch.tensor([d["pos"][i] for i in sel], dtype=torch.bool, device=dev)
            if pos.sum() == 0:
                continue
            logits = model(d["goal"], texts, aux)
            loss = -(torch.logsumexp(logits[pos], 0) - torch.logsumexp(logits, 0))
            if args.margin > 0 and (~pos).any():
                gap = logits[pos].min() - logits[~pos].max()
                loss = loss + args.margin_w * F.relu(args.margin - gap)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss.detach())
            run_n += 1
            step_n += 1
            if step_n % 2000 == 0:
                print(f"[train] ep{epoch} step{step_n} loss {run_loss / max(run_n, 1):.4f}", flush=True)
                run_loss = run_n = 0
    model.lm.save_pretrained(os.path.join(args.out, "reranker_lm"))
    torch.save({"head": model.head.state_dict()}, os.path.join(args.out, "reranker_head.pt"))
    json.dump({"model_name": args.model_name, "aux_dim": AUX_DIM},
              open(os.path.join(args.out, "meta.json"), "w"))
    print(f"[train] done -> {args.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cache", required=True)
    b.add_argument("--texts", default=None)
    b.add_argument("--cand-texts", default=None)
    b.add_argument("--split", default="train")
    b.add_argument("--max-eps", type=int, default=0)
    b.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--model-name", default=DEFAULT_RERANKER_LM)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lr-lm", type=float, default=2e-5)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--cap", type=int, default=48)
    t.add_argument("--margin", type=float, default=0.0)
    t.add_argument("--margin-w", type=float, default=0.5)
    t.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cmd_build(args) if args.cmd == "build" else cmd_train(args)
