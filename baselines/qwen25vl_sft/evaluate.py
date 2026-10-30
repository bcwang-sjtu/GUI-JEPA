"""Evaluate the Qwen2.5-VL-7B reference on AndroidControl IDD with the GUI-JEPA judge.

  python baselines/qwen25vl_sft/evaluate.py --model Qwen/Qwen2.5-VL-7B-Instruct \
      --adapter $SFT/lora_high/final --cache $CACHE --prep $SFT/prep --images $IMAGES \
      --mode high --out $SFT/eval_high_idd.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_prompt, hist_text_of, judge, parse_action, cand_center, load_texts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for zero-shot)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--mode", choices=("high", "low"), default="high")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    ap.add_argument("--click-topk", action="store_true",
                    help="also report click top-1/top-5 recall by candidate likelihood")
    ap.add_argument("--topk-chunk", type=int, default=8)
    args = ap.parse_args()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels,
                                         max_pixels=args.max_pixels)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda").eval()

    entries = json.load(open(os.path.join(args.prep, "test_idd_entries.json")))
    ct_get, it_get = load_texts(args.cache)

    def chat(prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def gen_action(img, prompt):
        enc = proc(text=[chat(prompt)], images=[img], return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=24, do_sample=False)
        return proc.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def click_topk(img, prompt, n_cand, k=5):
        """Rank candidate indices by the likelihood of '{"action_type":"click","target":i}'."""
        prefix = chat(prompt) + '{"action_type":"click","target":'
        scores = [None] * n_cand
        for c0 in range(0, n_cand, args.topk_chunk):
            chunk = list(range(c0, min(n_cand, c0 + args.topk_chunk)))
            enc = proc(text=[prefix + f"{i}}}" for i in chunk], images=[img] * len(chunk),
                       return_tensors="pt", padding=True).to("cuda")
            logits = model(**enc).logits
            ids = enc["input_ids"]
            for bi, i in enumerate(chunk):
                L = len(proc.tokenizer(f"{i}}}", add_special_tokens=False)["input_ids"])
                end = int(torch.nonzero(enc["attention_mask"][bi]).squeeze(-1)[-1])
                lp = sum(torch.log_softmax(logits[bi, p - 1], -1)[ids[bi, p]].item()
                         for p in range(end - L + 1, end + 1))
                scores[i] = lp / max(L, 1)
        return sorted(range(n_cand), key=lambda i: -scores[i])[:k]

    tot, cor, type_cor = Counter(), Counter(), Counter()
    g_tot = g_cor = n_eval = 0
    top1 = top5 = n_scored = 0
    dump, t0 = [], time.time()
    for ei, e in enumerate(entries):
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
            gt_type = gt["type"]
            labels = ct_e[si] if si < len(ct_e) else []
            imgp = os.path.join(args.images, str(eid), f"s{si}.jpg")
            if not os.path.exists(imgp):  # counted as an error, like every evaluated step
                tot[gt_type] += 1
                n_eval += 1
                continue
            img = Image.open(imgp).convert("RGB")
            hist = hist_text_of(steps, ct_e, si)
            if args.mode == "low":
                instrs = it_get(eid).get("instrs", [])
                prompt = build_prompt(goal, hist, labels, step_instr=instrs[si] if si < len(instrs) else "")
            else:
                prompt = build_prompt(goal, hist, labels)
            raw = gen_action(img, prompt)
            pred = parse_action(raw)
            ok, t_ok, _ = judge(pred, gt, step["cand_bounds"], step["screen_wh"])
            tot[gt_type] += 1
            n_eval += 1
            type_cor[gt_type] += int(t_ok)
            cor[gt_type] += int(ok)
            if gt_type in ("click", "long_press") and gt.get("gt_bbox"):
                g_tot += 1
                g_cor += int(ok)
                if args.click_topk and step["cand_bounds"].shape[0] > 0:
                    cb, (w, h), b = step["cand_bounds"], step["screen_wh"], gt["gt_bbox"]
                    correct = {i for i in range(cb.shape[0])
                               if b[0] <= cand_center(cb[i])[0] * w <= b[2]
                               and b[1] <= cand_center(cb[i])[1] * h <= b[3]}
                    if correct:
                        ranked = click_topk(img, prompt, cb.shape[0])
                        n_scored += 1
                        top1 += int(ranked[0] in correct)
                        top5 += int(bool(set(ranked) & correct))
            dump.append({"eid": eid, "si": si, "gt_type": gt_type, "raw": raw[:80],
                         "pred_kind": pred.get("action_type"), "ok": bool(ok), "t_ok": bool(t_ok)})
        if ei % 50 == 0:
            print(f"[eval] {n_eval} steps, {(time.time() - t0) / 60:.1f}min, "
                  f"acc {100 * sum(cor.values()) / max(n_eval, 1):.2f}", flush=True)

    n = sum(tot.values())
    res = {"mode": args.mode, "adapter": args.adapter, "n_steps": n,
           "step_acc": round(100 * sum(cor.values()) / max(n, 1), 2),
           "type_acc": round(100 * sum(type_cor.values()) / max(n, 1), 2),
           "grounding_acc_clicks": round(100 * g_cor / max(g_tot, 1), 2),
           "per_type": {k: {"n": tot[k], "step_acc": round(100 * cor[k] / max(tot[k], 1), 1),
                            "type_acc": round(100 * type_cor[k] / max(tot[k], 1), 1)} for k in tot}}
    if args.click_topk:
        res["click_topk"] = {"n_scored": n_scored, "top1_recall": round(100 * top1 / max(n_scored, 1), 2),
                             "top5_recall": round(100 * top5 / max(n_scored, 1), 2)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"result": res, "dump": dump}, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
