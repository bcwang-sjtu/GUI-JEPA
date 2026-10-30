"""Per-step latency of the Qwen2.5-VL-7B reference with HF transformers (batch 1).

Same IDD steps (manifest order), prompt, pixel budget and greedy decoding as evaluate.py.
Configurations: base (zero-shot), sft_lora_unmerged (PEFT adapter) and sft_lora_merged
(adapter merged into the weights, the deployment setting used for the speed ratios).
Timed: chat template + processor + host-to-device copy + generate. Screenshot capture,
a11y retrieval and JPEG decoding are excluded.

  python baselines/qwen25vl_sft/bench_latency_hf.py --model Qwen/Qwen2.5-VL-7B-Instruct \
      --adapter $SFT/lora_high/final --cache $CACHE --prep $SFT/prep --images $IMAGES \
      --out latency_qwen_hf.json [--save-merged $SFT/merged_high]
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_prompt, hist_text_of, load_texts  # noqa: E402


def stats(xs):
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[max(0, min(n - 1, int(round(p * (n - 1)))))]
    return {"n": n, "median": round(statistics.median(s), 3), "mean": round(sum(s) / n, 3),
            "p90": round(q(0.9), 3), "min": round(s[0], 3), "max": round(s[-1], 3)}


def collect_steps(args, n_total):
    entries = json.load(open(os.path.join(args.prep, "test_idd_entries.json")))
    ct_get, it_get = load_texts(args.cache)
    items = []
    for e in entries:
        eid = int(e["eid"])
        ep = torch.load(e["path"], map_location="cpu", weights_only=False)
        ct_e = ct_get(eid)
        goal = it_get(eid).get("goal", ep.get("goal", ""))
        for si, step in enumerate(ep["steps"]):
            imgp = os.path.join(args.images, str(eid), f"s{si}.jpg")
            if not os.path.exists(imgp):
                continue
            labels = ct_e[si] if si < len(ct_e) else []
            items.append({"eid": eid, "si": si, "img": imgp,
                          "prompt": build_prompt(goal, hist_text_of(ep["steps"], ct_e, si), labels)})
            if len(items) >= n_total:
                return items
    return items


def bench(name, model, proc, items, args):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    recs = []
    for k, it in enumerate(items):
        img = Image.open(it["img"]).convert("RGB")
        t_b = time.perf_counter()
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["prompt"]}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t_d = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        t_e = time.perf_counter()
        with torch.no_grad():  # prefill approximation: a single generated token
            model.generate(**enc, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        t_g = time.perf_counter()
        n_new = int(out.shape[1] - enc["input_ids"].shape[1])
        rec = {"warmup": k < args.n_warmup, "input_tokens": int(enc["input_ids"].shape[1]),
               "new_tokens": n_new, "gen_ms": (t_e - t_d) * 1e3, "prefill_ms": (t_g - t_e) * 1e3,
               "e2e_ms": (t_e - t_b) * 1e3}
        if n_new > 1:
            rec["decode_ms_per_token"] = (rec["gen_ms"] - rec["prefill_ms"]) / (n_new - 1)
        recs.append(rec)
    meas = [r for r in recs if not r["warmup"]]
    summ = {key: stats([r[key] for r in meas if key in r])
            for key in ("e2e_ms", "gen_ms", "prefill_ms", "decode_ms_per_token", "input_tokens", "new_tokens")}
    summ["peak_mem_allocated_GB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    print(f"[{name}] e2e p50 {summ['e2e_ms']['median']} ms, p90 {summ['e2e_ms']['p90']} ms, "
          f"peak {summ['peak_mem_allocated_GB']} GB", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", default=["base", "sft_lora_unmerged", "sft_lora_merged"])
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    ap.add_argument("--save-merged", default=None, help="save merged weights (for vLLM serving)")
    args = ap.parse_args()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    items = collect_steps(args, args.n_warmup + args.n_steps)
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda").eval()
    results = {"device": torch.cuda.get_device_name(0), "configs": {}}
    if "base" in args.configs:
        results["configs"]["base"] = bench("base", model, proc, items, args)
    if args.adapter and any(c.startswith("sft_lora") for c in args.configs):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        if "sft_lora_unmerged" in args.configs:
            results["configs"]["sft_lora_unmerged"] = bench("sft_lora_unmerged", model, proc, items, args)
        model = model.merge_and_unload().eval()
        if "sft_lora_merged" in args.configs:
            results["configs"]["sft_lora_merged"] = bench("sft_lora_merged", model, proc, items, args)
        if args.save_merged:
            os.makedirs(args.save_merged, exist_ok=True)
            model.save_pretrained(args.save_merged, safe_serialization=True)
            proc.save_pretrained(args.save_merged)
            for fn in ("chat_template.json", "generation_config.json", "preprocessor_config.json"):
                src = os.path.join(args.model, fn)
                if os.path.exists(src) and not os.path.exists(os.path.join(args.save_merged, fn)):
                    shutil.copy(src, args.save_merged)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
