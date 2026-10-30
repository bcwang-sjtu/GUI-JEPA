"""Per-step latency of the Qwen2.5-VL-7B reference with vLLM (batch 1, single request per step).

Same steps, prompt, pixel budget and greedy decoding as bench_latency_hf.py; use the merged
SFT weights (--model). Timed: chat template + one llm.generate() call (multimodal
preprocessing, scheduling and decoding). The prefill time is approximated with max_tokens=1.

  python baselines/qwen25vl_sft/bench_latency_vllm.py --model $SFT/merged_high \
      --cache $CACHE --prep $SFT/prep --images $IMAGES --no-prefix-caching --out latency_qwen_vllm.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_latency_hf import collect_steps  # noqa: E402


def stats(xs):
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[max(0, min(n - 1, int(round(p * (n - 1)))))]
    return {"n": n, "median": round(statistics.median(s), 3), "mean": round(sum(s) / n, 3),
            "p90": round(q(0.9), 3), "min": round(s[0], 3), "max": round(s[-1], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged SFT weights (or the base model)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="matches the model's generation_config used by the HF run")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--no-prefix-caching", action="store_true")
    args = ap.parse_args()

    from PIL import Image
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    items = collect_steps(args, args.n_warmup + args.n_steps)
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
              seed=0, max_num_seqs=4, enable_prefix_caching=not args.no_prefix_caching)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, repetition_penalty=args.repetition_penalty)

    recs = []
    for k, it in enumerate(items):
        img = Image.open(it["img"]).convert("RGB")
        t_b = time.perf_counter()
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["prompt"]}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        req = {"prompt": text, "multi_modal_data": {"image": img}}
        t_c = time.perf_counter()
        o = llm.generate([req], sp, use_tqdm=False)[0]
        t_d = time.perf_counter()
        llm.generate([req], sp1, use_tqdm=False)
        t_e = time.perf_counter()
        n_new = len(o.outputs[0].token_ids)
        rec = {"warmup": k < args.n_warmup, "input_tokens": len(o.prompt_token_ids), "new_tokens": n_new,
               "gen_ms": (t_d - t_c) * 1e3, "prefill_ms": (t_e - t_d) * 1e3, "e2e_ms": (t_d - t_b) * 1e3}
        if n_new > 1:
            rec["decode_ms_per_token"] = (rec["gen_ms"] - rec["prefill_ms"]) / (n_new - 1)
        recs.append(rec)
    meas = [r for r in recs if not r["warmup"]]
    res = {key: stats([r[key] for r in meas if key in r])
           for key in ("e2e_ms", "gen_ms", "prefill_ms", "decode_ms_per_token", "input_tokens", "new_tokens")}
    res["device"] = torch.cuda.get_device_name(0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[vllm] e2e p50 {res['e2e_ms']['median']} ms, p90 {res['e2e_ms']['p90']} ms -> {args.out}")


if __name__ == "__main__":
    main()
