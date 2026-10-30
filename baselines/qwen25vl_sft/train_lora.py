"""LoRA SFT of Qwen2.5-VL-7B-Instruct on AndroidControl steps (loss on the target tokens only).

Paper recipe (one adapter per setting): 4,000 training episodes, 1 epoch, LR 1e-4 with
30 warm-up updates and cosine decay, gradient accumulation 16, LoRA r=16 / alpha=32 on
the language-model attention and MLP projections, bf16.

  python baselines/qwen25vl_sft/train_lora.py --model Qwen/Qwen2.5-VL-7B-Instruct \
      --data $SFT/train_high.jsonl --out $SFT/lora_high
"""
import argparse
import json
import math
import os
import random
import time

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "train_args.json"), "w"), indent=2)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    proc = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels,
                                         max_pixels=args.max_pixels)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    except Exception as e:
        print("[train] flash-attn unavailable, using sdpa:", e)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    data = [json.loads(line) for line in open(args.data)]
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"[train] {len(data)} samples, epochs={args.epochs}, accum={args.accum}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    total_updates = (len(data) * args.epochs) // args.accum + 1

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_updates - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    def build(sample):
        img = Image.open(sample["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": sample["prompt"]}]}]
        prompt_text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[prompt_text + sample["target"] + "<|im_end|>\n"], images=[img],
                   return_tensors="pt")
        plen = proc(text=[prompt_text], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :plen] = -100
        enc["labels"] = labels
        return enc

    step = upd = nb = 0
    running, t0 = 0.0, time.time()
    order = list(range(len(data)))
    for ep in range(args.epochs):
        random.shuffle(order)
        opt.zero_grad()
        for idx in order:
            try:
                enc = build(data[idx])
            except Exception:
                continue
            enc = {k: v.to("cuda") for k, v in enc.items()}
            out = model(**enc)
            loss = out.loss / args.accum
            if torch.isnan(loss):
                continue
            loss.backward()
            running += out.loss.item()
            nb += 1
            step += 1
            if step % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(upd)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                upd += 1
                if upd % 20 == 0:
                    print(f"[train] ep{ep} update {upd}/{total_updates} loss {running / max(nb, 1):.4f} "
                          f"lr {lr_at(upd):.2e} {(time.time() - t0) / 60:.1f}min", flush=True)
                    running, nb = 0.0, 0
            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(os.path.join(args.out, f"step{step}"))
    model.save_pretrained(os.path.join(args.out, "final"))
    proc.save_pretrained(os.path.join(args.out, "final"))
    print(f"[train] done in {(time.time() - t0) / 60:.1f}min -> {args.out}/final")


if __name__ == "__main__":
    main()
