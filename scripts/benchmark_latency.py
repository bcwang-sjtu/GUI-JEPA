"""Real-step serving latency of the full GUI-JEPA pipeline (AC-High, batch 1).

Each step runs the deployed path on a real IDD test step:
JPEG screenshot -> 224 px + ImageNet normalisation -> frozen JEPA ViT-L -> candidate actions
from the step's a11y candidates -> ROI-conditioned action encoding -> residual predictor +
structural head -> MiniLM text matcher -> argmax -> span ranker (predicted input_text) /
app heuristic (predicted open_app) / click reranker (predicted click).

Every segment is timed with torch.cuda.synchronize() at its boundaries. Screenshot capture,
a11y-tree retrieval and parsing, and action execution are not measured; the CLIP goal
embedding is computed once per episode and reported separately.

  python scripts/benchmark_latency.py --cache $CACHE --images $IMAGES \
      --encoder-ckpt $CKPT/encoder/encoder_1.pt --ckpt-dir $CKPT/policy_s1 \
      --span-ranker $CKPT/span_ranker --click-reranker $CKPT/click_reranker_s1 \
      --n-steps 100 [--bf16] --out latency.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gui_jepa.actions import A_DIM, build_candidate_actions, kind_id, gt_action_vector  # noqa: E402
from gui_jepa.clip_text import CLIPTextEncoder  # noqa: E402
from gui_jepa.data.cache import load_manifest, load_json_int_keys  # noqa: E402
from gui_jepa.models import (LatentPredictor, StructuralHead, CrossEncoderMatcher,  # noqa: E402
                             DEFAULT_MATCHER, load_encoder, load_span_ranker)
from gui_jepa.rankers import load_reranker, click_cand_aux  # noqa: E402
from gui_jepa.roi import batch_roi, roi_pool  # noqa: E402
from gui_jepa.text_utils import (extract_type_text, predict_app, text_match,  # noqa: E402
                                 span_candidates, history_text)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
ED = 1024
SEGS = ["decode_resize", "encode_vit", "build_cands", "roi_action_enc", "predictor_head",
        "matcher", "argmax", "span_ranker", "app_heuristic", "click_reranker"]


def stats(xs):
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[max(0, min(n - 1, int(round(p * (n - 1)))))]
    return {"n": n, "p50": round(statistics.median(s), 3), "mean": round(sum(s) / n, 3),
            "p90": round(q(0.9), 3), "min": round(s[0], 3), "max": round(s[-1], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--images", required=True, help="exported JPEGs (<eid>/s<step>.jpg)")
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="policy checkpoint (heads.pt + matcher_lm/)")
    ap.add_argument("--span-ranker", required=True)
    ap.add_argument("--click-reranker", required=True)
    ap.add_argument("--cand-texts", default=None)
    ap.add_argument("--rerank-beta", type=float, default=1.0)
    ap.add_argument("--rerank-gate", type=float, default=1.0)
    ap.add_argument("--span-max-words", type=int, default=16)
    ap.add_argument("--span-cap", type=int, default=800)
    ap.add_argument("--matcher-model", default=DEFAULT_MATCHER)
    ap.add_argument("--pred-depth", type=int, default=3)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--bf16", action="store_true", help="bf16 weights + autocast")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(1)
    torch.manual_seed(1)
    dev = torch.device(args.device)
    log = lambda m: print(m, flush=True)
    torch.cuda.set_device(dev)  # synchronize / memory statistics refer to this device

    # ------------------------------------------------------------------ load the full pipeline
    vit = load_encoder(args.encoder_ckpt, device=dev)
    clip = CLIPTextEncoder(device=dev)
    predictor = LatentPredictor(embed_dim=ED, action_dim=A_DIM + ED, num_patches_side=14,
                                depth=args.pred_depth).to(dev)
    wm_head = StructuralHead(zdim=ED, cond_extra=A_DIM + ED).to(dev)
    matcher = CrossEncoderMatcher(dev, args.matcher_model).to(dev)
    ck = torch.load(os.path.join(args.ckpt_dir, "heads.pt"), map_location=dev)
    wm_head.load_state_dict(ck["wm_head"])
    matcher.head.load_state_dict(ck["matcher_head"])
    predictor.load_state_dict(ck["predictor"])
    from transformers import AutoModel
    matcher.lm = AutoModel.from_pretrained(os.path.join(args.ckpt_dir, "matcher_lm")).to(dev)
    span_ranker = load_span_ranker(args.span_ranker, dev)
    reranker = load_reranker(args.click_reranker, dev)
    modules = (vit, predictor, wm_head, matcher, span_ranker, reranker)
    for m in modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        if args.bf16:
            m.to(torch.bfloat16)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.bf16 \
        else (lambda: torch.autocast("cuda", enabled=False))
    app_vocab = list(json.load(open(os.path.join(args.cache, "app_vocab.json"))).keys())
    texts = load_json_int_keys(os.path.join(args.cache, "instr_texts.json"))
    cand_texts = load_json_int_keys(args.cand_texts or os.path.join(args.cache, "cand_texts.json"))
    nparams = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    params = {"vit": nparams(vit), "clip_text": nparams(clip.model), "predictor": nparams(predictor),
              "structural_head": nparams(wm_head), "matcher": nparams(matcher),
              "span_ranker": nparams(span_ranker), "click_reranker": nparams(reranker)}
    log(f"parameters (M): { {k: round(v, 1) for k, v in params.items()} }")

    # ------------------------------------------------------------------ IDD steps in manifest order
    test_e = [e for e in load_manifest(args.cache)["test"] if e.get("subsplit") == "IDD"]
    items, n_total = [], args.n_warmup + args.n_steps
    for e in test_e:
        ep = torch.load(e["path"], map_location="cpu", weights_only=False)
        for si in range(len(ep["steps"])):
            imgp = os.path.join(args.images, str(ep["eid"]), f"s{si}.jpg")
            if os.path.exists(imgp):
                items.append((ep, si, imgp))
            if len(items) >= n_total:
                break
        if len(items) >= n_total:
            break

    def sync():
        torch.cuda.synchronize()

    recs, goal_ms = [], []
    state = {"eid": None, "goal_emb": None, "z_prev": None}
    torch.cuda.reset_peak_memory_stats()
    for k, (ep, si, imgp) in enumerate(items):
        step = ep["steps"][si]
        eid = int(ep["eid"])
        gt = step["gt"]
        goal = texts.get(eid, {}).get("goal", ep.get("goal", ""))
        ct_e = cand_texts.get(eid, [])
        ctexts = ct_e[si] if si < len(ct_e) else []
        if state["eid"] != eid:  # once per episode: CLIP goal embedding
            sync()
            tg = time.perf_counter()
            with torch.no_grad():
                g = clip.encode([goal]).half().float()[0].to(dev)
            sync()
            goal_ms.append((time.perf_counter() - tg) * 1e3)
            state.update(eid=eid, goal_emb=g, z_prev=None)
        seg = {}
        sync()
        t_begin = t_last = time.perf_counter()

        def tick(name):
            nonlocal t_last
            sync()
            now = time.perf_counter()
            seg[name] = (now - t_last) * 1e3
            t_last = now

        # 1. JPEG decode + resize + normalise (CPU)
        im = Image.open(imgp).convert("RGB").resize((224, 224))
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)
        tick("decode_resize")
        # 2. frozen ViT-L
        with torch.no_grad(), ac():
            feats = vit.forward_features(x.to(dev, dtype=torch.bfloat16 if args.bf16 else torch.float32))
        if feats.dim() == 3 and feats.size(1) == 197:
            feats = feats[:, 1:, :]
        z_t = feats[0].half().float()
        tick("encode_vit")
        # 3. candidate actions + condition text with the action history
        acts, metas = build_candidate_actions(step)
        N = acts.size(0)
        cond_txt = goal if si == 0 else f"{goal} || done: {history_text(ep['steps'], ct_e, si)}"
        kinds = torch.tensor([kind_id(m) for m in metas], device=dev)
        tick("build_cands")
        # 4. ROI-conditioned action encoding
        rz = batch_roi(z_t.cpu(), metas, step["cand_bounds"]).to(dev)
        for j, m in enumerate(metas):
            if m.get("cand") is None:
                rz[j].zero_()
        acts_full = torch.cat([acts.to(dev), rz], -1)
        tick("roi_action_enc")
        # 5. residual predictor + structural head
        with torch.no_grad(), ac():
            z_pred = z_t.unsqueeze(0) + predictor(z_t.unsqueeze(0).expand(N, -1, -1), acts_full).float()
            roi_pred = torch.stack([roi_pool(z_pred[j], step["cand_bounds"][m["cand"]].tolist())
                                    if m.get("cand") is not None else z_pred[j].mean(0)
                                    for j, m in enumerate(metas)])
            if si == 0 or state["z_prev"] is None:
                hv = torch.zeros(A_DIM + ED, device=dev)
            else:
                prev = ep["steps"][si - 1]
                hv = torch.cat([gt_action_vector(prev["gt"], prev["screen_wh"]).to(dev),
                                (z_t - state["z_prev"]).mean(0)])
            cond = torch.cat([state["goal_emb"], hv]).unsqueeze(0).expand(N, -1)
            total = wm_head(roi_pred, z_pred.mean(1), cond, kinds).float()
        tick("predictor_head")
        # 6. text matcher over element candidates
        elem_js = [j for j, m in enumerate(metas) if m.get("cand") is not None and m["cand"] < len(ctexts)]
        if elem_js and cond_txt:
            with torch.no_grad(), ac():
                ce = matcher(cond_txt, [ctexts[metas[j]["cand"]] for j in elem_js]).float()
            total = total.clone()
            total[torch.tensor(elem_js, device=dev)] += ce
        tick("matcher")
        # 7. base decision
        argmax_idx = int(total.argmax())
        m = metas[argmax_idx]
        tick("argmax")
        # 8. arguments, only for the predicted type
        pred_text = pred_app = None
        if m["kind"] == "input_text":
            cands = span_candidates(goal, max_words=args.span_max_words, cap=args.span_cap)
            if cands:
                with torch.no_grad(), ac():
                    pred_text = cands[int(span_ranker(goal, cands).float().argmax())]
            else:
                pred_text = extract_type_text(goal, goal)
            tick("span_ranker")
        elif m["kind"] == "open_app":
            pred_app = predict_app(goal, goal, app_vocab)
            tick("app_heuristic")
        # 9. type-gated click reranker
        if m["kind"] == "click":
            cjs = [j for j, mm in enumerate(metas) if mm["kind"] == "click"]
            if len(cjs) >= 2 and argmax_idx in cjs:
                cb, cf = step["cand_bounds"], step["cand_flags"]
                rtexts = [ctexts[metas[j]["cand"]] if metas[j]["cand"] < len(ctexts) else "[empty]"
                          for j in cjs]
                raux = [click_cand_aux(cb[metas[j]["cand"]], cf[metas[j]["cand"]]) for j in cjs]
                with torch.no_grad(), ac():
                    r = reranker(goal, rtexts, torch.tensor(raux, device=dev)).float()
                rrz = (r - r.mean()) / (r.std() + 1e-6)
                final = total[torch.tensor(cjs, device=dev)] + args.rerank_beta * rrz
                new_local, base_local = int(final.argmax()), cjs.index(argmax_idx)
                if float(rrz[new_local] - rrz[base_local]) > args.rerank_gate:
                    m = metas[cjs[new_local]]
            tick("click_reranker")
        sync()
        total_ms = (time.perf_counter() - t_begin) * 1e3
        state["z_prev"] = z_t

        # judge (AC-High, same rules as scripts/train_policy.py)
        gt_type = gt["type"]
        ok = (m["kind"] == gt_type) or (gt_type == "long_press" and m["kind"] == "click")
        if ok:
            if gt_type in ("click", "long_press"):
                w_, h_ = step["screen_wh"]
                b = gt.get("gt_bbox")
                ok = bool(b) and b[0] <= m.get("cx", 0) * w_ <= b[2] and b[1] <= m.get("cy", 0) * h_ <= b[3]
            elif gt_type == "scroll":
                ok = m.get("dir") == gt["scroll_dir"]
            elif gt_type == "input_text":
                ok = text_match(pred_text, gt["text"])
            elif gt_type == "open_app":
                ok = pred_app == gt["app_name"]
        recs.append({"warmup": k < args.n_warmup, "eid": eid, "si": si, "n_actions": N,
                     "seg_ms": seg, "total_ms": total_ms, "correct": bool(ok)})

    meas = [r for r in recs if not r["warmup"]]
    n = len(meas)
    tot = [r["total_ms"] for r in meas]
    res = {
        "device": torch.cuda.get_device_name(dev), "bf16": args.bf16, "n_measured": n,
        "params_M": {k: round(v, 1) for k, v in params.items()},
        "total_ms": stats(tot),
        "segments_ms": {s: stats([r["seg_ms"].get(s, 0.0) for r in meas]) for s in SEGS},
        "goal_clip_ms_per_episode": stats(goal_ms),
        "n_actions": stats([r["n_actions"] for r in meas]),
        "acc_high_on_slice": round(100 * sum(r["correct"] for r in meas) / max(n, 1), 2),
        "peak_mem_allocated_GB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    log(f"[latency] p50 {res['total_ms']['p50']} ms  p90 {res['total_ms']['p90']} ms  "
        f"mean {res['total_ms']['mean']} ms  peak {res['peak_mem_allocated_GB']} GB -> {args.out}")


if __name__ == "__main__":
    main()
