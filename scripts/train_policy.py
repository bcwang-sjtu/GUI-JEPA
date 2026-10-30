"""Train and evaluate the GUI-JEPA policy on AndroidControl (cached features).

score(c) = StructuralHead(candidate-conditioned JEPA latent of c, condition)
         + MiniLM cross-encoder(condition text, text of c)        (element candidates only)

The residual latent predictor turns the frozen screen latent z_t into one latent per
candidate, z~_c = z_t + Delta(z_t, a_c, roi(z_t, c)); the policy picks argmax_c score(c).
Training uses a listwise cross-entropy over the candidates of each step, for both the
step instruction (AC-Low) and the goal (AC-High) as condition.

At evaluation, the typed text of an input_text action comes from the span ranker
(--span-ranker) and click picks can be re-ranked by the type-gated click reranker
(--click-reranker). Mechanism analyses are available as flags:
  --identity-predictor   z~_c := z_t (no predictor), at training and/or evaluation
  --shuffle-actions      permute candidate action inputs (all / within_type) at evaluation
  --fresh-predictor      random predictor initialisation instead of the dynamics-pretrained one
  --freeze-predictor     keep the predictor frozen during policy training
  --no-latent            text matcher only (no latent pathway)

Example (train + evaluate on IDD):
  python scripts/train_policy.py --cache $CACHE --init-predictor $CKPT/dynamics/predictor.pt \
      --out $CKPT/policy_s1 --seed 1
Example (evaluate the final system):
  python scripts/train_policy.py --cache $CACHE --eval-only --ckpt-dir $CKPT/policy_s1 \
      --span-ranker $CKPT/span_ranker --click-reranker $CKPT/click_reranker_s1 \
      --out $OUT/policy_s1_IDD
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gui_jepa.actions import A_DIM, build_candidate_actions, gt_action_index, kind_id, gt_action_vector  # noqa: E402
from gui_jepa.data.cache import (load_manifest, iter_step_records, select_eval_episodes,  # noqa: E402
                                 load_json_int_keys)
from gui_jepa.models import (LatentPredictor, StructuralHead, CrossEncoderMatcher,  # noqa: E402
                             DEFAULT_MATCHER, load_span_ranker)
from gui_jepa.rankers import load_reranker, rerank_click  # noqa: E402
from gui_jepa.roi import batch_roi, roi_pool  # noqa: E402
from gui_jepa.text_utils import (extract_type_text, predict_app, text_match,  # noqa: E402
                                 span_candidates, history_text)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # data
    ap.add_argument("--cache", required=True, help="feature cache dir (manifest.json + ep_*.pt)")
    ap.add_argument("--texts", default=None, help="instr_texts.json (default: <cache>/instr_texts.json)")
    ap.add_argument("--cand-texts", default=None, help="candidate texts (default: <cache>/cand_texts.json)")
    ap.add_argument("--app-vocab", default=None, help="app_vocab.json (default: <cache>/app_vocab.json)")
    ap.add_argument("--subsplits-json", default=None, help="official test_subsplits.json (OOD subsplits)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1)
    # model
    ap.add_argument("--init-predictor", default=None, help="dynamics-pretrained predictor.pt")
    ap.add_argument("--pred-depth", type=int, default=3)
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--matcher-model", default=DEFAULT_MATCHER)
    # training
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4, help="predictor-free heads")
    ap.add_argument("--lr-lm", type=float, default=2e-5, help="MiniLM matcher and predictor")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--nonclick-weight", type=float, default=1.5)
    ap.add_argument("--max-train-eps", type=int, default=0)
    ap.add_argument("--ckpt-every-epoch", action="store_true")
    # mechanism variants
    ap.add_argument("--freeze-predictor", action="store_true")
    ap.add_argument("--fresh-predictor", action="store_true")
    ap.add_argument("--identity-predictor", action="store_true")
    ap.add_argument("--no-latent", action="store_true")
    ap.add_argument("--shuffle-actions", choices=["none", "all", "within_type"], default="none")
    # evaluation
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt-dir", default=None, help="dir with heads.pt + matcher_lm/")
    ap.add_argument("--eval-split", choices=["test", "validation"], default="test")
    ap.add_argument("--subsplit", default="IDD",
                    help="IDD | app_unseen | task_unseen | category_unseen | full")
    ap.add_argument("--max-test-steps", type=int, default=0, help="0 = all steps")
    ap.add_argument("--span-ranker", default=None, help="dir with ranker_lm/ + ranker_head.pt")
    ap.add_argument("--span-max-words", type=int, default=16)
    ap.add_argument("--span-cap", type=int, default=800)
    ap.add_argument("--click-reranker", default=None, help="dir with reranker_lm/ + reranker_head.pt")
    ap.add_argument("--rerank-beta", type=float, default=1.0)
    ap.add_argument("--rerank-gate", type=float, default=1.0)
    ap.add_argument("--dump-predictions", default=None, help="write per-step predictions (json)")
    args = ap.parse_args()
    args.texts = args.texts or os.path.join(args.cache, "instr_texts.json")
    args.cand_texts = args.cand_texts or os.path.join(args.cache, "cand_texts.json")
    args.app_vocab = args.app_vocab or os.path.join(args.cache, "app_vocab.json")
    if args.eval_only and not args.ckpt_dir:
        ap.error("--eval-only requires --ckpt-dir")
    if not args.eval_only and not args.fresh_predictor and not args.init_predictor:
        ap.error("training needs --init-predictor (or --fresh-predictor)")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    train_pred = not args.freeze_predictor

    man = load_manifest(args.cache)
    train_e = man["train"][:args.max_train_eps] if args.max_train_eps else man["train"]
    test_e = select_eval_episodes(man, args.eval_split, args.subsplit, args.subsplits_json)
    texts = load_json_int_keys(args.texts)
    cand_texts = load_json_int_keys(args.cand_texts)
    with open(args.app_vocab) as f:
        app_vocab = list(json.load(f).keys())
    print(f"[policy] train {len(train_e)} eps, eval {len(test_e)} eps "
          f"({args.eval_split}/{args.subsplit})", flush=True)

    # ------------------------------------------------------------------ modules
    adim = A_DIM + args.embed_dim  # action vector + ROI feature of the candidate's element
    predictor = LatentPredictor(embed_dim=args.embed_dim, action_dim=adim,
                                num_patches_side=args.grid, depth=args.pred_depth).to(dev)
    if not args.fresh_predictor and not args.eval_only:
        predictor.load_state_dict(torch.load(args.init_predictor, map_location=dev))
    if train_pred:
        predictor.train()
    else:
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad_(False)

    cond_extra = A_DIM + args.embed_dim  # history: previous action vector + mean latent change
    wm_head = StructuralHead(zdim=args.embed_dim, cond_extra=cond_extra).to(dev)
    matcher = CrossEncoderMatcher(dev, args.matcher_model).to(dev)

    span_ranker = load_span_ranker(args.span_ranker, dev) if args.span_ranker else None

    if args.eval_only:
        ck = torch.load(os.path.join(args.ckpt_dir, "heads.pt"), map_location=dev)
        wm_head.load_state_dict(ck["wm_head"])
        matcher.head.load_state_dict(ck["matcher_head"])
        if "predictor" in ck:
            predictor.load_state_dict(ck["predictor"])
        else:
            predictor.load_state_dict(torch.load(args.init_predictor, map_location=dev))
        from transformers import AutoModel
        matcher.lm = AutoModel.from_pretrained(os.path.join(args.ckpt_dir, "matcher_lm")).to(dev)
    click_reranker = load_reranker(args.click_reranker, dev) if args.click_reranker else None

    params = [{"params": matcher.lm.parameters(), "lr": args.lr_lm},
              {"params": matcher.head.parameters(), "lr": args.lr}]
    if not args.no_latent:
        params.append({"params": wm_head.parameters(), "lr": args.lr})
    if train_pred:
        params.append({"params": predictor.parameters(), "lr": args.lr_lm})
    opt = torch.optim.AdamW(params, weight_decay=args.wd)

    # ------------------------------------------------------------------ helpers
    def step_texts(ep, si):
        ct = cand_texts.get(ep["eid"])
        return ct[si] if ct and si < len(ct) else []

    def history_vec(ep, si):
        """Previous recorded action vector + mean latent change from the previous frame."""
        if si == 0:
            return torch.zeros(A_DIM + args.embed_dim, device=dev)
        prev_step = ep["steps"][si - 1]
        a = gt_action_vector(prev_step["gt"], prev_step["screen_wh"]).to(dev)
        fz = ep["frames_z"]
        d = (fz[si].float() - fz[si - 1].float()).mean(0).to(dev)
        return torch.cat([a, d])

    def forward_step(ep, si, step, cond_key, train_mode):
        acts, metas = build_candidate_actions(step)
        N = acts.size(0)
        z_t = ep["frames_z"][si].float().to(dev)
        rois = batch_roi(z_t.cpu(), metas, step["cand_bounds"], args.grid).to(dev)
        rz = rois.clone()
        for j, m in enumerate(metas):
            if m.get("cand") is None:  # global actions carry a zero region feature
                rz[j].zero_()
        acts_full = torch.cat([acts.to(dev), rz], -1)

        tinfo = texts.get(ep["eid"], {})
        instrs = tinfo.get("instrs", [])
        cond_txt = (instrs[si] if si < len(instrs) else "") if cond_key == "low" \
            else tinfo.get("goal", "")
        if si > 0:
            cond_txt = f"{cond_txt} || done: {history_text(ep['steps'], cand_texts.get(ep['eid'], []), si)}"

        total = torch.zeros(N, device=dev)
        if not args.no_latent:
            ctx = torch.enable_grad() if (train_pred and train_mode) else torch.no_grad()
            with ctx:
                if args.shuffle_actions != "none":
                    g_ = torch.Generator(device="cpu").manual_seed(1000 + int(si))
                    if args.shuffle_actions == "within_type":
                        kd_ = torch.tensor([kind_id(m) for m in metas])
                        perm_ = torch.arange(N)
                        for k_ in kd_.unique():
                            idx_ = torch.nonzero(kd_ == k_).squeeze(-1)
                            if idx_.numel() > 1:
                                perm_[idx_] = idx_[torch.randperm(idx_.numel(), generator=g_)]
                    else:
                        perm_ = torch.randperm(N, generator=g_)
                    acts_full = acts_full[perm_.to(acts_full.device)]
                z_pred = predictor(z_t.unsqueeze(0).expand(N, -1, -1), acts_full)
                z_pred = z_t.unsqueeze(0) + z_pred  # residual: z~_c = z_t + Delta
                if args.identity_predictor:
                    z_pred = z_t.unsqueeze(0).expand(N, -1, -1)
            roi_pred = torch.stack([
                roi_pool(z_pred[j], step["cand_bounds"][m["cand"]].tolist(), args.grid)
                if m.get("cand") is not None else z_pred[j].mean(0)
                for j, m in enumerate(metas)])
            cond = (ep["instr_embs"][si] if cond_key == "low" else ep["goal_emb"]).float().to(dev)
            cond = torch.cat([cond, history_vec(ep, si)]).unsqueeze(0).expand(N, -1)
            kinds = torch.tensor([kind_id(m) for m in metas], device=dev)
            total = total + wm_head(roi_pred, z_pred.mean(1), cond, kinds)

        ctexts = step_texts(ep, si)
        elem_js = [j for j, m in enumerate(metas)
                   if m.get("cand") is not None and m["cand"] < len(ctexts)]
        if elem_js and cond_txt:
            ce_scores = matcher(cond_txt, [ctexts[metas[j]["cand"]] for j in elem_js])
            total = total.clone()
            total[torch.tensor(elem_js, device=dev)] += ce_scores
        return total, metas

    def save_checkpoint(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        save = {"wm_head": wm_head.state_dict(), "matcher_head": matcher.head.state_dict()}
        if train_pred:
            save["predictor"] = predictor.state_dict()
        torch.save(save, os.path.join(out_dir, "heads.pt"))
        matcher.lm.save_pretrained(os.path.join(out_dir, "matcher_lm"))

    # ------------------------------------------------------------------ training
    step_n = 0
    for epoch in range(0 if args.eval_only else args.epochs):
        for ep, si, step in iter_step_records(train_e, shuffle=True):
            _, metas = build_candidate_actions(step)
            gt_j = gt_action_index(step, metas)
            if gt_j is None:  # ground-truth element not enumerated
                continue
            w = 1.0 if step["gt"]["type"] == "click" else args.nonclick_weight
            for cond_key in ("low", "high"):
                scores, _ = forward_step(ep, si, step, cond_key, True)
                if not scores.requires_grad:
                    continue
                loss = w * F.cross_entropy(scores.unsqueeze(0), torch.tensor([gt_j], device=dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
            step_n += 1
            if step_n % 2000 == 0:
                print(f"[policy] ep{epoch} step{step_n} loss {loss.item():.3f}", flush=True)
        if args.ckpt_every_epoch:
            save_checkpoint(os.path.join(args.out, f"ep{epoch}"))
    if not args.eval_only:
        save_checkpoint(args.out)

    # ------------------------------------------------------------------ evaluation
    predictor.eval()
    matcher.eval()
    wm_head.eval()

    @torch.no_grad()
    def predict_text(src_txt, goal_txt):
        s = src_txt or goal_txt
        cands = span_candidates(s, max_words=args.span_max_words, cap=args.span_cap)
        if not cands:
            return extract_type_text(src_txt, goal_txt)
        return cands[int(span_ranker(s, cands).argmax())]

    dumps = {"low": [], "high": []}

    @torch.no_grad()
    def eval_setting(cond_key):
        tot, cor, type_cor, pred_kind = Counter(), Counter(), Counter(), Counter()
        g_tot = g_cor = n_eval = 0
        for ep, si, step in iter_step_records(test_e):
            if args.max_test_steps and n_eval >= args.max_test_steps:
                break
            gt = step["gt"]
            tinfo = texts.get(ep["eid"], {})
            instrs = tinfo.get("instrs", [])
            instr_txt = instrs[si] if si < len(instrs) else ""
            goal_txt = tinfo.get("goal", "")
            scores, metas = forward_step(ep, si, step, cond_key, False)
            argmax_idx = int(scores.argmax())
            m = metas[argmax_idx]
            if click_reranker is not None and m["kind"] == "click":
                m = metas[rerank_click(click_reranker, goal_txt, scores, metas, argmax_idx, step,
                                       step_texts(ep, si), args.rerank_beta, args.rerank_gate)]
            gt_type = gt["type"]
            pred_kind[m["kind"]] += 1
            t_ok = (m["kind"] == gt_type) or (gt_type == "long_press" and m["kind"] == "click")
            tot[gt_type] += 1
            n_eval += 1
            if t_ok:
                type_cor[gt_type] += 1
            ok = False
            pred_text = pred_app = None
            if t_ok:
                if gt_type in ("click", "long_press"):
                    if gt["gt_bbox"]:
                        w_, h_ = step["screen_wh"]
                        b = gt["gt_bbox"]
                        px, py = m.get("cx", 0) * w_, m.get("cy", 0) * h_
                        ok = b[0] <= px <= b[2] and b[1] <= py <= b[3]
                        g_tot += 1
                        g_cor += int(ok)
                elif gt_type == "scroll":
                    ok = m.get("dir") == gt["scroll_dir"]
                elif gt_type == "input_text":
                    src = instr_txt if cond_key == "low" else goal_txt
                    pred_text = (predict_text(src, goal_txt) if span_ranker is not None
                                 else extract_type_text(src, goal_txt))
                    ok = text_match(pred_text, gt["text"])
                elif gt_type == "open_app":
                    src = instr_txt if cond_key == "low" else goal_txt
                    pred_app = predict_app(src, goal_txt, app_vocab)
                    ok = pred_app == gt["app_name"]
                else:
                    ok = True
            cor[gt_type] += int(ok)
            if args.dump_predictions:
                dumps[cond_key].append({
                    "eid": ep["eid"], "si": si, "gt_type": gt_type,
                    "pred_kind": m["kind"], "pred_cand": m.get("cand"), "pred_dir": m.get("dir"),
                    "pred_text": pred_text, "pred_app": pred_app,
                    "type_ok": bool(t_ok), "correct": bool(ok)})
        n = sum(tot.values())
        return {"setting": cond_key, "n_steps": n,
                "step_acc": round(100 * sum(cor.values()) / max(n, 1), 2),
                "type_acc": round(100 * sum(type_cor.values()) / max(n, 1), 2),
                "grounding_acc_clicks": round(100 * g_cor / max(g_tot, 1), 2),
                "click_grounding_n": g_tot,
                "pred_kind_hist": dict(sorted(pred_kind.items())),
                "per_type": {k: {"n": tot[k], "step_acc": round(100 * cor[k] / max(tot[k], 1), 1)}
                             for k in tot}}

    results = {"args": vars(args)}
    for ck in ("low", "high"):
        results[f"ac_{ck}"] = eval_setting(ck)
        print(json.dumps(results[f"ac_{ck}"], indent=1), flush=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if args.dump_predictions:
        with open(args.dump_predictions, "w") as f:
            json.dump(dumps, f)
    print(f"[policy] AC-Low {results['ac_low']['step_acc']}  AC-High {results['ac_high']['step_acc']}"
          f"  -> {args.out}/results.json", flush=True)


if __name__ == "__main__":
    main()
