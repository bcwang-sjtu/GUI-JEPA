"""Candidate texts read by the text matcher, the click reranker and the 7B reference.

For every enumerated candidate: its own label, up to four texts of nodes inside its box,
its class name, a coarse screen region and an "editable" tag, joined by " | ".

  python scripts/preprocess/dump_candidate_texts.py --shards "$AC/android_control-*" \
      --out $CACHE/cand_texts.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from android_env.proto.a11y import android_accessibility_forest_pb2 as forest_pb2  # noqa: E402
from gui_jepa.data.androidcontrol import iter_episodes, episode_steps, extract_candidates  # noqa: E402


def all_text_nodes(forest_bytes):
    """[(bounds, text)] for every node with a non-empty text or content description."""
    forest = forest_pb2.AndroidAccessibilityForest()
    forest.ParseFromString(forest_bytes)
    out = []
    for w in forest.windows:
        for n in w.tree.nodes:
            b = n.bounds_in_screen
            t = (n.text or "").strip() or (n.content_description or "").strip()
            if t and b.right > b.left and b.bottom > b.top:
                out.append(((b.left, b.top, b.right, b.bottom), t[:40]))
    return out


def region(cx, cy, w, h):
    v = "top" if cy < h / 3 else ("mid" if cy < 2 * h / 3 else "bottom")
    hz = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
    return f"{v}-{hz}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-candidates", type=int, default=80)
    args = ap.parse_args()
    texts = {}
    for sf in sorted(glob.glob(args.shards)):
        for ex in iter_episodes(sf):
            steps = episode_steps(ex)
            if not steps:
                continue
            per_step = []
            for st in steps:
                cands = extract_candidates(st.forest_bytes, st.screen_w, st.screen_h,
                                           max_candidates=args.max_candidates)
                tnodes = all_text_nodes(st.forest_bytes)
                w, h = max(st.screen_w, 1), max(st.screen_h, 1)
                labels = []
                for c in cands:
                    own = c.label_text()[:40]
                    l, t, r, b = c.bounds
                    inside = [tx for (bb, tx) in tnodes
                              if bb[0] >= l - 2 and bb[1] >= t - 2 and bb[2] <= r + 2 and bb[3] <= b + 2
                              and tx.lower() != own.lower()][:4]
                    cx, cy = c.center
                    parts = ([own] if own else []) + inside
                    parts.append(c.class_name.split(".")[-1])
                    parts.append(region(cx, cy, w, h))
                    if c.editable:
                        parts.append("editable")
                    if not own and not inside:
                        parts.insert(0, "icon")
                    labels.append(" | ".join(p for p in parts if p))
                per_step.append(labels)
            texts[steps[0].episode_id] = per_step
    with open(args.out, "w") as f:
        json.dump(texts, f)
    print(f"[cand_texts] {len(texts)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
