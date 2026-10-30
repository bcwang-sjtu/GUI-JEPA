"""Dump goals, step instructions and the open_app vocabulary from the AndroidControl shards.

Outputs:  <out-dir>/instr_texts.json  {eid: {"goal": str, "instrs": [str, ...]}}
          <out-dir>/app_vocab.json    {app_name: count}  (training-split open_app actions)

  python scripts/preprocess/dump_texts.py --shards "$AC/android_control-*" \
      --splits $AC/splits.json --out-dir $CACHE
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gui_jepa.data.androidcontrol import iter_tfrecords, parse_example  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob of AndroidControl TFRecord shards")
    ap.add_argument("--splits", required=True, help="official splits.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.splits) as f:
        train_ids = {int(x) for x in json.load(f)["train"]}

    texts, app_vocab = {}, collections.Counter()
    for sf in sorted(glob.glob(args.shards)):
        for payload in iter_tfrecords(sf):
            ex = parse_example(payload)
            if "episode_id" not in ex:
                continue
            eid = int(ex["episode_id"][0])
            goal = ex["goal"][0].decode("utf-8", "replace")
            instrs = [s.decode("utf-8", "replace") for s in ex.get("step_instructions", [])]
            texts[eid] = {"goal": goal, "instrs": instrs}
            if eid in train_ids:
                for a in ex.get("actions", []):
                    act = json.loads(a.decode("utf-8", "replace"))
                    if act.get("action_type") == "open_app" and act.get("app_name"):
                        app_vocab[act["app_name"]] += 1
    with open(os.path.join(args.out_dir, "instr_texts.json"), "w") as f:
        json.dump(texts, f)
    with open(os.path.join(args.out_dir, "app_vocab.json"), "w") as f:
        json.dump(dict(app_vocab.most_common()), f, indent=1)
    print(f"[texts] {len(texts)} episodes, {len(app_vocab)} apps")


if __name__ == "__main__":
    main()
