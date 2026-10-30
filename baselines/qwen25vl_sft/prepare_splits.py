"""Select the first N training episodes and all IDD test episodes from the feature cache.

  python baselines/qwen25vl_sft/prepare_splits.py --cache $CACHE --out $SFT/prep
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    man = json.load(open(os.path.join(args.cache, "manifest.json")))
    train = man["train"][:args.n_train]
    test_idd = [e for e in man["test"] if e.get("subsplit") == "IDD"]
    train_eids = [int(e["eid"]) for e in train]
    test_eids = [int(e["eid"]) for e in test_idd]
    json.dump(sorted(set(train_eids) | set(test_eids)), open(os.path.join(args.out, "eids_all.json"), "w"))
    json.dump(train, open(os.path.join(args.out, "train_entries.json"), "w"))
    json.dump(test_idd, open(os.path.join(args.out, "test_idd_entries.json"), "w"))
    print(f"train {len(train_eids)} episodes, IDD test {len(test_eids)} episodes")


if __name__ == "__main__":
    main()
