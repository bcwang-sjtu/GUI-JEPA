"""Export pre-action screenshots as JPEG files, aligned to the cache step index.

Used by the real-step latency benchmark and by the Qwen2.5-VL-7B reference.
Images are written to <out>/<eid>/s<step>.jpg (long side <= --long-side).

  python scripts/preprocess/export_screenshots.py --shards "$AC/android_control-*" \
      --eids eids.json --out $IMAGES
"""
import argparse
import glob
import io
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gui_jepa.data.androidcontrol import iter_episodes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob of AndroidControl TFRecord shards")
    ap.add_argument("--eids", required=True, help="json list of episode ids to export")
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-side", type=int, default=1024)
    args = ap.parse_args()
    need = {int(x) for x in json.load(open(args.eids))}
    os.makedirs(args.out, exist_ok=True)
    done, saved, t0 = set(), 0, time.time()
    for sh in sorted(glob.glob(args.shards)):
        if not need - done:
            break
        for ex in iter_episodes(sh):
            eid = int(ex["episode_id"][0])
            if eid not in need or eid in done:
                continue
            d = os.path.join(args.out, str(eid))
            os.makedirs(d, exist_ok=True)
            for si, png in enumerate(ex.get("screenshots", [])):
                try:
                    im = Image.open(io.BytesIO(png)).convert("RGB")
                except Exception:
                    continue
                w, h = im.size
                sc = args.long_side / max(w, h)
                if sc < 1.0:
                    im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.BILINEAR)
                im.save(os.path.join(d, f"s{si}.jpg"), quality=88)
                saved += 1
            done.add(eid)
    print(f"[export] {len(done)}/{len(need)} episodes, {saved} images, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
