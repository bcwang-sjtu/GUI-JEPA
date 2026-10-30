"""Extract VideoCUA desktop frames at action timestamps (encoder pretraining data).

For every task folder (action_log.json + video/video.mp4) writes <out>/<task_id>/<i>.png,
resized so the shorter side is at most --image-size.

  python scripts/preprocess/extract_videocua_frames.py --app-dir $VIDEOCUA/<app> --out $PRETRAIN/videocua
"""
import argparse
import glob
import json
import os
import time

import cv2


def extract_task(task_dir, out_dir, image_size=448):
    """Returns (n_frames_extracted, n_actions, error_or_None)."""
    anno_path = os.path.join(task_dir, "action_log.json")
    video_path = os.path.join(task_dir, "video", "video.mp4")
    if not os.path.exists(anno_path) or not os.path.exists(video_path):
        return 0, 0, "missing files"
    try:
        actions = json.load(open(anno_path)).get("action_log", [])
    except Exception as e:
        return 0, 0, f"annotation: {e}"
    if not actions:
        return 0, 0, "no actions"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, "cannot open video"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
    if duration <= 0:
        cap.release()
        return 0, 0, "bad duration"
    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    for i, action in enumerate(actions):
        ts = action.get("timestamp", 0)
        if ts < 0 or ts > duration + 1:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        h, w = frame.shape[:2]
        if h > image_size or w > image_size:
            scale = image_size / min(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, f"{i}.png"), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        n_ok += 1
    cap.release()
    return n_ok, len(actions), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True, help="one unzipped VideoCUA app folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", type=int, default=448)
    args = ap.parse_args()
    task_dirs = sorted(d for d in glob.glob(os.path.join(args.app_dir, "*"))
                       if os.path.isdir(d) and os.path.exists(os.path.join(d, "action_log.json")))
    print(f"[videocua] {len(task_dirs)} tasks in {args.app_dir}", flush=True)
    frames = n_err = 0
    t0 = time.time()
    for ti, td in enumerate(task_dirs):
        n_ok, _, err = extract_task(td, os.path.join(args.out, os.path.basename(td)), args.image_size)
        frames += n_ok
        n_err += int(err is not None)
        if (ti + 1) % 20 == 0:
            print(f"[videocua] {ti + 1}/{len(task_dirs)} frames={frames} errors={n_err}", flush=True)
    print(f"[videocua] done: {frames} frames, {n_err} errors, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
