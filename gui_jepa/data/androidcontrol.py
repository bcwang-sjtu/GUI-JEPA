"""AndroidControl TFRecord reader and accessibility-tree candidate enumeration.

Data: gs://gresearch/android_control (tf.train.Example shards). Per episode:
  episode_id, goal, screenshots (PNG), accessibility_trees (AndroidAccessibilityForest),
  screenshot_widths / screenshot_heights, actions (JSON), step_instructions.

Candidates: visible a11y nodes flagged clickable, long-clickable, editable or
scrollable, de-duplicated by bounds and capped at the 80 smallest by area.
A recorded click is mapped to the smallest enumerated candidate that contains it.
"""
import json
import struct
from dataclasses import dataclass
from typing import Iterator, Optional

from android_env.proto.a11y import android_accessibility_forest_pb2 as forest_pb2


# --------------------------------------------------------------------------- TFRecord IO

def iter_tfrecords(path: str) -> Iterator[bytes]:
    """Pure-Python TFRecord reader (GZIP-aware, CRCs skipped)."""
    import gzip
    with open(path, "rb") as raw:
        magic = raw.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            length, = struct.unpack("<Q", header)
            f.read(4)  # length crc
            payload = f.read(length)
            f.read(4)  # payload crc
            yield payload


def parse_example(payload: bytes) -> dict:
    from tfrecord import example_pb2
    ex = example_pb2.Example()
    ex.ParseFromString(payload)
    out = {}
    for key, feat in ex.features.feature.items():
        kind = feat.WhichOneof("kind")
        if kind == "bytes_list":
            out[key] = list(feat.bytes_list.value)
        elif kind == "int64_list":
            out[key] = list(feat.int64_list.value)
        elif kind == "float_list":
            out[key] = list(feat.float_list.value)
    return out


# --------------------------------------------------------------------------- candidates

@dataclass
class Candidate:
    bounds: tuple          # (l, t, r, b) in screen pixels
    text: str
    content_desc: str
    resource_id: str
    class_name: str
    clickable: bool
    long_clickable: bool
    editable: bool
    scrollable: bool

    @property
    def center(self):
        l, t, r, b = self.bounds
        return ((l + r) / 2.0, (t + b) / 2.0)

    @property
    def area(self):
        l, t, r, b = self.bounds
        return max(0, r - l) * max(0, b - t)

    def label_text(self) -> str:
        for s in (self.text, self.content_desc,
                  self.resource_id.split("/")[-1] if self.resource_id else ""):
            if s:
                return s
        return self.class_name.split(".")[-1]


def _get_prop(node, name, default=None):
    try:
        return getattr(node, name)
    except AttributeError:
        return default


def extract_candidates(forest_bytes: bytes, screen_w: int, screen_h: int,
                       max_candidates: int = 80) -> list:
    """Enumerate interactive elements from a serialised a11y forest."""
    forest = forest_pb2.AndroidAccessibilityForest()
    forest.ParseFromString(forest_bytes)
    cands = []
    for window in forest.windows:
        for node in window.tree.nodes:
            b = node.bounds_in_screen
            l, t, r, bm = b.left, b.top, b.right, b.bottom
            if r <= l or bm <= t:
                continue
            if r < 0 or bm < 0 or l > screen_w or t > screen_h:
                continue
            if not _get_prop(node, "is_visible_to_user", True):
                continue
            clickable = bool(node.is_clickable)
            long_clickable = bool(_get_prop(node, "is_long_clickable", False))
            editable = bool(_get_prop(node, "is_editable", False))
            scrollable = bool(_get_prop(node, "is_scrollable", False))
            if not (clickable or long_clickable or editable or scrollable):
                continue
            cands.append(Candidate(
                bounds=(l, t, r, bm),
                text=(node.text or "")[:128],
                content_desc=(node.content_description or "")[:128],
                resource_id=(node.view_id_resource_name or "")[:128],
                class_name=node.class_name or "",
                clickable=clickable, long_clickable=long_clickable,
                editable=editable, scrollable=scrollable,
            ))
    # de-duplicate identical bounds (prefer a node with text), keep the smallest first
    seen = {}
    for c in cands:
        if c.bounds not in seen or (not seen[c.bounds].label_text() and c.label_text()):
            seen[c.bounds] = c
    cands = sorted(seen.values(), key=lambda c: c.area)
    return cands[:max_candidates]


def match_point_to_candidate(cands: list, x: float, y: float) -> Optional[int]:
    """Smallest-area candidate containing the point."""
    best, best_area = None, None
    for i, c in enumerate(cands):
        l, t, r, b = c.bounds
        if l <= x <= r and t <= y <= b:
            if best is None or c.area < best_area:
                best, best_area = i, c.area
    return best


# --------------------------------------------------------------------------- episodes

@dataclass
class Step:
    episode_id: int
    step_idx: int
    goal: str
    step_instruction: str
    action: dict
    screenshot_png: bytes
    forest_bytes: bytes
    screen_w: int
    screen_h: int


def iter_episodes(shard_path: str) -> Iterator[dict]:
    for payload in iter_tfrecords(shard_path):
        ex = parse_example(payload)
        if "episode_id" in ex:
            yield ex


def episode_steps(ex: dict) -> list:
    """Convert a parsed episode into per-step records."""
    eid = ex["episode_id"][0]
    goal = ex["goal"][0].decode("utf-8", "replace")
    actions = [json.loads(a.decode("utf-8", "replace")) for a in ex.get("actions", [])]
    instrs = [s.decode("utf-8", "replace") for s in ex.get("step_instructions", [])]
    shots = ex.get("screenshots", [])
    forests = ex.get("accessibility_trees", [])
    ws = ex.get("screenshot_widths", [])
    hs = ex.get("screenshot_heights", [])
    steps = []
    for i, act in enumerate(actions):
        if i >= len(shots):
            break
        steps.append(Step(
            episode_id=eid, step_idx=i, goal=goal,
            step_instruction=instrs[i] if i < len(instrs) else "",
            action=act,
            screenshot_png=shots[i],
            forest_bytes=forests[i] if i < len(forests) else b"",
            screen_w=int(ws[i]) if i < len(ws) else 0,
            screen_h=int(hs[i]) if i < len(hs) else 0,
        ))
    return steps
