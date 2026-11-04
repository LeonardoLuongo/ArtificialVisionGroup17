"""Render tracker output on real SoccerNet frames: boxes + IDs only, no other overlays.

Usage:
    python docs/render_tracking.py <seq> <tracking_txt> <out_dir> [--frames 120 300] [--video START END]
"""
import argparse
import colorsys
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "dataset" / "tracking-2023" / "test"


def load_tracks(path):
    per_frame = defaultdict(list)
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        per_frame[int(float(p[0]))].append((int(float(p[1])), *map(float, p[2:6])))
    return per_frame


def id_color(tid):
    # golden-ratio hue spacing: neighbouring IDs get clearly different colours
    h = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


MONO = (245, 245, 245)  # BGR, single colour for every box


def draw(img, dets, mono=True):
    for tid, x, y, w, h in dets:
        c = MONO if mono else id_color(tid)
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1 if mono else 2, cv2.LINE_AA)
        label = str(tid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 7), (x1 + tw + 7, y1), c, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    return img


def frame_path(seq, fid):
    return TEST_DIR / f"SNMOT-{seq}" / "img1" / f"{fid:06d}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("tracking_txt")
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, nargs="*", default=[])
    ap.add_argument("--video", type=int, nargs=2)
    ap.add_argument("--ffmpeg", default=None)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(a.tracking_txt)

    for fid in a.frames:
        img = draw(cv2.imread(str(frame_path(a.seq, fid))), tracks.get(fid, []))
        cv2.imwrite(str(out / f"frame_{a.seq}_{fid:04d}.png"), img)

    if a.video:
        start, end = a.video
        tmp = out / f"_tmp_{a.seq}.mp4"
        first = cv2.imread(str(frame_path(a.seq, start)))
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        for fid in range(start, end + 1):
            img = cv2.imread(str(frame_path(a.seq, fid)))
            if img is None:
                break
            vw.write(draw(img, tracks.get(fid, [])))
        vw.release()
        final = out / f"tracking_{a.seq}.mp4"
        if a.ffmpeg:
            # H.264 + yuv420p so LinkedIn and phones play it
            subprocess.run([a.ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart",
                            str(final)], check=True)
            tmp.unlink()
        else:
            tmp.rename(final)


if __name__ == "__main__":
    main()
