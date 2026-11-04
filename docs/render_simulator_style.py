"""Render GT vs PRED frames in the same style as SIMULATOR/simulator.py (ROI-coloured boxes + counts).

Reuses the simulator's own drawing helpers. ROIs use the same layout as Figure 2 of the report;
counts are recomputed per frame with the base-center rule.

Usage:
    python docs/render_simulator_style.py <seq> <tracking_txt> <out_dir> [--frames 300 500] [--video START END]
"""
import argparse
import subprocess
import sys
import types
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "SIMULATOR"))
sys.modules.setdefault("trackeval", types.ModuleType("trackeval"))  # only the drawing helpers are needed
from simulator import (read_mot_txt_per_frame, draw_rois, draw_roi_counts, draw_boxes_roi_colored,  # noqa: E402
                       roi_pixel_rect, bbox_base_center, point_in_rect)

TEST_DIR = ROOT / "dataset" / "tracking-2023" / "test"

# same layout as the ROIs in Figure 2 of the report
ROIS = {
    "roi1": {"x": 0.14, "y": 0.25, "width": 0.31, "height": 0.46},
    "roi2": {"x": 0.49, "y": 0.02, "width": 0.43, "height": 0.67},
}
ROI_COLORS = {"roi1": (255, 0, 0), "roi2": (0, 0, 255)}  # blue, red (BGR), as in the simulator


def counts(items, w, h):
    rects = [roi_pixel_rect(ROIS[n], w, h) for n in ("roi1", "roi2")]
    c = {1: 0, 2: 0}
    for (_, x, y, bw, bh) in items:
        cx, cy = bbox_base_center(x, y, bw, bh)
        for rid, r in enumerate(rects, 1):
            if point_in_rect(cx, cy, r):
                c[rid] += 1
                break
    return c


def panel(frame, items, tag, seq, fid, n_frames):
    img = frame.copy()
    h, w = img.shape[:2]
    cnt = counts(items, w, h)
    cv2.putText(img, f"TOTAL: {cnt[1] + cnt[2]}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    draw_rois(img, ROIS, ROI_COLORS)
    draw_roi_counts(img, ROIS, ROI_COLORS, cnt, roi_ids=(1, 2))
    draw_boxes_roi_colored(img, items, ROIS, ROI_COLORS, out_color=(0, 255, 255), draw_ids=True)
    cv2.putText(img, f"{tag} | video {seq} | 17 | frame {fid}/{n_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def compose(seq, fid, gt, pred, n_frames):
    frame = cv2.imread(str(TEST_DIR / f"SNMOT-{seq}" / "img1" / f"{fid:06d}.jpg"))
    if frame is None:
        return None
    left = panel(frame, gt.get(fid, []), "GT", seq, fid, n_frames)
    right = panel(frame, pred.get(fid, []), "PRED", seq, fid, n_frames)
    side = cv2.hconcat([left, right])
    cv2.line(side, (left.shape[1], 0), (left.shape[1], side.shape[0] - 1), (255, 255, 255), 2)
    stacked = cv2.vconcat([left, right])
    cv2.line(stacked, (0, left.shape[0]), (stacked.shape[1] - 1, left.shape[0]), (255, 255, 255), 2)
    return side, stacked, right


def encode(ffmpeg, src, dst, scale=None):
    vf = ["-vf", f"scale={scale}"] if scale else []
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), *vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(dst)], check=True)
    Path(src).unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("tracking_txt")
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, nargs="*", default=[])
    ap.add_argument("--video", type=int, nargs=2)
    ap.add_argument("--ffmpeg", required=False)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seq_dir = TEST_DIR / f"SNMOT-{a.seq}"
    gt = read_mot_txt_per_frame(str(seq_dir / "gt" / "gt.txt"))
    pred = read_mot_txt_per_frame(a.tracking_txt)
    n_frames = len(list((seq_dir / "img1").glob("*.jpg")))

    for fid in a.frames:
        side, stacked, right = compose(a.seq, fid, gt, pred, n_frames)
        cv2.imwrite(str(out / f"sim_{a.seq}_{fid:04d}_side.png"), side)
        cv2.imwrite(str(out / f"sim_{a.seq}_{fid:04d}_stacked.png"), stacked)
        cv2.imwrite(str(out / f"sim_{a.seq}_{fid:04d}_pred.png"), right)

    if a.video:
        start, end = a.video
        tmp_side, tmp_stack = out / "_side.mp4", out / "_stack.mp4"
        vs = vt = None
        for fid in range(start, end + 1):
            r = compose(a.seq, fid, gt, pred, n_frames)
            if r is None:
                break
            side, stacked, _ = r
            if vs is None:
                four = cv2.VideoWriter_fourcc(*"mp4v")
                vs = cv2.VideoWriter(str(tmp_side), four, 25, (side.shape[1], side.shape[0]))
                vt = cv2.VideoWriter(str(tmp_stack), four, 25, (stacked.shape[1], stacked.shape[0]))
            vs.write(side)
            vt.write(stacked)
        vs.release()
        vt.release()
        if a.ffmpeg:
            encode(a.ffmpeg, tmp_side, out / f"sim_{a.seq}_side.mp4", "1920:540")
            encode(a.ffmpeg, tmp_stack, out / f"sim_{a.seq}_stacked.mp4", "1080:1216")


if __name__ == "__main__":
    main()
