import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import font_manager
from PIL import Image

REPO = Path(sys.argv[1])
IMG = REPO / "docs" / "images"
LI = REPO / "docs" / "linkedin"
FRAME = IMG / "gt_vs_pred.png"

fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in fonts else "DejaVu Sans"
plt.rcParams["font.family"] = FONT

BG = "#0e1712"
CARD = "#16241c"
LINE = "#2a3d31"
TXT = "#eef4f0"
MUTED = "#98ad9f"
GREEN = "#3ddc84"
BLUE = "#4ea8ff"
RED = "#ff6b6b"
AMBER = "#ffc857"


def canvas(w=1080, h=1350, bg=BG):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def card(ax, x, y, w, h, fc=CARD, ec=LINE, r=18, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, p1, p2, color=MUTED, lw=2.5, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=22, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def footer(ax, n, total, w=1080, h=1350):
    ax.text(60, h - 55, "SoccerNet tracking  ·  Artificial Vision  ·  UNISA", color=MUTED, fontsize=15, va="center")
    ax.text(w - 60, h - 55, f"{n}/{total}", color=MUTED, fontsize=15, va="center", ha="right")


def save(fig, path):
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", path)


frame = Image.open(FRAME).convert("RGB")
fw, fh = frame.size
pred_half = frame.crop((fw // 2 + 2, 0, fw, fh))
gt_half = frame.crop((0, 0, fw // 2 - 2, fh))

TOTAL = 5

# ---------------------------------------------------------------- 1. cover
fig, ax = canvas()
ax.text(60, 120, "COMPUTER VISION PROJECT", color=GREEN, fontsize=20, weight="bold")
ax.text(60, 205, "Who's who", color=TXT, fontsize=64, weight="bold")
ax.text(60, 290, "on the pitch?", color=TXT, fontsize=64, weight="bold")
ax.text(60, 365, "Tracking every player in broadcast football video,\nwith a camera that never stops moving.",
        color=MUTED, fontsize=23, va="top", linespacing=1.4)
pw = 960
ph = int(pred_half.size[1] * pw / pred_half.size[0])
ax.imshow(pred_half.resize((pw, ph), Image.LANCZOS), extent=(60, 60 + pw, 480 + ph, 480), zorder=2)
ax.add_patch(FancyBboxPatch((60, 480), pw, ph, boxstyle="round,pad=0,rounding_size=4", fc="none", ec=LINE, lw=2, zorder=3))
y0 = 480 + ph + 60
stats = [("82.4%", "HOTA@0.5", GREEN), ("+20 pts", "association acc.\nvs. baseline", BLUE), ("1.807", "final score\n(PTBS, max 2)", AMBER)]
cw = (960 - 2 * 24) / 3
for i, (v, l, c) in enumerate(stats):
    x = 60 + i * (cw + 24)
    card(ax, x, y0, cw, 190)
    ax.text(x + 28, y0 + 62, v, color=c, fontsize=40, weight="bold", va="center")
    ax.text(x + 28, y0 + 120, l, color=MUTED, fontsize=17, va="top", linespacing=1.3)
footer(ax, 1, TOTAL)
save(fig, LI / "01_cover.png")

# ---------------------------------------------------------------- 2. pipeline (portrait)
fig, ax = canvas()
ax.text(60, 120, "THE PIPELINE", color=GREEN, fontsize=20, weight="bold")
ax.text(60, 195, "Four stages, one job:", color=TXT, fontsize=44, weight="bold")
ax.text(60, 255, "never lose track of a player.", color=TXT, fontsize=44, weight="bold")
steps = [
    ("1", "Detect", "YOLOv8m fine-tuned on SoccerNet", "Backbone frozen, fitness on mAP@50, 1280 px input", BLUE),
    ("2", "Track", "BoT-SORT + camera motion compensation", "Short-term continuity frame to frame", BLUE),
    ("3", "Describe", "OSNet Re-ID embeddings", "Each player crop becomes a 512-d appearance vector", BLUE),
    ("4", "Remember", "RobustMapper (ours)", "Gallery memory, probation, merge & recovery of lost IDs", GREEN),
]
y = 330
bh = 160
gap = 44
for i, (n, title, sub, desc, c) in enumerate(steps):
    card(ax, 60, y, 960, bh, ec=c if c == GREEN else LINE, lw=2.5 if c == GREEN else 1.5)
    ax.add_patch(Circle((130, y + bh / 2), 36, fc=c, ec="none"))
    ax.text(130, y + bh / 2, n, color=BG, fontsize=30, weight="bold", ha="center", va="center")
    ax.text(200, y + 50, title, color=TXT, fontsize=30, weight="bold", va="center")
    ax.text(200, y + 95, sub, color=c, fontsize=20, va="center", weight="bold")
    ax.text(200, y + 132, desc, color=MUTED, fontsize=18, va="center")
    if i < len(steps) - 1:
        arrow(ax, (130, y + bh + 6), (130, y + bh + gap - 6), color=MUTED)
    y += bh + gap
ax.text(60, y + 10, "Output per frame: player boxes with persistent IDs\n+ how many players stand inside each region of interest.",
        color=MUTED, fontsize=19, va="top", linespacing=1.4)
footer(ax, 2, TOTAL)
save(fig, LI / "02_pipeline.png")

# ---------------------------------------------------------------- 3. progression chart (portrait)
names = ["ByteTrack", "BoT-SORT\n+ GMC", "BoT-SORT\n+ OSNet", "RobustMapper\n(final)"]
hota = [69.51, 71.60, 77.71, 82.39]
assa = [54.84, 58.97, 67.93, 75.26]
deta = [86.12, 86.94, 87.40, 90.21]

fig, ax = canvas()
ax.text(60, 120, "WHAT ACTUALLY MOVED THE NEEDLE", color=GREEN, fontsize=20, weight="bold")
ax.text(60, 195, "Detection was already good.", color=TXT, fontsize=42, weight="bold")
ax.text(60, 255, "Identity was the hard part.", color=TXT, fontsize=42, weight="bold")
cax = fig.add_axes([0.11, 0.31, 0.83, 0.44])
cax.set_facecolor(BG)
xs = range(4)
cax.plot(xs, deta, color=BLUE, lw=3, marker="o", ms=10, label="DetA  (finding players)")
cax.plot(xs, hota, color=GREEN, lw=4, marker="o", ms=12, label="HOTA  (overall)")
cax.plot(xs, assa, color=RED, lw=3, marker="o", ms=10, label="AssA  (keeping their ID)")
for vals, c, dy in [(deta, BLUE, 2.2), (hota, GREEN, 2.2), (assa, RED, -4.2)]:
    for x, v in zip(xs, vals):
        cax.text(x, v + dy, f"{v:.1f}", color=c, fontsize=16, ha="center", weight="bold")
cax.set_xticks(list(xs))
cax.set_xticklabels(names, color=TXT, fontsize=16)
cax.set_ylim(48, 96)
cax.set_xlim(-0.35, 3.35)
cax.tick_params(axis="y", colors=MUTED, labelsize=14)
cax.tick_params(axis="x", length=0, pad=12)
cax.grid(axis="y", color=LINE, lw=1)
for s in cax.spines.values():
    s.set_visible(False)
cax.set_ylabel("score (%)", color=MUTED, fontsize=15)
leg = cax.legend(loc="lower right", frameon=False, fontsize=16, labelcolor=TXT)
card(ax, 60, 1055, 960, 160)
ax.text(90, 1100, "AssA: 54.8  →  75.3", color=RED, fontsize=30, weight="bold", va="center")
ax.text(90, 1160, "Most of the gain came from remembering players,\nnot from a better detector.",
        color=MUTED, fontsize=18, va="center", linespacing=1.35)
footer(ax, 3, TOTAL)
save(fig, LI / "03_progression.png")

# ---------------------------------------------------------------- 4. RobustMapper lifecycle
fig, ax = canvas()
ax.text(60, 120, "INSIDE ROBUSTMAPPER", color=GREEN, fontsize=20, weight="bold")
ax.text(60, 195, "Don't trust a new ID", color=TXT, fontsize=44, weight="bold")
ax.text(60, 255, "until it earns it.", color=TXT, fontsize=44, weight="bold")


def state(x, y, w, h, title, body, c):
    card(ax, x, y, w, h, ec=c, lw=2.5)
    ax.text(x + w / 2, y + 48, title, color=c, fontsize=26, weight="bold", ha="center", va="center")
    ax.text(x + w / 2, y + 88, body, color=MUTED, fontsize=16, ha="center", va="top", linespacing=1.35)


state(60, 340, 400, 200, "PROBATION", "new BoT-SORT ID\nwatched for 8 frames\nbefore it counts", AMBER)
state(620, 340, 400, 200, "LOCKED", "confirmed player\nkeeps a gallery of\nup to 50 poses", GREEN)
state(620, 700, 400, 200, "DORMANT", "out of frame or occluded\nremembered for\nup to 1500 frames", BLUE)
state(60, 700, 400, 200, "MERGED", "the \"new\" player looked\nlike an old one (sim > 0.85)\nso it gets the old ID back", RED)

arrow(ax, (468, 440), (612, 440), color=GREEN, lw=3)
ax.text(540, 415, "no match", color=GREEN, fontsize=15, ha="center")
arrow(ax, (760, 545), (760, 695), color=BLUE, lw=3)
ax.text(745, 620, "lost", color=BLUE, fontsize=16, va="center", ha="right")
arrow(ax, (820, 695), (820, 545), color=GREEN, lw=3)
ax.text(838, 620, "recovered:\nsimilar look +\nplausible position", color=GREEN, fontsize=15, va="center", linespacing=1.3)
arrow(ax, (260, 545), (260, 695), color=RED, lw=3)
ax.text(278, 620, "matches an old\nlocked track", color=RED, fontsize=15, va="center", linespacing=1.3)

card(ax, 60, 960, 960, 250)
ax.text(90, 1000, "Why a gallery and not a running average?", color=TXT, fontsize=22, weight="bold", va="center")
ax.text(90, 1045, "A player seen from the back, from the side, and half-hidden\nbehind a teammate looks like three different people.\n"
                  "Keeping several poses instead of one averaged vector\nmade re-identification after occlusions much more reliable.",
        color=MUTED, fontsize=18, va="top", linespacing=1.4)
footer(ax, 4, TOTAL)
save(fig, LI / "04_robustmapper.png")

# ---------------------------------------------------------------- 5. GT vs pred + results
fig, ax = canvas()
ax.text(60, 120, "RESULT", color=GREEN, fontsize=20, weight="bold")
ax.text(60, 195, "Ground truth vs. our output", color=TXT, fontsize=42, weight="bold")
iw = 960
ih = int(gt_half.size[1] * iw / gt_half.size[0])
ih2 = int(ih * 0.68)
ax.text(60, 250, "Ground truth", color=MUTED, fontsize=17, weight="bold", va="center")
g = gt_half.resize((iw, ih), Image.LANCZOS)
p = pred_half.resize((iw, ih), Image.LANCZOS)
# show the top 80% of each half so both fit
g = g.crop((0, 0, iw, ih2))
p = p.crop((0, 0, iw, ih2))
ax.imshow(g, extent=(60, 60 + iw, 270 + ih2, 270), zorder=2)
ax.text(60, 270 + ih2 + 27, "Prediction", color=GREEN, fontsize=17, weight="bold", va="center")
ax.imshow(p, extent=(60, 60 + iw, 270 + ih2 + 45 + ih2, 270 + ih2 + 45), zorder=2)
ytxt = 270 + 2 * ih2 + 62
ax.text(60, ytxt, "Players counted in the blue and red regions: 5 and 4, same as the ground truth.",
        color=MUTED, fontsize=16, va="top")
yk = ytxt + 45
metrics_ = [("HOTA", "82.4", GREEN), ("DetA", "90.2", BLUE), ("AssA", "75.3", RED), ("nMAE", "0.983", AMBER)]
cw = (960 - 3 * 18) / 4
for i, (k, v, c) in enumerate(metrics_):
    x = 60 + i * (cw + 18)
    card(ax, x, yk, cw, 110)
    ax.text(x + cw / 2, yk + 44, v, color=c, fontsize=30, weight="bold", ha="center", va="center")
    ax.text(x + cw / 2, yk + 84, k, color=MUTED, fontsize=16, ha="center", va="center")
footer(ax, 5, TOTAL)
save(fig, LI / "05_results.png")

# ---------------------------------------------------------------- PDF carousel
slides = [Image.open(LI / n).convert("RGB") for n in
          ["01_cover.png", "02_pipeline.png", "03_progression.png", "04_robustmapper.png", "05_results.png"]]
slides[0].save(LI / "carousel.pdf", save_all=True, append_images=slides[1:], resolution=100)
print("saved carousel.pdf")

# ================================================================ README figures (light, landscape)
plt.rcParams.update({"axes.edgecolor": "#cccccc"})

# tracker progression
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
import numpy as np
x = np.arange(4)
wbar = 0.26
for off, vals, c, lab in [(-wbar, hota, "#2e9e5b", "HOTA@0.5"), (0, deta, "#2f7fd1", "DetA@0.5"), (wbar, assa, "#d9534f", "AssA@0.5")]:
    b = ax.bar(x + off, vals, wbar, color=c, label=lab)
    ax.bar_label(b, fmt="%.1f", fontsize=9, padding=2)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("\n", " ") for n in names])
ax.set_ylim(50, 95)
ax.set_ylabel("score (%)")
ax.set_title("Tracking results on our local evaluation split (same fine-tuned YOLOv8m detector)", fontsize=12)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(IMG / "tracker_comparison.png", facecolor="white")
plt.close(fig)
print("saved tracker_comparison.png")

# detector comparison
models = ["YOLOv5mu", "YOLOv8m", "YOLO11m", "YOLO12m"]
m50 = [98.097, 98.106, 98.083, 98.354]
rec = [95.117, 95.613, 95.240, 95.865]
prec = [97.150, 97.598, 97.090, 97.890]
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
for off, vals, c, lab in [(-wbar, m50, "#2f7fd1", "mAP@0.50"), (0, rec, "#d9534f", "Recall"), (wbar, prec, "#8a8f98", "Precision")]:
    b = ax.bar(x + off, vals, wbar, color=c, label=lab)
    ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.get_xticklabels()[1].set_fontweight("bold")
ax.set_ylim(94, 99.2)
ax.set_ylabel("score (%)")
ax.set_title("Detector benchmark on the validation set (YOLOv8m chosen for speed/accuracy)", fontsize=12)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(IMG / "detector_comparison.png", facecolor="white")
plt.close(fig)
print("saved detector_comparison.png")

# pipeline (landscape, light)
fig, ax = canvas(1600, 410, bg="white")
blocks = [("Video frame", "1920x1080\nmoving camera", "#f2f4f3", "#555"),
          ("YOLOv8m", "fine-tuned detector", "#e7f0fb", "#2f7fd1"),
          ("BoT-SORT", "Kalman + camera\nmotion compensation", "#e7f0fb", "#2f7fd1"),
          ("OSNet Re-ID", "appearance\nembeddings", "#e7f0fb", "#2f7fd1"),
          ("RobustMapper", "probation · merge ·\nrecovery · gallery", "#e6f6ec", "#2e9e5b")]
bw, bh, gap = 260, 170, 60
x0 = (1600 - (5 * bw + 4 * gap)) / 2
for i, (t, s, fc, ec) in enumerate(blocks):
    x = x0 + i * (bw + gap)
    card(ax, x, 60, bw, bh, fc=fc, ec=ec, lw=2.5, r=16)
    ax.text(x + bw / 2, 60 + 55, t, color="#1d2b22", fontsize=22, weight="bold", ha="center", va="center")
    ax.text(x + bw / 2, 60 + 95, s, color="#4a5a50", fontsize=15, ha="center", va="top", linespacing=1.3)
    if i < 4:
        arrow(ax, (x + bw + 6, 60 + bh / 2), (x + bw + gap - 6, 60 + bh / 2), color="#666")
c4 = x0 + 3 * (bw + gap) + bw / 2
c5 = x0 + 4 * (bw + gap) + bw / 2
outs = [("tracking_K_17.txt", "frame, global ID, box", c4), ("behavior_K_17.txt", "players per ROI", c5)]
for t, s_, cx in outs:
    card(ax, cx - 148, 300, 296, 90, fc="#fafafa", ec="#999", lw=1.5, r=12)
    ax.text(cx, 330, t, color="#1d2b22", fontsize=16, weight="bold", ha="center", va="center", family="monospace")
    ax.text(cx, 365, s_, color="#4a5a50", fontsize=14, ha="center", va="center")
arrow(ax, (c5, 60 + bh + 6), (c5, 294), color="#666")
arrow(ax, (c5 - 40, 60 + bh + 6), (c4 + 40, 294), color="#666")
fig.savefig(IMG / "pipeline.png", dpi=100, facecolor="white")
plt.close(fig)
print("saved pipeline.png")
