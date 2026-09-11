"""Быстрый рендер .ldr в PNG с реальными цветами LDraw: три ракурса."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from legobot.colors import load_palette

DIM = {"3001": (4, 2), "3003": (2, 2), "3010": (4, 1), "3004": (2, 1), "3005": (1, 1)}
rgb = {c.code: np.array(c.rgb) / 255 for c in load_palette()}

bricks = []
for line in open(sys.argv[1]):
    if not line.startswith("1 "):
        continue
    t = line.split(); color = int(t[1]); cx, cy, cz = map(float, t[2:5]); rot = list(map(float, t[5:14]))
    w, l = DIM[t[14][:-4]]
    if rot[0] == 0: w, l = l, w
    bricks.append((int(round(cx / 20 - w / 2)), int(round(cz / 20 - l / 2)), int(round(-cy / 24)), w, l, color))
nx = max(x + w for x, _, _, w, _, _ in bricks); nz = max(z + l for _, z, _, _, l, _ in bricks); nk = max(k for _, _, k, _, _, _ in bricks) + 1
vol = np.zeros((nx, nz, nk), bool); fc = np.zeros((nx, nz, nk, 3))
for x, z, k, w, l, color in bricks:
    vol[x:x + w, z:z + l, k] = True; fc[x:x + w, z:z + l, k] = rgb[color]
fig = plt.figure(figsize=(15, 5))
for i, (elev, azim, title) in enumerate([(15, -35, "3/4"), (0, -90, "front"), (0, 0, "side")]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.voxels(vol, facecolors=fc, edgecolor="k", linewidth=0.1)
    ax.view_init(elev, azim); ax.set_box_aspect((nx, nz, nk * 1.2)); ax.set_axis_off(); ax.set_title(title)
plt.tight_layout()
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".ldr", ".png")
plt.savefig(out, dpi=110); print("->", out)
