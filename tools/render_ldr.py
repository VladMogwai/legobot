"""Быстрый рендер .ldr в PNG с реальными цветами LDraw: три ракурса."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit('/', 1)[0]); sys.path.insert(0, '.')
from legobot.colors import load_palette
from ldr_util import load_bricks, load_fixtures
from legobot.fixtures import WHEEL_BY_PART

rgb = {c.code: np.array(c.rgb) / 255 for c in load_palette()}
bricks, height = load_bricks(sys.argv[1])
aspect = height / 20
nx = max(x + w for x, _, _, w, _, _ in bricks); nz = max(z + l for _, z, _, _, l, _ in bricks); nk = max(k for _, _, k, _, _, _ in bricks) + 1
vol = np.zeros((nx, nz, nk), bool); fc = np.zeros((nx, nz, nk, 3))
for x, z, k, w, l, color in bricks:
    vol[x:x + w, z:z + l, k] = True; fc[x:x + w, z:z + l, k] = rgb[color]
# колёса рисуем дисками из вокселей по их реальному диаметру
specs = WHEEL_BY_PART
X, Z, Y = np.meshgrid((np.arange(nx) + .5) * 20, (np.arange(nz) + .5) * 20, -np.arange(nk) * height + height / 2, indexing="ij")
for part, color, (px, py, pz), _ in load_fixtures(sys.argv[1]):
    if part in specs:
        spec = specs[part]
        disc = (np.abs(X - px) <= spec.width / 2) & (np.hypot(Z - pz, Y - py) <= spec.diameter / 2)
        vol |= disc; fc[disc] = (0.15, 0.15, 0.15)
fig = plt.figure(figsize=(15, 5))
for i, (elev, azim, title) in enumerate([(15, -35, "3/4"), (0, -90, "front"), (0, 0, "side")]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.voxels(vol, facecolors=fc, edgecolor="k", linewidth=0.1)
    ax.view_init(elev, azim); ax.set_box_aspect((nx, nz, nk * aspect)); ax.set_axis_off(); ax.set_title(title)
plt.tight_layout()
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].rsplit(".", 1)[0] + ".png"
plt.savefig(out, dpi=110); print("->", out)
