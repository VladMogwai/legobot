import sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
grid = int(sys.argv[2]) if len(sys.argv) > 2 else 30

mesh = trimesh.load(path, force="mesh")
print("faces:", len(mesh.faces), "watertight:", mesh.is_watertight)
print("extents (mm):", np.round(mesh.extents, 1))

# pitch so that the longest side fits in `grid` voxels
pitch = mesh.extents.max() / grid
vox = mesh.voxelized(pitch).fill()
m = vox.matrix
print("grid:", m.shape, "filled voxels:", int(m.sum()))

# 1 voxel = 1 stud (20 LDU). Vertical: brick = 1.2 studs, so we count layers as-is for now.
fig = plt.figure(figsize=(14, 5))
for i, (elev, azim, title) in enumerate([(20, -60, "3/4 view"), (0, 0, "front"), (90, -90, "top")]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.voxels(m, facecolors="#f2c94c", edgecolor="#8a6d1a", linewidth=0.2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(m.shape)
    ax.set_axis_off()
    ax.set_title(f"{title}  {m.shape}  {int(m.sum())} voxels")
plt.tight_layout()
out = f"duck_grid{grid}.png"
plt.savefig(out, dpi=110)
print("saved", out)
