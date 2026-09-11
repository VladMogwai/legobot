"""Демо: раскрашивает STL утки по геометрическим правилам и сохраняет PLY с цветами вершин.
Жёлтое тело, оранжевый клюв, чёрные глаза. Цвета — точные значения из LDConfig."""
import sys
import numpy as np
import trimesh

YELLOW, ORANGE, BLACK = (242, 205, 55), (254, 138, 24), (5, 19, 29)

mesh = trimesh.load(sys.argv[1], force="mesh")
lo, hi = mesh.bounds
V = mesh.vertices
height = hi[2] - lo[2]
cx = (lo[0] + hi[0]) / 2

colors = np.tile(YELLOW, (len(V), 1))
beak = (V[:, 1] < lo[1] + 3.3) & (V[:, 2] > lo[2] + 0.40 * height) & (V[:, 2] < lo[2] + 0.78 * height)
colors[beak] = ORANGE
for side in (-1, 1):
    eye_center = np.array([cx + side * 8.5, lo[1] + 7.3, lo[2] + 0.75 * height])
    eye = np.linalg.norm(V - eye_center, axis=1) < 2.6
    colors[eye] = BLACK

mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.hstack([colors, np.full((len(V), 1), 255)]))
out = sys.argv[2]
mesh.export(out)
print(f"{out}: вершин {len(V)}, клюв {beak.sum()}, глаза {int((colors == BLACK).all(1).sum())}")
