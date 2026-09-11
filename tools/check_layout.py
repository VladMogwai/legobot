"""Диагностика кладки: связность модели, кирпичи без перекрытия шва, на одном штырьке."""
import sys
import numpy as np
from legobot.voxelize import drop_floating, voxelize_mesh
from legobot.layout import layout_bricks
from legobot.parts import BRICKS

v = drop_floating(voxelize_mesh(sys.argv[1], int(sys.argv[2]), BRICKS.aspect)).occupancy
nx, nz, nl = v.shape
bricks = layout_bricks(v, np.zeros(v.shape, int), 14, BRICKS)
ids = np.full((nx, nz, nl), -1, int)
for i, b in enumerate(bricks):
    ids[b.x:b.x+b.width, b.z:b.z+b.length, b.layer] = i
assert (ids >= 0).sum() == v.sum() == sum(b.part.area for b in bricks), "coverage"

parent = list(range(len(bricks)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
stacked = single = 0
for i, b in enumerate(bricks):
    if b.layer == 0:
        continue
    under = ids[b.x:b.x+b.width, b.z:b.z+b.length, b.layer-1]
    below = set(under[under >= 0].tolist())
    for j in below:
        parent[find(i)] = find(j)
    if len(below) == 1: stacked += 1
    if (under >= 0).sum() <= 1: single += 1
components = len({find(i) for i in range(len(bricks))})
print(f"bricks {len(bricks)}  связных кусков: {components}  на одном кирпиче снизу: {stacked}  одним штырьком: {single}")
from collections import Counter
sizes = Counter(find(i) for i in range(len(bricks)))
print("размеры кусков:", sorted(sizes.values(), reverse=True))
