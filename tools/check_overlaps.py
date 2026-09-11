"""Независимая проверка .ldr на пересечения деталей.

Детали каталога — чистые коробки тела (без штырьков). Колёса — цилиндры по спецификации.
Ось внутри кирпича с отверстием и внутри колеса — не пересечение, а соединение.
"""
import sys, itertools
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); sys.path.insert(0, '.')
from ldraw_geometry import bbox as geometry_bbox
from ldr_util import _PARTS
from legobot.fixtures import AXLE, AXLE_BRICK, WHEEL_BY_PART

CONNECTED = {frozenset((AXLE, AXLE_BRICK))} | {frozenset((AXLE, w)) for w in WHEEL_BY_PART}


def box(part):
    p = _PARTS.get(part)
    if p is not None:
        return np.array([-10.0 * p.width, 0.0, -10.0 * p.length]), np.array([10.0 * p.width, float(p.height), 10.0 * p.length])
    if part == AXLE_BRICK:  # кирпич 1x2: тело без штырьков
        return np.array([-20.0, 0.0, -10.0]), np.array([20.0, 24.0, 10.0])
    return geometry_bbox(part + ".dat")


items = []
for line in open(sys.argv[1]):
    if not line.startswith("1 "):
        continue
    t = line.split(); name = t[14][:-4]
    pos = np.array(list(map(float, t[2:5]))); M = np.array(list(map(float, t[5:14]))).reshape(3, 3)
    lo, hi = box(name)
    w = np.array(list(itertools.product(*zip(lo, hi)))) @ M.T + pos
    items.append((name, w.min(0), w.max(0), pos))


def boxes_overlap(a, b):
    return all(min(a[2][k], b[2][k]) - max(a[1][k], b[1][k]) > 1.0 for k in range(3))


def wheel_hits_box(wheel, other):
    """Колесо — цилиндр вдоль X: пересечение, если коробка заходит в круг сечения."""
    spec = WHEEL_BY_PART[wheel[0]]; px, py, pz = wheel[3]
    lo, hi = other[1], other[2]
    if min(hi[0], px + spec.width / 2) - max(lo[0], px - spec.width / 2) <= 1.0:
        return False
    dz = max(lo[2] - pz, 0, pz - hi[2]); dy = max(lo[1] - py, 0, py - hi[1])
    return np.hypot(dz, dy) < spec.diameter / 2 - 1.0


n = 0
for a, b in itertools.combinations(items, 2):
    if frozenset((a[0], b[0])) in CONNECTED or not boxes_overlap(a, b):
        continue
    if a[0] in WHEEL_BY_PART:
        n += wheel_hits_box(a, b)
    elif b[0] in WHEEL_BY_PART:
        n += wheel_hits_box(b, a)
    else:
        n += 1
print(f"{sys.argv[1]}: деталей {len(items)}, пересечений {n}")
