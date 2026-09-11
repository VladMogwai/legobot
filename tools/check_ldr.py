"""Метрики прочности по готовому .ldr: связность, слабо держащиеся кирпичи, столбики 1x1."""
import sys
from collections import Counter
import numpy as np

DIM = {"3001": (4, 2), "3003": (2, 2), "3010": (4, 1), "3004": (2, 1), "3005": (1, 1)}

def load(path):
    bricks = []
    for line in open(path):
        if not line.startswith("1 "):
            continue
        t = line.split()
        cx, cy, cz = map(float, t[2:5]); rot = list(map(float, t[5:14])); part = t[14][:-4]
        w, l = DIM[part]
        if rot[0] == 0: w, l = l, w
        bricks.append((int(round(cx / 20 - w / 2)), int(round(cz / 20 - l / 2)), int(round(-cy / 24)), w, l))
    return bricks

bricks = load(sys.argv[1])
xs = [x + w for x, _, _, w, _ in bricks]; zs = [z + l for _, z, _, _, l in bricks]; ks = [k for _, _, k, _, _ in bricks]
ids = np.full((max(xs), max(zs), max(ks) + 1), -1, int)
for i, (x, z, k, w, l) in enumerate(bricks):
    assert (ids[x:x+w, z:z+l, k] == -1).all(), "overlap"
    ids[x:x+w, z:z+l, k] = i
occ = ids >= 0
nl = ids.shape[2]

parent = list(range(len(bricks)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
weak_total = weak_loaded = stack11 = 0
conn_hist = Counter()
for i, (x, z, k, w, l) in enumerate(bricks):
    sl = (slice(x, x+w), slice(z, z+l))
    under = ids[sl][:, :, k-1] if k > 0 else None
    over = ids[sl][:, :, k+1] if k+1 < nl else None
    below = int((under >= 0).sum()) if under is not None else w * l  # земля
    above = int((over >= 0).sum()) if over is not None else 0
    if under is not None:
        for j in set(under[under >= 0].tolist()): parent[find(i)] = find(j)
    conn_hist[min(below + above, 8)] += 1
    if below + above <= 2: weak_total += 1
    if below <= 1 and above > 0 and k > 0: weak_loaded += 1
    if (w, l) == (1, 1) and k > 0 and under is not None and under[0, 0] >= 0:
        ux, uz, uk, uw, ul = bricks[under[0, 0]]
        if (uw, ul) == (1, 1): stack11 += 1
comps = len({find(i) for i in range(len(bricks))})
sizes = Counter(f"{min(w,l)}x{max(w,l)}" for _, _, _, w, l in bricks)
print(f"{sys.argv[1]}")
print(f"  кирпичей {len(bricks)}  {dict(sorted(sizes.items()))}")
print(f"  кусков {comps}  |  всего ≤2 штырьков: {weak_total}  |  ≤1 снизу и есть нагрузка сверху: {weak_loaded}  |  1x1 на 1x1: {stack11}")
print(f"  штырьков на кирпич (0..8+): {[conn_hist[i] for i in range(9)]}")
