"""Независимая проверка .ldr: габариты деталей берутся из .dat библиотеки Studio,
матрицы — из файла; ищем пересечения тел кирпичей в мировых LDU."""
import sys, itertools
import numpy as np

LIB = "/Applications/Studio 2.0/ldraw/parts/"

def bbox(part):
    pts = []
    for line in open(LIB + part + ".dat", encoding="utf-8", errors="ignore"):
        t = line.split()
        if t and t[0] in ("3", "4"):
            n = int(t[0]); vals = list(map(float, t[2:2 + 3 * n]))
            pts += [vals[i:i + 3] for i in range(0, 3 * n, 3)]
    p = np.array(pts); return p.min(0), p.max(0)

boxes = []
for line in open(sys.argv[1]):
    if not line.startswith("1 "):
        continue
    t = line.split()
    pos = np.array(list(map(float, t[2:5]))); M = np.array(list(map(float, t[5:14]))).reshape(3, 3)
    lo, hi = bbox(t[14][:-4])
    w = np.array(list(itertools.product(*zip(lo, hi)))) @ M.T + pos
    lo, hi = w.min(0), w.max(0)
    lo[1] = max(lo[1], hi[1] - 24)  # тело без штырьков (у пластин тело 8, штырьки 4 — тоже отрезаются)
    boxes.append((lo, hi))
n = 0
for (alo, ahi), (blo, bhi) in itertools.combinations(boxes, 2):
    if all(min(ahi[k], bhi[k]) - max(alo[k], blo[k]) > 1.0 for k in range(3)):
        n += 1
print(f"{sys.argv[1]}: кирпичей {len(boxes)}, пересечений {n}")
