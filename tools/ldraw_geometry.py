"""Габариты детали LDraw с рекурсивным разбором подфайлов (parts/, p/, UnOfficial)."""
import os
from functools import lru_cache
import numpy as np

ROOT = "/Applications/Studio 2.0/ldraw"
SEARCH = ["parts", "p", "UnOfficial/parts", "UnOfficial/p", "parts/s", "p/48", "UnOfficial/parts/s"]


def find_file(name):
    name = name.replace("\\", "/")
    for d in SEARCH:
        path = os.path.join(ROOT, d, name)
        if os.path.exists(path):
            return path
        path = os.path.join(ROOT, d, name.lower())
        if os.path.exists(path):
            return path
    return None


@lru_cache(maxsize=None)
def points(name, depth=0):
    """Все вершины детали в её локальных координатах (N x 3)."""
    path = find_file(name)
    if path is None or depth > 12:
        return np.zeros((0, 3))
    pts = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        t = line.split()
        if not t:
            continue
        if t[0] in ("2", "3", "4"):
            n = int(t[0]); vals = list(map(float, t[2:2 + 3 * n]))
            pts += [vals[i:i + 3] for i in range(0, 3 * n, 3)]
        elif t[0] == "1" and len(t) >= 15:
            pos = np.array(list(map(float, t[2:5]))); M = np.array(list(map(float, t[5:14]))).reshape(3, 3)
            sub = points(t[14], depth + 1)
            if len(sub):
                pts += (sub @ M.T + pos).tolist()
    return np.array(pts) if pts else np.zeros((0, 3))


def bbox(name):
    p = points(name)
    return (p.min(0), p.max(0)) if len(p) else None


def header(name):
    path = find_file(name)
    return open(path, encoding="utf-8", errors="ignore").readline().strip().lstrip("0 ") if path else "?"
