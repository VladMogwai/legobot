"""Подбор сетки под заданный размер: число деталей или высоту в сантиметрах."""
from typing import Callable

import trimesh

from .parts import STUD_LDU, Vocabulary
from .pipeline import STUD_MM, Build

MIN_GRID, MAX_GRID = 6, 120


def fit_parts(build_at: Callable[[int], Build], target: int) -> Build:
    return _fit(build_at, lambda b: b.part_count, target, MIN_GRID, MAX_GRID)


def fit_height(build_at: Callable[[int], Build], mesh_path: str, height_cm: float, vocabulary: Vocabulary) -> Build:
    estimate = _grid_estimate_for_height(mesh_path, height_cm, vocabulary)
    return _fit(build_at, lambda b: b.size_mm[2], height_cm * 10, estimate - 3, estimate + 3)


def _grid_estimate_for_height(mesh_path: str, height_cm: float, vocabulary: Vocabulary) -> int:
    mesh = trimesh.load(mesh_path, force="mesh")
    width, length, height = mesh.extents
    layers = height_cm * 10 / (STUD_MM * vocabulary.height / STUD_LDU)
    # Модель масштабируется по вертикали в aspect раз, чтобы воксель стал пропорциями детали;
    # шаг сетки — масштабированная высота, делённая на число слоёв.
    pitch = height / vocabulary.aspect / layers
    return round(max(width, length) / pitch)


def _fit(build_at, measure, target, lo, hi) -> Build:
    """Бинарный поиск по сетке; возвращает сборку, у которой measure ближе всего к target."""
    lo, hi = max(lo, MIN_GRID), min(hi, MAX_GRID)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        result = build_at(mid)
        if best is None or abs(measure(result) - target) < abs(measure(best) - target):
            best = result
        if measure(result) < target:
            lo = mid + 1
        elif measure(result) > target:
            hi = mid - 1
        else:
            break
    return best
