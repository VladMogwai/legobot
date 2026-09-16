"""Существует ли деталь в таком цвете: по catalog/part_colors.csv (строится tools/catalog.py
из инвентарей Rebrickable). Нет в списке — либо не выпускалась, либо о ней нет данных."""
import csv
from functools import lru_cache
from pathlib import Path

PART_COLORS = Path(__file__).resolve().parent.parent / "catalog" / "part_colors.csv"


@lru_cache
def _known() -> dict[str, set[int]]:
    known: dict[str, set[int]] = {}
    if not PART_COLORS.exists():
        return known
    with open(PART_COLORS, newline="") as f:
        for row in csv.DictReader(f):
            known.setdefault(row["ldraw"], set()).add(int(row["color_code"]))
    return known


def exists(number: str, color: int) -> bool | None:
    """True/False, или None — о детали нет данных о цветах."""
    colors = _known().get(number.lower())
    return None if colors is None else color in colors


def unavailable(parts: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Пары (деталь, цвет), которых по данным не выпускалось."""
    return sorted({(n, c) for n, c in parts if exists(n, c) is False})
