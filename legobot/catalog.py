"""Существует ли деталь в таком цвете: по catalog/part_colors.csv (строится tools/catalog.py
из инвентарей Rebrickable). Нет в списке — либо не выпускалась, либо о ней нет данных."""
import csv
from functools import lru_cache
from pathlib import Path

PART_COLORS = Path(__file__).resolve().parent.parent / "catalog" / "part_colors.csv"


MIN_SETS = 2   # пара деталь+цвет из меньшего числа наборов считается редкой: на BrickLink её обычно не купить


@lru_cache
def _known() -> dict[str, dict[int, int]]:
    """номер детали -> {код цвета: в скольких наборах}."""
    known: dict[str, dict[int, int]] = {}
    if not PART_COLORS.exists():
        return known
    with open(PART_COLORS, newline="") as f:
        for row in csv.DictReader(f):
            known.setdefault(row["ldraw"], {})[int(row["color_code"])] = int(row.get("sets") or 0)
    return known


def exists(number: str, color: int, min_sets: int = 0) -> bool | None:
    """True/False, или None — о детали нет данных о цветах. min_sets — не меньше стольких наборов."""
    colors = _known().get(number.lower())
    return None if colors is None else colors.get(color, -1) >= min_sets


PAB = Path(__file__).resolve().parent.parent / "catalog" / "pab.csv"


@lru_cache
def _pab() -> dict[tuple[str, int], int | None] | None:
    """(деталь, цвет) -> цена в центах на Pick a Brick, None — цены нет, но деталь продаётся.
    Весь словарь None — выгрузки нет (см. tools/pab.py). В браузер уезжает список без цен:
    доступность деталей и цветов — факт каталога, цены lego.com мы не публикуем."""
    if not PAB.exists():
        return None
    with open(PAB, newline="") as f:
        return {(row["ldraw"], int(row["color_code"])): int(row["cents"]) if row.get("cents") else None
                for row in csv.DictReader(f)}


def available(number: str, color: int) -> bool | None:
    """Покупаемая пара. Есть выгрузка Pick a Brick — продаётся ли там (для деталей, которые PaB
    вообще предлагает); иначе — выпускалась хотя бы в MIN_SETS наборах."""
    pab = _pab()
    if pab is not None and any(n == number.lower() for n, _ in pab):
        return (number.lower(), color) in pab
    return exists(number, color, MIN_SETS)


def price_cents(number: str, color: int) -> int | None:
    """Цена на Pick a Brick, если известна."""
    pab = _pab()
    return None if pab is None else pab.get((number.lower(), color))


def purchasable_parts() -> set[tuple[str, int]] | None:
    """Пары деталь+цвет, которые продаются; None — выгрузки нет."""
    pab = _pab()
    return None if pab is None else set(pab)


def unavailable(parts: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Пары (деталь, цвет), которых не купить: не выпускались или редкие."""
    return sorted({(n, c) for n, c in parts if available(n, c) is False})


def purchasable_colors() -> set[int] | None:
    """Коды цветов, в которых на Pick a Brick есть хотя бы кирпич 1x1 или пластина 1x1;
    None — выгрузки нет."""
    pab = _pab()
    if pab is None:
        return None
    return {code for (number, code) in pab if number in ("3005", "3024")}
