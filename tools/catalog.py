"""Каталог: python tools/catalog.py -> catalog/colors.csv, catalog/parts.csv, catalog/part_colors.csv.

Запускается на машине со Studio; результат лежит в git, и legobot читает его, а не Studio —
так бот работает и на сервере, где Studio нет.

Источники:
- Studio: data/StudioPartDefinition2.txt (номер LDraw, номер BrickLink, название, категория)
  и ldraw/parts/*.dat (есть ли геометрия — можем ли мы деталь положить).
- Rebrickable (corpus/rebrickable/*.csv, свободные выгрузки): в каких цветах деталь
  выпускалась — по инвентарям всех наборов (inventory_parts) и элементам (elements).
  Цвета сопоставляются по имени, а телесные (названы иначе: Nougat = Flesh) — по id и RGB.
- Studio: data/elementInfoList.json — список элементов LEGO (деталь BrickLink + цвет BrickLink),
  цвет BrickLink переводится в LDraw по StudioColorDefinition.txt.
Деталь считается выпускавшейся в цвете, если так говорит хотя бы один источник.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from legobot.colors import _LINE, _SPECIAL, COMMON_COLORS, load_palette
from legobot.parts import LIBRARY

STUDIO_DATA = Path("/Applications/Studio 2.0/data")
REBRICKABLE = Path("corpus/rebrickable")
OUT = Path("catalog")


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def studio_parts() -> dict[str, dict]:
    """{номер LDraw без .dat: {bl, name, category}} — только детали с геометрией у нас."""
    categories = {}
    with open(STUDIO_DATA / "StudioCategoryDefinition.txt", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            categories[(row["BL Index"], row["subIndex"])] = row["BL CatalogName"].strip()
    parts = {}
    with open(STUDIO_DATA / "StudioPartDefinition2.txt", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ldraw = row["LDraw ItemNo"].strip().lower().removesuffix(".dat")
            if not ldraw or row["IsAssembly?"] == "True":
                continue
            if not (LIBRARY / "parts" / f"{ldraw}.dat").exists() and not (LIBRARY / "UnOfficial/parts" / f"{ldraw}.dat").exists():
                continue
            parts.setdefault(ldraw, {
                "bl": row["BL ItemNo"].strip(),
                "name": row["Description"].strip(),
                "category": categories.get((row["BLCatalogIndex"], row["BLCatalogSubIndex"]), ""),
            })
    return parts


def rebrickable_colors() -> dict[int, int | None]:
    """id цвета Rebrickable -> код LDraw. Сначала по имени; иначе по id, если под тем же кодом в
    LDConfig тот же цвет (телесные: 92 Nougat = Flesh). Только по id нельзя: 326 у Rebrickable —
    Olive Green, у LDraw — Yellowish_Green. None — нет такого."""
    palette = {c.code: c for c in load_palette(common_only=False)}
    by_name = {normalize(c.name): c.code for c in palette.values()}
    out = {}
    with open(REBRICKABLE / "colors.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid, name = int(row["id"]), normalize(row["name"])
            if name in by_name:
                out[rid] = by_name[name]
            elif rid in palette and _rgb_distance(row["rgb"], palette[rid].rgb) < SAME_COLOR_RGB:
                out[rid] = rid
            else:
                out[rid] = None
    return out


SAME_COLOR_RGB = 100   # евклидово расстояние RGB, ниже которого это один цвет в двух справочниках


def _rgb_distance(hexrgb: str, rgb: tuple) -> float:
    a = [int(hexrgb[i:i + 2], 16) for i in (0, 2, 4)]
    return sum((x - y) ** 2 for x, y in zip(a, rgb)) ** 0.5


def studio_part_colors() -> dict[str, set[int]]:
    """номер детали BrickLink -> {коды LDraw}, по списку элементов Studio."""
    bl_to_ldraw = {}
    with open(STUDIO_DATA / "StudioColorDefinition.txt", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                bl_to_ldraw[row["BL Color Code"].strip()] = int(row["LDraw Color Code"])
            except (ValueError, TypeError):
                continue
    colors = defaultdict(set)
    with open(STUDIO_DATA / "elementInfoList.json", encoding="utf-8") as f:
        for e in json.load(f):
            code = bl_to_ldraw.get(e["blColorCode"])
            if code is not None:
                colors[e["blItemNo"].lower()].add(code)
    return colors


def rebrickable_part_colors() -> dict[str, dict[int, int]]:
    """номер детали -> {id цвета Rebrickable: в скольких инвентарях наборов}; элементы без
    наборов — 0. Число наборов — мера покупаемости: пара из одного набора на BrickLink обычно
    не продаётся (Studio такие помечает восклицательным знаком)."""
    colors: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    with open(REBRICKABLE / "inventory_parts.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            colors[row["part_num"].lower()][int(row["color_id"])] += 1
    with open(REBRICKABLE / "elements.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            colors[row["part_num"].lower()].setdefault(int(row["color_id"]), 0)
    return colors


def colors() -> None:
    """Все цвета LDConfig (код, имя, RGB справочника, RGB Studio, сплошной ли, ходовой ли)."""
    studio = {}
    with open(STUDIO_DATA / "StudioColorDefinition.txt", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                code = int(row["LDraw Color Code"])
            except (ValueError, TypeError):
                continue
            if row["RGB value"].startswith("#"):
                studio.setdefault(code, row["RGB value"].lstrip("#").lower())
    with open(LIBRARY / "LDConfig.ldr", encoding="utf-8", errors="ignore") as f, open(OUT / "colors.csv", "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["code", "name", "rgb", "studio_rgb", "solid", "common"])
        for line in f:
            m = _LINE.match(line)
            if not m:
                continue
            name, code, rgb = m.groups()
            solid = not any(tag in line for tag in _SPECIAL)
            w.writerow([code, name, rgb.lower(), studio.get(int(code), rgb.lower()), int(solid), int(int(code) in COMMON_COLORS)])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    colors()
    parts = studio_parts()
    color_map = rebrickable_colors()
    part_colors = rebrickable_part_colors()
    studio_colors = studio_part_colors()
    names = {c.code: c.name for c in load_palette(common_only=False)}
    with_colors = 0
    with open(OUT / "parts.csv", "w", newline="") as fp, open(OUT / "part_colors.csv", "w", newline="") as fc:
        wp, wc = csv.writer(fp), csv.writer(fc)
        wp.writerow(["ldraw", "bricklink", "name", "category", "colors"])
        wc.writerow(["ldraw", "color_code", "color", "sets"])
        for number, p in sorted(parts.items()):
            # Rebrickable нумерует как LDraw, иногда как BrickLink
            rb = part_colors.get(number) or part_colors.get(p["bl"].lower()) or {}
            sets: dict[int, int] = defaultdict(int)
            for c, n in rb.items():
                if color_map.get(c) is not None:
                    sets[color_map[c]] = max(sets[color_map[c]], n)
            for code in studio_colors.get(p["bl"].lower(), set()) | studio_colors.get(number, set()):
                sets.setdefault(code, 0)
            codes = sorted(sets.keys() & names.keys())
            with_colors += bool(codes)
            wp.writerow([number, p["bl"], p["name"], p["category"], len(codes)])
            for code in codes:
                wc.writerow([number, code, names[code], sets[code]])
    print(f"деталей с геометрией: {len(parts)}, из них с известными цветами: {with_colors}")
    print(f"-> {OUT / 'colors.csv'}, {OUT / 'parts.csv'}, {OUT / 'part_colors.csv'}")


if __name__ == "__main__":
    main()
