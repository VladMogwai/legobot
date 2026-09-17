"""Каталог: python tools/catalog.py -> catalog/colors.csv, catalog/parts.csv, catalog/part_colors.csv.

Запускается на машине со Studio; результат лежит в git, и legobot читает его, а не Studio —
так бот работает и на сервере, где Studio нет.

Источники:
- Studio: data/StudioPartDefinition2.txt (номер LDraw, номер BrickLink, название, категория)
  и ldraw/parts/*.dat (есть ли геометрия — можем ли мы деталь положить).
- Rebrickable (corpus/rebrickable/*.csv, свободные выгрузки): в каких цветах деталь
  выпускалась — по инвентарям всех наборов (inventory_parts) и элементам (elements).
  Id цветов Rebrickable совпадают с кодами LDraw (85 Dark Purple, 92 Nougat = Flesh);
  по имени сопоставлять нельзя — телесные названы иначе.
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
    """id цвета Rebrickable -> код LDraw: по id, если такой код есть в LDConfig, иначе по имени
    (None — нет такого)."""
    codes = {c.code for c in load_palette(common_only=False)}
    by_name = {normalize(c.name): c.code for c in load_palette(common_only=False)}
    out = {}
    with open(REBRICKABLE / "colors.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = int(row["id"])
            out[rid] = rid if rid in codes else by_name.get(normalize(row["name"]))
    return out


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


def rebrickable_part_colors() -> dict[str, set[int]]:
    """номер детали -> {id цветов Rebrickable}, по инвентарям наборов и элементам."""
    colors = defaultdict(set)
    for file, part_col, color_col in (("inventory_parts.csv", "part_num", "color_id"), ("elements.csv", "part_num", "color_id")):
        with open(REBRICKABLE / file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                colors[row[part_col].lower()].add(int(row[color_col]))
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
        wc.writerow(["ldraw", "color_code", "color"])
        for number, p in sorted(parts.items()):
            # Rebrickable нумерует как LDraw, иногда как BrickLink
            rb = part_colors.get(number) or part_colors.get(p["bl"].lower()) or set()
            codes = sorted(({color_map[c] for c in rb if color_map.get(c) is not None}
                            | studio_colors.get(p["bl"].lower(), set()) | studio_colors.get(number, set())) & names.keys())
            with_colors += bool(codes)
            wp.writerow([number, p["bl"], p["name"], p["category"], len(codes)])
            for code in codes:
                wc.writerow([number, code, names[code]])
    print(f"деталей с геометрией: {len(parts)}, из них с известными цветами: {with_colors}")
    print(f"-> {OUT / 'colors.csv'}, {OUT / 'parts.csv'}, {OUT / 'part_colors.csv'}")


if __name__ == "__main__":
    main()
