"""Pick a Brick: python tools/pab.py -> catalog/pab.csv (деталь LDraw, цвет LDraw, цена в центах, канал).

Вход — corpus/pab_us.csv: выгрузка каталога lego.com Pick a Brick (элемент; design ID; название;
LEGO-код цвета; имя цвета; hex; в наличии; канал pab/bap; цена в центах), только кирпичи, пластины
и тайлы. Это то, что LEGO продаёт поштучно сейчас; для покупки это точнее любого каталога.
Данные региональные и меняются — оба файла не в git, без них укладчик оценивает покупаемость
по числу наборов (см. legobot/catalog.py).

Название -> деталь: «BRICK 2X8» — кирпич 2x8 из словаря legobot; LEGO-код цвета -> LDraw —
по StudioColorDefinition.txt (колонка «LDD color code»).
"""
import csv
import re
from pathlib import Path

from legobot.finish import TILES
from legobot.parts import BRICKS, PLATES

STUDIO_COLORS = Path("/Applications/Studio 2.0/data/StudioColorDefinition.txt")
SRC = Path("corpus/pab_us.csv")
OUT = Path("catalog/pab.csv")
KIND = {"BRICK": BRICKS, "PLATE": PLATES}


def lego_to_ldraw() -> dict[int, int]:
    out = {}
    with open(STUDIO_COLORS, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                out.setdefault(int(row["LDD color code"]), int(row["LDraw Color Code"]))
            except (ValueError, TypeError):
                continue
    return out


def part_number(name: str) -> str | None:
    m = re.fullmatch(r"(BRICK|PLATE|FLAT TILE) (\d+)X(\d+)", name)
    if not m:
        return None
    kind, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    if kind == "FLAT TILE":
        tile = TILES.get((max(a, b), min(a, b)))
        return tile.number if tile else None
    part = KIND[kind].by_size(a, b) or KIND[kind].by_size(b, a)
    return part.number if part else None


def main() -> None:
    colors = lego_to_ldraw()
    rows, skipped = [], set()
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            number, code = part_number(r["name"]), colors.get(int(r["color_key"]))
            if number is None or code is None or r["available"] != "1":
                skipped.add(r["name"] if number is None else r["color_name"])
                continue
            rows.append((number, code, int(r["cents"] or 0), r["channel"]))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ldraw", "color_code", "cents", "channel"])
        w.writerows(sorted(set(rows)))
    print(f"-> {OUT}: {len(set(rows))} пар деталь+цвет; пропущено (не наши детали / неизвестный цвет): {sorted(skipped)[:12]}…")


if __name__ == "__main__":
    main()
