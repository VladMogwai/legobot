"""Копирует в vendor/ldraw/ файлы LDraw, нужные для деталей, которые кладёт бот (с подфайлами
и примитивами): так вьюшка и сборка работают без установленного Studio. Библиотека LDraw —
CC BY 2.0, см. vendor/ldraw/NOTICE.

Запускается сам при выкладывании сайта (tools/deploy_web.py): стоит добавить деталь в словарь —
и её геометрия должна уехать в браузер, иначе модель соберётся, но во вьюшке не покажется.
"""
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from legobot.finish import TILES            # noqa: E402 — после настройки пути
from legobot.fixtures import AXLE, AXLE_BRICK, WHEELS   # noqa: E402
from legobot.parts import RETIRED, VOCABULARIES      # noqa: E402
from legobot.slopes import SLOPES           # noqa: E402
from tools.ldraw_geometry import ROOT, find_file        # noqa: E402

OUT = PROJECT / "vendor" / "ldraw"


def part_numbers() -> set[str]:
    """Все детали, которые бот может положить."""
    # RETIRED — детали, которыми бот уже не выкладывает, но которые читает в чужих моделях:
    # без их геометрии такая модель не покажется во вьюшке.
    roots = ({p.number for v in VOCABULARIES.values() for p in v.parts}
             | {p.number for p in RETIRED} | {t.number for t in TILES.values()})
    roots |= set(SLOPES) | {AXLE_BRICK, AXLE}
    for wheel in WHEELS:
        roots |= {v for v in vars(wheel).values() if isinstance(v, str)}
    return roots


def main() -> None:
    seen: dict[str, Path] = {}
    missing: list[str] = []
    stack = [f"{number}.dat" for number in part_numbers()]
    while stack:
        name = stack.pop()
        key = name.replace("\\", "/").lower()
        if key in seen:
            continue
        path = find_file(name)
        if path is None:
            missing.append(name)
            continue
        seen[key] = path
        for line in open(path, encoding="utf-8", errors="ignore"):
            t = line.split()
            if len(t) >= 15 and t[0] == "1":
                stack.append(t[14])
    for path in seen.values():
        dst = OUT / os.path.relpath(path, ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dst)
    (OUT / "NOTICE").write_text("Файлы библиотеки LDraw (ldraw.org), лицензия CC BY 2.0. Подмножество, нужное деталям legobot.\n")
    print(f"-> {OUT}: {len(seen)} файлов" + (f"; не нашлись: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
