"""Копирует в vendor/ldraw/ файлы LDraw, нужные для деталей, которые кладёт бот (с подфайлами
и примитивами): так вьюшка и сборка работают без установленного Studio. Библиотека LDraw —
CC BY 2.0, см. vendor/ldraw/NOTICE."""
import os
import shutil
from pathlib import Path

from legobot.finish import TILES
from legobot.fixtures import AXLE, AXLE_BRICK, WHEELS
from legobot.parts import VOCABULARIES
from legobot.slopes import SLOPES
from tools.ldraw_geometry import ROOT, find_file

OUT = Path("vendor/ldraw")

roots = {p.number for v in VOCABULARIES.values() for p in v.parts} | {t.number for t in TILES.values()} | set(SLOPES) | {AXLE_BRICK, AXLE}
for w in WHEELS:
    roots |= {v for v in vars(w).values() if isinstance(v, str)}
seen = {}
stack = [f"{r}.dat" for r in roots]
while stack:
    name = stack.pop()
    key = name.replace("\\", "/").lower()
    if key in seen:
        continue
    path = find_file(name)
    if path is None:
        print("нет файла:", name)
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
print(f"скопировано {len(seen)} файлов в {OUT}")
