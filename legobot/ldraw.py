"""Раскладка кирпичей -> файл LDraw (.ldr), который импортирует Studio.

Формат заголовка и `0 STEP` — как в model.ldr внутри .io-файлов Studio.
"""
from itertools import groupby

from .fixtures import Fixture
from .layout import PlacedBrick
from .finish import TILES
from .parts import STUD_LDU, VOCABULARIES
from .slopes import PlacedSlope

_IDENTITY = "1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000"
_ROTATE_90 = "0.000000 0.000000 1.000000 0.000000 1.000000 0.000000 -1.000000 0.000000 0.000000"


def write_ldr(bricks: list[PlacedBrick], path: str, name: str, fixtures: list[Fixture] = (),
              slopes: list[PlacedSlope] = (), steps: list[list] | None = None) -> None:
    """steps — шаги инструкции (списки деталей); без них шаг = слой. Одни и те же шаги
    видят Studio, вьюшка и PDF."""
    lines = [f"0 FILE {name}.ldr", f"0 {name}", f"0 Name:  {name}", "0 Author:  legobot"]
    if steps is None:
        parts = sorted([*bricks, *slopes], key=lambda b: b.layer)   # скос — на своём нижнем слое
        steps = [list(g) for _, g in groupby(parts, key=lambda b: b.layer)]
    for step in steps:
        lines.extend(p.ldraw_line() if isinstance(p, PlacedSlope) else _brick_line(p) for p in step)
        lines.append("0 STEP")
    if fixtures:
        lines.extend(_fixture_line(f) for f in fixtures)
        lines.append("0 STEP")  # фиксированные детали — последним шагом
    lines.append("0 NOFILE")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _fixture_line(f: Fixture) -> str:
    x, y, z = f.position
    rot = " ".join(f"{v:.6f}" for v in f.rotation)
    return f"1 {f.color} {x:.6f} {y:.6f} {z:.6f} {rot} {f.part}.dat"


def _brick_line(b: PlacedBrick) -> str:
    cx = (b.x + b.width / 2) * STUD_LDU
    cz = (b.z + b.length / 2) * STUD_LDU
    cy = -b.layer * b.part.height  # в LDraw вверх — это -Y
    rot = _ROTATE_90 if b.rotated else _IDENTITY
    return f"1 {b.color} {cx:.6f} {cy:.6f} {cz:.6f} {rot} {b.part.number}.dat"


_KNOWN = {p.number: p for v in VOCABULARIES.values() for p in v.parts} | {t.number: t for t in TILES.values()}


def read_bricks(ldraw_text: str) -> tuple[list[PlacedBrick], int]:
    """Кирпичи, пластины и тайлы из текста LDraw (например, модели, отредактированной в Studio).
    Возвращает (детали, сколько строк с незнакомыми деталями пропущено)."""
    bricks, skipped = [], 0
    for line in ldraw_text.splitlines():
        t = line.split()
        if len(t) < 15 or t[0] != "1":
            continue
        part = _KNOWN.get(t[14].lower().removesuffix(".dat"))
        if part is None:
            skipped += 1
            continue
        color = int(t[1]); cx, cy, cz = map(float, t[2:5]); rot = list(map(float, t[5:14]))
        rotated = abs(rot[0]) < 0.5
        w, l = (part.length, part.width) if rotated else (part.width, part.length)
        bricks.append(PlacedBrick(part, int(round(cx / STUD_LDU - w / 2)), int(round(cz / STUD_LDU - l / 2)),
                                  int(round(-cy / part.height)), rotated, color))
    return bricks, skipped
