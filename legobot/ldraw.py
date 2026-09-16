"""Раскладка кирпичей -> файл LDraw (.ldr), который импортирует Studio.

Формат заголовка и `0 STEP` — как в model.ldr внутри .io-файлов Studio.
"""
from itertools import groupby

from .fixtures import Fixture
from .layout import PlacedBrick
from .parts import STUD_LDU
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
