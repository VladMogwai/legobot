"""Инструкция по сборке: шаги, списки деталей, PDF.

Шаг — слой; слой больше MAX_PER_STEP деталей делится на части по порядку кладки.
PDF: страница на шаг — собранное ранее приглушено, новые детали в цвете с обводкой,
внизу детали шага. Последняя страница — полный список деталей.
"""
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colors import load_palette
from .fixtures import Fixture
from .layout import PlacedBrick
from .parts import STUD_LDU, part_name

MAX_PER_STEP = 30


@dataclass(frozen=True)
class PartLine:
    number: str
    name: str
    color: int
    color_name: str
    quantity: int


def split_steps(bricks: list[PlacedBrick], max_per_step: int = MAX_PER_STEP) -> list[list[PlacedBrick]]:
    steps = []
    for layer in sorted({b.layer for b in bricks}):
        in_layer = sorted((b for b in bricks if b.layer == layer), key=lambda b: (b.z, b.x))
        pieces = max(1, -(-len(in_layer) // max_per_step))
        size = -(-len(in_layer) // pieces)
        steps.extend(in_layer[i:i + size] for i in range(0, len(in_layer), size))
    return steps


def bill_of_materials(bricks: list[PlacedBrick], fixtures: list[Fixture] = ()) -> list[PartLine]:
    names = {c.code: c.name for c in load_palette(common_only=False)}
    counts = Counter((b.part.number, b.color) for b in bricks)
    counts.update((f.part, f.color) for f in fixtures)
    lines = [PartLine(n, part_name(n), c, names.get(c, str(c)), q) for (n, c), q in counts.items()]
    return sorted(lines, key=lambda l: (-l.quantity, l.name, l.color_name))


def write_bom(lines: list[PartLine], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "name", "color_code", "color", "quantity"])
        for l in lines:
            w.writerow([l.number, l.name, l.color, l.color_name, l.quantity])


def write_pdf(bricks: list[PlacedBrick], steps: list[list[PlacedBrick]], path: str, title: str,
              fixtures: list[Fixture] = ()) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rgb = {c.code: np.array(c.rgb) / 255 for c in load_palette(common_only=False)}
    nx = max(b.x + b.width for b in bricks); nz = max(b.z + b.length for b in bricks)
    nk = max(b.layer for b in bricks) + 1
    aspect = bricks[0].part.height / STUD_LDU
    total = bill_of_materials(bricks, fixtures)

    with PdfPages(path) as pdf:
        _cover(pdf, plt, title, bricks, total, rgb, nx, nz, nk, aspect)
        done: list[PlacedBrick] = []
        for i, step in enumerate(steps, 1):
            fig = plt.figure(figsize=(8.27, 11.69))  # A4
            ax = fig.add_axes([0.02, 0.32, 0.96, 0.62], projection="3d")
            _draw(ax, done, step, rgb, nx, nz, nk, aspect)
            ax.set_title(f"Шаг {i} из {len(steps)}", fontsize=16, loc="left")
            fig.text(0.05, 0.28, "Детали шага:", fontsize=12, weight="bold")
            for j, line in enumerate(bill_of_materials(step)[:14]):
                fig.text(0.07, 0.25 - j * 0.017, f"{line.quantity:3d} ×  {line.name}  —  {line.color_name}", fontsize=10, family="monospace")
            pdf.savefig(fig); plt.close(fig)
            done = done + step
        _bom_page(pdf, plt, total)


def _draw(ax, done, step, rgb, nx, nz, nk, aspect):
    vol = np.zeros((nx, nz, nk), bool); fc = np.zeros((nx, nz, nk, 4))
    for b in done:
        vol[b.x:b.x + b.width, b.z:b.z + b.length, b.layer] = True
        fc[b.x:b.x + b.width, b.z:b.z + b.length, b.layer] = (*rgb.get(b.color, (0.5, 0.5, 0.5)), 0.25)
    for b in step:
        vol[b.x:b.x + b.width, b.z:b.z + b.length, b.layer] = True
        fc[b.x:b.x + b.width, b.z:b.z + b.length, b.layer] = (*rgb.get(b.color, (0.5, 0.5, 0.5)), 1.0)
    ax.voxels(vol, facecolors=fc, edgecolor=(0, 0, 0, 0.15), linewidth=0.2)
    ax.set_xlim(0, nx); ax.set_ylim(0, nz); ax.set_zlim(0, nk)
    ax.set_box_aspect((nx, nz, nk * aspect)); ax.view_init(28, -55); ax.set_axis_off()


def _cover(pdf, plt, title, bricks, total, rgb, nx, nz, nk, aspect):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.02, 0.35, 0.96, 0.55], projection="3d")
    _draw(ax, [], bricks, rgb, nx, nz, nk, aspect)
    fig.text(0.05, 0.93, title, fontsize=22, weight="bold")
    fig.text(0.05, 0.30, f"Деталей: {sum(l.quantity for l in total)}   Типов деталей: {len(total)}", fontsize=12)
    fig.text(0.05, 0.27, f"Размер: {nx * 0.8:.1f} × {nz * 0.8:.1f} × {nk * aspect * 0.8:.1f} см", fontsize=12)
    pdf.savefig(fig); plt.close(fig)


def _bom_page(pdf, plt, total):
    per_page = 40
    for start in range(0, len(total), per_page):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.05, 0.95, "Список деталей", fontsize=18, weight="bold")
        for j, line in enumerate(total[start:start + per_page]):
            fig.text(0.05, 0.91 - j * 0.021, f"{line.quantity:4d} ×  {line.number:<10} {line.name:<34} {line.color_name}", fontsize=9, family="monospace")
        pdf.savefig(fig); plt.close(fig)
