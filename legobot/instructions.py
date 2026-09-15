"""Инструкция по сборке: шаги, списки деталей, PDF в стиле Studio Instruction Maker.

Страница шага: альбомная, крупный номер, всё собранное в цвете со штырьками, новые
детали обведены красным, слева внизу голубая плашка с картинками деталей шага и «N×».
Шаг — несколько соседних деталей одного слоя (MAX_PER_STEP), как в настоящих инструкциях.
"""
import csv
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .colors import load_palette
from .fixtures import Fixture
from .layout import PlacedBrick
from .parts import STUD_LDU, part_name

MAX_PER_STEP = 6
PAGE = (11.69, 8.27)          # A4 альбомная, дюймы
CALLOUT_BG = "#dff0fb"
OUTLINE = "#d80000"
GROUND_RGB = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class PartLine:
    number: str
    name: str
    color: int
    color_name: str
    quantity: int


def split_steps(bricks: list[PlacedBrick], max_per_step: int = MAX_PER_STEP) -> list[list[PlacedBrick]]:
    """Слой за слоем, внутри слоя — соседние детали подряд (по рядам), по max_per_step на шаг."""
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
    scene = _Scene(bricks, rgb)
    total = bill_of_materials(bricks, fixtures)

    with PdfPages(path) as pdf:
        _cover(pdf, plt, scene, bricks, title, total)
        done: list[PlacedBrick] = []
        for i, step in enumerate(steps, 1):
            fig = plt.figure(figsize=PAGE)
            fig.text(0.03, 0.93, str(i), fontsize=34, weight="bold")
            ax = fig.add_axes([0.12, 0.08, 0.86, 0.86], projection="3d")
            scene.draw(ax, done, step, zoom=True)
            _callout(fig, step, rgb)
            pdf.savefig(fig); plt.close(fig)
            done = done + step
        _bom_pages(pdf, plt, total)


class _Scene:
    def __init__(self, bricks, rgb):
        self.rgb = rgb
        self.nx = max(b.x + b.width for b in bricks)
        self.nz = max(b.z + b.length for b in bricks)
        self.nk = max(b.layer for b in bricks) + 1
        self.aspect = bricks[0].part.height / STUD_LDU

    def draw(self, ax, done, step, zoom):
        """Каждая деталь — один параллелепипед: без внутренней сетки, обводка по контуру детали."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        shown = done + step
        occupied = np.zeros((self.nx, self.nz, self.nk), bool)
        for b in shown:
            occupied[b.x:b.x + b.width, b.z:b.z + b.length, b.layer] = True
        faces, colors, edges, widths = [], [], [], []
        for b in shown:
            color = self.rgb.get(b.color, GROUND_RGB)
            new = b in step
            for face, shade in _box_faces(b.x, b.z, b.layer, b.width, b.length):
                faces.append(face); colors.append(np.clip(color * shade, 0, 1))
                edges.append(OUTLINE if new else (0, 0, 0, 0.8)); widths.append(1.8 if new else 0.7)
            for x in range(b.x, b.x + b.width):
                for z in range(b.z, b.z + b.length):
                    if b.layer + 1 >= self.nk or not occupied[x, z, b.layer + 1]:
                        for face, shade in _stud_faces(x, z, b.layer + 1):
                            faces.append(face); colors.append(np.clip(color * shade, 0, 1))
                            edges.append((0, 0, 0, 0.6)); widths.append(0.5)
        if faces:
            ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors=edges, linewidths=widths))
        if zoom and shown:
            x0 = min(b.x for b in shown) - 1; x1 = max(b.x + b.width for b in shown) + 1
            z0 = min(b.z for b in shown) - 1; z1 = max(b.z + b.length for b in shown) + 1
            k1 = max(b.layer for b in shown) + 2
            side = max(x1 - x0, z1 - z0)
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
            ax.set_xlim(cx - side / 2, cx + side / 2); ax.set_ylim(cz - side / 2, cz + side / 2)
            ax.set_zlim(0, k1)
            ax.set_box_aspect((side, side, k1 * self.aspect))
        else:
            ax.set_xlim(0, self.nx); ax.set_ylim(0, self.nz); ax.set_zlim(0, self.nk)
            ax.set_box_aspect((self.nx, self.nz, self.nk * self.aspect))
        ax.view_init(32, -52); ax.set_axis_off()


_STUD_R, _STUD_H = 0.3, 0.2
_OCTAGON = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]


def _stud_faces(x, z, k):
    """Штырёк — восьмигранная призма на верхней грани: крышка и стенки, с затенением."""
    cx, cz = x + 0.5, z + 0.5
    ring = [(cx + _STUD_R * a, cz + _STUD_R * b) for a, b in _OCTAGON]
    faces = [([(px, pz, k + _STUD_H) for px, pz in ring], 1.0)]
    for i in range(8):
        (ax_, az), (bx, bz) = ring[i], ring[(i + 1) % 8]
        faces.append(([(ax_, az, k), (bx, bz, k), (bx, bz, k + _STUD_H), (ax_, az, k + _STUD_H)], 0.75))
    return faces


def _box_faces(x, z, k, w, l):
    """Шесть граней параллелепипеда детали с коэффициентом затенения: верх светлее, бока темнее."""
    x0, x1, z0, z1, k0, k1 = x, x + w, z, z + l, k, k + 1
    p = lambda X, Z, K: (X, Z, K)
    return [
        ([p(x0, z0, k1), p(x1, z0, k1), p(x1, z1, k1), p(x0, z1, k1)], 1.0),    # верх
        ([p(x0, z0, k0), p(x1, z0, k0), p(x1, z0, k1), p(x0, z0, k1)], 0.85),   # передняя (z0)
        ([p(x1, z0, k0), p(x1, z1, k0), p(x1, z1, k1), p(x1, z0, k1)], 0.7),    # правая (x1)
        ([p(x0, z1, k0), p(x1, z1, k0), p(x1, z1, k1), p(x0, z1, k1)], 0.85),   # задняя
        ([p(x0, z0, k0), p(x0, z1, k0), p(x0, z1, k1), p(x0, z0, k1)], 0.7),    # левая
        ([p(x0, z0, k0), p(x1, z0, k0), p(x1, z1, k0), p(x0, z1, k0)], 0.6),    # низ
    ]


# --- Плашка деталей: изометрия в 2D, один масштаб на всех страницах ---
_ISO_C, _ISO_S = np.cos(np.radians(30)), np.sin(np.radians(30))
_BRICK_H = 1.2       # высота кирпича в штырьках
_CALLOUT_UNIT = 0.9  # см на штырёк в плашке


def _iso(x, y, z):
    """Точка (x — вправо-вниз, y — вправо-вверх, z — вверх) -> 2D."""
    return ((x - y) * _ISO_C, (x + y) * _ISO_S + z)


def _callout(fig, step, rgb):
    """Плашка слева внизу: детали шага в изометрии, один масштаб на всех страницах, «N×» под каждой."""
    from matplotlib.patches import Rectangle
    lines = bill_of_materials(step)
    parts = [(next(b for b in step if b.part.number == l.number and b.color == l.color), l) for l in lines]
    gap = 1.4
    widths = [max(b.width, b.length) for b, _ in parts]
    total_w = sum(w * _ISO_C + 2 * _ISO_C for w in widths) + gap * (len(parts) + 1)   # в штырьках, по горизонтали
    fig_w, fig_h = fig.get_size_inches()
    box_w = total_w * _CALLOUT_UNIT / 2.54 / fig_w
    box_h = 0.22
    ax = fig.add_axes([0.03, 0.04, box_w, box_h])
    ax.set_xlim(0, total_w); ax.set_ylim(-0.6, box_h * fig_h * 2.54 / _CALLOUT_UNIT - 0.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(CALLOUT_BG)
    for side in ax.spines.values():
        side.set_linewidth(1.8)
    x = gap
    for (brick, line), w in zip(parts, widths):
        l = min(brick.part.width, brick.part.length)
        color = rgb.get(line.color, GROUND_RGB)
        # кирпич кладём так, чтобы его левый нижний угол в изометрии оказался в точке x
        origin_x = x + l * _ISO_C     # сдвиг, т.к. -y уводит влево
        ax.set_autoscale_on(False)
        # рисуем в локальных координатах: смещаем через трансформацию точек
        u0, _ = _iso(0, 0, 0)
        _draw_iso_brick_at(ax, origin_x, 0.9, w, l, color)
        ax.text(x + 0.2, -0.35, f"{line.quantity}×", fontsize=11, weight="bold")
        x += w * _ISO_C + 2 * _ISO_C + gap


def _draw_iso_brick_at(ax, u_left, v_bottom, w, l, color):
    """Кирпич w×l, у которого левый нижний угол проекции стоит в (u_left, v_bottom)."""
    from matplotlib.patches import Polygon
    corner = _iso(0, l, 0)  # самая левая точка (x=0, y=l, z=0)
    lowest = _iso(0, 0, 0)  # самая нижняя точка
    du, dv = u_left - corner[0], v_bottom - lowest[1]

    def P(x, y, z):
        u, v = _iso(x, y, z); return (u + du, v + dv)
    h = _BRICK_H
    # видны верх и две грани, обращённые к зрителю: y=0 (уходит вправо) и x=0 (уходит влево)
    for pts, shade in (
        ([(0, 0, h), (w, 0, h), (w, l, h), (0, l, h)], 1.0),
        ([(0, 0, 0), (w, 0, 0), (w, 0, h), (0, 0, h)], 0.8),
        ([(0, 0, 0), (0, l, 0), (0, l, h), (0, 0, h)], 0.62),
    ):
        ax.add_patch(Polygon([P(*p) for p in pts], closed=True, facecolor=np.clip(color * shade, 0, 1), edgecolor="black", linewidth=1.2))
    for sx in range(w):
        for sy in range(l):
            cx, cy, r, sh = sx + 0.5, sy + 0.5, 0.3, 0.2
            ring = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
            low = [P(px, py, h) for px, py in ring]; top = [P(px, py, h + sh) for px, py in ring]
            i0 = int(np.argmin([p[1] for p in low])); order = [(i0 + k) % 24 for k in range(-6, 7)]
            ax.add_patch(Polygon([low[i] for i in order] + [top[i] for i in reversed(order)], closed=True,
                                 facecolor=np.clip(color * 0.75, 0, 1), edgecolor="black", linewidth=0.9))
            ax.add_patch(Polygon(top, closed=True, facecolor=np.clip(color * 0.95, 0, 1), edgecolor="black", linewidth=0.9))


def _cover(pdf, plt, scene, bricks, title, total):
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.05, 0.1, 0.6, 0.8], projection="3d")
    scene.draw(ax, bricks, [], zoom=False)
    fig.text(0.68, 0.85, title, fontsize=24, weight="bold")
    fig.text(0.68, 0.78, f"Деталей: {sum(l.quantity for l in total)}", fontsize=13)
    fig.text(0.68, 0.74, f"Типов деталей: {len(total)}", fontsize=13)
    fig.text(0.68, 0.70, f"Размер: {scene.nx * 0.8:.1f} × {scene.nz * 0.8:.1f} × {scene.nk * scene.aspect * 0.8:.1f} см", fontsize=13)
    pdf.savefig(fig); plt.close(fig)


def _bom_pages(pdf, plt, total):
    per_page = 28
    for start in range(0, len(total), per_page):
        fig = plt.figure(figsize=PAGE)
        fig.text(0.04, 0.92, "Список деталей", fontsize=18, weight="bold")
        for j, line in enumerate(total[start:start + per_page]):
            fig.text(0.04, 0.86 - j * 0.029, f"{line.quantity:4d} ×  {line.number:<10} {line.name:<34} {line.color_name}",
                     fontsize=10, family="monospace")
        pdf.savefig(fig); plt.close(fig)
