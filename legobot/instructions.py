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
from .slopes import FACINGS, PlacedSlope

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


def split_steps(bricks: list, max_per_step: int = MAX_PER_STEP) -> list[list]:
    """Слой за слоем (скос — по своему нижнему слою), внутри слоя — соседние детали подряд
    (по рядам), по max_per_step на шаг."""
    steps = []
    for layer in sorted({b.layer for b in bricks}):
        in_layer = sorted((b for b in bricks if b.layer == layer), key=lambda b: (b.z, b.x))
        pieces = max(1, -(-len(in_layer) // max_per_step))
        size = -(-len(in_layer) // pieces)
        steps.extend(in_layer[i:i + size] for i in range(0, len(in_layer), size))
    return steps


def bill_of_materials(bricks: list, fixtures: list[Fixture] = ()) -> list[PartLine]:
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


def write_pdf(bricks: list, steps: list[list], path: str, title: str, fixtures: list[Fixture] = ()) -> None:
    """bricks — пластины и скосы вместе, в порядке сборки."""
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
            box = _callout(fig, step, rgb)
            ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
            scene.draw(ax, done, step, avoid=box)
            pdf.savefig(fig); plt.close(fig)
            done = done + step
        _bom_pages(pdf, plt, total)


class _Scene:
    def __init__(self, bricks, rgb):
        self.rgb = rgb
        self.aspect = min(b.part.height for b in bricks) / STUD_LDU   # слой = самая низкая деталь (пластина)
        self.nx = max(b.x + b.width for b in bricks)
        self.nz = max(b.z + b.length for b in bricks)
        self.nk = max(b.layer for b in bricks) + 1

    def draw(self, ax, done, step, avoid=None):
        """Собранное на этот шаг, кадр по его габаритам; avoid — прямоугольник плашки
        (доли страницы), модель ставится в большую из свободных областей: над ней или справа."""
        shown = done + step
        new = set(range(len(done), len(shown)))
        u0, v0, u1, v1 = _draw_iso(ax, shown, new, self.aspect, self.rgb)
        u0, v0, u1, v1 = u0 - 1, v0 - 1, u1 + 1, v1 + 1
        fig_w, fig_h = ax.figure.get_size_inches()
        ax0, ay0, aw, ah = ax.get_position().bounds
        regions = [(ax0, ay0, aw, ah)]
        if avoid:
            bx, by, bw, bh = avoid
            regions = [(ax0, by + bh, aw, ay0 + ah - by - bh), (bx + bw, ay0, ax0 + aw - bx - bw, ah)]
        scale, (rx, ry, rw, rh) = max((min(rw * fig_w / (u1 - u0), rh * fig_h / (v1 - v0)), r) for r in regions for rw, rh in [r[2:]])
        # центр модели -> центр свободной области; масштаб (дюймов на штырёк) общий по обеим осям
        x_range, y_range = aw * fig_w / scale, ah * fig_h / scale
        xlim0 = (u0 + u1) / 2 - (rx + rw / 2 - ax0) / aw * x_range
        ylim0 = (v0 + v1) / 2 - (ry + rh / 2 - ay0) / ah * y_range
        ax.set_xlim(xlim0, xlim0 + x_range); ax.set_ylim(ylim0, ylim0 + y_range)
        ax.set_aspect("equal"); ax.set_axis_off()


# --- Изометрия в 2D: одна проекция для модели и плашки ---
_ISO_C, _ISO_S = np.cos(np.radians(30)), np.sin(np.radians(30))
_STUD_R, _STUD_H = 0.3, 0.2
_STUD_RING = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
_SHADE_TOP, _SHADE_SLOPE, _SHADE_RIGHT, _SHADE_LEFT = 1.0, 0.92, 0.84, 0.68
_LINE, _LINE_DARK, _LINE_NEW = ("black", 0.8), ("#9a9a9a", 0.8), (OUTLINE, 2.0)


def _iso(x, y, z):
    """Точка (x — вправо-вверх, y — влево-вверх, z — вверх) -> 2D. Зритель смотрит с -x, -y."""
    return ((x - y) * _ISO_C, (x + y) * _ISO_S + z)


def _draw_iso(ax, bricks, new, aspect, rgb, shift=(0.0, 0.0)):
    """Рисует детали методом художника по колонкам штырьковой сетки: дальние раньше, видны только
    верх и две грани к зрителю, линии — только по границам деталей. Так ближние детали сами
    закрывают линии и штырьки дальних. Пластина — колонка в один слой; скос — колонки на всю
    его высоту с наклонным верхом. Возвращает bbox в 2D."""
    from itertools import groupby
    from matplotlib.collections import LineCollection, PolyCollection
    cells, units, unit_at = {}, {}, {}
    for i, b in enumerate(bricks):
        for u in _units(b, aspect):
            units[(u.x, u.y, u.k)] = (i, u)
            for k in range(u.k, u.k + u.layers):
                cells[(u.x, u.y, k)] = i
                unit_at[(u.x, u.y, k)] = u
    du, dv = shift
    P = lambda x, y, z: tuple(np.add(_iso(x, y, z), (du, dv)))
    corners = []
    # колонки с равной «глубиной» x+y не перекрываются на экране, поэтому рисуем их одной коллекцией
    order = sorted(units, key=lambda c: (-(c[0] + c[1]), c[2]))
    for depth, (_, group) in enumerate(groupby(order, key=lambda c: c[0] + c[1])):
        polys, fills, edges, widths, segs, seg_colors, seg_widths = [], [], [], [], [], [], []
        for x, y, k in group:
            i, u = units[(x, y, k)]; b = bricks[i]
            color = rgb.get(b.color, GROUND_RGB)
            line = _LINE if color.mean() > 0.3 else _LINE_DARK
            z0, ktop = u.z0, k + u.layers - 1
            h00, h10, h11, h01 = u.heights
            corners += [P(x, y, z0), P(x + 1, y + 1, max(u.heights) + _STUD_H), P(x + 1, y, z0), P(x, y + 1, z0)]

            def face(pts, shade):
                fill = np.clip(color * shade, 0, 1)
                polys.append([P(*p) for p in pts]); fills.append(fill); edges.append(fill); widths.append(0.4)   # обводка в цвет: без щелей между клетками

            def same(dx, dy, dk=0):
                return cells.get((x + dx, y + dy, k + dk)) == i

            def edge(p, q, dx, dy, dz, skip=False):
                """Линия по ребру, если сосед за ним — не та же деталь (иначе грань продолжается).
                Красная — только на границе новых деталей с остальным: общий контур группы, как в Studio."""
                j = cells.get((x + dx, y + dy, k + dz))
                if skip or j == i:
                    return
                c, w = _LINE_NEW if (i in new) != (j in new) else line
                segs.append([P(*p), P(*q)]); seg_colors.append(c); seg_widths.append(w)

            top = [(x, y, h00), (x + 1, y, h10), (x + 1, y + 1, h11), (x, y + 1, h01)]
            if (x, y, ktop + 1) not in cells and _ccw([P(*p) for p in top]):                 # верх
                face(top, _SHADE_TOP if u.flat else _SHADE_SLOPE)
                edge(top[0], top[3], -1, 0, 0, skip=same(-1, 0) and u.coplanar(unit_at.get((x - 1, y, k))))
                edge(top[1], top[2], 1, 0, 0, skip=same(1, 0) and u.coplanar(unit_at.get((x + 1, y, k))))
                edge(top[0], top[1], 0, -1, 0, skip=same(0, -1) and u.coplanar(unit_at.get((x, y - 1, k))))
                edge(top[3], top[2], 0, 1, 0, skip=same(0, 1) and u.coplanar(unit_at.get((x, y + 1, k))))
                if u.stud:
                    for pts, shade in _stud(x + 0.5, y + 0.5, h00):
                        polys.append([P(*p) for p in pts]); fills.append(np.clip(color * shade, 0, 1))
                        edges.append(line[0]); widths.append(line[1] * 0.6)
            # боковая грань пластины скрыта соседом, если он не ниже её по всему ребру (скос может
            # быть ниже — тогда грань над ним видна); колонку скоса рисуем всегда — ближние
            # соседи (они рисуются позже) закроют её сами
            def covered(n, near, far):
                return n is not None and n.heights[near] >= h00 - 1e-6 and n.heights[far] >= (h01 if near == 1 else h10) - 1e-6

            if u.layers > 1 or not covered(unit_at.get((x - 1, y, k)), 1, 2):           # левая грань
                face([(x, y, z0), (x, y + 1, z0), (x, y + 1, h01), (x, y, h00)], _SHADE_LEFT)
                edge((x, y, z0), (x, y, h00), 0, -1, 0); edge((x, y + 1, z0), (x, y + 1, h01), 0, 1, 0)
                edge((x, y, z0), (x, y + 1, z0), 0, 0, -1); edge((x, y, h00), (x, y + 1, h01), 0, 0, u.layers)
            if u.layers > 1 or not covered(unit_at.get((x, y - 1, k)), 3, 2):           # правая грань
                face([(x, y, z0), (x + 1, y, z0), (x + 1, y, h10), (x, y, h00)], _SHADE_RIGHT)
                edge((x, y, z0), (x, y, h00), -1, 0, 0); edge((x + 1, y, z0), (x + 1, y, h10), 1, 0, 0)
                edge((x, y, z0), (x + 1, y, z0), 0, 0, -1); edge((x, y, h00), (x + 1, y, h10), 0, 0, u.layers)
        # zorder явно: у LineCollection он по умолчанию выше, чем у граней, а нам нужен порядок глубины
        ax.add_collection(PolyCollection(polys, facecolors=fills, edgecolors=edges, linewidths=widths, joinstyle="round", zorder=2 * depth))
        ax.add_collection(LineCollection(segs, colors=seg_colors, linewidths=seg_widths, capstyle="round", zorder=2 * depth + 1))
    us, vs = zip(*corners)
    return min(us), min(vs), max(us), max(vs)


@dataclass
class _Unit:
    """Колонка рисования: клетка (x, y), нижний слой k, высота в слоях, низ z0 и высоты четырёх
    углов верха ((x,y), (x+1,y), (x+1,y+1), (x,y+1)) в штырьках; column — номер колонки скоса."""
    x: int
    y: int
    k: int
    layers: int
    z0: float
    heights: tuple[float, float, float, float]
    stud: bool
    column: int | None = None
    facing: str | None = None

    @property
    def flat(self) -> bool:
        return len(set(self.heights)) == 1

    def coplanar(self, other) -> bool:
        """Верх продолжается в соседнюю колонку той же детали без ребра: у пластины всегда,
        у скоса — если обе колонки плоские или обе под наклоном (ребро только на гребне)."""
        if self.column is None or other is None:
            return True
        return self.flat == other.flat


def _units(b, aspect) -> list[_Unit]:
    if not isinstance(b, PlacedSlope):
        h = b.layer * aspect + b.part.height / STUD_LDU
        return [_Unit(x, y, b.layer, 1, b.layer * aspect, (h, h, h, h), True)
                for x in range(b.x, b.x + b.width) for y in range(b.z, b.z + b.length)]
    dx, dz = FACINGS[b.facing]
    z0 = b.layer * aspect
    profile = b.slope.profile(b.layer_ldu)
    out = []
    for x, y, j in b.columns():
        hb, hf = profile[j]

        def h(cx, cy):
            s = dx * (cx - x) + dz * (cy - y)            # доля пути от заднего края к переднему
            s = s if (dx + dz) > 0 else s + 1
            return z0 + (hb + (hf - hb) * s) * aspect
        heights = (h(x, y), h(x + 1, y), h(x + 1, y + 1), h(x, y + 1))
        out.append(_Unit(x, y, b.layer, b.layers, z0, heights, hb == hf, j, b.facing))
    return out


def _ccw(pts) -> bool:
    """Грань видна, если её контур на экране идёт против часовой стрелки (обход задан снаружи)."""
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return area > 1e-9


def _stud(cx, cy, z):
    """Штырёк-цилиндр: видимая половина стенки и крышка (грани в 3D, с затенением)."""
    ring = [(cx + _STUD_R * a, cy + _STUD_R * b) for a, b in _STUD_RING]
    n = len(ring)
    low = [(px, py, z) for px, py in ring]; top = [(px, py, z + _STUD_H) for px, py in ring]
    i0 = int(np.argmin([_iso(*p)[1] for p in low]))          # самая нижняя точка на экране
    near = [(i0 + k) % n for k in range(-n // 4, n // 4 + 1)]
    return [([low[i] for i in near] + [top[i] for i in reversed(near)], 0.75), (top, 0.95)]


# --- Плашка деталей шага ---
_CALLOUT_UNIT = 0.9                       # см на штырёк, если детали помещаются
_CALLOUT_H, _CALLOUT_MAX_W = 0.22, 0.9    # доли страницы


def _callout(fig, step, rgb):
    """Плашка слева внизу: детали шага в той же изометрии, «N×» под каждой.
    Масштаб один для всех деталей и уменьшается, если они не влезают в рамку."""
    lines = bill_of_materials(step)
    parts = []
    for l in lines:
        b = next(b for b in step if b.part.number == l.number and b.color == l.color)
        if isinstance(b, PlacedSlope):
            sl = b.slope   # скос — гранью к зрителю, вправо-вниз
            parts.append((PlacedSlope(sl, 0, sl.depth - 1, b.layers - 1, "-z", b.color, b.layer_ldu), sl.width, sl.depth, l.quantity))
        else:
            w, d = max(b.width, b.length), min(b.width, b.length)   # длинной стороной вправо
            parts.append((PlacedBrick(b.part, 0, 0, 0, b.part.width < b.part.length, b.color), w, d, l.quantity))
    gap, label_h = 1.2, 0.9
    extents = [((w + d) * _ISO_C, (w + d) * _ISO_S + b.part.height / STUD_LDU + _STUD_H) for b, w, d, _ in parts]
    unit_layer = min(b.part.height for b in step) / STUD_LDU
    total_w = sum(e[0] for e in extents) + gap * (len(parts) + 1)
    total_h = label_h + max(e[1] for e in extents) + 0.3
    fig_w, fig_h = fig.get_size_inches()
    box_h_cm, max_w_cm = _CALLOUT_H * fig_h * 2.54, _CALLOUT_MAX_W * fig_w * 2.54
    unit = min(_CALLOUT_UNIT, box_h_cm / total_h, max_w_cm / total_w)
    box = (0.03, 0.04, total_w * unit / 2.54 / fig_w, _CALLOUT_H)
    ax = fig.add_axes(box)
    ax.set_xlim(0, total_w); ax.set_ylim(0, box_h_cm / unit)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(CALLOUT_BG)
    for side in ax.spines.values():
        side.set_linewidth(1.8)
    u = gap
    for (brick, w, d, quantity), (ext_w, _) in zip(parts, extents):
        _draw_iso(ax, [brick], set(), unit_layer if isinstance(brick, PlacedSlope) else brick.part.height / STUD_LDU, rgb, shift=(u + d * _ISO_C, label_h))
        ax.text(u + 0.2, 0.3, f"{quantity}×", fontsize=11, weight="bold")
        u += ext_w + gap
    return box


def _cover(pdf, plt, scene, bricks, title, total):
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.05, 0.1, 0.6, 0.8])
    scene.draw(ax, bricks, [])
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
