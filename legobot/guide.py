"""Инструкция для пиксельных моделей: PDF, который печатают и по которому собирают руками.

Универсальный изометрический рендерер (instructions.py) писался под утку и машину: у пиксельной
фигурки он даёт сотню-другую почти одинаковых кадров и файл на десятки мегабайт. Здесь другое:

- фигурка (стоячая, объёмная) — ряд за рядом снизу вверх. Шаг = ряд, нарисован сверху: ось X —
  ширина, ось Z — глубина, каждый кирпич прямоугольником своего цвета с подписью размера.
  Несколько рядов на страницу, у каждого — детали именно этого ряда;
- панно — как вышивка: секции SECTION×SECTION клеток с координатами и списком деталей секции,
  плюс страница подложки.

Обложка, страница «перед сборкой» со всеми деталями по цветам и, если известна, цена Pick a Brick.
Рисуем прямоугольниками (вектор): файл выходит в сотни килобайт и строится за секунды — так что
инструкция делается и в браузере, где matplotlib приходит вместе с scikit-image.
"""
import textwrap
from collections import Counter

import numpy as np

from .catalog import price_cents
from .colors import studio_palette

PAGE = (8.27, 11.69)      # A4 в дюймах
SECTION = 16              # клеток в стороне секции панно
ROWS_PER_PAGE = 6
MIN_LABEL_STUDS = 2       # в детали уже этого подпись размера не влезает


def write_guide(bricks: list, path: str, title: str, kind: str) -> int:
    """kind: 'панно' — секции, иначе ряды. Возвращает число страниц."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rgb = {c.code: tuple(v / 255 for v in c.rgb) for c in studio_palette(common_only=False)}
    names = {c.code: c.name for c in studio_palette(common_only=False)}
    pages = 0
    with PdfPages(path) as pdf:
        _cover(pdf, plt, bricks, title, kind, rgb, names)
        pages += 2
        if kind == "панно":
            pages += _panel_pages(pdf, plt, bricks, rgb, names)
        else:
            pages += _row_pages(pdf, plt, bricks, rgb, names)
    return pages


# --- обложка и список деталей ---

def _cover(pdf, plt, bricks, title, kind, rgb, names) -> None:
    totals = Counter((b.part.label, b.color) for b in bricks)
    width = max(b.x + b.width for b in bricks) - min(b.x for b in bricks)
    depth = max(b.z + b.length for b in bricks) - min(b.z for b in bricks)
    height = max(b.layer for b in bricks) + 1
    cm = lambda studs, unit=0.8: round(studs * unit, 1)
    size = f"{cm(width)} × {cm(depth)} × {cm(height, bricks[0].part.height / 20 * 0.8)} см"
    prices = [price_cents(b.part.number, b.color) for b in bricks]
    price = sum(p for p in prices if p is not None) / 100 if any(prices) else None

    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.88, title, fontsize=26, weight="bold")
    fig.text(0.08, 0.84, f"{kind} фигура · {len(bricks)} деталей · {size}", fontsize=13, color="#444444")
    lines = [
        "Как собирать",
        "",
        "Фигурка: ряд за рядом снизу вверх. На каждой странице несколько рядов;" if kind != "панно"
        else "Панно: подложка из пластин, поверх неё тайлы 1×1 по секциям 16×16 клеток.",
        "ряд нарисован сверху — перёд фигурки внизу полоски." if kind != "панно"
        else "Номера столбцов и рядов на схеме совпадают с общей картой на первой странице секций.",
        "",
        "Цвета названы так же, как в каталоге LEGO Pick a Brick и BrickLink.",
    ]
    if kind == "панно" and len({b.layer for b in bricks}) == 1:   # плитки без своей подложки
        studs_x = max(b.x + b.width for b in bricks) - min(b.x for b in bricks)
        studs_z = max(b.z + b.length for b in bricks) - min(b.z for b in bricks)
        plates = -(-studs_x // 48) * -(-studs_z // 48)
        lines += ["", f"Подложки в наборе нет: {studs_x} × {studs_z} штырьков нужно закрыть готовыми "
                      f"строительными пластинами — это {plates} шт. 48×48 или пластины 16×16 по площади."]
    fig.text(0.08, 0.76, "\n".join(lines), fontsize=11, va="top", linespacing=1.6)
    if price is not None:
        fig.text(0.08, 0.60, f"Детали на Pick a Brick: ${price:.2f}", fontsize=12, weight="bold")
    pdf.savefig(fig); plt.close(fig)

    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.94, "Перед сборкой: все детали", fontsize=18, weight="bold")
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.85]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rows = sorted(totals.items(), key=lambda kv: (-kv[1], names.get(kv[0][1], "")))
    per_column = 34
    for i, ((label, color), n) in enumerate(rows):
        col, row = divmod(i, per_column)
        x, y = col * 0.5, 1 - (row + 1) / (per_column + 1)
        ax.add_patch(plt.Rectangle((x, y), 0.022, 0.016, facecolor=rgb.get(color, "#888888"), edgecolor="#333333", lw=0.4))
        ax.text(x + 0.03, y + 0.002, f"{label}  {names.get(color, color).replace('_', ' ')} — {n}", fontsize=8.5)
    pdf.savefig(fig); plt.close(fig)


# --- фигурка: ряды ---

def _row_pages(pdf, plt, bricks, rgb, names) -> int:
    by_layer: dict[int, list] = {}
    for b in bricks:
        by_layer.setdefault(b.layer, []).append(b)
    layers = sorted(by_layer)
    x0 = min(b.x for b in bricks)
    x1 = max(b.x + b.width for b in bricks)
    depth = max(b.z + b.length for b in bricks)
    pages = 0
    for start in range(0, len(layers), ROWS_PER_PAGE):
        chunk = layers[start:start + ROWS_PER_PAGE]
        fig = plt.figure(figsize=PAGE)
        fig.text(0.06, 0.96, f"Ряды {chunk[0] + 1}–{chunk[-1] + 1} из {len(layers)}", fontsize=14, weight="bold")
        block = 0.88 / ROWS_PER_PAGE
        for i, layer in enumerate(chunk):
            _row_strip(fig, plt, by_layer[layer], layer, len(layers), x0, x1, depth, rgb, names,
                       top=0.91 - i * block, block=block)
        pdf.savefig(fig); plt.close(fig)
        pages += 1
    return pages


def _row_strip(fig, plt, row, layer, total_layers, x0, x1, depth, rgb, names, top, block) -> None:
    """Блок одного ряда: заголовок, детали ряда, вид сверху (X — ширина, Z — глубина, перёд внизу)."""
    fig.text(0.06, top, f"Ряд {layer + 1} из {total_layers}", fontsize=9.5, weight="bold")
    fig.text(0.94, top, "вид сверху, перёд снизу", fontsize=7, color="#888888", ha="right")
    counts = Counter((b.part.label, b.color) for b in row).most_common()
    parts = ", ".join(f"{label} {names.get(color, color).replace('_', ' ')} ×{n}" for (label, color), n in counts)
    lines = textwrap.wrap(parts, 128)
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1].rsplit(",", 1)[0] + f" и ещё {len(counts) - lines[0].count(',') - lines[1].count(',') - 1} видов"
    for k, line in enumerate(lines):
        fig.text(0.06, top - 0.018 - k * 0.012, line, fontsize=6.5, color="#333333")

    ax_height = block * 0.52
    ax_top = top - 0.022 - len(lines) * 0.012
    ax = fig.add_axes([0.06, ax_top - ax_height, 0.88, ax_height])
    ax.set_xlim(x0, x1); ax.set_ylim(depth, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(x0, x1 + 1, 5)); ax.set_yticks([])
    ax.tick_params(labelsize=6, length=2, pad=1)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    for b in row:
        ax.add_patch(plt.Rectangle((b.x, b.z), b.width, b.length, facecolor=rgb.get(b.color, "#888888"),
                                   edgecolor="#1b1b1b", lw=0.5))
        if b.width >= MIN_LABEL_STUDS:
            ax.text(b.x + b.width / 2, b.z + b.length / 2, b.part.label, ha="center", va="center",
                    fontsize=5.5, color=_ink(rgb.get(b.color, (0.5, 0.5, 0.5))))


def _ink(color) -> str:
    return "#111111" if sum(color[:3]) / 3 > 0.55 else "#f2f2f2"


# --- панно: секции ---

def _panel_pages(pdf, plt, bricks, rgb, names) -> int:
    top = max(b.layer for b in bricks)
    tiles = [b for b in bricks if b.layer == top]
    base = [b for b in bricks if b.layer < top]
    x0, z0 = min(b.x for b in tiles), min(b.z for b in tiles)
    nx = max(b.x + b.width for b in tiles) - x0
    nz = max(b.z + b.length for b in tiles) - z0
    grid = np.full((nx, nz), -1)
    for b in tiles:
        grid[b.x - x0:b.x - x0 + b.width, b.z - z0:b.z - z0 + b.length] = b.color
    sections = [(sx, sz) for sz in range(0, nz, SECTION) for sx in range(0, nx, SECTION)]

    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.95, "Карта секций", fontsize=16, weight="bold")
    fig.text(0.06, 0.92, f"{nx} × {nz} клеток, {len(sections)} секций по {SECTION}×{SECTION}", fontsize=10, color="#444444")
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.76])
    ax.set_xlim(0, nx); ax.set_ylim(nz, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.imshow(np.transpose([[rgb.get(c, (1, 1, 1)) for c in col] for col in grid], (1, 0, 2)),
              extent=(0, nx, nz, 0), interpolation="nearest")
    for i, (sx, sz) in enumerate(sections, 1):
        w, h = min(SECTION, nx - sx), min(SECTION, nz - sz)
        ax.add_patch(plt.Rectangle((sx, sz), w, h, fill=False, edgecolor="#ffffff", lw=1.2))
        ax.text(sx + w / 2, sz + h / 2, str(i), ha="center", va="center", fontsize=11,
                color="#ffffff", weight="bold", path_effects=None)
    if base:
        counts = Counter((b.part.label, b.color) for b in base)
        text = "Подложка: " + ", ".join(f"{label} {names.get(color, color).replace('_', ' ')} ×{n}"
                                        for (label, color), n in counts.most_common())
        fig.text(0.06, 0.07, text, fontsize=8, color="#333333", wrap=True)
    pdf.savefig(fig); plt.close(fig)

    for i, (sx, sz) in enumerate(sections, 1):
        w, h = min(SECTION, nx - sx), min(SECTION, nz - sz)
        block = grid[sx:sx + w, sz:sz + h]
        fig = plt.figure(figsize=PAGE)
        fig.text(0.06, 0.95, f"Секция {i} из {len(sections)}", fontsize=15, weight="bold")
        fig.text(0.06, 0.92, f"столбцы {sx + 1}–{sx + w}, ряды {sz + 1}–{sz + h}", fontsize=10, color="#444444")
        ax = fig.add_axes([0.10, 0.34, 0.80, 0.54])
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect("equal")
        for cx in range(w):
            for cz in range(h):
                code = block[cx, cz]
                ax.add_patch(plt.Rectangle((cx, cz), 1, 1, facecolor=rgb.get(code, (1, 1, 1)) if code >= 0 else "#ffffff",
                                           edgecolor="#bbbbbb", lw=0.3))
        ax.set_xticks([c + 0.5 for c in range(w)]); ax.set_xticklabels(range(sx + 1, sx + w + 1), fontsize=6)
        ax.set_yticks([c + 0.5 for c in range(h)]); ax.set_yticklabels(range(sz + 1, sz + h + 1), fontsize=6)
        ax.tick_params(length=0, pad=2)
        inside = [b for b in tiles if sx <= b.x - x0 < sx + w and sz <= b.z - z0 < sz + h]
        for b in inside:                      # границы деталей: видно, где 1×2, а где 2×2
            ax.add_patch(plt.Rectangle((b.x - x0 - sx, b.z - z0 - sz), b.width, b.length,
                                       fill=False, edgecolor="#222222", lw=0.9))
        counts = Counter((b.part.label, b.color) for b in inside)
        lines = [f"{label} {names.get(code, code).replace('_', ' ')} — {n}"
                 for (label, code), n in counts.most_common()]
        ax.figure.text(0.10, 0.28, "Детали этой секции:", fontsize=9, weight="bold")
        for k, line in enumerate(lines):
            col, row = divmod(k, 12)
            fig.text(0.10 + col * 0.30, 0.25 - row * 0.018, line, fontsize=8)
        pdf.savefig(fig); plt.close(fig)
    return len(sections) + 1
