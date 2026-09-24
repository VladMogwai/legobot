"""Схема панно: PDF для мозаик, которые не собрать по шагам.

Пошаговая инструкция (instructions.py) рисует на каждой странице всю модель целиком, поэтому
у мозаики в тысячи деталей шаг приходится брать таким крупным, что шагов остаётся три-четыре —
собирать по ним нечего. Такие панно и в жизни выкладывают не по шагам, а по схеме, как вышивку:
карта секций SECTION×SECTION клеток, дальше страница на каждую секцию с координатами и списком
её деталей.
"""
from collections import Counter

import numpy as np

from .catalog import price_cents
from .colors import studio_palette

PAGE = (8.27, 11.69)      # A4 в дюймах
SECTION = 16              # клеток в стороне секции
PARTS_COLUMNS, PARTS_ROWS = 3, 38   # список деталей: колонок и строк на странице


def write_schema(bricks: list, path: str, title: str) -> int:
    """Схема панно: обложка, все детали, карта секций и страница на секцию.
    Текст английский, как и в пошаговой инструкции. Возвращает число страниц."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rgb = {c.code: tuple(v / 255 for v in c.rgb) for c in studio_palette(common_only=False)}
    names = {c.code: c.name for c in studio_palette(common_only=False)}
    with PdfPages(path) as pdf:
        pages = _cover(pdf, plt, bricks, title)
        pages += _parts_pages(pdf, plt, bricks, rgb, names)
        pages += _section_pages(pdf, plt, bricks, rgb, names)
    return pages


def _cover(pdf, plt, bricks, title) -> int:
    width = max(b.x + b.width for b in bricks) - min(b.x for b in bricks)
    depth = max(b.z + b.length for b in bricks) - min(b.z for b in bricks)
    height = max(b.layer for b in bricks) + 1
    cm = lambda studs, unit=0.8: round(studs * unit, 1)
    size = f"{cm(width)} × {cm(depth)} × {cm(height, bricks[0].part.height / 20 * 0.8)} cm"
    prices = [price_cents(b.part.number, b.color) for b in bricks]
    price = sum(p for p in prices if p is not None) / 100 if any(p is not None for p in prices) else None

    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.88, title, fontsize=26, weight="bold")
    fig.text(0.08, 0.84, f"Panel · {len(bricks)} pieces · {size}", fontsize=13, color="#444444")
    lines = ["How to build", "",
             "Too many pieces for step-by-step: build it from the schema instead.",
             "Lay the base plates first, then the tiles section by section.",
             f"Each section is {SECTION} × {SECTION} cells; column and row numbers match the section map."]
    lines += ["", "Colour names are the ones LEGO Pick a Brick and BrickLink use."]
    if len({b.layer for b in bricks}) == 1:          # плитки без своей подложки
        studs_x = max(b.x + b.width for b in bricks) - min(b.x for b in bricks)
        studs_z = max(b.z + b.length for b in bricks) - min(b.z for b in bricks)
        plates = -(-studs_x // 48) * -(-studs_z // 48)
        lines += ["", f"No base included: cover {studs_x} × {studs_z} studs with baseplates —",
                  f"that is {plates} of 48 × 48, or 16 × 16 plates by area."]
    fig.text(0.08, 0.76, "\n".join(lines), fontsize=11, va="top", linespacing=1.6)
    if price is not None:
        fig.text(0.08, 0.50, f"Pick a Brick total: ${price:.2f}", fontsize=12, weight="bold")
    pdf.savefig(fig); plt.close(fig)
    return 1


def _parts_pages(pdf, plt, bricks, rgb, names) -> int:
    """Все детали по цветам: три колонки на страницу, сколько надо страниц."""
    totals = Counter((b.part.label, b.color) for b in bricks)
    rows = sorted(totals.items(), key=lambda kv: (-kv[1], names.get(kv[0][1], "")))
    per_page = PARTS_COLUMNS * PARTS_ROWS
    pages = 0
    for start in range(0, len(rows), per_page):
        chunk = rows[start:start + per_page]
        fig = plt.figure(figsize=PAGE)
        title = "Parts you need" + ("" if start == 0 else " (continued)")
        fig.text(0.08, 0.94, title, fontsize=18, weight="bold")
        fig.text(0.08, 0.915, f"{sum(totals.values())} pieces, {len(rows)} kinds", fontsize=10, color="#555555")
        for i, ((label, color), n) in enumerate(chunk):
            col, row = divmod(i, PARTS_ROWS)
            x = 0.08 + col * 0.29
            y = 0.88 - (row + 1) * 0.021
            fig.patches.append(plt.Rectangle((x, y), 0.018, 0.013, transform=fig.transFigure,
                                             facecolor=rgb.get(color, "#888888"), edgecolor="#333333", lw=0.4))
            fig.text(x + 0.024, y, f"{label}  {names.get(color, color).replace('_', ' ')} — {n}", fontsize=8)
        pdf.savefig(fig); plt.close(fig)
        pages += 1
    return pages


def _section_pages(pdf, plt, bricks, rgb, names) -> int:
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
    fig.text(0.06, 0.95, "Section map", fontsize=16, weight="bold")
    fig.text(0.06, 0.92, f"{nx} × {nz} cells, {len(sections)} sections of {SECTION} × {SECTION}", fontsize=10, color="#444444")
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
        text = "Base plates: " + ", ".join(f"{label} {names.get(color, color).replace('_', ' ')} ×{n}"
                                        for (label, color), n in counts.most_common())
        fig.text(0.06, 0.07, text, fontsize=8, color="#333333", wrap=True)
    pdf.savefig(fig); plt.close(fig)

    for i, (sx, sz) in enumerate(sections, 1):
        w, h = min(SECTION, nx - sx), min(SECTION, nz - sz)
        block = grid[sx:sx + w, sz:sz + h]
        fig = plt.figure(figsize=PAGE)
        fig.text(0.06, 0.95, f"Section {i} of {len(sections)}", fontsize=15, weight="bold")
        fig.text(0.06, 0.92, f"columns {sx + 1}–{sx + w}, rows {sz + 1}–{sz + h}", fontsize=10, color="#444444")
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
        ax.figure.text(0.10, 0.28, "Pieces in this section:", fontsize=9, weight="bold")
        for k, line in enumerate(lines):
            col, row = divmod(k, 12)
            fig.text(0.10 + col * 0.30, 0.25 - row * 0.018, line, fontsize=8)
        pdf.savefig(fig); plt.close(fig)
    return len(sections) + 1


def _ink(color) -> str:
    return "#111111" if sum(color[:3]) / 3 > 0.55 else "#f2f2f2"
