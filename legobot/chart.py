"""Схема панно: детали с цветом и границами, номера рядов и столбцов, легенда по цветам.

Для плоской мозаики из тысяч тайлов пошаговый изометрический PDF бессмыслен (сотни страниц,
сотни мегабайт); собирают такие панно по схеме, как вышивку: ряд за рядом, по номерам.
"""
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

from .colors import load_palette, studio_palette

CELL = 14          # пиксель на клетку
MARGIN = 40        # поле под номера
LEGEND_ROW = 22


def write_chart(bricks: list, path: str) -> int:
    """PNG-схема по плиткам — это верхний слой: 1, когда у панно есть своя подложка, иначе 0.
    Возвращает число плиток."""
    tiles = [b for b in bricks if b.layer == max(x.layer for x in bricks)]
    rgb = {c.code: c.rgb for c in studio_palette(common_only=False)}
    names = {c.code: c.name for c in load_palette(common_only=False)}
    x0, z0 = min(b.x for b in tiles), min(b.z for b in tiles)
    nx = max(b.x + b.width for b in tiles) - x0
    nz = max(b.z + b.length for b in tiles) - z0
    counts = Counter()
    for b in tiles:
        counts[b.color] += b.part.area      # в легенде — клетки цвета, а не число деталей
    legend_h = LEGEND_ROW * len(counts) + 20
    img = Image.new("RGB", (MARGIN + nx * CELL + 10, MARGIN + nz * CELL + legend_h), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for b in tiles:
        px, py = MARGIN + (b.x - x0) * CELL, MARGIN + (b.z - z0) * CELL   # у плоской мозаики z растёт сверху вниз, как строки картинки
        d.rectangle([px, py, px + b.width * CELL - 1, py + b.length * CELL - 1],
                    fill=rgb[b.color], outline=(90, 90, 90))              # граница = граница детали
    for i in range(nx):
        if i % 5 == 0:
            d.text((MARGIN + i * CELL + 2, MARGIN - 14), str(i + 1), fill="black", font=font)
            d.line([MARGIN + i * CELL, MARGIN, MARGIN + i * CELL, MARGIN + nz * CELL], fill=(90, 90, 90))
    for j in range(nz):
        if j % 5 == 0:
            d.text((4, MARGIN + j * CELL + 2), str(j + 1), fill="black", font=font)
            d.line([MARGIN, MARGIN + j * CELL, MARGIN + nx * CELL, MARGIN + j * CELL], fill=(90, 90, 90))
    y = MARGIN + nz * CELL + 10
    for code, n in counts.most_common():
        d.rectangle([MARGIN, y, MARGIN + 16, y + 16], fill=rgb[code], outline="black")
        d.text((MARGIN + 24, y + 3), f"{names.get(code, code)}: {n}", fill="black", font=font)
        y += LEGEND_ROW
    img.save(path)
    return len(tiles)
