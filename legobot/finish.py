"""Отделка: пластины, над которыми ничего нет, заменяются тайлами — гладкие поверхности без штырьков."""
from dataclasses import replace

import numpy as np

from .layout import PlacedBrick
from .parts import PLATE_HEIGHT_LDU, Part

# Тайлы из библиотеки Studio; та же ориентация, что у пластин: «1 x 4» лежит вдоль X.
TILES = {
    (8, 1): Part("4162", 8, 1, PLATE_HEIGHT_LDU),
    (6, 1): Part("6636", 6, 1, PLATE_HEIGHT_LDU),
    (4, 1): Part("2431", 4, 1, PLATE_HEIGHT_LDU),
    (3, 1): Part("63864", 3, 1, PLATE_HEIGHT_LDU),
    (2, 1): Part("3069b", 2, 1, PLATE_HEIGHT_LDU),
    (1, 1): Part("3070b", 1, 1, PLATE_HEIGHT_LDU),
    (6, 6): Part("10202", 6, 6, PLATE_HEIGHT_LDU),
    (4, 2): Part("87079", 4, 2, PLATE_HEIGHT_LDU),
    (3, 2): Part("26603", 3, 2, PLATE_HEIGHT_LDU),
    (2, 2): Part("3068b", 2, 2, PLATE_HEIGHT_LDU),
}


# Размеры, для которых тайла нет: делим вдоль длинной стороны на те, для которых есть.
SPLITS = {(8, 2): ((4, 2), (4, 2)), (6, 2): ((4, 2), (2, 2))}


def tile_exposed_tops(bricks: list[PlacedBrick], occupancy: np.ndarray) -> list[PlacedBrick]:
    """Пластина становится тайлом, если над всей её площадью пусто. 2x6 и 2x8 делятся на два тайла."""
    nlayers = occupancy.shape[2]
    out = []
    for b in bricks:
        exposed = b.layer + 1 >= nlayers or not occupancy[b.x:b.x + b.width, b.z:b.z + b.length, b.layer + 1].any()
        if b.part.height != PLATE_HEIGHT_LDU or not exposed:
            out.append(b)
            continue
        size = (b.part.width, b.part.length)
        if size in TILES:
            out.append(replace(b, part=TILES[size]))
        elif size in SPLITS:
            out.extend(_split(b, SPLITS[size]))
        else:
            out.append(b)
    return out


def _split(b: PlacedBrick, sizes) -> list[PlacedBrick]:
    """Режем вдоль длинной стороны детали (локальная ось X, в мире — X или Z в зависимости от поворота)."""
    pieces, offset = [], 0
    for w, l in sizes:
        if b.rotated:
            pieces.append(replace(b, part=TILES[(w, l)], z=b.z + offset))
        else:
            pieces.append(replace(b, part=TILES[(w, l)], x=b.x + offset))
        offset += w
    return pieces
