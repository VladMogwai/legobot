"""Объёмная фигурка из пиксельной картинки: силуэт «надувается».

Толщина клетки растёт с расстоянием до края силуэта — по штырьку на клетку — и упирается
в максимум; объём симметричен относительно средней плоскости. Край выходит скруглённым, как
у плюшевой игрушки, а не срезанным, как у доски. Работает и для профиля (дракон), и для анфаса.

Поверхность красится цветом пикселя картинки над ней: бок и спина повторяют переднюю грань.
Для профиля это точно то, что нужно (вторая сторона зверя — зеркало первой); для анфаса спина
повторит лицо — честная первая версия.

Внутренние воксели цвета не требуют (ANY_COLOR) — раскладчик кладёт туда что удобнее.
"""
import numpy as np
from scipy import ndimage

from .layout import ANY_COLOR, PlacedBrick, layout_bricks
from .mosaic import Mosaic, _loose_pieces
from .parts import BRICKS
from .voxelize import interior

VOLUME_DEPTH = 8   # максимальная толщина в штырьках (переопределяется --depth)
SLOPE = 1.0        # штырьков толщины на клетку расстояния от края: 1 — скругление под 45°


def inflate(mosaic: Mosaic, max_depth: int = VOLUME_DEPTH) -> tuple[np.ndarray, np.ndarray]:
    """(воксели bool [x, z, y], цвета int той же формы); y — слой 0 внизу, z — глубина."""
    mask = mosaic.codes >= 0
    xs, ys = np.nonzero(mask)
    mask = mask[xs.min():xs.max() + 1, ys.min():ys.max() + 1]
    codes = mosaic.codes[xs.min():xs.max() + 1, ys.min():ys.max() + 1]
    mask, codes = np.flip(mask, axis=1), np.flip(codes, axis=1)      # слой 0 — нижний ряд
    half_max = max(1, max_depth // 2)
    distance = ndimage.distance_transform_edt(mask)                   # до ближайшей пустой клетки, в клетках
    half = np.clip(np.ceil(distance * SLOPE), 1, half_max).astype(int) * mask
    nx, ny = mask.shape
    depth = 2 * half_max
    z = np.arange(depth)
    voxels = np.abs(z[None, :, None] - (depth - 1) / 2) < half[:, None, :]  # [x, z, y]
    colors = np.full(voxels.shape, ANY_COLOR)
    surface = voxels & ~interior(voxels)
    colors[surface] = np.broadcast_to(codes[:, None, :], voxels.shape)[surface]
    return voxels, colors


def volume_bricks(mosaic: Mosaic, max_depth: int = VOLUME_DEPTH) -> list[PlacedBrick]:
    voxels, colors = inflate(mosaic, max_depth)
    body = int(np.bincount(colors[colors >= 0].ravel()).argmax())
    bricks = layout_bricks(voxels, colors, body, BRICKS)
    loose = _loose_pieces(bricks)
    return [b for i, b in enumerate(bricks) if i not in loose]
